from typing import Any, Dict, Union, List
import os 
import logging
import MySQLdb  
from mcp.server.fastmcp import FastMCP
import json
from anthropic import Anthropic
from datetime import date, datetime
from decimal import Decimal
import traceback
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("sql_server_detailed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_mysql_server")

load_dotenv() 


mcp = FastMCP("mysql-server")

# Database connection configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "passwd": os.getenv("DB_PASSWORD", "your_password"), 
    "db": os.getenv("DB_NAME", "your_database"),  
    "port": int(os.getenv("DB_PORT", 3306))
}


api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    logger.warning("ANTHROPIC_API_KEY environment variable not set")
anthropic_client = Anthropic(api_key=api_key)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super(CustomJSONEncoder, self).default(obj)


def serialize_data(data: Any) -> Any:
    """
    Recursively serialize data to ensure JSON compatibility.
    Handles dates, decimals, etc.
    """
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_data(item) for item in data]
    elif isinstance(data, (date, datetime)):
        return data.isoformat()
    elif isinstance(data, Decimal):
        return float(data)
    elif hasattr(data, '__dict__'):
        return serialize_data(data.__dict__)
    else:
        return data

def get_connection():
    try:
        connection = MySQLdb.connect(**DB_CONFIG)
        return connection
    except MySQLdb.Error as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise

# Checking for unsafe queries, which can be used to edit the database
def is_safe_query(sql: str) -> bool:
    """Basic check for potentially unsafe queries"""
    sql_lower = sql.lower()
    unsafe_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    is_safe = not any(keyword in sql_lower for keyword in unsafe_keywords)
    
    if not is_safe:
        found_keywords = [kw for kw in unsafe_keywords if kw in sql_lower]
        logger.warning(f"Unsafe SQL detected - found keywords: {found_keywords} in query: {sql}")
    
    return is_safe


### Resources implementation
# Schema of the database
@mcp.resource("mysql://schema")
async def get_schema() -> Dict[str, Any]:
    """Provide database table structure information"""
    logger.info("Resource requested: mysql://schema")
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Get all table names
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [list(table.values())[0] for table in tables]

        # Get structure for each table
        schema = {}
        for table_name in table_names:
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            table_schema = []
            
            for column in columns:
                table_schema.append({
                    "name": column["Field"],
                    "type": column["Type"],
                    "null": column["Null"],
                    "key": column["Key"],
                    "default": column["Default"],
                    "extra": column["Extra"]
                })
            
            # Add table statistics (row count)
            cursor.execute(f"SELECT COUNT(*) as row_count FROM `{table_name}`")
            count_result = cursor.fetchone()
            row_count = count_result['row_count'] if count_result else 0
            
            # Get table indexes
            cursor.execute(f"SHOW INDEX FROM `{table_name}`")
            indexes = cursor.fetchall()
            index_info = []
            for idx in indexes:
                index_info.append({
                    "name": idx["Key_name"],
                    "column": idx["Column_name"],
                    "non_unique": idx["Non_unique"],
                    "index_type": idx["Index_type"]
                })
            
            schema[table_name] = {
                "columns": table_schema,
                "row_count": row_count,
                "indexes": index_info
            }
        
        return {
            "database": DB_CONFIG["db"],
            "tables": schema
        }
    except Exception as e:
        logger.error(f"Error while retrieving schema: {str(e)}", exc_info=True)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Tables list
@mcp.resource("mysql://tables")
async def get_tables() -> Dict[str, Any]:
    """Provide database table list"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [list(table.values())[0] for table in tables]
        
        return {
            "database": DB_CONFIG["db"],
            "tables": table_names
        }
    except Exception as e:
        logger.error(f"Error while retrieving tables: {str(e)}", exc_info=True)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


### Tools implementation
# Create a plan for database queries
@mcp.tool()
async def create_query_plan(query: str) -> Dict[str, Any]:
    """
    Tool to create a structured plan for answering user queries.
    
    Args:
        query (str): The user's natural language query about the database
    
    Returns:
        Dict[str, Any]: A structured plan with steps to answer the query
    """
    
    # Get database schema to inform the planning
    try:
        # logger.debug("Fetching schema data for planning")
        schema_data = await get_schema()
        tables_data = await get_tables()
        # logger.debug(f"Schema fetched successfully for {len(schema_data['tables'])} tables")
    except Exception as e:
        logger.error(f"Error fetching schema for planning: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to fetch database schema: {str(e)}"
        }
    
    conn = None
    cursor = None
    
    try:
        # Create connection
        conn = get_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Get sample data from each table (just 3 rows max)
        sample_data = {}
        for table_name in tables_data["tables"]:
            try:
                # logger.debug(f"Fetching sample data from table: {table_name}")
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
                sample_rows = cursor.fetchall()
                if sample_rows:
                    # logger.debug(f"Got {len(sample_rows)} sample rows from {table_name}")
                    sample_data[table_name] = sample_rows
            except Exception as e:
                logger.warning(f"Could not get sample data for table {table_name}: {str(e)}")
        
        # Format database information for the LLM
        # logger.debug("Preparing database information for planning")
        db_info = {
            "database": schema_data["database"],
            "tables": schema_data["tables"],
            "table_names": tables_data["tables"],
            "sample_data": serialize_data(sample_data)
        }
        
        prompt = f"""
        You are a database query planning assistant. Given a user query and database schema, 
        create a detailed step-by-step plan to answer the query.
        
        Database Information:
        Database Name: {db_info['database']}
        Available Tables: {', '.join(db_info['table_names'])}
        
        Schema Details (table structure, row counts, and indexes):
        {json.dumps(serialize_data(db_info['tables']), indent=2, cls=CustomJSONEncoder)}
        
        Sample Data (up to 3 rows per table):
        {json.dumps(db_info['sample_data'], indent=2, cls=CustomJSONEncoder)}
        
        User Query: {query}
        
        IMPORTANT: Do not attempt to load entire tables. For large tables with many rows, use appropriate 
        filtering, aggregations, or LIMIT clauses.

        Wherever you need to create an in-detailed report, towards the end after curating all the results after the actions, 
        create a final summary of the results like a report and make sure to mentioned about this in the analysis plan.
        
        Please create a specific plan with these components:
        1. Query Understanding: Clarify what the user is asking for
        2. Data Needs: What tables and columns will be needed
        3. SQL Query Plan: What SQL queries should be executed (in order if multiple are needed)
        4. Performance Considerations: How to ensure queries are efficient with large tables
        5. Analysis Plan: How to interpret and present the results
        
        Format your response as a JSON structure with these keys:
        - understanding: A summary of what the user is asking
        - required_data: List of tables and columns needed
        - sql_queries: List of SQL queries to execute (with explanations)
        - performance_notes: How queries are optimized for large tables
        - analysis_steps: How to interpret and format the results
        """
        # logger.debug("Prompt prepared for planning")
    except Exception as e:
        logger.error(f"Error preparing planning data: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to prepare planning data: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
   
    
    try:
        
        # Call Anthropic API to generate the plan
        # logger.info("Calling Anthropic API for plan generation")
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4098,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        logger.debug(f"Input tokens : {response.usage.input_tokens}")
        logger.debug(f"Ouput tokens : {response.usage.output_tokens}")

        
        plan_text = response.content[0].text
        
        result = {
            "success": True,
            "plan": plan_text 
        }
        
        
        return result
        
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error creating query plan: {str(e)}", exc_info=True)
        logger.error(f"Error details: {error_details}")
        return {
            "success": False,
            "error": f"Failed to create query plan: {str(e)}"
        }

# Tool for executing read-only SQL queries
@mcp.tool()
async def query_data(sql: str) -> Dict[str, Any]:
    """
    Tool to execute read-only SQL queries.

    Args:
        sql (str): SQL query to execute
    
    Returns:
        Dict[str, Any]: Query result
    
    """    
    if not is_safe_query(sql):
        logger.warning(f"Unsafe query rejected: {sql}")
        return {
            "success": False,
            "error": "Potentially unsafe query detected. Only SELECT queries are allowed."
        }
    

    conn = None
    cursor = None
    execution_start_time = datetime.now()
    
    try:
        # Create connection
        conn = get_connection()
        # Create dictionary cursor
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        
        # Start read-only transaction
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("START TRANSACTION")
        
        try:
            # Execute with timeout to prevent long-running queries
            cursor.execute("SET SESSION MAX_EXECUTION_TIME=10000")  # 10 second timeout
            
            cursor.execute(sql)
            
            results = cursor.fetchall()

            execution_end_time = datetime.now()

            conn.commit()
            
            # Convert results to serializable format and include metadata
            serialized_results = serialize_data(results)
            result_count = len(results)
            truncated = result_count >= 200
            

            return {
                "success": True,
                "results": serialized_results,
                "rowCount": result_count,
                "truncated": truncated,
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"Error executing query: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Database connection error in query_data: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Database connection error: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as e:
        logger.critical(f"Fatal error starting server: {str(e)}", exc_info=True)