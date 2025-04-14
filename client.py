import asyncio
import logging
import os
import sys
import json
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from mcp.types import Resource


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

logger = logging.getLogger("mcp_mysql_client")



class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return str(obj)


def safe_json_dumps(obj, **kwargs):
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)
    except Exception as e:
        return f"<Non-serializable object: {str(obj)[:100]}... (Error: {str(e)})>"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client_detailed_trace.log"),
        logging.FileHandler("all_logs.log"),  
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("mcp_client")

load_dotenv()  

class MCPClient:
    def __init__(self):

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

        self.conversation = []

        self.resources = []
        self.added_resources = False


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # logger.info(f"Connecting to server: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            logger.error("Server script must be a .py or .js file")
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # logger.debug(f"Setting up stdio transport with params: {command} {server_script_path}")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # logger.debug("Initializing session")
        init_result = await self.session.initialize()
        # logger.debug(f"Session initialized with result: {safe_json_dumps(init_result)}")


        ### TOOLS

        # List available tools
        # logger.debug("Fetching available tools")
        response = await self.session.list_tools()
        tools = response.tools
        
        # Log detailed tool information
        # logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")


        ### RESOURCES
        # Preload resources
        # logger.info("Preloading resources...")
        resources = await self.session.list_resources()


        resources_from_server = resources.resources
        resource_content = {}
        for resource in resources_from_server:
            # logger.debug(f"Loaded resource: {resource.name}")
            # logger.debug(f"Resource: {safe_json_dumps(resource)}")

            resource_data = await self.session.read_resource(resource.name)
            # logger.debug(f"Resource data: {safe_json_dumps(resource_data)}")
            resource_content[resource.name] = resource_data.contents
            
        
        # logger.debug(f"Resource content: {safe_json_dumps(resource_content)}")

        self.resources = resource_content



    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        if not self.added_resources:
            # Add resource context to messages
            resource_context = "\nResources:\n" + "\n".join([f"- {resource} : {self.resources[resource]}" for resource in self.resources.keys()])
            self.conversation.append({
                "role": "user",
                "content": resource_context
            })
            self.added_resources = True
            
        # logger.info(f"Processing query: {query}")
        messages = self.conversation.copy()
        # Add user query to conversation history
        user_message = {
            "role": "user",
            "content": query
        }
        self.conversation.append(user_message)
        messages.append(user_message)

        # logger.debug("Fetching available tools")
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        # logger.debug(f"Available tools for Claude: {safe_json_dumps(available_tools)}")

        logger.debug("Making initial Claude API call")
        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            logger.debug(f"Input tokens : {response.usage.input_tokens}")
            logger.debug(f"Ouput tokens : {response.usage.output_tokens}")


            logger.debug(f"Claude response received: {len(response.content)} content items")
            logger.debug(f"Claude response received: {response.content}")
        except Exception as e:
            logger.error(f"Error in Claude API call: {str(e)}", exc_info=True)
            raise

        final_text = []
        final_answer = False
        
        while not final_answer:
            assistant_message_content = []
            has_tool_call = False
            
            for content in response.content:
                if content.type == 'text':
                    # logger.debug(f"Processing text response from Claude: {content.text[:100]}...")
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                    final_answer = True  # We have text content, which might be a final answer
                    
                elif content.type == 'tool_use':
                    has_tool_call = True
                    final_answer = False  # Reset since we're still in tool calls
                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call
                    logger.debug(f"Executing tool call: {tool_name}")
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        # logger.debug(f"Tool call result received for {tool_name}: {safe_json_dumps(result)}")
                        result_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                        # logger.debug(f"Tool call result received for {tool_name}: {result_preview}")
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
                        result = type('obj', (object,), {'content': f"Error: {str(e)}"})
                    
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    assistant_message_content.append(content)
                    
                    # Add the assistant message to both messages and conversation history
                    assistant_message = {
                        "role": "assistant",
                        "content": assistant_message_content.copy()  # Important: create a copy
                    }
                    messages.append(assistant_message)
                    self.conversation.append(assistant_message)

                    # Add the tool result to both messages and conversation history
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    }
                    messages.append(tool_result_message)
                    self.conversation.append(tool_result_message)
            
            if has_tool_call:
                # Get next response from Claude
                logger.debug("Making follow-up Claude API call with tool results")
                try:

                    response = self.anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4098,
                        messages=messages,
                        tools=available_tools
                    )


                    logger.debug(f"Input tokens : {response.usage.input_tokens}")
                    logger.debug(f"Ouput tokens : {response.usage.output_tokens}")

                
                    logger.debug(f"Follow-up Claude response received: {len(response.content)} content items")
                    logger.debug(f"Follow-up Claude response content: {safe_json_dumps(response.content)}")

                except Exception as e:
                    logger.error(f"Error in follow-up Claude API call: {str(e)}", exc_info=True)
                    raise
            else:
                break
        
        if final_answer and len(response.content) > 0 and response.content[0].type == 'text':
            last_response_text = response.content[0].text
            if last_response_text not in final_text:
                final_text.append(last_response_text)
                
            if not has_tool_call:
                final_assistant_message = {
                    "role": "assistant",
                    "content": [content for content in response.content]
                }
                self.conversation.append(final_assistant_message)

        final_response = "\n".join(final_text)
        return final_response

    async def chat_loop(self):
        """Run an interactive chat loop"""
        # logger.info("Starting chat loop")
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                print(f"User input received: {query}")
                logger.debug(f"User input received: {query}")

                if query.lower() == 'quit':
                    break
                    
                if query.lower() == 'clear history':
                    self.clear_conversation_history()
                    print("Conversation history cleared.")
                    continue

                logger.debug(f"Processing user query: {query}")
                response = await self.process_query(query)
                logger.debug("Displaying response to user")
                print("\n" + response)

            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        logger.debug("Resources cleaned up successfully")
        
    def get_conversation_history(self):
        """Get the current conversation history"""
        return self.conversation
        
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation = []
        self.added_resources = False  # Reset so resources will be added again


async def main():

    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    client = MCPClient()

    # Get server path
    server_path = "Downloads/MCP/mcp-mysql-server/sql_server.py"
    if len(sys.argv) > 1:
        server_path = sys.argv[1]
    
    server_path = os.path.abspath(server_path)

    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        await client.cleanup()
        # logger.info("Application terminated")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)