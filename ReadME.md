# MCP MySQL Server & Client

A connecteor between a MySQL database and LLMs, allowing for natural language querying of SQL databases using the Model Context Protocol.

## Overview

This repo provides a set of tools that enable natural language queries against MySQL databases. It consists of two main components:

1. **MCP MySQL Server**: A FastMCP server that provides tools for executing read-only SQL queries and planning database operations.
2. **MCP Client**: A command-line interface that connects to the server and uses LLM API (Ex :- Claude) to interpret natural language queries.


## Features

- Database schema introspection for intelligent query planning
- Read-only query execution with safety checks
- Support for complex, multi-step database operations
- Currently, this implementation uses **local mysql database and communicates via stdio between the mcp server and client**



## Installation

1. Clone the repository

2. Install required packages:

    We use uv to install and run the applications, you are free to choose anything of your choice.

   ```
   uv pip install -r requirements.txt

   pip install MySQL-python
   pip install mysqlclient
   ```


3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   DB_HOST=your_database_host
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_NAME=your_database_name
   DB_PORT=3306
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Starting the Server

Run the MySQL Server component:

```
uv run server_mysql.py
```

### Running the Client

Start the client and connect to the server:

```
uv run client.py
```


### Adding New Features

To add new tools/resources or resources to the server:

1. To define new or change the exsiting ones, use the `@mcp.tool()` or `@mcp.resource()` decorators
2. Implement the required functionality
3. Restart the server for changes to take effect

In this way, only the server will need to be maintained and client can plug in without any changes. 