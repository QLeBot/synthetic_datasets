"""
SQL Server Reader Module
Handles reading data from SQL Server databases using SQLAlchemy
"""
from sqlalchemy import create_engine, Engine
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv('.env.local')


def get_sql_server_engine(
    server: Optional[str] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    driver: str = "ODBC Driver 17 for SQL Server",
    trusted_connection: bool = False,
    encrypt: bool = True,
    trust_server_certificate: bool = False,
    timeout: int = 30
) -> Engine:
    """
    Create a SQLAlchemy engine for SQL Server
    
    Args:
        server (str): SQL Server instance name or IP address
        database (str): Database name
        username (str): Username for authentication (if not using trusted connection)
        password (str): Password for authentication (if not using trusted connection)
        driver (str): ODBC driver name. Default is "ODBC Driver 17 for SQL Server"
        trusted_connection (bool): Use Windows Authentication if True
        encrypt (bool): Use encryption for the connection (default: True)
        trust_server_certificate (bool): Trust server certificate without validation (default: False)
        timeout (int): Connection timeout in seconds (default: 30)
    
    Returns:
        Engine: SQLAlchemy engine object
    
    Raises:
        ValueError: If required connection parameters are missing
    """
    # Get connection parameters from environment variables if not provided
    server = server or os.getenv("SQL_SERVER")
    database = database or os.getenv("SQL_DATABASE")
    username = username or os.getenv("SQL_USER") or os.getenv("SQL_USERNAME")
    password = password or os.getenv("SQL_PASSWORD")
    
    if not server:
        raise ValueError("SQL Server name is required. Provide it as parameter or set SQL_SERVER environment variable.")
    if not database:
        raise ValueError("Database name is required. Provide it as parameter or set SQL_DATABASE environment variable.")
    
    # Check if we should use Windows Authentication
    use_trusted = trusted_connection or os.getenv("SQL_TRUSTED_CONNECTION", "False").lower() == "true"
    
    # Build connection string matching the working pattern from sqlalchemy.py
    # URL-encode the driver name (spaces become +)
    driver_encoded = driver.replace(' ', '+')
    
    # Build query parameters (matching the working pattern)
    query_params = []
    query_params.append(f"driver={driver_encoded}")
    query_params.append(f"timeout={timeout}")
    query_params.append(f"connect_timeout={timeout}")
    
    # Add encryption and certificate trust parameters only if explicitly set
    # (matching the pattern where these are optional)
    if not use_trusted:
        # Only add Encrypt if explicitly disabled (default behavior may vary)
        if not encrypt:
            query_params.append("Encrypt=no")
        
        # Add TrustServerCertificate if explicitly requested (needed for self-signed certs)
        if trust_server_certificate:
            query_params.append("TrustServerCertificate=yes")
    
    query_string = "&".join(query_params)
    
    # Build connection string
    if use_trusted:
        # Windows Authentication
        conn_str = (
            f"mssql+pyodbc://{server}/{database}"
            f"?{query_string}"
            f"&Trusted_Connection=yes"
        )
    else:
        if not username or not password:
            raise ValueError("Username and password are required for SQL authentication. "
                           "Provide them as parameters or set SQL_USER/SQL_USERNAME and SQL_PASSWORD environment variables. "
                           "Alternatively, set SQL_TRUSTED_CONNECTION=True to use Windows Authentication.")
        
        # SQL Authentication
        conn_str = (
            f"mssql+pyodbc://{username}:{password}"
            f"@{server}/{database}"
            f"?{query_string}"
        )
    
    # Create engine with connection pooling (matching working pattern)
    try:
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,  # Verify connections before using them
            pool_recycle=3600,   # Recycle connections after 1 hour
            connect_args={
                'timeout': timeout,
                'connect_timeout': timeout,
            }
        )
        return engine
    except Exception as e:
        error_msg = str(e)
        # Show connection info (without password) for debugging
        debug_info = f"Server: {server}, Database: {database}"
        if username:
            debug_info += f", Username: {username}"
        
        # Show connection string without password
        debug_conn_str = conn_str.replace(f":{password}@", ":***@") if password else conn_str
        
        if "Login failed" in error_msg or "authentication" in error_msg.lower():
            if use_trusted:
                raise Exception(
                    f"Failed to connect to SQL Server using Windows Authentication.\n"
                    f"Error: {error_msg}\n"
                    f"{debug_info}\n"
                    f"Connection string: {debug_conn_str}\n"
                    f"Tip: Your Windows account may not have access. Try SQL authentication with --username and --password."
                )
            else:
                raise Exception(
                    f"Failed to connect to SQL Server using SQL Authentication.\n"
                    f"Error: {error_msg}\n"
                    f"{debug_info}\n"
                    f"Connection string: {debug_conn_str}\n"
                    f"Tip: Verify credentials."
                )
        elif "certificate" in error_msg.lower() or "ssl" in error_msg.lower() or "certificat" in error_msg.lower():
            raise Exception(
                f"Connection SSL/certificate error.\n"
                f"Error: {error_msg}\n"
                f"{debug_info}\n"
                f"Connection string: {debug_conn_str}\n"
                f"Tip: Try using --trust-cert flag to trust the server certificate."
            )
        else:
            raise Exception(f"Failed to create SQL Server connection: {error_msg}\n{debug_info}\nConnection string: {debug_conn_str}")


def read_table(
    table_name: str,
    schema: str = "dbo",
    server: Optional[str] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    query: Optional[str] = None,
    engine: Optional[Engine] = None,
    trusted_connection: bool = False,
    encrypt: bool = True,
    trust_server_certificate: bool = False,
    **connection_kwargs
) -> pd.DataFrame:
    """
    Read a table from SQL Server into a pandas DataFrame
    
    Args:
        table_name (str): Name of the table to read
        schema (str): Schema name (default: "dbo")
        server (str): SQL Server instance name or IP address (optional if using env vars)
        database (str): Database name (optional if using env vars)
        username (str): Username for authentication (optional if using env vars or trusted connection)
        password (str): Password for authentication (optional if using env vars or trusted connection)
        query (str): Custom SQL query. If provided, table_name and schema are ignored
        engine (Engine): Existing SQLAlchemy engine. If provided, uses this instead of creating new one
        trusted_connection (bool): Use Windows Authentication (optional)
        encrypt (bool): Use encryption for the connection (default: True)
        trust_server_certificate (bool): Trust server certificate (default: False)
        **connection_kwargs: Additional arguments to pass to get_sql_server_engine()
    
    Returns:
        pd.DataFrame: DataFrame containing the table data
    """
    # Use existing engine or create new one
    if engine is None:
        engine = get_sql_server_engine(
            server=server,
            database=database,
            username=username,
            password=password,
            trusted_connection=trusted_connection,
            encrypt=encrypt,
            trust_server_certificate=trust_server_certificate,
            **connection_kwargs
        )
        close_engine = False
    else:
        close_engine = False  # Don't close provided engine
    
    try:
        # Use custom query if provided, otherwise build SELECT query
        if query:
            sql_query = query
        else:
            sql_query = f"SELECT * FROM [{schema}].[{table_name}]"
        
        # Read data into DataFrame using SQLAlchemy engine
        df = pd.read_sql(sql_query, engine)
        return df
    
    except Exception as e:
        raise Exception(f"Error reading table: {str(e)}")
    
    finally:
        # Note: We don't close the engine here as it manages its own connection pool
        # The engine will be garbage collected when it goes out of scope
        pass
