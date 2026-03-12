from sqlalchemy import create_engine, event
from .config import Settings

def get_sqlalchemy_engine(settings: Settings, database: str, timeout: int = 300, query_timeout: int = 0):
    """
    Create SQLAlchemy engine with connection timeout and retry settings.
    
    Args:
        settings: Settings object
        database: Database name
        timeout: Connection timeout in seconds (default: 300 = 5 minutes)
        query_timeout: Query execution timeout in seconds (default: 0 = no timeout)
                      Use 0 for no timeout, or set a value in seconds (e.g., 300 for 5 minutes)
    
    Returns:
        SQLAlchemy engine with timeout and retry configuration
    """
    driver = "ODBC Driver 17 for SQL Server"
    
    # Connection timeout parameters
    # timeout: Total timeout for the connection attempt
    # connect_timeout: Timeout for establishing the connection
    # QueryTimeout: Timeout for individual queries in seconds (0 = no timeout)
    # Default to 0 (no timeout) for large queries, but allow override
    
    timeout_params = (
        f"&timeout={timeout}"
        f"&connect_timeout={timeout}"
        f"&QueryTimeout={query_timeout}"
    )
    
    # Also set in connect_args for pyodbc
    pyodbc_timeout_args = {
        'timeout': timeout,
        'connect_timeout': timeout,
    }

    if settings.sql_user and settings.sql_password:
        conn_str = (
            f"mssql+pyodbc://{settings.sql_user}:{settings.sql_password}"
            f"@{settings.sql_server}/{database}"
            f"?driver={driver.replace(' ', '+')}"
            f"{timeout_params}"
        )
        # Debug: show connection string without password
        debug_conn_str = (
            f"mssql+pyodbc://{settings.sql_user}:***"
            f"@{settings.sql_server}/{database}"
            f"?driver={driver.replace(' ', '+')}"
            f"{timeout_params}"
        )
    else:
        conn_str = (
            f"mssql+pyodbc://{settings.sql_server}/{database}"
            f"?driver={driver.replace(' ', '+')}"
            f"&Trusted_Connection=yes"
            f"{timeout_params}"
        )
        debug_conn_str = conn_str
    
    # Debug output
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Connection string: {debug_conn_str}")

    # Create engine
    engine = create_engine(
        conn_str,
        pool_pre_ping=True,  # Verify connections before using them
        pool_recycle=3600,   # Recycle connections after 1 hour
        connect_args=pyodbc_timeout_args
    )
    
    # Set QueryTimeout directly on pyodbc connection using event listener
    # This ensures QueryTimeout is actually applied even if connection string doesn't work
    @event.listens_for(engine, "connect")
    def set_query_timeout(dbapi_conn, connection_record):
        """Set QueryTimeout directly on the pyodbc connection."""
        try:
            import pyodbc
            # For SQLAlchemy with pyodbc, dbapi_conn should be the pyodbc connection
            if isinstance(dbapi_conn, pyodbc.Connection):
                dbapi_conn.timeout = query_timeout
            elif hasattr(dbapi_conn, 'timeout'):
                dbapi_conn.timeout = query_timeout
        except Exception:
            # Silently fail - QueryTimeout setting is optional
            pass
    
    return engine