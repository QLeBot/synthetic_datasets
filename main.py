"""
Main Workflow: Generate Synthetic Dataset from SQL Server Table

This script:
1. Reads data from a SQL Server table
2. Analyzes patterns for each column (min, max, avg, mean, occurrence)
3. Generates synthetic data based on those patterns
"""
import os
import pandas as pd
from dotenv import load_dotenv
import argparse

from src.sql_server_reader import read_table
from src.pattern_analyzer import analyze_dataframe_patterns, print_pattern_summary, save_patterns_to_file
from src.data_generator import generate_synthetic_data

load_dotenv('.env.local')


def generate_synthetic_dataset_from_sql(
    table_name: str,
    schema: str = "dbo",
    num_samples: int = 1000,
    output_path: str = "output/synthetic_dataset.csv",
    server: str = None,
    database: str = None,
    username: str = None,
    password: str = None,
    trusted_connection: bool = False,
    encrypt: bool = True,
    trust_server_certificate: bool = False,
    query: str = None,
    show_patterns: bool = True,
    save_source: bool = False,
    save_patterns: bool = True,
    patterns_format: str = 'json'
) -> pd.DataFrame:
    """
    Main workflow: Read from SQL Server, analyze patterns, generate synthetic data
    
    Args:
        table_name: Name of the SQL Server table to read
        schema: Schema name (default: "dbo")
        num_samples: Number of synthetic samples to generate
        output_path: Path to save the synthetic dataset
        server: SQL Server instance (optional, uses SQL_SERVER env var if not provided)
        database: Database name (optional, uses SQL_DATABASE env var if not provided)
        username: SQL Server username (optional, uses SQL_USER/SQL_USERNAME env var if not provided)
        password: SQL Server password (optional, uses SQL_PASSWORD env var if not provided)
        trusted_connection: Use Windows Authentication (optional, uses SQL_TRUSTED_CONNECTION env var)
        query: Custom SQL query (optional, overrides table_name/schema)
        show_patterns: Whether to print pattern analysis summary
        save_source: Whether to save the source data to a file
        save_patterns: Whether to save pattern analysis results to a file
        patterns_format: Format for patterns file - 'json' or 'txt' (default: 'json')
    
    Returns:
        DataFrame containing synthetic data
    """
    print("="*80)
    print("SYNTHETIC DATASET GENERATOR")
    print("="*80)
    
    # Step 1: Read data from SQL Server
    print(f"\n[Step 1/3] Reading data from SQL Server...")
    print(f"  Table: {schema}.{table_name}" if not query else f"  Query: {query[:50]}...")
    
    try:
        source_data = read_table(
            table_name=table_name,
            schema=schema,
            server=server,
            database=database,
            username=username,
            password=password,
            query=query,
            trusted_connection=trusted_connection,
            encrypt=encrypt,
            trust_server_certificate=trust_server_certificate
        )
        print(f"  ✓ Loaded {len(source_data)} rows, {len(source_data.columns)} columns")
        
        if save_source:
            source_path = output_path.replace('.csv', '_source.csv')
            os.makedirs(os.path.dirname(source_path) if os.path.dirname(source_path) else '.', exist_ok=True)
            source_data.to_csv(source_path, index=False)
            print(f"  ✓ Source data saved to: {source_path}")
    
    except Exception as e:
        print(f"  ✗ Error reading from SQL Server: {str(e)}")
        raise
    
    # Step 2: Analyze patterns
    print(f"\n[Step 2/3] Analyzing data patterns...")
    patterns = analyze_dataframe_patterns(source_data)
    print(f"  ✓ Analyzed patterns for {len(patterns)} columns")
    
    if show_patterns:
        print_pattern_summary(patterns)
    
    # Save patterns to file
    if save_patterns:
        patterns_path = output_path.rsplit('.', 1)[0] + '_patterns.' + patterns_format
        save_patterns_to_file(patterns, patterns_path, format=patterns_format)
        print(f"  ✓ Pattern analysis saved to: {patterns_path}")
    
    # Step 3: Generate synthetic data
    print(f"\n[Step 3/3] Generating {num_samples} synthetic samples...")
    synthetic_data = generate_synthetic_data(source_data, num_samples, patterns)
    print(f"  ✓ Generated {len(synthetic_data)} synthetic rows")
    
    # Save synthetic data (support both CSV and Parquet)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if output_path.lower().endswith('.parquet'):
        try:
            synthetic_data.to_parquet(output_path, index=False, engine='pyarrow')
            print(f"  ✓ Synthetic data saved to: {output_path} (Parquet format)")
        except ImportError:
            print(f"  ✗ Error: pyarrow is required for Parquet format. Install it with: pip install pyarrow")
            print(f"  Saving as CSV instead...")
            csv_path = output_path.rsplit('.', 1)[0] + '.csv'
            synthetic_data.to_csv(csv_path, index=False)
            print(f"  ✓ Synthetic data saved to: {csv_path} (CSV format)")
    else:
        # Default to CSV if extension not specified or not parquet
        if not output_path.lower().endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
        synthetic_data.to_csv(output_path, index=False)
        print(f"  ✓ Synthetic data saved to: {output_path} (CSV format)")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    
    return synthetic_data


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset from SQL Server table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables
  python main.py --table Customers --schema Sales --samples 5000
  
  # Using explicit connection parameters
  python main.py --table Customers --server localhost --database MyDB --username user --password pass
  
  # Using custom query
  python main.py --query "SELECT TOP 1000 * FROM Sales.Customers WHERE Age > 25"
        """
    )
    
    parser.add_argument('--table', type=str, help='Table name to read from')
    parser.add_argument('--schema', type=str, default='dbo', help='Schema name (default: dbo)')
    parser.add_argument('--query', type=str, help='Custom SQL query (overrides table/schema)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='output/synthetic_dataset.csv', help='Output file path (supports .csv or .parquet)')
    parser.add_argument('--server', type=str, help='SQL Server instance (or use SQL_SERVER env var)')
    parser.add_argument('--database', type=str, help='Database name (or use SQL_DATABASE env var)')
    parser.add_argument('--username', type=str, help='SQL Server username (or use SQL_USER/SQL_USERNAME env var)')
    parser.add_argument('--password', type=str, help='SQL Server password (or use SQL_PASSWORD env var)')
    parser.add_argument('--trusted-connection', action='store_true', help='Use Windows Authentication (or set SQL_TRUSTED_CONNECTION=True)')
    parser.add_argument('--no-encrypt', action='store_true', help='Disable encryption for SQL Server connection')
    parser.add_argument('--trust-cert', action='store_true', help='Trust server certificate (for self-signed certificates)')
    parser.add_argument('--no-patterns', action='store_true', help='Do not print pattern analysis summary')
    parser.add_argument('--no-save-patterns', action='store_true', help='Do not save pattern analysis to file')
    parser.add_argument('--patterns-format', type=str, choices=['json', 'txt'], default='json', help='Format for patterns file (default: json)')
    parser.add_argument('--save-source', action='store_true', help='Save source data to file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.query and not args.table:
        parser.error("Either --table or --query must be provided")
    
    try:
        generate_synthetic_dataset_from_sql(
            table_name=args.table or '',
            schema=args.schema,
            num_samples=args.samples,
            output_path=args.output,
            server=args.server,
            database=args.database,
            username=args.username,
            password=args.password,
            trusted_connection=args.trusted_connection,
            encrypt=not args.no_encrypt,
            trust_server_certificate=args.trust_cert,
            query=args.query,
            show_patterns=not args.no_patterns,
            save_patterns=not args.no_save_patterns,
            patterns_format=args.patterns_format,
            save_source=args.save_source
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
