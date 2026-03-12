# Synthetic Datasets Generator

A Python project to generate synthetic datasets based on existing data from SQL Server. The tool analyzes patterns in real data (statistical measures, value occurrences, etc.) and generates realistic synthetic data that closely matches the original data characteristics.

## Goal

Generate synthetic datasets based on existing ones by:
1. **Reading data** from SQL Server tables
2. **Analyzing patterns** for each column (min, max, avg, mean, occurrence frequencies)
3. **Generating synthetic data** that closely matches the real data patterns

## Features

- 📊 **Statistical Pattern Analysis**: Analyzes min, max, mean, median, std dev, quartiles, skewness, and kurtosis for numeric columns
- 📈 **Occurrence Patterns**: Tracks value frequencies and distributions for categorical/text columns
- 🔍 **Pattern Detection**: Identifies common prefixes/suffixes and data type patterns
- 🎲 **Realistic Generation**: Uses statistical distributions to generate values close to real data
- 🔐 **SQL Server Integration**: Reads directly from SQL Server with Windows or SQL authentication
- 📁 **Flexible Output**: Saves synthetic data to CSV files

## Project Structure

```
synthetic_datasets/
├── src/
│   ├── __init__.py
│   ├── sql_server_reader.py    # SQL Server connection and data reading
│   ├── pattern_analyzer.py     # Pattern analysis (stats, occurrences)
│   └── data_generator.py        # Synthetic data generation
├── data/                        # Data storage (gitignored)
├── input/                       # Input files (gitignored)
├── output/                      # Generated synthetic datasets (gitignored)
├── main.py                      # Main workflow script
├── .env.local                   # Environment variables (gitignored)
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
```bash
pip install pandas numpy pyodbc python-dotenv scipy scikit-learn
```

3. **Set up environment variables**:
   
   Create a `.env.local` file in the project root with your SQL Server credentials:
   ```env
   # SQL Server Connection
   SQL_SERVER=your_server_name
   SQL_DATABASE=your_database_name
   SQL_USERNAME=your_username
   SQL_PASSWORD=your_password
   
   # OR for Windows Authentication:
   SQL_TRUSTED_CONNECTION=True
   ```

## Usage

### Command Line Interface

**Basic usage** (using environment variables):
```bash
python main.py --table Customers --schema Sales --samples 5000
```

**With explicit connection parameters**:
```bash
python main.py --table Customers --server localhost --database MyDB --username user --password pass --samples 1000
```

**Using custom SQL query**:
```bash
python main.py --query "SELECT TOP 1000 * FROM Sales.Customers WHERE Age > 25" --samples 2000
```

**Save source data and suppress pattern output**:
```bash
python main.py --table Customers --save-source --no-patterns --output output/my_synthetic_data.csv
```

### Python API

```python
from main import generate_synthetic_dataset_from_sql

# Generate synthetic data
synthetic_df = generate_synthetic_dataset_from_sql(
    table_name="Customers",
    schema="Sales",
    num_samples=5000,
    output_path="output/synthetic_customers.csv"
)
```

### Direct Module Usage

```python
from src.sql_server_reader import read_table
from src.pattern_analyzer import analyze_dataframe_patterns
from src.data_generator import generate_synthetic_data

# Read from SQL Server
source_data = read_table("Customers", schema="Sales")

# Analyze patterns
patterns = analyze_dataframe_patterns(source_data)

# Generate synthetic data
synthetic_data = generate_synthetic_data(source_data, num_samples=1000, patterns=patterns)
```

## Visualize generated coordinates (Streamlit)

If you're using the coordinate generator in `src/coords_generator.py`, you can visualize points on an interactive map with Streamlit:

```bash
streamlit run streamlit_app.py
```

## How It Works

### 1. Data Reading
- Connects to SQL Server using pyodbc
- Reads table data into pandas DataFrame
- Supports Windows Authentication or SQL Authentication

### 2. Pattern Analysis
For each column, the tool analyzes:

**Numeric Columns:**
- Min, Max, Mean, Median
- Standard Deviation
- Quartiles (Q25, Q75)
- Skewness and Kurtosis

**Categorical/Text Columns:**
- Value occurrence frequencies
- Most common values
- Common prefixes/suffixes
- String length statistics

**Date Columns:**
- Date range
- Min/Max dates

### 3. Synthetic Data Generation
- **Numeric**: Uses normal distribution (with clipping) or uniform distribution based on statistics
- **Categorical**: Samples from value frequency distribution
- **Text**: Uses common patterns and frequency distributions
- **Dates**: Random dates within the observed range
- **Nulls**: Preserves original null percentage

## Command Line Options

```
--table TABLE          Table name to read from
--schema SCHEMA        Schema name (default: dbo)
--query QUERY          Custom SQL query (overrides table/schema)
--samples N            Number of synthetic samples (default: 1000)
--output PATH          Output file path (default: output/synthetic_dataset.csv)
--server SERVER        SQL Server instance (or use SQL_SERVER env var)
--database DATABASE    Database name (or use SQL_DATABASE env var)
--username USER        SQL Server username (or use SQL_USERNAME env var)
--password PASS        SQL Server password (or use SQL_PASSWORD env var)
--no-patterns          Do not print pattern analysis summary
--save-source          Save source data to file
```

## Requirements

- Python 3.7+
- pandas
- numpy
- pyodbc
- python-dotenv
- scipy (for statistical functions)
- scikit-learn (for evaluation tools)

## Examples

### Example 1: Generate 10,000 synthetic customer records
```bash
python main.py --table Customers --schema Sales --samples 10000 --output output/synthetic_customers.csv
```

### Example 2: Generate from filtered query
```bash
python main.py --query "SELECT * FROM Sales.Orders WHERE OrderDate > '2023-01-01'" --samples 5000
```

### Example 3: Save both source and synthetic data
```bash
python main.py --table Products --save-source --output output/synthetic_products.csv
```

## Output

The tool generates:
- **Synthetic dataset CSV**: Contains generated data matching original patterns
- **Pattern summary** (optional): Console output showing analyzed patterns
- **Source data CSV** (optional): Original data if `--save-source` is used

## Notes

- The generated data maintains statistical properties of the original data
- Value distributions are preserved for categorical columns
- Null percentages are maintained
- Generated data is suitable for testing, development, and privacy-preserving analytics

## License

This project is for educational and development purposes.
