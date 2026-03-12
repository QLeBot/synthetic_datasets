"""
Pattern Analyzer Module
Analyzes data patterns including statistical measures (min, max, avg, mean) and occurrence patterns
"""
import pandas as pd
import numpy as np
import json
import warnings
from typing import Dict, Tuple, Any, Optional
from collections import Counter
import os

# Suppress pandas date parsing warnings globally
warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Parsed string.*timezone.*')


def analyze_column_patterns(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze patterns for a single column including statistical measures and occurrence patterns
    
    Args:
        df: DataFrame containing the data
        column: Column name to analyze
    
    Returns:
        Dictionary containing pattern information:
        - data_type: pandas dtype
        - pattern_type: 'numeric', 'date', 'categorical', 'text'
        - stats: Statistical measures (min, max, mean, median, std, etc.)
        - occurrence: Value occurrence patterns for categorical/text data
        - common_pattern: Common prefix/suffix pattern
        - null_count: Number of null values
        - unique_count: Number of unique values
    """
    series = df[column]
    pattern_info = {
        'column_name': column,
        'data_type': str(series.dtype),
        'null_count': series.isnull().sum(),
        'null_percentage': (series.isnull().sum() / len(series)) * 100,
        'unique_count': series.nunique(),
        'total_count': len(series)
    }
    
    # Determine pattern type
    pattern_type = _determine_pattern_type(series)
    pattern_info['pattern_type'] = pattern_type
    
    # Calculate statistical measures for numeric columns
    if pattern_type == 'numeric':
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 0:
            pattern_info['stats'] = {
                'min': float(numeric_series.min()),
                'max': float(numeric_series.max()),
                'mean': float(numeric_series.mean()),
                'median': float(numeric_series.median()),
                'std': float(numeric_series.std()),
                'q25': float(numeric_series.quantile(0.25)),
                'q75': float(numeric_series.quantile(0.75)),
                'skewness': float(numeric_series.skew()) if len(numeric_series) > 2 else 0.0,
                'kurtosis': float(numeric_series.kurtosis()) if len(numeric_series) > 2 else 0.0
            }
        else:
            pattern_info['stats'] = None
    
    # Calculate occurrence patterns for categorical/text columns
    elif pattern_type in ['categorical', 'text']:
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) > 0:
            # Value occurrence (frequency distribution)
            value_counts = non_null_series.value_counts()
            pattern_info['occurrence'] = {
                'top_values': value_counts.head(20).to_dict(),
                'value_frequencies': (value_counts / len(non_null_series)).head(20).to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_frequency': float(value_counts.iloc[0] / len(non_null_series)) if len(value_counts) > 0 else 0.0
            }
            
            # Common prefix/suffix pattern
            pattern_info['common_pattern'] = _find_common_pattern(non_null_series.head(100))
            
            # String length statistics if applicable
            str_lengths = non_null_series.str.len()
            pattern_info['string_stats'] = {
                'min_length': int(str_lengths.min()),
                'max_length': int(str_lengths.max()),
                'mean_length': float(str_lengths.mean()),
                'median_length': float(str_lengths.median())
            }
        else:
            pattern_info['occurrence'] = None
            pattern_info['common_pattern'] = None
    
    # Date pattern analysis
    elif pattern_type == 'date':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', FutureWarning)
            date_series = pd.to_datetime(series, errors='coerce', format='mixed').dropna()
        if len(date_series) > 0:
            pattern_info['stats'] = {
                'min_date': str(date_series.min()),
                'max_date': str(date_series.max()),
                'date_range_days': int((date_series.max() - date_series.min()).days),
                'mean_date': str(date_series.mean())
            }
        else:
            pattern_info['stats'] = None
    
    return pattern_info


def analyze_dataframe_patterns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze patterns for all columns in a DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary mapping column names to their pattern information
    """
    patterns = {}
    for column in df.columns:
        patterns[column] = analyze_column_patterns(df, column)
    return patterns


def _determine_pattern_type(series: pd.Series) -> str:
    """
    Determine the pattern type of a column
    
    Returns:
        'numeric', 'date', 'categorical', or 'text'
    """
    # Check for numeric pattern
    numeric_series = pd.to_numeric(series, errors='coerce')
    if numeric_series.notna().sum() / len(series) > 0.8:  # 80% numeric
        return 'numeric'
    
    # Check for date pattern (suppress warnings for format inference)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', FutureWarning)
        date_series = pd.to_datetime(series, errors='coerce', format='mixed')
    if date_series.notna().sum() / len(series) > 0.8:  # 80% dates
        return 'date'
    
    # Check for categorical (low cardinality)
    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.5:  # Less than 50% unique values
        return 'categorical'
    
    # Default to text
    return 'text'


def _find_common_pattern(values: pd.Series) -> Optional[str]:
    """
    Find common prefix/suffix pattern in a series of string values
    
    Args:
        values: Series of string values
    
    Returns:
        Common pattern string or None
    """
    if len(values) == 0:
        return None
    
    str_values = [str(v) for v in values if pd.notna(v)]
    if not str_values:
        return None
    
    # Find common prefix
    first = str_values[0]
    common_prefix = ""
    
    for i in range(len(first)):
        if all(len(x) > i and x[i] == first[i] for x in str_values):
            common_prefix += first[i]
        else:
            break
    
    # Find common suffix
    common_suffix = ""
    first_rev = first[::-1]
    for i in range(1, len(first_rev) + 1):
        suffix = first[-i:]
        if all(x.endswith(suffix) for x in str_values):
            common_suffix = suffix
        else:
            break
    
    # Return the longer pattern
    if len(common_prefix) > len(common_suffix):
        return common_prefix if len(common_prefix) > 2 else None
    else:
        return common_suffix if len(common_suffix) > 2 else None


def print_pattern_summary(patterns: Dict[str, Dict[str, Any]]):
    """
    Print a summary of analyzed patterns
    
    Args:
        patterns: Dictionary of pattern information from analyze_dataframe_patterns
    """
    print("\n" + "="*80)
    print("PATTERN ANALYSIS SUMMARY")
    print("="*80)
    
    for col_name, pattern_info in patterns.items():
        print(f"\nColumn: {col_name}")
        print(f"  Type: {pattern_info['pattern_type']} ({pattern_info['data_type']})")
        print(f"  Nulls: {pattern_info['null_count']} ({pattern_info['null_percentage']:.2f}%)")
        print(f"  Unique values: {pattern_info['unique_count']}")
        
        if pattern_info['pattern_type'] == 'numeric' and pattern_info.get('stats'):
            stats = pattern_info['stats']
            print(f"  Statistics:")
            print(f"    Min: {stats['min']:.2f}")
            print(f"    Max: {stats['max']:.2f}")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Std Dev: {stats['std']:.2f}")
        
        elif pattern_info['pattern_type'] in ['categorical', 'text'] and pattern_info.get('occurrence'):
            occ = pattern_info['occurrence']
            print(f"  Most common value: {occ['most_common']} ({occ['most_common_frequency']*100:.2f}%)")
            if pattern_info.get('common_pattern'):
                print(f"  Common pattern: {pattern_info['common_pattern']}")
        
        elif pattern_info['pattern_type'] == 'date' and pattern_info.get('stats'):
            stats = pattern_info['stats']
            print(f"  Date range: {stats['min_date']} to {stats['max_date']}")
            print(f"  Range: {stats['date_range_days']} days")


def save_patterns_to_file(
    patterns: Dict[str, Dict[str, Any]],
    output_path: str,
    format: str = 'json'
) -> str:
    """
    Save pattern analysis results to a file
    
    Args:
        patterns: Dictionary of pattern information from analyze_dataframe_patterns
        output_path: Path to save the patterns file
        format: File format - 'json' or 'txt' (default: 'json')
    
    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format.lower() == 'json':
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        serializable_patterns = convert_to_serializable(patterns)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_patterns, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PATTERN ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for col_name, pattern_info in patterns.items():
                f.write(f"Column: {col_name}\n")
                f.write(f"  Type: {pattern_info['pattern_type']} ({pattern_info['data_type']})\n")
                f.write(f"  Nulls: {pattern_info['null_count']} ({pattern_info['null_percentage']:.2f}%)\n")
                f.write(f"  Unique values: {pattern_info['unique_count']}\n")
                f.write(f"  Total count: {pattern_info['total_count']}\n")
                
                if pattern_info['pattern_type'] == 'numeric' and pattern_info.get('stats'):
                    stats = pattern_info['stats']
                    f.write(f"  Statistics:\n")
                    f.write(f"    Min: {stats['min']:.2f}\n")
                    f.write(f"    Max: {stats['max']:.2f}\n")
                    f.write(f"    Mean: {stats['mean']:.2f}\n")
                    f.write(f"    Median: {stats['median']:.2f}\n")
                    f.write(f"    Std Dev: {stats['std']:.2f}\n")
                    f.write(f"    Q25: {stats['q25']:.2f}\n")
                    f.write(f"    Q75: {stats['q75']:.2f}\n")
                    if 'skewness' in stats:
                        f.write(f"    Skewness: {stats['skewness']:.2f}\n")
                    if 'kurtosis' in stats:
                        f.write(f"    Kurtosis: {stats['kurtosis']:.2f}\n")
                
                elif pattern_info['pattern_type'] in ['categorical', 'text'] and pattern_info.get('occurrence'):
                    occ = pattern_info['occurrence']
                    f.write(f"  Most common value: {occ['most_common']} ({occ['most_common_frequency']*100:.2f}%)\n")
                    if pattern_info.get('common_pattern'):
                        f.write(f"  Common pattern: {pattern_info['common_pattern']}\n")
                    
                    if pattern_info.get('string_stats'):
                        str_stats = pattern_info['string_stats']
                        f.write(f"  String length stats:\n")
                        f.write(f"    Min length: {str_stats['min_length']}\n")
                        f.write(f"    Max length: {str_stats['max_length']}\n")
                        f.write(f"    Mean length: {str_stats['mean_length']:.2f}\n")
                        f.write(f"    Median length: {str_stats['median_length']:.2f}\n")
                    
                    if occ.get('top_values'):
                        f.write(f"  Top 10 values:\n")
                        for i, (value, count) in enumerate(list(occ['top_values'].items())[:10], 1):
                            freq = occ['value_frequencies'].get(value, 0) * 100
                            f.write(f"    {i}. {value}: {count} occurrences ({freq:.2f}%)\n")
                
                elif pattern_info['pattern_type'] == 'date' and pattern_info.get('stats'):
                    stats = pattern_info['stats']
                    f.write(f"  Date range: {stats['min_date']} to {stats['max_date']}\n")
                    f.write(f"  Range: {stats['date_range_days']} days\n")
                
                f.write("\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'")
    
    return output_path
