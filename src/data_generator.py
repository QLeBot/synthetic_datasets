"""
Data Generator Module
Generates synthetic data based on statistical patterns from real data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import random
import string

from .pattern_analyzer import analyze_dataframe_patterns


def generate_synthetic_data(
    source_data: pd.DataFrame,
    num_samples: int,
    patterns: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data based on patterns from source data
    
    Args:
        source_data: Original DataFrame to learn patterns from
        num_samples: Number of synthetic samples to generate
        patterns: Pre-computed patterns (optional, will compute if not provided)
    
    Returns:
        DataFrame with synthetic data
    """
    if patterns is None:
        patterns = analyze_dataframe_patterns(source_data)
    
    synthetic_data = {}
    
    for column in source_data.columns:
        pattern_info = patterns[column]
        synthetic_data[column] = generate_column_data(
            pattern_info,
            num_samples,
            source_data[column] if column in source_data.columns else None
        )
    
    return pd.DataFrame(synthetic_data)


def generate_column_data(
    pattern_info: Dict[str, Any],
    num_samples: int,
    reference_series: Optional[pd.Series] = None
) -> list:
    """
    Generate synthetic data for a single column based on its pattern
    
    Args:
        pattern_info: Pattern information from analyze_column_patterns
        num_samples: Number of samples to generate
        reference_series: Original column data for reference (optional)
    
    Returns:
        List of generated values
    """
    pattern_type = pattern_info['pattern_type']
    null_percentage = pattern_info['null_percentage']
    
    # Determine how many nulls to generate
    num_nulls = int(num_samples * (null_percentage / 100))
    num_values = num_samples - num_nulls
    
    if pattern_type == 'numeric':
        values = _generate_numeric_values(pattern_info, num_values)
    elif pattern_type == 'date':
        values = _generate_date_values(pattern_info, num_values)
    elif pattern_type == 'categorical':
        values = _generate_categorical_values(pattern_info, num_values)
    elif pattern_type == 'text':
        values = _generate_text_values(pattern_info, num_values, reference_series)
    else:
        values = [None] * num_values
    
    # Add nulls to match original null percentage
    values.extend([None] * num_nulls)
    
    # Shuffle to randomize null positions
    random.shuffle(values)
    
    return values


def _generate_numeric_values(pattern_info: Dict[str, Any], num_samples: int) -> list:
    """Generate numeric values based on statistical patterns"""
    if not pattern_info.get('stats'):
        return [0] * num_samples
    
    stats = pattern_info['stats']
    min_val = stats['min']
    max_val = stats['max']
    mean = stats['mean']
    std = stats['std']
    
    # Use normal distribution if std > 0, otherwise uniform
    if std > 0:
        # Clip values to min/max range
        values = np.random.normal(mean, std, num_samples)
        values = np.clip(values, min_val, max_val)
    else:
        # Uniform distribution
        values = np.random.uniform(min_val, max_val, num_samples)
    
    # Convert to appropriate type
    if pattern_info['data_type'] in ['int64', 'int32', 'int']:
        values = [int(round(v)) for v in values]
    else:
        values = [float(v) for v in values]
    
    return values.tolist()


def _generate_date_values(pattern_info: Dict[str, Any], num_samples: int) -> list:
    """Generate date values based on date range patterns"""
    if not pattern_info.get('stats'):
        return [None] * num_samples
    
    stats = pattern_info['stats']
    min_date = pd.to_datetime(stats['min_date'])
    max_date = pd.to_datetime(stats['max_date'])
    
    # Generate random dates within the range
    date_range = (max_date - min_date).days
    random_days = np.random.randint(0, date_range, num_samples)
    
    dates = [min_date + timedelta(days=int(d)) for d in random_days]
    
    # Format as strings in the same format as original
    return [d.strftime('%Y-%m-%d') for d in dates]


def _generate_categorical_values(pattern_info: Dict[str, Any], num_samples: int) -> list:
    """Generate categorical values based on occurrence patterns"""
    if not pattern_info.get('occurrence'):
        return ['Unknown'] * num_samples
    
    occ = pattern_info['occurrence']
    top_values = occ.get('top_values', {})
    value_frequencies = occ.get('value_frequencies', {})
    
    if not top_values:
        return ['Unknown'] * num_samples
    
    # Create probability distribution from frequencies
    values = list(value_frequencies.keys())
    probabilities = list(value_frequencies.values())
    
    # Normalize probabilities
    total_prob = sum(probabilities)
    if total_prob > 0:
        probabilities = [p / total_prob for p in probabilities]
    else:
        # Equal probability if no frequencies available
        probabilities = [1.0 / len(values)] * len(values)
    
    # Generate values based on probability distribution
    generated = np.random.choice(values, size=num_samples, p=probabilities)
    
    return generated.tolist()


def _generate_text_values(
    pattern_info: Dict[str, Any],
    num_samples: int,
    reference_series: Optional[pd.Series] = None
) -> list:
    """Generate text values based on patterns"""
    if not pattern_info.get('occurrence'):
        # Generate random strings
        string_stats = pattern_info.get('string_stats', {})
        min_len = string_stats.get('min_length', 5)
        max_len = string_stats.get('max_length', 20)
        mean_len = int(string_stats.get('mean_length', 10))
        
        values = []
        for _ in range(num_samples):
            length = max(min_len, min(max_len, int(np.random.normal(mean_len, 2))))
            value = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            values.append(value)
        return values
    
    # Use occurrence patterns if available
    occ = pattern_info['occurrence']
    top_values = occ.get('top_values', {})
    value_frequencies = occ.get('value_frequencies', {})
    common_pattern = pattern_info.get('common_pattern')
    
    if top_values and len(top_values) > 0:
        # Generate based on frequency distribution
        values = list(value_frequencies.keys())
        probabilities = list(value_frequencies.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(values)] * len(values)
        
        generated = []
        for _ in range(num_samples):
            # Sometimes use existing pattern, sometimes generate new
            if random.random() < 0.7 and common_pattern:  # 70% chance to use pattern
                base = random.choice(list(top_values.keys())[:10])
                # Modify base value slightly
                if len(base) > len(common_pattern):
                    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
                    generated.append(common_pattern + suffix)
                else:
                    generated.append(base)
            else:
                generated.append(np.random.choice(values, p=probabilities))
        
        return generated
    
    # Fallback: generate random strings
    string_stats = pattern_info.get('string_stats', {})
    min_len = string_stats.get('min_length', 5)
    max_len = string_stats.get('max_length', 20)
    mean_len = int(string_stats.get('mean_length', 10))
    
    values = []
    for _ in range(num_samples):
        length = max(min_len, min(max_len, int(np.random.normal(mean_len, 2))))
        prefix = common_pattern if common_pattern else ''
        suffix_len = max(1, length - len(prefix))
        suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=suffix_len))
        values.append(prefix + suffix)
    
    return values
