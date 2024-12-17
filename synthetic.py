import openai
import os
import pandas as pd

# Function to read csv file and return individual columns, their length and their type
def read_csv(file_path):
    data = pd.read_csv(file_path)
    patterns = {}
    for col in data.columns:
        # Get a sample of values from the column
        sample_values = data[col].dropna().astype(str).head(10)
        
        # Analyze the pattern in the values
        pattern = analyze_pattern(sample_values)
        patterns[col] = pattern
        
    return {col: (len(data[col]), data[col].dtype, patterns[col]) for col in data.columns}

def analyze_pattern(values):
    """Analyze the pattern in a sequence of values."""
    if not len(values):
        return "empty"
    
    # Convert all values to strings for pattern analysis
    sample = values.iloc[0]
    
    # Check for numeric pattern
    if str(sample).replace('.','').replace('-','').isdigit():
        return "numeric"
    
    # Check for date pattern (simple check)
    if pd.to_datetime(values, errors='coerce').notna().all():
        return "date"
    
    # Check for categorical/text pattern
    if values.nunique() / len(values) < 0.5:  # If less than 50% unique values
        return "categorical"
    
    # Default to text
    return "text"

print(read_csv("example_data.csv"))


