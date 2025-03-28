import openai
import os
import pandas as pd

# Function to read csv file and return individual columns, their length and their type
def read_csv(file_path):
    data = pd.read_csv(file_path)
    patterns = {}
    common_patterns = {}
    for col in data.columns:
        # Get a sample of values from the column
        sample_values = data[col].dropna().astype(str).head(10)
        
        # Analyze the pattern in the values
        pattern = analyze_pattern(sample_values)
        patterns[col] = pattern

        # Find common pattern/prefix/suffix between values in the column
        common_pattern = data_pattern(sample_values)
        common_patterns[col] = common_pattern
        
    return {col: (len(data[col]), data[col].dtype, patterns[col], common_patterns[col]) for col in data.columns}

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

# function to find similarity between rows values in a column
# example : 
# data = [User_1, User_2, User_3, User_4, User_5, User_6, User_7, User_8, User_9, User_10]
# similarity between all rows values in the column is User_
# return the similarity
def data_pattern(values):
    """
    Find common pattern/prefix/suffix between values in a column.
    
    Args:
        values: pandas Series or list of values
    
    Returns:
        str: Common pattern found between values, or empty string if no pattern
    """
    if not len(values):
        return ""
    
    # Convert values to list of strings
    values = [str(x) for x in values if pd.notna(x)]
    if not values:
        return ""
    
    # Find common prefix
    first = values[0]
    common = ""
    
    # Compare each character position across all values
    for i in range(len(first)):
        if all(len(x) > i and x[i] == first[i] for x in values):
            common += first[i]
        else:
            break
            
    # Find common suffix by reversing strings if no prefix found
    if not common:
        first = values[0]
        for i in range(1, len(first) + 1):
            if all(x.endswith(first[-i:]) for x in values):
                common = first[-i:]
            else:
                break
    
    return common if common else ""

print(read_csv("sample_data/example_data.csv"))


