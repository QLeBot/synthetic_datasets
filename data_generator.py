from openai import OpenAI
import os
import data_pattern

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def generate_synthetic_data(input_data, num_samples=10):
    """
    Generate synthetic data based on either a CSV file or column specifications
    
    Args:
        input_data: Either a path to CSV file or a dict of column specifications
        num_samples: Number of samples to generate per column
    
    Returns:
        dict: Generated synthetic data
    """
    synthetic_data = {}
    
    # Handle CSV file input
    if isinstance(input_data, str):
        data_patterns = data_pattern.read_csv(input_data)
    # Handle direct column specifications
    elif isinstance(input_data, dict):
        data_patterns = {
            col_name: (num_samples, dtype, dtype, None) 
            for col_name, (dtype, _) in input_data.items()
        }
    else:
        raise ValueError("input_data must be either a CSV file path or a dictionary of column specifications")

    for column_name, (length, dtype, pattern, common_pattern) in data_patterns.items():
        # Create a prompt based on the column characteristics
        prompt = create_column_prompt(column_name, pattern, common_pattern, num_samples, dtype)
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data generation assistant. Generate data that matches the specified pattern."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Process the response into a list
            generated_values = process_api_response(response.choices[0].message.content)
            
            # Store the generated data
            synthetic_data[column_name] = generated_values
            
        except Exception as e:
            print(f"Error generating data for column {column_name}: {str(e)}")
    
    return synthetic_data

def create_column_prompt(column_name, pattern, common_pattern, num_samples, dtype):
    """
    Create a prompt for generating synthetic data for a column
    """
    prompt = f"Generate {num_samples} samples of data for a column named '{column_name}' with the following characteristics:\n"
    prompt += f"- Data type: {dtype}\n"
    
    if common_pattern:
        prompt += f"- Common pattern found in values: '{common_pattern}'\n"
    
    if dtype == "numeric" or pattern == "numeric":
        if column_name.lower() == "age":
            prompt += "Generate realistic human age values between 0 and 100.\n"
        else:
            prompt += "Generate numeric values that make sense for this column.\n"
    elif dtype == "date" or pattern == "date":
        prompt += "Generate dates in YYYY-MM-DD format.\n"
    elif dtype == "categorical" or pattern == "categorical":
        if column_name.lower() == "occupation":
            prompt += "Generate realistic job titles or professions.\n"
        else:
            prompt += "Generate categorical values that would be reasonable for this column.\n"
    elif dtype == "text" or pattern == "text":
        if column_name.lower() == "name":
            prompt += "Generate realistic full names of people.\n"
        elif column_name.lower() == "email":
            prompt += "Generate realistic email addresses.\n"
        else:
            prompt += "Generate text values that would be reasonable for this column.\n"
    
    prompt += "Return the values as a comma-separated list."
    
    return prompt

def process_api_response(response_text):
    """
    Process the API response text into a list of values
    """
    # Split the response into individual values and clean them
    values = [v.strip() for v in response_text.split(',')]
    return values 