import thispersondoesnotexist
import synthetic
from openai import OpenAI
import os

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def generate_synthetic_data(csv_file, num_samples=10):
    # Get data patterns from synthetic.py
    data_patterns = synthetic.read_csv(csv_file)
    synthetic_data = {}

    for column_name, (length, dtype, pattern, common_pattern) in data_patterns.items():
        # Create a prompt based on the column characteristics
        prompt = create_column_prompt(column_name, pattern, common_pattern, length, dtype)
        
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
            
            # Store the generated data
            synthetic_data[column_name] = response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating data for column {column_name}: {str(e)}")
    
    return synthetic_data

def create_column_prompt(column_name, pattern, common_pattern, length, dtype):
    prompt = f"Generate {length} samples of data for a column named '{column_name}' with the following characteristics:\n"
    prompt += f"- Data type: {dtype}\n"
    prompt += f"- Pattern type: {pattern}\n"
    
    if common_pattern:
        prompt += f"- Common pattern found in values: '{common_pattern}'\n"
    
    if pattern == "numeric":
        prompt += "Generate numeric values that make sense for this column.\n"
    elif pattern == "date":
        prompt += "Generate dates in a consistent format.\n"
    elif pattern == "categorical":
        prompt += "Generate categorical values that would be reasonable for this column.\n"
    elif pattern == "text":
        prompt += "Generate text values that would be reasonable for this column.\n"
    
    prompt += "Return the values as a comma-separated list."
    
    return prompt

# Example usage
if __name__ == "__main__":
    synthetic_data = generate_synthetic_data("synthetic.csv")
    for column, generated_values in synthetic_data.items():
        print(f"\nColumn: {column}")
        print(f"Generated values: {generated_values}")
