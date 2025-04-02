import thispersondoesnotexist
import data_pattern
from data_generator import generate_synthetic_data
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env.local')

def generate_dataset(
    sample_data_path=None,
    output_path="generated_data/synthetic_dataset.csv",
    use_openai=True,
    num_samples=10,
    api_key=os.getenv("OPENAI_API_KEY")
):
    """
    Main function to generate synthetic dataset with multiple options
    
    Args:
        sample_data_path (str): Path to sample dataset (optional)
        output_path (str): Where to save the generated data
        use_openai (bool): Whether to use OpenAI for generation
        num_samples (int): Number of samples to generate
        api_key (str): OpenAI API key (optional)
    
    Returns:
        dict: Generated synthetic data
    """
    # Set OpenAI API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif use_openai and "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    synthetic_data = {}
    
    if sample_data_path and os.path.exists(sample_data_path):
        # Generate data based on patterns from sample data
        synthetic_data = generate_synthetic_data(sample_data_path, num_samples)
    elif use_openai:
        # Generate completely new data using OpenAI
        default_columns = {
            "name": ("text", 10),
            "age": ("numeric", 10),
            "occupation": ("categorical", 10),
            "email": ("text", 10)
        }
        synthetic_data = generate_synthetic_data(default_columns, num_samples)
    else:
        raise ValueError("Either provide sample_data_path or set use_openai=True")
    
    # Save generated data to CSV
    if synthetic_data:
        df = pd.DataFrame(synthetic_data)
        df.to_csv(output_path, index=False)
        print(f"Generated data saved to: {output_path}")
    
    return synthetic_data

def display_results(synthetic_data):
    """
    Display the generated synthetic data
    """
    for column, generated_values in synthetic_data.items():
        print(f"\nColumn: {column}")
        print(f"Generated values: {generated_values}")

if __name__ == "__main__":
    # Remove the hardcoded API key since we're loading from .env.local
    synthetic_data = generate_dataset(
        sample_data_path="sample_data/retail_sales_dataset.csv",
        output_path="generated_data/retail_sales_dataset_generated.csv",
        use_openai=False,
        num_samples=10000,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    display_results(synthetic_data)
