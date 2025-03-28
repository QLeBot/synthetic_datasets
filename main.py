import thispersondoesnotexist
import data_pattern
from data_generator import generate_synthetic_data
import os
import pandas as pd

def generate_dataset(
    sample_data_path=None,
    output_path="generated_data/synthetic_dataset.csv",
    use_openai=True,
    num_samples=10,
    num_faces=0
):
    """
    Main function to generate synthetic dataset with multiple options
    
    Args:
        sample_data_path (str): Path to sample dataset (optional)
        output_path (str): Where to save the generated data
        use_openai (bool): Whether to use OpenAI for generation
        num_samples (int): Number of samples to generate
        num_faces (int): Number of AI faces to generate
    
    Returns:
        dict: Generated synthetic data
    """
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
    
    # Generate AI faces if requested
    if num_faces > 0:
        thispersondoesnotexist.download_ai_face(num_faces)
    
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
    # Example usage
    synthetic_data = generate_dataset(
        sample_data_path="sample_data/example_data.csv",  # Optional: path to sample data
        output_path="generated_data/synthetic_dataset.csv",
        use_openai=True,
        num_samples=10,
        num_faces=5
    )
    display_results(synthetic_data)
