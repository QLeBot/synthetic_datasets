import thispersondoesnotexist
import data_pattern
from data_generator import generate_synthetic_data

def generate_dataset(csv_file, num_faces=0):
    """
    Main function to generate synthetic dataset including optional AI faces
    """
    # Generate synthetic data from CSV patterns
    synthetic_data = generate_synthetic_data(csv_file)
    
    # Generate AI faces if requested
    if num_faces > 0:
        thispersondoesnotexist.download_ai_face(num_faces)
    
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
    csv_file = "synthetic.csv"
    synthetic_data = generate_dataset(csv_file, num_faces=5)
    display_results(synthetic_data)
