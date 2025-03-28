import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

class DataQualityChecker:
    def __init__(self, data):
        """
        Initialize the DataQualityChecker with a pandas DataFrame
        """
        self.data = data
        self.report = {}

    def check_missing_values(self):
        """Check for missing values in each column"""
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        self.report['missing_values'] = {
            'total_missing': missing.to_dict(),
            'missing_percentage': missing_percent.to_dict()
        }

    def check_duplicates(self):
        """Check for duplicate rows"""
        duplicates = self.data.duplicated().sum()
        self.report['duplicates'] = {
            'total_duplicates': duplicates,
            'duplicate_percentage': (duplicates / len(self.data)) * 100
        }

    def check_outliers(self, columns=None, threshold=3):
        """
        Check for outliers using Z-score method
        Only applies to numerical columns
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for column in columns:
            z_scores = np.abs(stats.zscore(self.data[column].dropna()))
            outliers[column] = {
                'total_outliers': len(z_scores[z_scores > threshold]),
                'outlier_percentage': (len(z_scores[z_scores > threshold]) / len(z_scores)) * 100
            }
        self.report['outliers'] = outliers

    def check_basic_stats(self):
        """Calculate basic statistics for numerical columns"""
        numeric_stats = self.data.describe()
        self.report['basic_stats'] = numeric_stats.to_dict()

    def check_data_types(self):
        """Check data types of each column"""
        self.report['data_types'] = self.data.dtypes.to_dict()

    def evaluate_ml_potential(self, target_column, problem_type='classification', test_size=0.2):
        """
        Evaluate dataset quality using machine learning performance metrics
        
        Args:
            target_column: Name of the target variable
            problem_type: 'classification' or 'regression'
            test_size: Proportion of dataset to use for testing
        """
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=categorical_cols)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate model
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            metric_name = 'accuracy'
        else:  # regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            metric_name = 'r2_score'
        
        # Store results
        self.report['ml_evaluation'] = {
            'problem_type': problem_type,
            f'{metric_name}': score,
            'feature_importance': feature_importance
        }
        if problem_type == 'regression':
            self.report['ml_evaluation']['rmse'] = rmse

    def generate_report(self, target_column=None, problem_type=None):
        """Generate a complete data quality report"""
        self.check_missing_values()
        self.check_duplicates()
        self.check_outliers()
        self.check_basic_stats()
        self.check_data_types()
        
        if target_column is not None:
            self.evaluate_ml_potential(target_column, problem_type)
        
        return self.report

    def visualize_missing_values(self):
        """Create a heatmap of missing values"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load data from sample_data folder
    try:
        sample_data = pd.read_csv('sample_data/dataset.csv')
    except FileNotFoundError:
        print("Error: Could not find the dataset file in sample_data folder.")
        # Create a more complex example dataset
        np.random.seed(42)
        n_samples = 1000
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.binomial(1, 0.5, n_samples)
        })
        print("Using example dataset instead.")

    # Initialize the checker
    checker = DataQualityChecker(sample_data)
    
    # Generate the report with ML evaluation
    # Assuming 'target' is the column to predict and it's a classification problem
    quality_report = checker.generate_report(
        target_column='target',
        problem_type='classification'
    )
    
    # Print the report
    print("\nData Quality Report:")
    print("===================")
    for metric, results in quality_report.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(results)
    
    # Visualize missing values
    checker.visualize_missing_values()
