import numpy as np
import pandas as pd
from Cleaning.cleaning import cleaning_data  # Import the cleaning function
from Cleaning.removing_outliers import IQR_for_outliers

# Define file paths
input_file_path = r"C:\Users\mgabi\Desktop\becode\becode_projects\immo-eliza-ml\Cleaning\properties.csv"
cleaned_output_path = r"C:\Users\mgabi\Desktop\becode\becode_projects\immo-eliza-ml\Cleaning\properties_cleaned.csv"
""" it the same path for the cleaning output 
 (predefined inside the cleaning file) BE CARFUL"""
#
outliers_output_dir = (
    r"C:\Users\mgabi\Desktop\becode\becode_projects\immo-eliza-ml\Preproc_ML"
)

data = pd.read_csv(input_file_path)
cleaned_data = cleaning_data(data)
property_types = ["APARTMENT", "HOUSE"]
combined_data_no_outliers = IQR_for_outliers(
    cleaned_data, property_types, outliers_output_dir
)

print("Data cleaning and outlier removal complete.")
