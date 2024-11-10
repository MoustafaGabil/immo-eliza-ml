import numpy as np
import pandas as pd


def IQR_for_outliers(data, property_types, output_dir):
    combined_data_no_outlier = pd.DataFrame()  # Empty DataFrame to hold combined data

    for property_type in property_types:
        pro = data[data["property_type"] == property_type]
        percentile25 = pro["price"].quantile(0.25)
        percentile75 = pro["price"].quantile(0.75)
        IQR = percentile75 - percentile25

        upper_limit = ( percentile75 + 1.5 * IQR )  # Determine upper and lower limits for outliers
        lower_limit = percentile25 - 1.5 * IQR
        data_no_outlier = pro[ (pro["price"] <= upper_limit) & (pro["price"] >= lower_limit)]  # Remove outliers

        individual_filename = ( f"{property_type}_without_outliers.csv")  # Saving the data separetly
        data_no_outlier.to_csv(f"{output_dir}/{individual_filename}", index=False)

        combined_data_no_outlier = pd.concat([combined_data_no_outlier, data_no_outlier], ignore_index=True)  # Append to combined DataFrame
        
    # Save the combined data without outliers to a single CSV file
    combined_filename = "Houses_and_Apartments_combined_without_outliers.csv"
    combined_data_no_outlier.to_csv(f"{output_dir}/{combined_filename}", index=False)

    return combined_data_no_outlier
