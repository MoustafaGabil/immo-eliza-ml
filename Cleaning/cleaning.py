import pandas as pd
import numpy as np


def cleaning_data(data):
    print(f"Total rows before cleaning: {len(data)}")
    # Removing unnecessarily columns #
    data.drop(
        columns=["cadastral_income", "primary_energy_consumption_sqm"], inplace=True
    )

    # Calculating the missing data for each column and its percentage
    missing_values = data.isna().sum()  # Count missing values in each column
    total_rows = len(data)
    missing_percentage = round(((missing_values / total_rows) * 100), 2)
    missing_data = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_percentage}
    )

    # REmoving duplication only of the id and zip_codes matches
    data_cleaned = data.drop_duplicates(subset=["id", "zip_code"], keep=False)

    # Remove leading, trailing ,and inside spaces from all string columns
    data_cleaned_spaces = data_cleaned.apply(
        lambda x: x.str.strip() if x.dtype == "object" else x
    )  # using .str.strip() this is only to remove the head and tail spaces
    data_cleaned_spaces = data_cleaned.apply(
        lambda x: x.str.replace(r"\s+", " ", regex=True) if x.dtype == "object" else x
    )  # x.str.replace(r'\s+', ' ', regex=True) using regex to remove unwanted spaces between words.

    # Filling the empty cells with 'NaN' if it is numeric and 'None' if it is 'string'  to be encoded later in the preprocessing step
    data_cleaned_spaces.replace({"MISSING": None, "": None}, inplace=True)

    def fill_missing_values(col):
        if col.dtype == "object":  # Check if the column is of type object (string)
            return col.where(col.notna(), None)
        else:  # For numeric columns
            return col.where(col.notna(), np.nan)

    data_cleaned_final_1 = data_cleaned_spaces.apply(fill_missing_values)

    # Solving the encoding errors for price if it is exists to avoid any errors that can be occured
    data_cleaned_final_1["price"] = data_cleaned_final_1["price"].replace(
        {"\€": "", ",": ""}, regex=True
    )

    # removing raws with empty values
    columns_to_check = [
        "province",
        "zip_code",
        "region",
        "id",
        "price",
        "locality",
    ]  # Columns to check for empty values
    data_cleaned_final = data_cleaned_final_1.dropna(subset=columns_to_check, how="any")

    """ remove rows with empty strings. This is for confirmation, 
    # it can be useless as explained in the previous 2 comments".
    #however it has to be applied if I used it before conversion """
    data_cleaned_final = data_cleaned_final[
        (data_cleaned_final[columns_to_check] != "").all(axis=1)
    ]
    print(f"Total rows after cleaning: {len(data_cleaned_final)}")
    # Saving the cleaned file in the choesen path.
    data_cleaned_final.to_csv(
        r"C:\Users\mgabi\Desktop\becode\becode_projects\immo-eliza-ml\Cleaning\properties_cleaned.csv",
        index=False,
    )
    return data_cleaned_final
