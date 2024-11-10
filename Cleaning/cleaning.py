import pandas as pd
import numpy as np


def cleaning_data(data, cleaned_output_path):
    print(f"Total rows before cleaning: {len(data)}")
    # Removing unnecessarily columns #
    data.drop(columns=["cadastral_income", "primary_energy_consumption_sqm"], inplace=True)
    # Removing duplication only of the id and zip_codes matches
    data_cleaned = data.drop_duplicates(subset=["id", "zip_code"], keep=False) # keep = fales so, Removes all occurrences of duplicates rather than keeping the first or last instance.

    # Remove leading, trailing ,and inside spaces from all string columns
    data_cleaned_spaces = data_cleaned.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # using .str.strip() this is only to remove the head and tail spaces
     # x.str.replace(r'\s+', ' ', regex=True) using regex to remove unwanted spaces between words.
    data_cleaned_spaces = data_cleaned.apply(lambda x: x.str.replace(r"\s+", " ", regex=True) if x.dtype == "object" else x) 
    # Filling the empty cells with 'NaN' if it is numeric and 'None' if it is 'string'  to be encoded later in the preprocessing step
    data_cleaned_spaces.replace({"MISSING": None, "": None}, inplace=True)
    def fill_missing_values(col):
        if col.dtype == "object":  # Check if the column is of type object (string) as here "object == strings"
            return col.where(col.notna(), None) # where: it is a method, which is used for conditional replacement in pandas
        else:  # For numeric columns
            return col.where(col.notna(), np.nan)  # the missing values are replaced by nan
    data_cleaned_final_1 = data_cleaned_spaces.apply(fill_missing_values) # applying the fill_missing_values function on each column on the Datafram.

    # Solving the encoding errors for price if it is exists to avoid any errors that can be occured
    data_cleaned_final_1["price"] = data_cleaned_final_1["price"].replace({"\€": "", ",": ""}, regex=True)

    # removing raws with empty values as thoes columns are important
    columns_to_check = [
        "province",
        "zip_code",
        "region",
        "id",
        "price",
        "locality",
    ]  
    data_cleaned_final = data_cleaned_final_1.dropna(subset=columns_to_check, how="any")

    """ remove rows with empty strings. This is for confirmation, 
    # it can be useless as explained in the previous 2 comments".
    #however it has to be applied if I used it before conversion """
    data_cleaned_final = data_cleaned_final[(data_cleaned_final[columns_to_check] != "").all(axis=1)] #axis = 1 for rows
    print(f"Total rows after cleaning: {len(data_cleaned_final)}")
    # Saving the cleaned file in the choesen path.
    data_cleaned_final.to_csv(cleaned_output_path,index=False,)
    return data_cleaned_final
