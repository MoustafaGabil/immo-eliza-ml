import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib

# Constants
parameters = [
    "construction_year",
    "total_area_sqm",
    "nbr_frontages",
    "nbr_bedrooms",
    "kitchen_type_encoded",
    "Bulding_sta_encoded",
    "epc_encoded",
    "garden_sqm",
    "surface_land_sqm",
    "fl_double_glazing",
    "fl_terrace",
    "fl_swimming_pool",
    "fl_floodzone",
    "latitude",
    "longitude",
    "zip_code",
]
Heat_map_parmeters = [
    "price",
    "construction_year",
    "total_area_sqm",
    "nbr_frontages",
    "nbr_bedrooms",
    "kitchen_type_encoded",
    "Bulding_sta_encoded",
    "epc_encoded",
    "garden_sqm",
    "latitude",
    "longitude",
    "zip_code",
]

# ------------------------------------------------  Feature Engineering ---------------------------------------
#drop rows if they are empty for the below columns
def cleaning(dataset):
    dataset.dropna(
        subset=[
            "total_area_sqm",
            "terrace_sqm",
            "garden_sqm",
            "nbr_frontages",
            "equipped_kitchen",
            "latitude",
            "longitude",
        ],
        inplace=True,
    )
    # droping the property if it has 0 bed rooms as it is not logic 
    dataset = dataset[dataset["nbr_bedrooms"] != 0]
    dataset["construction_year"] = dataset["construction_year"].fillna(dataset["construction_year"].mode()[0])  # using mode gives the best results based on the corellation with the price
    dataset["nbr_bedrooms"] = dataset["nbr_bedrooms"].fillna(dataset["nbr_bedrooms"].mean()) # using mean gives the best results based on the corellation with the price
    dataset["equipped_kitchen"] = (dataset["equipped_kitchen"].replace(0, "unknown").fillna("unknown")) # replacing 0s and missing values with "unknown"
    dataset["state_building"] = (dataset["state_building"].replace(0, "unknown").fillna("unknown"))  # replacing 0s and missing values with "unknown"
    dataset["nbr_frontages"] = dataset["nbr_frontages"].fillna(dataset["nbr_frontages"].median()) # using median gives the best results based on the corellation with the price
    dataset["epc"] = dataset["epc"].fillna(dataset["epc"].mode()[0]) # using mode gives the best results based on the corellation with the price
    dataset["garden_sqm"] = dataset["garden_sqm"].fillna(dataset["garden_sqm"].mean()) # using mean gives the best results based on the corellation with the price
    dataset["surface_land_sqm"] = dataset["surface_land_sqm"].fillna(dataset["surface_land_sqm"].median()) # using median gives the best results based on the corellation with the price
    return dataset

#----------------------------------------------------- Encoding -----------------------------------------------

# Define encoding function It has to be ascending ordered.
def encoding(dataset):
    kitchen_order = [
        "unknown",
        "NOT_INSTALLED",
        "USA_UNINSTALLED",
        "SEMI_EQUIPPED",
        "USA_SEMI_EQUIPPED",
        "INSTALLED",
        "USA_INSTALLED",
        "HYPER_EQUIPPED",
        "USA_HYPER_EQUIPPED",
    ]
    building_con_order = [
        "unknown",
        "TO_RESTORE",
        "TO_RENOVATE",
        "TO_BE_DONE_UP",
        "GOOD",
        "JUST_RENOVATED",
        "AS_NEW",
    ]
    epc_order = ['G','F','E','D','C','B','A','A+','A++']  
    # using Ordinal Encoder for encoding the categorical variables
    encoder_kit = OrdinalEncoder(categories=[kitchen_order])  
    encoder_bul = OrdinalEncoder(categories=[building_con_order])
    encoder_epc = OrdinalEncoder(categories=[epc_order])

    dataset["kitchen_type_encoded"] = encoder_kit.fit_transform(dataset[["equipped_kitchen"]]) 
    dataset["Bulding_sta_encoded"] = encoder_bul.fit_transform(dataset[["state_building"]])
    dataset["epc_encoded"] = encoder_epc.fit_transform(dataset[["epc"]])

    joblib.dump(encoder_kit, "Results and Evaluation\Encoders\encoder_kitchen_type.joblib")
    joblib.dump(encoder_bul, "Results and Evaluation\Encoders\encoder_building_state.joblib")
    joblib.dump(encoder_epc, "Results and Evaluation\Encoders\encoder_epc.joblib")

    locality_encoder = OneHotEncoder(sparse_output=False, drop="first")  
     # sparse_output=False to avoid creating sparse matrix but will creat dense array (a regular 2D numpy array or DataFrame).
     #Drops the first category for each feature to avoid multicollinearity 
     #If you encode all categories, one category can be perfectly predicted from the others, introducing redundancy
    locality_encoded = locality_encoder.fit_transform(dataset[["locality"]])
    #The function get_feature_names_out(["locality"]) creates a list of column's names based on the input column.
    locality_encoded_df = pd.DataFrame(locality_encoded, columns=locality_encoder.get_feature_names_out(["locality"])) 
    #drop=True: Prevents the old index from being added as a new column and avioid Misalignment as the new DF for the encoded locality has its own indeces (new) might not be the same as the dataset index.
    dataset = pd.concat([dataset.reset_index(drop=True), locality_encoded_df.reset_index(drop=True)],axis=1,)
    # Save the encoder using joblib (this will save the encoding process for later use)
    joblib.dump(locality_encoder, "Results and Evaluation\Encoders\locality_encoder.joblib")
    return dataset, locality_encoded_df.columns.tolist()


# Define evaluation function
def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = {}  # to save the evaluation results
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mae)
        r2 = r2_score(y_test, predictions)

        results[name] = {
            "MAE": mae,
            "R²": r2,
            "Train Score": train_score,
            "Test Score": test_score,
            "Model name": name,
            "RMSE": rmse,
        }

    results_evaluation = pd.DataFrame(results).T 
    return results_evaluation


# Heat map function and saving
def heat_map(data, title, save_path=None):
    selected_parameters = data[Heat_map_parmeters]
    correlation_matrix = selected_parameters.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        linewidths=0.05,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show(block=False)  ## It has to be block=False to avoid


# Processing each CSV file
directory = r"Results and Evaluation"
all_results = []

for file_name in os.listdir(directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(directory, file_name)
        print(f"Processing file: {file_name}")

        dataset = pd.read_csv(file_path)
        dataset, locality_encoded_columns = encoding(cleaning(dataset))

        file_name_no_ext = os.path.splitext(file_name)[0] # using splitext to extract only the name which will have index [0] and avoid includind the extention .csv
        heat_map(dataset,title=f"Correlation map for {file_name_no_ext}",
            save_path=f"Results and Evaluation\Heat_maps\heatmap_{file_name_no_ext}.png",)

        X = dataset[parameters + locality_encoded_columns]
        y = dataset["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

        models = {
            "RandomForest Regression": RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=20,
                bootstrap=True,
            ),
            "XGB Regression": XGBRegressor(
                n_estimators=900,
                subsample=0.8,
                min_child_weight=3,
                max_depth=8,
                learning_rate=0.05,
                gamma=0.2,
                colsample_bytree=0.36,
            ),
            "Hist Gradient Boosting Regression": HistGradientBoostingRegressor(
                min_samples_leaf=10,
                max_leaf_nodes=20,
                max_iter=500,
                max_depth=10,
                learning_rate=0.05,
                l2_regularization=0.0,
            ),
        }

        # Specify the directory where you want to save the models
        model_save_directory = r"Results and Evaluation\models"

        # looping Inside  model evaluation , saving the models
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Saving the model
            model_filename = os.path.join(model_save_directory, f"{name.replace(' ', '_')}_{file_name_no_ext}.pkl")
            with open(model_filename, "wb") as model_file:
                pickle.dump(model, model_file)
                print(f"Saved {name} model to {model_filename}")

        results_evaluation = evaluate_models(X_train, X_test, y_train, y_test, models)
        results_evaluation["properties type"] = file_name_no_ext
        all_results.append(results_evaluation)

final_results_evaluation = pd.concat(all_results)
column_order = ["properties type", "Model name", "MAE", "RMSE", "R²", "Train Score", "Test Score"]
final_results_evaluation = final_results_evaluation[column_order]
print(final_results_evaluation)

final_results_evaluation.to_csv(r"Results and Evaluation\Evaluation_results\combined_model_evaluation_results.csv",index=False,)
