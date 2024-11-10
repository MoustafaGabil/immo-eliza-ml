
import pandas as pd
import numpy as np
import joblib


def predict_price(features,property_type, model_paths,mse_values,encoder_paths):
    
    # Load the encoder and the model
    locality_encoder = joblib.load(encoder_paths["locality"])
    kitchen_encoder = joblib.load(encoder_paths['kitchen_type'])
    building_state_encoder = joblib.load(encoder_paths['building_state'])
    epc_encoder = joblib.load(encoder_paths['epc'])
    
    if property_type == "house":
        model = joblib.load(model_paths[0])
        mse = mse_values["house"]
    elif  property_type == "apartment":
          model = joblib.load(model_paths[1])
          mse = mse_values["apartment"]
    else:
        raise ValueError("Invalid property type. Choose 'house' or 'apartment'.")

    # Extract each encoded value from the feature dictionary so it can be encoded ( 1 , 3 .. etc)
    locality = features.pop("locality")
    locality_encoded = locality_encoder.transform([[locality]])
    
    kitchen_type = features.pop("kitchen_type_encoded")
    kitchen_encoded = kitchen_encoder.transform([[kitchen_type]])[0][0] # [0][0] as the output of the encoder is 2D array eg. [[1]]

    building_state = features.pop("Bulding_sta_encoded")
    building_state_encoded = building_state_encoder.transform([[building_state]])[0][0]
    
    epc = features.pop("epc_encoded")
    epc_encoded = epc_encoder.transform([[epc]])[0][0]

    # Add encoded features back to the dictionary
    features["kitchen_type_encoded"] = kitchen_encoded
    features["Bulding_sta_encoded"] = building_state_encoded
    features["epc_encoded"] = epc_encoded
    features_df = pd.DataFrame([features])  # Create a DataFrame from the other features

    # Combine the encoded locality with other features
    input_data = np.concatenate([features_df.values, locality_encoded], axis=1)

    predicted_price = model.predict(input_data)
    # Predict the price using the loaded model
    predict_price_range = [predicted_price-mse , predicted_price+mse]

    return predicted_price[0] , predict_price_range

features = {
    "locality": "Antwerp",  
    "construction_year": 2020,
    "total_area_sqm": 125,
    "nbr_frontages": 4,
    "nbr_bedrooms": 3,
    "kitchen_type_encoded": "NOT_INSTALLED",  #"unknown","NOT_INSTALLED","USA_UNINSTALLED","SEMI_EQUIPPED","USA_SEMI_EQUIPPED","INSTALLED","USA_INSTALLED","HYPER_EQUIPPED","USA_HYPER_EQUIPPED"
    "Bulding_sta_encoded": "TO_BE_DONE_UP",     # "unknown","TO_RESTORE","TO_RENOVATE","TO_BE_DONE_UP", "GOOD","JUST_RENOVATED", "AS_NEW"
    "epc_encoded": "B",                    # 'A++','A+','A','B','C','D','E','F','G'
    "garden_sqm": 25,
    "surface_land_sqm": 36,
    "fl_double_glazing": 1,
    "fl_terrace": 0,
    "fl_swimming_pool": 0,
    "fl_floodzone": 1,
    "latitude": 51.174944,
    "longitude": 4.3345427,
    "zip_code": 9000,
}

# Paths to the saved encoder and model
encoder_paths = {"locality":"Results and Evaluation/Encoders/locality_encoder.joblib",
                 "kitchen_type":"Results and Evaluation/Encoders/encoder_kitchen_type.joblib",
                 "building_state":"Results and Evaluation/Encoders/encoder_building_state.joblib",
                 "epc":"Results and Evaluation/Encoders/encoder_epc.joblib"
                 }

model_path_Houses = "Results and Evaluation/models/XGB_Regression_HOUSE_without_outliers.pkl"
model_path_apartment = "Results and Evaluation/models/XGB_Regression_APARTMENT_without_outliers.pkl"
mse_values = { "house": 49956,  
              "apartment": 31540  }
model_paths=[model_path_Houses,model_path_apartment]
property_type = "apartment" 
predicted_price,predict_price_range = predict_price(features,property_type, model_paths,mse_values,encoder_paths)
print(f"Predicted Price: {int(predicted_price)}")
print(f'predicted price range [ min , max] : [{int(predict_price_range[0][0])} , {int(predict_price_range[1][0])}]')
