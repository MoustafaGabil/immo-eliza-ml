# Real Estate Price Prediction Project (immo-eliza-ml) 🏠
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Pandas](https://img.shields.io/badge/uses-Pandas-blue.svg)
![Scikit-learn](https://img.shields.io/badge/uses-Scikit--learn-orange.svg)




## 🏢 Description

This project is focused on predicting real estate prices using machine learning models. It includes preprocessing of the data, feature engineering, training XGB_Regression,RandomForest,Hist_Gradient_Boosting , and evaluating their performance.Additionally, Exploring the the importance of features for XGB_Regression when Using zip code , longitude and longitude ( without ziip code) and using (zip code,longitude and longitude)
The project structure is designed to separate the processes into modules for better readability and maintenance.

<img src="https://miro.medium.com/v2/resize:fit:640/1*D6s2K1y7kjE14swcgITB1w.png"/>



### Issues and update requests
- If you encounter any issues or have suggestions for improvements, please feel free to open an issue in the repository.
- Contributions to enhance the functionality or performance of the models are always welcome.

Find me on [LinkedIn](https://www.linkedin.com/in/viktor-cosaert/) for collaboration, feedback, or to connect.

## 📦 Repo structure
```.
├── Cleaning/
│      ├── cleaning.py
│      ├── properties.csv
│      ├── properties_cleaned.csv
│      └── removing_outliers.py
├── Heat_maps/
│      ├── Heat map for apartments feature
│      ├── Heat map for Houses feature
│      └── Heat map for apartments and houses feature 
├── models/
│      └── 9 models for best models (XGB_Regression,RandomForest,Hist_Gradient_Boosting) for different properties
├── imo_models
│      └── virtual environment files
├── Notebooks/
│      ├── Evaluation_results
│      ├── features_importance.ipynb
│      ├── outliers_removal.ipynb   
├── Preproc_ML/
│      ├── cleaning.ipynb
│      │     ├── combined_model_evaluation_results
│      │     ├── comparasion of features results (with/without zip code,longitude and longitude)
│      │     └── graphs of important features (with/without zip code,longitude and longitude)
│      │     
│      │
│      ├── APARTMENT_without_outliers
│      ├── HOUSE_without_outliers 
       └── Houses_and_Apartments_combined_without_outliers
├── .gitignore
├── main.py
├── README.md
└── requirements.txt 
```
cleaning.ipynb
features_importance.ipynb
outliers_removal.ipynb
preprocessing and models test.ipynb
## 🚧 Installation 

1. Clone the repository to your local machine.

    ```
    git clone https://github.com/MoustafaGabil/immo-eliza-ml.git
    ```

2. Navigate to the project directory and install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

### pipeline for excuting the files: 

## Note: 
The whole data is separated into Houses, Apartment and combined file for both Houses and Apartments.

1. The main row data is called properties.csv and exists in Cleaning folder, the same folder where the cleaned data will be saved.

2. Running the **cleaning and outliers removal.py** which calls the the functions inside the Cleaning folder for cleaning the data and the removing the outlier.

    ```python
        # Define file paths
        input_file_path = r"immo-eliza-ml\Cleaning\properties.csv"
        cleaned_output_path = r"immo-eliza-ml\Cleaning\properties_cleaned.csv"
        """ it the same path for the cleaning output 
        (predefined inside the cleaning file) BE CARFUL"""
        #
        outliers_output_dir = (
            r"immo-eliza-ml\Preproc_ML"
        )

        data = pd.read_csv(input_file_path)
        cleaned_data = cleaning_data(data)
        property_types = ["APARTMENT", "HOUSE"]
        combined_data_no_outliers = IQR_for_outliers(
            cleaned_data, property_types, outliers_output_dir
        )

        print("Data cleaning and outlier removal complete.")

3. Running the **train.py** which contains the script for preprocessing, feature engineering and testing of 3 different models (XGB_Regression,RandomForest,Hist_Gradient_Boosting).
    The following are the parameters used for modeling and creating the heat maps.
     
    ''' python
        parameters = ["construction_year","total_area_sqm","nbr_frontages","nbr_bedrooms","kitchen_type_encoded","Bulding_sta_encoded","epc_encoded","garden_sqm","surface_land_sqm","fl_double_glazing","fl_terrace","fl_swimming_pool","fl_floodzone","latitude","longitude","zip_code",]

       Heat_map_parmeters = ["price","construction_year","total_area_sqm","nbr_frontages","nbr_bedrooms","kitchen_type_encoded","Bulding_sta_encoded","epc_encoded","garden_sqm","latitude","longitude","zip_code"]

4. The **Preproc_ML** folder contains the results for different models and comparison based on using different features.

### Results: 
There are more results available for testing the models and visualization can be found in folders such as:

-  Heat_maps: where you can find the correlation between the price and different features as indicated in the below figure.
![alt text](heatmap_Houses_and_Apartments_combined_without_outliers.png).</b>

- The results for evaluations are as follow:

| File                                             | Model name                         | MAE         | RMSE        | RA³          | Train Score | Test Score|
|--------------------------------------------------|------------------------------------|-------------|-------------|--------------|-------------|-----------|
| APARTMENT_without_outliers                       | Random Forest Regression           | 38847.56735 | 197.0978624 | 0.746259935  | 0.920989357 | 0.74626   |
| APARTMENT_without_outliers                       | XGB Regression                     | 31681.84959 | 177.9939594 | 0.816293401  | 0.986465603 | 0.816293  |
| APARTMENT_without_outliers.csv                   | Hist Gradient Boosting Regression  | 34233.0489  | 185.0217525 | 0.797021799  | 0.913430742 | 0.797022  |
| Houses_and_Apartments_combined_without_outliers  | Random Forest Regression           | 47965.06836 | 219.0092883 | 0.768380127  | 0.921555976 | 0.76838   |
| Houses_and_Apartments_combined_without_outliers  | XGB Regression                     | 41086.61647 | 202.6983386 | 0.817504332  | 0.980630054 | 0.817504  |
| Houses_and_Apartments_combined_without_outliers  | Hist Gradient Boosting Regression  | 46254.58674 | 215.0687954 | 0.791215598  | 0.86780864  | 0.791216  |
| HOUSE_without_outliers                           | Random Forest Regression           | 58860.51476 | 242.6118603 | 0.765119731  | 0.92104172  | 0.76512   |
| HOUSE_without_outliers                           | XGB Regression                     | 50102.96551 | 223.8369172 | 0.816225213  | 0.993652011 | 0.816225  |
| HOUSE_without_outliers.csv                       | Hist Gradient Boosting Regression  | 53392.92323 | 231.0690876 | 0.801762635  | 0.9124195   | 0.801763  |

- where: 
    - MAE: (Mean Absolute Error).
    - RMSE (Root Mean Squared Error).
    - R-squared Coefficient

- The some of the feature importance results can found below: </b>
![alt text](feature_importance_H&A_lan&lat&zip_code.png) </b>


## 🔧 Updates & Upgrades

The data can be further preprocessed to get more accurate results and more models can be tested. 
The importance of longitude and latitude appeared in the features importance so, It is advisable to have more exploration. 


```

## ⏱️ Project Timeline
The initial setup of this project was completed in 5 days.


