# Real Estate Price Prediction Project (immo-eliza-ml) 🏠
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Pandas](https://img.shields.io/badge/uses-Pandas-blue.svg)
![Scikit-learn](https://img.shields.io/badge/uses-Scikit--learn-orange.svg)




## 🏢 Description

This project is focused on predicting real estate prices using machine learning models. It includes preprocessing of the data, feature engineering, training XGB_Regression,RandomForest,Hist_Gradient_Boosting , and evaluating their performance.Additionally, Exploring the the importance of features for XGB_Regression when Using zip code , longitude and longitude ( without zip code) and using (zip code,longitude and longitude)
The project structure is designed to separate the processes into modules for better readability and maintenance.
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/1*D6s2K1y7kjE14swcgITB1w.png" />
</p>

### Issues and update requests
- If you have any questions, run into issues, or have ideas for improvement, please don’t hesitate to open an issue in the repository.
- Contributions to improve the models' functionality or performance are highly encouraged and appreciated.


Find me on [LinkedIn](https://www.linkedin.com/in/moustafa-gabil-8a4a6bab/) for collaboration, feedback, or to connect.

## 📦 Repo structure
```.
├── Cleaning/
│      ├── cleaning.py
│      ├── properties.csv
│      ├── properties_cleaned.csv
│      └── removing_outliers.py
├── imo_models
│      └── virtual environment files
├── Notebooks/
│      ├── cleaning.ipynb
│      ├── features_importance.ipynb
│      ├── outliers_removal.ipynb 
│      └── preprocessing and models test.ipynb  
├── Results and Evaluation/
│      ├── Evaluation_results
│      │     ├── combined_model_evaluation_results
│      │     └──  comparasion of features results (with/without zip code,longitude and longitude)
│      │     
│      ├── Features importance/
│      │     └── Three graphs of important features for XGB_Regression model (with/without zip code,longitude and longitude) 
│      │
│      ├── Heat_maps/ 
│      │     ├── Heat map for apartments features
│      │     ├── Heat map for Houses features
│      │     └── Heat map for apartments and houses feature
│      ├── models/
│      │     └── 9 models for best models (XGB_Regression,RandomForest,Hist_Gradient_Boosting) for different properties           
│      ├── APARTMENT_without_outliers
│      ├── HOUSE_without_outliers 
│      └── Houses_and_Apartments_combined_without_outliers
│ 
├── cleaning and outliers removal.py 
├── .gitignore
├── train.py
├── README.md
└── requirements.txt 

```
## 🚧 Installation 

  1. Clone the repository to your local machine.

    ```
    git clone https://github.com/MoustafaGabil/immo-eliza-ml.git
    ```

  2. Navigate to the project directory and install the required dependencies:

    ```
    pip install -r requirements.txt
    ```
  ```
## pipeline for excuting the files: 

**Notes**
- The whole data is separated during the outliers removal into Houses, Apartment ,and combined file for both Houses and Apartments.
- The Notebooks folder contains Notebooks that perform the same process as the scripted files but in form of notebooks. 

1. The main row data is called "properties.csv", it exists in the Cleaning folder, where the cleaned data will be saved too.

2. Running the **cleaning and outliers removal.py** which calls the functions inside the Cleaning folder (removing_outliers.py & cleaning.py ) for cleaning the data and removing the outlier.

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
    The following are the parameters used for model testing and creating heat/correlation maps.
     
    ''' python
        parameters = ["construction_year","total_area_sqm","nbr_frontages","nbr_bedrooms","kitchen_type_encoded","Bulding_sta_encoded","epc_encoded","garden_sqm","surface_land_sqm","fl_double_glazing","fl_terrace","fl_swimming_pool","fl_floodzone","latitude","longitude","zip_code",]

       Heat_map_parmeters = ["price","construction_year","total_area_sqm","nbr_frontages","nbr_bedrooms","kitchen_type_encoded","Bulding_sta_encoded","epc_encoded","garden_sqm","latitude","longitude","zip_code"]

4. The **Results and Evaluation** folder contains the results for different models and comparison based on using different features.
  ```
## Results: 
There are more results available for testing the models and visualization can be found in folders such as:

1. Heat_maps: where you can find the  as indicated in the below figure.
<p align="center">

  <![Correlation between the price and different features by using Houses & Apartments data](heatmap_Houses_and_Apartments_combined_without_outliers.png) />
</p>

2. Evaluation table for different models and differnt properties combinations as follow:

| File                                             | Model name                         | MAE         | RMSE        | R²           | Train Score | Test Score|
|--------------------------------------------------|------------------------------------|-------------|-------------|--------------|-------------|-----------|
| APARTMENT_without_outliers                       | Random Forest Regression           | 38847.567   | 197.098     | 0.7462       | 0.9210      | 0.7462    |
| APARTMENT_without_outliers                       | XGB Regression                     | 31681.849   | 177.994     | 0.8163       | 0.9865      | 0.8163    |
| APARTMENT_without_outliers.csv                   | Hist Gradient Boosting Regression  | 34233.048   | 185.022     | 0.7970       | 0.9134      | 0.7970    |
| Houses_and_Apartments_combined_without_outliers  | Random Forest Regression           | 47965.068   | 219.010     | 0.7684       | 0.9216      | 0.7684    |
| Houses_and_Apartments_combined_without_outliers  | XGB Regression                     | 41086.616   | 202.698     | 0.8175       | 0.9806      | 0.8175    |
| Houses_and_Apartments_combined_without_outliers  | Hist Gradient Boosting Regression  | 46254.587   | 215.069     | 0.7912       | 0.8678      | 0.7912    |
| HOUSE_without_outliers                           | Random Forest Regression           | 58860.515   | 242.612     | 0.7651       | 0.9210      | 0.7651    |
| HOUSE_without_outliers                           | XGB Regression                     | 50102.966   | 223.837     | 0.8162       | 0.9937      | 0.8162    |
| HOUSE_without_outliers.csv                       | Hist Gradient Boosting Regression  | 53392.923   | 231.069     | 0.8018       | 0.9124      | 0.8018    |

**where**

    - MAE:  Mean Absolute Error.
    - RMSE: Root Mean Squared Error.
    - R²:   The coefficient of determination

3.  The some of the feature importance results can found below: </b>
<p align="center">

  <![The some of the feature importance results](<only lan_lon_zip.png>) />
</p>

4. The models after training and testing is saved in the "models" folder to use them later


## 🔧 Updates & Upgrades

The data can be further preprocessed to get more accurate results and more models can be tested. 
The importance of longitude and latitude appeared in the features importance so, It is advisable to have more exploration. 


```

## ⏱️ Project Timeline
The initial setup of this project was completed in 5 days.


