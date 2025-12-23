
# for data manipulation
import pandas as pd
import numpy as np

import sklearn
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.pipeline import Pipeline # Import Pipeline
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# for model training, tuning, and evaluation (regression metrics)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error



# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/krishpvg/visit-with-us-data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# Drop unique identifier column (not useful for modeling)
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)
def build_preprocessor():
  # List of numerical features
  numeric_features = [
     'Age',
     'DurationOfPitch',
     'NumberOfFollowups',
     'NumberOfTrips',
     'NumberOfChildrenVisiting',
     'MonthlyIncome'
  ]

  # List of categorical features
  onehot_categorical_features = [
      'TypeOfContact',
     'Occupation',
     'Gender',
     'ProductPitched',
     'MaritalStatus',
     'Designation'
  ]

  ordinal_categorical_features = [
      'CityTier',
     'PreferredPropertyStar',
     'PitchSatisfactionScore'
  ] 

  binary_categorical_features = [
  'Passport',
  'Owncar'   
  ]

  ordinal_categories = [
     [1, 2, 3],
     [3, 4, 5],
     [1, 2, 3, 4, 5]
  ]


  preprocessor = make_column_transformer(
     (StandardScaler(), numeric_features),
     (OneHotEncoder(handle_unknown='ignore', drop='first'), onehot_categorical_features),
     (OrdinalEncoder(categories=ordinal_categories), ordinal_categorical_features),
     ('passthrough', binary_categorical_features)
  )
  
  return preprocessor


# Define the target variable for the regression task
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="krishpvg/visit-with-us-data",
        repo_type="dataset",
    )
