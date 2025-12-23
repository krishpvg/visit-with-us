# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score, log_loss
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/krishpvg/visit-with-us-data/Xtrain.csv"
Xtest_path = "hf://datasets/krishpvg/visit-with-us-data/Xtest.csv"
ytrain_path = "hf://datasets/krishpvg/visit-with-us-data/ytrain.csv"
ytest_path = "hf://datasets/krishpvg/visit-with-us-data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

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
      'TypeofContact',
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
  'OwnCar' # Corrected 'Owncar' to 'OwnCar'
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

  # Define base XGBoost Regressor
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Pipeline
preprocessor = build_preprocessor()
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

  # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Predicted probabilities for ROC AUC and log loss
    y_pred_train_prob = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_prob = best_model.predict_proba(Xtest)[:, 1]

    # Classification metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_roc_auc = roc_auc_score(ytrain, y_pred_train_prob)
    test_roc_auc = roc_auc_score(ytest, y_pred_test_prob)

    train_logloss = log_loss(ytrain, y_pred_train_prob)
    test_logloss = log_loss(ytest, y_pred_test_prob)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "train_logloss": train_logloss,
        "test_logloss": test_logloss
    })

    # Save the model locally
    model_path = "best_product_taken_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "krishpvg/visit-with-us"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_product_taken_model_v1.joblib",
        path_in_repo="product_taken_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
