#%% IMPORTS
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from zipfile import ZipFile

#%% DATA LOADING
# Load the training data
X = pd.read_parquet("./X_t0.parquet")
y = pd.read_parquet("./y_t0.parquet")

# Display initial data info
print("Initial Data Information:")
print(X.info())
print("Target Distribution:")
print(y.value_counts())

#%% DATA PREPROCESSING

# Drop the 'wallet_address' as it's a unique identifier with no predictive value
X = X.drop(columns=['wallet_address'])

# Identify numerical and categorical features based on variable descriptions
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Datetime features to be processed separately
datetime_features = [
    'borrow_timestamp', 
    'first_tx_timestamp', 
    'last_tx_timestamp', 
    'risky_first_tx_timestamp', 
    'risky_last_tx_timestamp'
]

# Drop datetime features for now; alternatively, engineer features from them
X = X.drop(columns=datetime_features)

# Update numerical features after dropping datetime
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Check for any remaining categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Display feature types
print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

#%% SPLIT DATA
# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#%% PREPROCESSING PIPELINE
# Define the preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define the preprocessing for categorical data (if any)
if categorical_features:
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
else:
    categorical_transformer = 'passthrough'

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#%% MODEL DEFINITIONS
# Define all model pipelines

# Baseline Pipeline
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('baseline_model', DummyClassifier(strategy='prior', random_state=42))
])

# Logistic Regression Pipeline
logistic_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('logistic_regression', LogisticRegression(
        solver='liblinear',
        random_state=42
    ))
])

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb_classifier', XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ))
])

#%% TRAIN AND EVALUATE MODELS
def train_evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities for the test set
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    
    # Generate classification report using a threshold of 0.5
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred_class)
    
    # Display results
    print(f"{model_name} Model Evaluation")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return roc_auc, logloss

# Evaluate Baseline Model
roc_auc_baseline, logloss_baseline = train_evaluate_model(
    baseline_pipeline, X_train, y_train, X_test, y_test, "Baseline"
)

# Evaluate Logistic Regression
roc_auc_lr, logloss_lr = train_evaluate_model(
    logistic_pipeline, X_train, y_train, X_test, y_test, "Logistic Regression"
)

# Evaluate Random Forest
roc_auc_rf, logloss_rf = train_evaluate_model(
    rf_pipeline, X_train, y_train, X_test, y_test, "Random Forest Classifier"
)

# Evaluate XGBoost
roc_auc_xgb, logloss_xgb = train_evaluate_model(
    xgb_pipeline, X_train, y_train, X_test, y_test, "XGBoost Classifier"
)

#%% SELECT THE BEST MODEL
# Compare models based on ROC AUC and Log Loss
models_performance = {
    'Baseline': {'ROC AUC': roc_auc_baseline, 'Log Loss': logloss_baseline},
    'Logistic Regression': {'ROC AUC': roc_auc_lr, 'Log Loss': logloss_lr},
    'Random Forest': {'ROC AUC': roc_auc_rf, 'Log Loss': logloss_rf},
    'XGBoost': {'ROC AUC': roc_auc_xgb, 'Log Loss': logloss_xgb}
}

# Display models performance
print("\nModels Performance:")
for model, metrics in models_performance.items():
    print(f"{model}: ROC AUC = {metrics['ROC AUC']:.4f}, Log Loss = {metrics['Log Loss']:.4f}")

# Assuming XGBoost performed the best
selected_pipeline = xgb_pipeline

#%% PREDICTION ON NEW DATA
# Load the new data for prediction
X_new = pd.read_parquet("./X_t1.parquet")

# Display new data info
print("\nNew Data Information:")
print(X_new.info())

# Preprocess the new data
# Drop the 'wallet_address' as done previously
X_new_processed = X_new.drop(columns=['wallet_address'])

# Drop the same datetime features as done during training
X_new_processed = X_new_processed.drop(columns=datetime_features)

# Ensure that the new data has the same features as the training data
missing_features = set(X_train.columns) - set(X_new_processed.columns)
if missing_features:
    raise ValueError(f"The following expected features are missing in the new data: {missing_features}")

#%% GENERATE FILES FUNCTION
# Assuming the generateFiles function is provided and should not be edited.
# Including it here for completeness. If it's already defined elsewhere, you can omit this.
def generateFiles(predict_data, clf_pipe):
    """Genera los archivos a subir en CodaLab

    Input
    ---------------
    predict_data: Dataframe con los datos de entrada a predecir
    clf_pipe: pipeline del clf

    Output
    ---------------
    archivo de txt
    """
    y_pred_clf = clf_pipe.predict_proba(predict_data)[:, 1]
    with open('./predictions.txt', 'w') as f:
        for item in y_pred_clf:
            f.write("%s\n" % item)
    
    with ZipFile('predictions.zip', 'w') as zipObj:
        zipObj.write('predictions.txt')
    os.remove('predictions.txt')

#%% GENERATE AND SAVE PREDICTIONS
# Generate and save predictions using the selected pipeline
generateFiles(X_new_processed, selected_pipeline)

# Confirm that the predictions.zip file has been created
if os.path.exists('predictions.zip'):
    print("\nPredictions have been successfully saved to 'predictions.zip'.")
else:
    print("\nError: 'predictions.zip' was not created.")
