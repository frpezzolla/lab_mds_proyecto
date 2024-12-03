#%% IMPORTS
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, classification_report, roc_curve
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np

#%% CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_STATE = 42
SEED = 42

#%% SET RANDOM SEED
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

#%% DATA LOADING
def load_data():
    X0 = pd.read_parquet("./data/X_t0.parquet")
    y0 = pd.read_parquet("./data/y_t0.parquet")
    X1 = pd.read_parquet("./data/X_t1.parquet")
    y1 = pd.read_parquet("./data/y_t1.parquet")
    X = pd.concat([X0, X1], axis=0).reset_index(drop=True)
    y = pd.concat([y0, y1], axis=0).reset_index(drop=True)
    X_for_predictions = pd.read_parquet("./data/X_t2.parquet")
    return X, y, X_for_predictions

#%% DATA PREPROCESSING
def preprocess_data(X, X_for_predictions):
    # Drop the 'wallet_address' as it's a unique identifier with no predictive value
    X = X.drop(columns=['wallet_address'])
    X_for_predictions = X_for_predictions.drop(columns=['wallet_address'])

    # Identify numerical and categorical features based on variable descriptions
    datetime_features = [
        'borrow_timestamp', 
        'first_tx_timestamp', 
        'last_tx_timestamp', 
        'risky_first_tx_timestamp', 
        'risky_last_tx_timestamp'
    ]
    X = X.drop(columns=datetime_features)
    X_for_predictions = X_for_predictions.drop(columns=datetime_features)

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

    # Handle missing values if any
    X = X.fillna(0)
    X_for_predictions = X_for_predictions.fillna(0)

    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    X_for_predictions[numerical_features] = scaler.transform(X_for_predictions[numerical_features])

    # One-hot encode categorical features
    if categorical_features:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat = encoder.fit_transform(X[categorical_features])
        X_for_predictions_cat = encoder.transform(X_for_predictions[categorical_features])

        # Get new feature names after encoding
        encoded_features = encoder.get_feature_names_out(categorical_features)

        # Create DataFrames from the encoded features
        X_cat_df = pd.DataFrame(X_cat, columns=encoded_features, index=X.index)
        X_for_predictions_cat_df = pd.DataFrame(X_for_predictions_cat, columns=encoded_features, index=X_for_predictions.index)

        # Drop original categorical columns and concatenate encoded columns
        X = X.drop(columns=categorical_features).reset_index(drop=True)
        X = pd.concat([X, X_cat_df.reset_index(drop=True)], axis=1)

        X_for_predictions = X_for_predictions.drop(columns=categorical_features).reset_index(drop=True)
        X_for_predictions = pd.concat([X_for_predictions, X_for_predictions_cat_df.reset_index(drop=True)], axis=1)

    return X, X_for_predictions

#%% CUSTOM DATASET
class CustomDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

#%% MODEL DEFINITIONS
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

#%% TRAINING FUNCTION
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

#%% EVALUATION FUNCTION
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)

            preds = outputs.cpu().numpy()
            labels = y_batch.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    roc_auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    return epoch_loss, roc_auc, logloss, all_labels, all_preds

#%% PLOTTING FUNCTIONS
def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_scores):.4f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

def plot_loss(train_losses, val_losses, model_name):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.legend()
    plt.savefig(f'loss_curve_{model_name}.png')
    plt.close()

#%% GENERATE FILES FUNCTION
def generateFiles(predict_data, model):
    """Generates prediction files for submission.

    Args:
        predict_data (pd.DataFrame): DataFrame containing input features.
        model (nn.Module): Trained PyTorch model.

    Outputs:
        predictions.zip containing predictions.txt
    """
    model.eval()
    dataset = CustomDataset(predict_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)
    # Save predictions to txt
    with open('./predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    # Zip the txt file
    with ZipFile('predictions.zip', 'w') as zipObj:
        zipObj.write('predictions.txt')
    os.remove('predictions.txt')

#%% MAIN FUNCTION
def main():
    set_seed()

    #%% DATA LOADING
    X, y, X_for_predictions = load_data()
    print("Initial Data Information:")
    print(X.info())
    print("Target Distribution:")
    print(y['target'].value_counts())

    #%% DATA PREPROCESSING
    X, X_for_predictions = preprocess_data(X, X_for_predictions)
    print("Preprocessed Data Information:")
    print(X.info())

    #%% SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y['target'], test_size=0.3, random_state=RANDOM_STATE, stratify=y['target']
    )

    #%% CREATE DATALOADERS
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #%% MODEL INITIALIZATION
    input_dim = X_train.shape[1]
    model = BinaryClassificationModel(input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #%% TRAINING LOOP
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, roc_auc, logloss, _, _ = evaluate_model(model, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - ROC AUC: {roc_auc:.4f} - Log Loss: {logloss:.4f}")

    #%% FINAL EVALUATION
    val_loss, roc_auc, logloss, y_true, y_scores = evaluate_model(model, test_loader, criterion)
    print("\nFinal Evaluation on Test Set")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    y_pred_class = (y_scores >= 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class))

    #%% PLOTTING
    plot_roc_curve(y_true, y_scores, "PyTorch_Model")
    plot_loss(train_losses, val_losses, "PyTorch_Model")
    print("\nPlots have been saved as PNG files.")

    #%% GENERATE AND SAVE PREDICTIONS
    print("\nGenerating Predictions on New Data...")
    generateFiles(X_for_predictions, model)
    if os.path.exists('predictions.zip'):
        print("Predictions have been successfully saved to 'predictions.zip'.")
    else:
        print("Error: 'predictions.zip' was not created.")

if __name__ == "__main__":
    main()
