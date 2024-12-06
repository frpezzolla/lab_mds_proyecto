#%%
import pandas as pd

X0 = pd.read_parquet("./data/X_t0.parquet")
y0 = pd.read_parquet("./data/y_t0.parquet")
X1 = pd.read_parquet("./data/X_t1.parquet")
y1 = pd.read_parquet("./data/y_t1.parquet")
X = pd.concat([X0, X1], axis=0)
y = pd.concat([y0, y1], axis=0)

X_for_predictions = pd.read_parquet("./data/X_t2.parquet") # for predictions


#%% DATA TYPES
data_types = {
    'borrow_block_number': 'numerical',  # Block number where the loan occurred
    'borrow_timestamp': 'datetime',  # Timestamp indicating when the loan occurred
    'wallet_address': 'categorical',  # Address of the wallet associated with the transactions
    'first_tx_timestamp': 'datetime',  # Date and time of the wallet's first transaction
    'last_tx_timestamp': 'datetime',  # Date and time of the wallet's last transaction
    'wallet_age': 'numerical',  # Age of the wallet in days
    'incoming_tx_count': 'numerical',  # Total number of incoming transactions
    'outgoing_tx_count': 'numerical',  # Total number of outgoing transactions
    'net_incoming_tx_count': 'numerical',  # Net difference between incoming and outgoing transactions
    'total_gas_paid_eth': 'numerical',  # Total gas fees paid in Ethereum (ETH)
    'avg_gas_paid_per_tx_eth': 'numerical',  # Average gas fee paid per transaction in ETH
    'risky_tx_count': 'numerical',  # Number of transactions classified as risky
    'risky_unique_contract_count': 'numerical',  # Number of unique contracts involved in risky transactions
    'risky_first_tx_timestamp': 'datetime',  # Timestamp of the first risky transaction
    'risky_last_tx_timestamp': 'datetime',  # Timestamp of the last risky transaction
    'risky_first_last_tx_timestamp_diff': 'numerical',  # Time difference between the first and last risky transactions
    'risky_sum_outgoing_amount_eth': 'numerical',  # Total outgoing ETH in risky transactions
    'outgoing_tx_sum_eth': 'numerical',  # Total outgoing ETH in all transactions
    'incoming_tx_sum_eth': 'numerical',  # Total incoming ETH in all transactions
    'outgoing_tx_avg_eth': 'numerical',  # Average outgoing ETH per transaction
    'incoming_tx_avg_eth': 'numerical',  # Average incoming ETH per transaction
    'max_eth_ever': 'numerical',  # Maximum ETH balance ever recorded in the wallet
    'min_eth_ever': 'numerical',  # Minimum ETH balance ever recorded in the wallet
    'total_balance_eth': 'numerical',  # Current total balance of the wallet in ETH
    'risk_factor': 'numerical',  # Calculated risk factor for the wallet
    'total_collateral_eth': 'numerical',  # Total collateral in ETH associated with the wallet
    'total_collateral_avg_eth': 'numerical',  # Average collateral in ETH
    'total_available_borrows_eth': 'numerical',  # Total ETH available for borrowing
    'total_available_borrows_avg_eth': 'numerical',  # Average ETH available for borrowing
    'avg_weighted_risk_factor': 'numerical',  # Weighted average of the risk factor
    'risk_factor_above_threshold_daily_count': 'numerical',  # Number of days the risk factor exceeded a threshold
    'avg_risk_factor': 'numerical',  # General average of the risk factor
    'max_risk_factor': 'numerical',  # Maximum risk factor value recorded
    'borrow_amount_sum_eth': 'numerical',  # Total amount borrowed in ETH
    'borrow_amount_avg_eth': 'numerical',  # Average amount borrowed in ETH
    'borrow_count': 'numerical',  # Total number of borrowing transactions
    'repay_amount_sum_eth': 'numerical',  # Total amount repaid in ETH
    'repay_amount_avg_eth': 'numerical',  # Average amount repaid in ETH
    'repay_count': 'numerical',  # Total number of repayment transactions
    'borrow_repay_diff_eth': 'numerical',  # Difference between the amount borrowed and repaid in ETH
    'deposit_count': 'numerical',  # Total number of deposits made
    'deposit_amount_sum_eth': 'numerical',  # Total deposit amount in ETH
    'time_since_first_deposit': 'numerical',  # Time elapsed since the first deposit
    'withdraw_amount_sum_eth': 'numerical',  # Total amount withdrawn in ETH
    'withdraw_deposit_diff_if_positive_eth': 'numerical',  # Positive difference between withdrawals and deposits in ETH
    'liquidation_count': 'numerical',  # Total number of liquidations recorded
    'time_since_last_liquidated': 'numerical',  # Time elapsed since the last liquidation
    'liquidation_amount_sum_eth': 'numerical',  # Total amount liquidated in ETH
    'market_adx': 'numerical',  # Average Directional Index of the market
    'market_adxr': 'numerical',  # Smoothed Average Directional Index of the market
    'market_apo': 'numerical',  # Absolute Price Oscillator of the market
    'market_aroonosc': 'numerical',  # Aroon Oscillator for the market
    'market_aroonup': 'numerical',  # Aroon-Up value of the market
    'market_atr': 'numerical',  # Average True Range of the market
    'market_cci': 'numerical',  # Commodity Channel Index of the market
    'market_cmo': 'numerical',  # Chande Momentum Oscillator of the market
    'market_correl': 'numerical',  # Market correlation
    'market_dx': 'numerical',  # Directional Index of the market
    'market_fastk': 'numerical',  # Fast %K stochastic component of the market
    'market_fastd': 'numerical',  # Fast %D stochastic component of the market
    'market_ht_trendmode': 'categorical',  # Hilbert trend mode of the market
    'market_linearreg_slope': 'numerical',  # Linear regression slope of the market
    'market_macd_macdext': 'numerical',  # Extended MACD line of the market
    'market_macd_macdfix': 'numerical',  # Fixed MACD line of the market
    'market_macd': 'numerical',  # MACD line of the market
    'market_macdsignal_macdext': 'numerical',  # Extended MACD signal line of the market
    'market_macdsignal_macdfix': 'numerical',  # Fixed MACD signal line of the market
    'market_macdsignal': 'numerical',  # MACD signal line of the market
    'market_max_drawdown_365d': 'numerical',  # Maximum drawdown over 365 days in the market
    'market_natr': 'numerical',  # Normalized Average True Range of the market
    'market_plus_di': 'numerical',  # Positive Directional Indicator of the market
    'market_plus_dm': 'numerical',  # Positive Directional Movement of the market
    'market_ppo': 'numerical',  # Percentage Price Oscillator of the market
    'market_rocp': 'numerical',  # Rate of Change Percentage of the market price
    'market_rocr': 'numerical',  # Rate of Change Ratio of the market price
    'unique_borrow_protocol_count': 'numerical',  # Number of unique borrowing protocols used
    'unique_lending_protocol_count': 'numerical',  # Number of unique lending protocols active
    # 'target': 'categorical'  # Target variable indicating client delinquency status
}

# Define lists for different data types
numerical_features = [col for col, dtype in data_types.items() if dtype == 'numerical']
categorical_features = [col for col, dtype in data_types.items() if dtype == 'categorical']
datetime_features = [col for col, dtype in data_types.items() if dtype == 'datetime']
other_features = [col for col, dtype in data_types.items() if dtype not in ['numerical', 'categorical', 'datetime']]

minmax_scaler_features = [
    'market_rocr', 'market_correl', 
    'unique_borrow_protocol_count', 'unique_lending_protocol_count'
]

power_transformer_features = [
    'avg_gas_paid_per_tx_eth', 'avg_weighted_risk_factor', 'borrow_block_number',
    'incoming_tx_sum_eth', 'max_risk_factor', 'repay_count',
    'risk_factor_above_threshold_daily_count', 'risky_tx_count',
    'risky_unique_contract_count', 'total_available_borrows_eth',
    'total_available_borrows_avg_eth', 'total_balance_eth',
    'total_collateral_avg_eth', 'total_collateral_eth', 'total_gas_paid_eth',
    'wallet_age', 'withdraw_amount_sum_eth', 'withdraw_deposit_diff_if_positive_eth'
]

robust_scaler_features = [
    'avg_risk_factor', 'borrow_amount_avg_eth', 'borrow_amount_sum_eth',
    'borrow_count', 'borrow_repay_diff_eth', 'deposit_amount_sum_eth',
    'deposit_count', 'incoming_tx_count', 'liquidation_amount_sum_eth',
    'liquidation_count', 'market_max_drawdown_365d', 'market_plus_dm',
    'net_incoming_tx_count', 'outgoing_tx_avg_eth', 'outgoing_tx_count',
    'outgoing_tx_sum_eth', 'repay_amount_avg_eth', 'repay_amount_sum_eth',
    'risky_first_last_tx_timestamp_diff', 'risky_sum_outgoing_amount_eth',
    'time_since_first_deposit', 'time_since_last_liquidated'
]

standard_scaler_features = [
    'market_adxr', 'market_apo', 'market_aroonosc', 'market_aroonup',
    'market_atr', 'market_cci', 'market_cmo', 'market_dx', 'market_fastd',
    'market_fastk', 'market_linearreg_slope', 'market_macd',
    'market_macd_macdext', 'market_macd_macdfix', 'market_macdsignal',
    'market_macdsignal_macdext', 'market_macdsignal_macdfix', 'market_natr',
    'market_plus_di', 'market_ppo', 'market_rocp', 'risk_factor'
]


# print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
# print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")
# print(f"Datetime Features ({len(datetime_features)}): {datetime_features}")
# print(f"Other Features ({len(other_features)}): {other_features}")

#%% DATA PREPROCESSING
import numpy as np

# It seems that 999... is a placeholder
X['time_since_last_liquidated'] = X['time_since_last_liquidated'].replace(999999999.0, np.nan)
# It seems that 100... is a placeholder
X['max_risk_factor'] = X.max_risk_factor.replace(1000000, np.nan)

#%% EXPLORATORY DATA ANALYSIS

import matplotlib.pyplot as plt
from tqdm import tqdm

for col in tqdm(minmax_scaler_features, desc="progress"):
    plt.hist(X[col], bins=25, log=True)
    plt.xlabel(col)
    plt.savefig(f'./plots/hist_{col}.png', bbox_inches='tight')
    plt.show()

    

#%%
import matplotlib.pyplot as plt
from tqdm import tqdm

for feature_set in [
    minmax_scaler_features, power_transformer_features,
    robust_scaler_features, standard_scaler_features
]:
    for col in feature_set:
        plt.hist(X[col], bins=25, log=True)
        plt.xlabel(col)
        plt.savefig(f'./plots/hist_{col}.png', bbox_inches='tight')
        plt.show()

#%% UNIQUE LENDING/BORROW PROTOCOL COUNT

unique_wallet_addresses = X['wallet_address'].unique()

max_count = 15
wallets_seen = []

for protocol_type in ['lending', 'borrow']:
    count = 0
    for _, row in X.iterrows():
        current_wallet = row['wallet_address']
        if current_wallet not in wallets_seen and count < max_count:
            if pd.notna(current_wallet):  # Ensure wallet address is valid
                mask = X['wallet_address'] == current_wallet
                plt.hist(X[mask][f'unique_{protocol_type}_protocol_count'], bins=25)
                plt.title(f'{current_wallet} {count}')
                plt.xlabel(f'{protocol_type}')
                plt.show()  # Display the plot
                wallets_seen.append(current_wallet)  # Add wallet to seen list
                count += 1
                
#%%

import seaborn as sns

sns.heatmap(X[numerical_features].corr())
plt.show()


#%%

stats_numerical = X[numerical_features].describe()
stats_numerical

#%%

stats_categorical = X[categorical_features].describe(include='all')
stats_categorical

#%%

from scipy.stats import shapiro, skew, kurtosis

shapiro_results = {}
skew_results = {}
kurtosis_results = {}

for col in numerical_features:
    data = X[col].dropna()
    shapiro_results[col] = shapiro(data)
    skew_results[col] = skew(data)
    kurtosis_results[col] = kurtosis(data)

print("Shapiro-Wilk Test Results:")
for col, result in shapiro_results.items():
    print(f"{col}: W={result[0]}, p-value={result[1]}")

print("\nSkewness Results:")
for col, result in skew_results.items():
    print(f"{col}: skewness={result}")

print("\nKurtosis Results:")
for col, result in kurtosis_results.items():
    print(f"{col}: kurtosis={result}")

#%% MODEL PREPARATION

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc


from sklearn.metrics import precision_recall_curve, auc

# def run_classification(X,y,model_name):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                         random_state=1936,
#                                                         test_size=.3,
#                                                         stratify=y)    
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('minmax', MinMaxScaler(), minmax_scaler_features),
#             ('power', PowerTransformer(method='yeo-johnson'), power_transformer_features),
#             ('robust', RobustScaler(), robust_scaler_features),
#             ('standard', StandardScaler(), standard_scaler_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
#         remainder='passthrough')
    
#     def objective(trial):
#         if model_name == 'xgboost':
#             gb_params = {
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#                 'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduce max depth
#                 'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),  # Increase range
#                 'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Focus on larger subsamples
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Avoid very small values
#                 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1 regularization
#                 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)  # L2 regularization
#             }

#             pipeline = Pipeline(steps=[
#                     ('preprocessor', preprocessor),
#                     ('classifier', XGBClassifier(**gb_params, random_state=1936))
#                 ])
            
#         else:
#             raise ValueError
        
#         pipeline.fit(X_train, y_train)
#         y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
#         precision, recall, _ = precision_recall_curve(y_train, y_train_pred_proba)
#         auc_pr = auc(recall, precision)
#         return auc_pr
    
#     study = optuna.create_study(direction='maximize',
#                                 sampler=TPESampler(),
#                                 pruner=MedianPruner())
#     study.optimize(
#         objective,
#         timeout=60*60,
#         # n_trials=15,
#         show_progress_bar=True)

#     print(f"Best Parameters: {study.best_params}")
#     print(f"Best AUC-PR Score: {study.best_value}")
    
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', XGBClassifier(**study.best_params, random_state=42))
#     ])
    
#     pipeline.fit(X_train, y_train)
#     test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
#     precision, recall, _ = precision_recall_curve(y_test, test_pred_proba)
#     test_auc_pr = auc(recall, precision)
#     print(f"Test AUC-PR Score: {test_auc_pr}")

#     return study

# if __name__ == "__main__":
#     best_study = run_classification(X, y, model_name='xgboost')


#%%

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc

def run_classification(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=1936,
                                                        test_size=.3,
                                                        stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), minmax_scaler_features),
            ('power', PowerTransformer(method='yeo-johnson'), power_transformer_features),
            ('robust', RobustScaler(), robust_scaler_features),
            ('standard', StandardScaler(), standard_scaler_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough')
    
    def objective(trial):
        gb_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
        }
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**gb_params, random_state=1936, device='cuda'))
        ])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            pipeline.fit(X_train_fold, y_train_fold)
            y_val_pred = pipeline.predict_proba(X_val_fold)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val_fold, y_val_pred)
            auc_pr = auc(recall, precision)
            aucs.append(auc_pr)
            
            # Report intermediate score for pruning
            trial.report(np.mean(aucs), step=fold_idx)
            
            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return np.mean(aucs)

    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
    study.optimize(objective, timeout=60*60, show_progress_bar=True)
    
    print(f"Best Parameters: {study.best_params}")
    print(f"Best AUC-PR Score (CV): {study.best_value}")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**study.best_params, random_state=1936, device='cuda'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    test_auc_pr = auc(recall, precision)
    print(f"Test AUC-PR Score: {test_auc_pr}")
    return study

if __name__ == "__main__":
    best_study = run_classification(X, y, model_name='xgboost')


# %%

from zipfile import ZipFile
import os


def generateFiles(predict_data, clf_pipe):
    """Genera los archivos a subir en CodaLab

    Input
    ---------------
    predict_data: Dataframe con los datos de entrada a predecir
    clf_pipe: pipeline del clf

    Ouput
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

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=1936,
                                                        test_size=.3,
                                                        stratify=y)     
    preprocessor = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), minmax_scaler_features),
            ('power', PowerTransformer(method='yeo-johnson'), power_transformer_features),
            ('robust', RobustScaler(), robust_scaler_features),
            ('standard', StandardScaler(), standard_scaler_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough')

    pipe_clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**best_study.best_params, random_state=1936, device='cuda'))
        ])
    
    pipe_clf.fit(X_train,y_train)    
    generateFiles(X_for_predictions, pipe_clf)