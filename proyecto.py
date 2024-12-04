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

for col in tqdm(numerical_features, desc="progress"):
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

#%% MODEL PREPARATION

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def run_classification(X,y,model_name):
    X_train, y_train, X_test, y_test = train_test_split(X, y, 
                                                        random_state=42,
                                                        test_size=.3,
                                                        stratify=y)    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough')
    
    def objective(trial):
        if model_name == 'xgboost':
            gb_params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', XGBClassifier(**gb_params, random_state=42))
                ])
            
        else:
            raise ValueError
            
        scores = cross_val_score(pipeline, X, y, scoring='roc_auc')
        return min([np.mean(scores), np.median(scores)])
    
    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(),
                                pruner=MedianPruner())
    study.optimize(
        objective,
        timeout=60*5,
        show_progress_bar=True)
    
    return study

if __name__ == "__main__":
    run_classification(X, y, model_name='xgboost')
    