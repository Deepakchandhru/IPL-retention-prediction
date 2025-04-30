import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load and preprocess data
df = pd.read_csv('processed_ipl.csv')
df = df[df['Season'].isin([22, 23, 24])]

df_grouped = df.groupby(['Season', 'Striker']).agg({
    'total_runs': 'sum',
    'balls_faced': 'sum',
    'strike_rate': 'mean',
    'total_wickets': 'sum',
    'total_runs_conceded': 'sum',
    'balls_bowled': 'sum',
    'economy_rate': 'mean'
}).reset_index()

df_grouped = df_grouped[(df_grouped['Striker'].notnull()) & (df_grouped['Striker'] != '0')]

# Retention rule (for training the retention model)
df_grouped['Retention'] = ((df_grouped['total_runs'] > 300) | (df_grouped['total_wickets'] > 10) |
                            ((df_grouped['total_runs'] > 150) & (df_grouped['strike_rate'] > 140)) |
                            ((df_grouped['total_wickets'] > 5) & (df_grouped['economy_rate'] < 8))).astype(int)

# Features and target for retention model
X = df_grouped[['total_runs', 'balls_faced', 'strike_rate', 'total_wickets', 'total_runs_conceded', 'balls_bowled', 'economy_rate']]
y = df_grouped['Retention']
X = X.fillna(0)

# Train-test split for retention model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retention model (XGBoost Classifier)
retention_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
retention_model.fit(X_train, y_train)

# Save retention model
retention_model.save_model('ipl_retention_model.json')

# Salary prediction model: Use performance score to map salary
retained_players = df_grouped[df_grouped['Retention'] == 1].copy()

# Compute performance score (for salary prediction)
def compute_score(row):
    score = (
            0.5 * row['total_runs'] +
            0.3 * row['strike_rate']
        )
    if row['total_wickets'] > 0:
        score += 150
        score += 10 * row['total_wickets']
        score -= 0.3 * row['economy_rate']
    return score

# Add performance score to retained players
retained_players['Performance Score'] = retained_players.apply(compute_score, axis=1)

# Normalize salary using MinMaxScaler
salary_range = (400, 1600)  # min and max salary (in Lakhs)
scaler = MinMaxScaler(feature_range=(salary_range[0], salary_range[1]))
retained_players['Calculated Salary (Lakhs)'] = scaler.fit_transform(retained_players[['Performance Score']])

# Train salary prediction model using performance score
X_salary = retained_players[['Performance Score']]
y_salary = retained_players['Calculated Salary (Lakhs)']

X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

# Salary model (XGBoost Regressor)
salary_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
salary_model.fit(X_train_salary, y_train_salary)

# Save salary model
salary_model.save_model('ipl_salary_prediction_model.json')

# Save data with predicted salary
retained_players.to_csv('retained_players_with_salary.csv', index=False)
