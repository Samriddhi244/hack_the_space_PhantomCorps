import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingRegressor
from scipy.stats import randint
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge


geomagnetic_data = pd.read_csv("daily_geomagnetic_data.csv")
solar_data = pd.read_csv("daily_solar_data.csv")

geomagnetic_head = geomagnetic_data.head()
solar_head = solar_data.head()
print(geomagnetic_data.columns)
print(geomagnetic_head)
print(solar_data.columns)
print(solar_head)

geomagnetic_data['Date'] = pd.to_datetime(geomagnetic_data['Timestamp']).dt.date
solar_data['Date'] = pd.to_datetime(solar_data['Date']).dt.date
geomagnetic_data.drop(columns=['Timestamp'], inplace=True)

merged_data = pd.merge(geomagnetic_data, solar_data, on='Date', how='inner')
merged_data.head()

replace_values = {'*': None, -1: None}
merged_data.replace(replace_values, inplace=True)

columns_to_numeric = ['Middle Latitude A', 'High Latitude A', 'Estimated A',
                      'Middle Latitude K', 'High Latitude K', 'Estimated K',
                      'Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)',
                      'Flares: C', 'Flares: M', 'Flares: X', 'Flares: S',
                      'Flares: 1', 'Flares: 2', 'Flares: 3']
for col in columns_to_numeric:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

cleaned_data = merged_data.dropna()
features = ['Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)',
            'Flares: C', 'Flares: M', 'Flares: X', 'Middle Latitude A', 'High Latitude A', 'Estimated A']
target = 'Estimated K'
X = cleaned_data[features]
y = cleaned_data[target]

cleaned_data.head(), X.head(), y.head()


# Define the number of rows and columns for the grid
num_rows = 2  # You can adjust the number of rows
num_cols = 5  # You can adjust the number of columns

# Create a new figure and axis objects
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# List of columns to plot
columns = ['Middle Latitude A', 'High Latitude A', 'Estimated A', 'Middle Latitude K', 'High Latitude K', 'Estimated K','Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)', 'New Regions']

# Plot histograms for each column in a separate subplot
for i, col in enumerate(columns):
    axes[i].hist(merged_data[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(col + ' Distribution')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Show the plot
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]

}

# Initialize Random Forest with RandomizedSearchCV
rf = RandomForestRegressor(random_state=42)

# Perform Randomized Search with Cross-Validation
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the randomized search model
rf_random.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best hyperparameters: {rf_random.best_params_}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_random.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the cross-validation scores
cv_mean = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"Cross-Validation MSE: {cv_mean} ± {cv_std}")

models = {
    'Random Forest': rf_random.best_estimator_,
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse}")

from sklearn.preprocessing import StandardScaler

# Initialize a scaler
scaler = StandardScaler()

# Fit and transform the training data, and transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SMOTE for oversampling
smote = SMOTE(random_state=42)

# Apply SMOTE to training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a model on the resampled dataset
rf.fit(X_train_resampled, y_train_resampled)

# Combine multiple models into a Voting Regressor
voting_model = VotingRegressor([('rf', rf_random.best_estimator_),
                                ('gb', GradientBoostingRegressor(random_state=42)),
                                ('xgb', XGBRegressor(random_state=42))])

# Train the ensemble model
voting_model.fit(X_train, y_train)

# Make predictions
y_pred = voting_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Voting Regressor MSE: {mse}")

y_pred = model.predict(X_test)
print("the predicted values are:")
print(y_pred)
print("the actual values are:")
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2',None]
}
# Initialize the model
gbr = GradientBoostingRegressor(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=gbr, param_distributions=param_grid, 
                                   n_iter=50,  # Number of parameter settings sampled
                                   scoring='neg_mean_squared_error', 
                                   cv=3,  # Cross-validation folds
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1)  # Use all processors

# Fit the model
random_search.fit(X_train, y_train)
# Best hyperparameters found
print("Best Hyperparameters:", random_search.best_params_)

# Predict using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Tuned Model Mean Squared Error:", mse)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

param_dist = {
    'n_estimators': randint(100, 1000),  # Test different tree numbers
    'max_depth': randint(3, 30),         # Explore depths from 3 to 30
    'min_samples_split': randint(2, 20), # Minimum samples to split nodes
    'min_samples_leaf': randint(1, 20),  # Minimum leaf size
    'max_features': [ 'sqrt', 'log2',None]  # Explore different max features
}

# Setup RandomForestRegressor or another model
rf = RandomForestRegressor(random_state=42)

# RandomizedSearchCV to find best hyperparameters
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                   n_iter=10, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print(random_search.best_params_)

# Define base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=200, random_state=42))
]

# Define meta model (can be any regressor)
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# Train stacking model
stacking_model.fit(X_train, y_train)

# Predict
y_pred_stack = stacking_model.predict(X_test)

print("the predicted values are:")
print(y_pred_stack)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')