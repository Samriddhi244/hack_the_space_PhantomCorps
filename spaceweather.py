#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#importing the dataset
geomagnetic_data = pd.read_csv("daily_geomagnetic_data.csv")
solar_data = pd.read_csv("daily_solar_data.csv")

#printing the top 5 columns along with the labels
geomagnetic_head = geomagnetic_data.head()
solar_head = solar_data.head()

print(geomagnetic_head)
print(solar_head)

#printing the column names
print(geomagnetic_data.columns)
print(solar_data.columns)

#changes the "timestamp" label to "Date" to date format
geomagnetic_data['Date'] = pd.to_datetime(geomagnetic_data['Timestamp']).dt.date
solar_data['Date'] = pd.to_datetime(solar_data['Date']).dt.date
geomagnetic_data.drop(columns=['Timestamp'], inplace=True)

#merging the solar data and the geomagnetic data 
merged_data = pd.merge(geomagnetic_data, solar_data, on='Date', how='inner')
merged_data.head()

#replacing values
replace_values = {'*': None, -1: None}
merged_data.replace(replace_values, inplace=True)

#converting the column values to numeric and if the cant it is converted to Nan
columns_to_numeric = ['Middle Latitude A', 'High Latitude A', 'Estimated A', 
                      'Middle Latitude K', 'High Latitude K', 'Estimated K', 
                      'Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)',
                      'Flares: C', 'Flares: M', 'Flares: X', 'Flares: S', 
                      'Flares: 1', 'Flares: 2', 'Flares: 3']
for col in columns_to_numeric:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

#cleaning the data and setting target and features
cleaned_data = merged_data.dropna()
features = ['Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)', 
            'Flares: C', 'Flares: M', 'Flares: X', 'Middle Latitude A', 'High Latitude A', 'Estimated A']
target = 'Estimated K'
X = cleaned_data[features]
y = cleaned_data[target]
cleaned_data.head(), X.head(), y.head()

#splitting the dataset to test set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Manual tuning example
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)

#using RandomizedSearchCV to search for the best hyperparameter combination.
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [ 'sqrt', 'log2',None]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train, y_train)

# Best parameters found
print(random_search.best_params_)

#the output :- Fitting 3 folds for each of 10 candidates, totalling 30 fits
#{'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10}


# Make predictions on the test data
y_pred = rf.predict(X_test)
print(y_pred)

#the output :- [2.48319092 0.30383771 1.7343742  ... 0.51653312 0.72879499 1.3000408 ]


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared to measure the proportion of variance explained by the model
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Mean Squared Error: 0.7667997279588548
# R-squared: 0.5505911564387093


# Get the feature importances
importances = rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

#the output :-                     Feature  Importance
#8                 Estimated A    0.896966
#7             High Latitude A    0.021553
#0           Radio Flux 10.7cm    0.021015
#1              Sunspot Number    0.018388
#2  Sunspot Area (10^6 Hemis.)    0.016658
#6           Middle Latitude A    0.015183
#3                   Flares: C    0.007298
#4                   Flares: M    0.002269
#5                   Flares: X    0.000671

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
