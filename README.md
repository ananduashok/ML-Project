---

**Assignment 1**: assignment-1.ipynb, Dataset: house_price.csv

**Assignment 2**: assignment-2.ipynb, Dataset: Employee.csv

**Assignment 3**: assignment-3.ipynb, Dataset: California Housing dataset available in the sklearn library, Note: assignment-3-note.txt

**Assignment 4**: assignment-4.ipynb, Dataset: Breast cancer dataset available in the sklearn library, Note: assignment-4-note.txt

**Assignment 5**: assignment-5.ipynb, Dataset: Iris dataset available in the sklearn library, Note: assignment-5-note.txt

**Project**: Project-ML.ipynb, Dataset: CarPrice_Assignment.csv, Note: README.md

# Car Price Prediction Project

## Project Overview

This project aims to predict the price of cars based on various features using different regression algorithms. The dataset used in this project contains information about different types of cars in the American market. The goal is to understand the factors affecting car prices and build a model that can accurately predict car prices.

## Dataset

The dataset used in this project is `CarPrice_Assignment.csv`. It contains the following columns:

- `car_ID`: Unique ID for each car
- `symboling`: Risk factor symbol assigned to the car
- `CarName`: Name of the car
- `fueltype`: Type of fuel used by the car (gas/diesel)
- `aspiration`: Aspiration type (std/turbo)
- `doornumber`: Number of doors (two/four)
- `carbody`: Body style of the car (convertible/hatchback/sedan/wagon/hardtop)
- `drivewheel`: Type of drive wheel (fwd/rwd/4wd)
- `enginelocation`: Location of the engine (front/rear)
- `wheelbase`: Wheelbase of the car
- `carlength`: Length of the car
- `carwidth`: Width of the car
- `carheight`: Height of the car
- `curbweight`: Curb weight of the car
- `enginetype`: Type of engine (dohc/ohc/ohcf/ohcv/rotor/l)
- `cylindernumber`: Number of cylinders in the engine
- `enginesize`: Size of the engine
- `fuelsystem`: Type of fuel system (mpfi/2bbl/mfi/1bbl/spdi/idi)
- `boreratio`: Bore ratio of the engine
- `stroke`: Stroke of the engine
- `compressionratio`: Compression ratio of the engine
- `horsepower`: Horsepower of the car
- `peakrpm`: Peak RPM of the car
- `citympg`: City mileage of the car
- `highwaympg`: Highway mileage of the car
- `price`: Price of the car

## Project Steps

### 1. Loading and Preprocessing

- Load the dataset using pandas.
- Perform necessary preprocessing steps such as handling missing values, encoding categorical variables, and splitting the data into features and target variable.

### 2. Model Implementation

- Implement the following five regression algorithms:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosting Regressor
  5. Support Vector Regressor

### 3. Model Evaluation

- Evaluate the performance of all the models based on R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- Identify the best performing model and justify why it is the best.

### 4. Feature Importance Analysis

- Identify the significant variables affecting car prices using feature importance from models like Random Forest Regressor and Gradient Boosting Regressor.

### 5. Hyperparameter Tuning

- Perform hyperparameter tuning for the best performing model to improve its performance.
- Evaluate the tuned model on the test set.

## Code

### Loading and Preprocessing

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Display the first few rows of the dataframe
print(df.head())
```

### Model Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Preprocessing
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Define the independent variables (features) and the dependent variable (target)
X = df.drop(['car_ID', 'price'], axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

# Train the models and evaluate their performance
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "R-squared": r2_score(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
    }

# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
```

### Feature Importance Analysis

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Preprocessing
df = pd.get_dummies(df, drop_first=True)

# Define the independent variables (features) and the dependent variable (target)
X = df.drop(['car_ID', 'price'], axis=1)
y = df['price']

# Train a Random Forest Regressor to get feature importances
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(feature_importances.head(10))
```

### Hyperparameter Tuning

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Preprocess the data
df['CarName'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df = pd.get_dummies(df, drop_first=True)

# Split the data into features and target variable
X = df.drop(['car_ID', 'price'], axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and its hyperparameters for tuning
model = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10]
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, params, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
results = {
    'R-squared': r2_score(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred)
}

# Print the results
print("Best Model (Random Forest Regressor) after Hyperparameter Tuning:")
for metric_name, value in results.items():
    print(f"  {metric_name}: {value}")
```

### Visualizing Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Preprocessing
df = pd.get_dummies(df, drop_first=True)

# Define the independent variables (features) and the dependent variable (target)
X = df.drop(['car_ID', 'price'], axis=1)
y = df['price']

# Train a Random Forest Regressor to get feature importances
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Feature Importances')
plt.show()
```

## Conclusion

This project demonstrates how to build and evaluate regression models to predict car prices based on various features. The Random Forest Regressor was identified as the best performing model.
