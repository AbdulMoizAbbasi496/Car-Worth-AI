from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Load data
data = pd.read_csv('cars_processed.csv')
X = data.drop('selling_price', axis=1)
y = data['selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Track best model
best_score = -np.inf
best_model = None
best_model_name = ""

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Use R² as main metric to track best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# Output best model
print(f"\n✅ Best Model: {best_model_name} with R² Score = {best_score:.4f}")

# Save best model
joblib.dump(best_model, 'model.joblib')
# So , random forest regressor will be ussed.