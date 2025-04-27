import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df_numeric1 = pd.read_csv("df_numeric1.csv")

X1 = df_numeric1.drop('sales_price', axis=1)  # Features
y1 = df_numeric1['sales_price']  # Target

# Scale the features
scaler_X = StandardScaler()
X_scaled1 = scaler_X.fit_transform(X1)

# Apply PCA
pca1 = PCA(n_components=0.95)  # Keep components explaining 95% of variance
X_pca1 = pca1.fit_transform(X_scaled1)

# Print PCA details
print("Explained Variance Ratio:", pca1.explained_variance_ratio_)
print("Cumulative Explained Variance Ratio:", np.cumsum(pca1.explained_variance_ratio_))
print("Number of components:", pca1.n_components_)

# Create a new DataFrame with PCA components
df_pca1 = pd.DataFrame(data=X_pca1, columns=[f"PC{i+1}" for i in range(X_pca1.shape[1])])

# Concatenate the target variable back
df_pca1['sales_price'] = y1

# Convert to final DataFrame
df_numeric1_pca = pd.DataFrame(df_pca1)

# Scale the target variable
scaler_y = MinMaxScaler()
df_numeric1_pca['sales_price_scaled'] = scaler_y.fit_transform(df_numeric1_pca[['sales_price']])

# Drop the original target variable
df_numeric1_pca = df_numeric1_pca.drop(columns=['sales_price'], errors='ignore')

# Splitting Features and Target
X = df_numeric1_pca.drop('sales_price_scaled', axis=1)
y = df_numeric1_pca['sales_price_scaled']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define the Extra Trees Regressor model
et = ExtraTreesRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=7,
    max_features='sqrt',
    random_state=123
)

# Train the model
et.fit(X_train, y_train)

# Predictions
y_train_pred = et.predict(X_train)
y_test_pred = et.predict(X_test)

# Evaluation on Training Data
print("Training R2 Score:", r2_score(y_train, y_train_pred))
print("MAE (Training):", mean_absolute_error(y_train, y_train_pred))
print("MSE (Training):", mean_squared_error(y_train, y_train_pred))
print("RMSE (Training):", np.sqrt(mean_squared_error(y_train, y_train_pred)))

# Evaluation on Test Data
print("\nTest R2 Score:", r2_score(y_test, y_test_pred))
print("MAE (Test):", mean_absolute_error(y_test, y_test_pred))
print("MSE (Test):", mean_squared_error(y_test, y_test_pred))
print("RMSE (Test):", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Save Model and Scalers inside `/models/`
joblib.dump(et, 'models/extra_trees_model.pkl')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(pca1, 'models/pca.pkl')

