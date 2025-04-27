import joblib
import numpy as np

# Load the trained model, scalers, and PCA
models = joblib.load("models/extra_trees_model.pkl")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")
pca = joblib.load("models/pca.pkl")

# Function to preprocess input and make a prediction
def predict_price(features):
    features = np.array(features).reshape(1, -1)

    # Apply the same StandardScaler transformation
    features_scaled = scaler_X.transform(features)

    # Apply PCA transformation (consistent with training)
    features_pca = pca.transform(features_scaled)

    # Predict using the trained model
    prediction_scaled = models.predict(features_pca)

    # Convert prediction back to original scale
    prediction_original = scaler_y.inverse_transform([[prediction_scaled[0]]])[0][0]

    return prediction_original
