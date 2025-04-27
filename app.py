from flask import Flask, render_template, request, jsonify
from deployment import predict_price, models, scaler_X, pca ,scaler_y # Import the function from deployment.py
import pandas as pd
import joblib

app = Flask(__name__)
# code that redirects the app to the main web page
@app.route("/")
def about():
    return render_template("about.html")
# predict is something that will allow us to take the input parameter from the user
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            'Country_encoded': data.get('Country_encoded'),
            'ProductName_encoded': data.get('ProductName_encoded'),
            'ProductType_encoded': data.get('ProductType_encoded'),
            'Product_category_encoded': data.get('Product_category_encoded'),
            'variant_size_id': data.get('variant_size_id'),
            'variant_id': data.get('variant_id'),
            'Review': float(data.get('Review', 0)),
            'discount': float(data.get('discount', 0))
        }])

        # Transform using trained scaler and PCA
        input_scaled = scaler_X.transform(input_df)
        input_pca = pca.transform(input_scaled)

        # Predict and inverse scale
        prediction_scaled = models.predict(input_pca)[0]
        prediction_price = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

        return jsonify({'predicted_price': round(prediction_price, 2)})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    price = request.args.get('price', default=None, type=float)
    return render_template('result.html', prediction=price)

if __name__ == "__main__":
    app.run(debug=True)
