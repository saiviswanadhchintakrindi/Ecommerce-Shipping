from flask import Flask, request, render_template
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Initializing
app = Flask(__name__)

# Loading models
model = pickle.load(open('Ecommerce_RF_74.h5', 'rb'))
scaler = pickle.load(open("EcommerceScaler.pkl","rb"))

# home route
@app.route('/')
def home():
    return render_template('index.html')

# predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        data = [
            request.form['Warehouse_block'],
            request.form['Mode_of_Shipment'],
            request.form['Customer_care_calls'],
            request.form['Customer_rating'],
            request.form['Cost_of_the_Product'],
            request.form['Prior_purchases'],
            request.form['Product_importance'],
            request.form['Gender'],
            request.form['Discount_offered'],
            request.form['Weight_in_gms']
        ]

        # Convert data to numpy array and reshape for scaler
        data_array = np.array(data, dtype=float).reshape(1, -1)
        
        # Scale data
        scaled_data = scaler.transform(data_array)
        prediction = model.predict(scaled_data)
        
        # prediction
        result = 'Your Product will reach On Time' if prediction == 0 else 'Your product will get Delayed'
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=False)
