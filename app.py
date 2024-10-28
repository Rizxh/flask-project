from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load the pre-trained model
model_loaded = pickle.load(open('final_model.sav', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define the main route to display the form and results
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect input from form
        Tenure = int(request.form["Tenure"])
        WarehouseToHome = int(request.form["WarehouseToHome"])
        NumberOfDeviceRegistered = int(request.form["NumberOfDeviceRegistered"])
        DaySinceLastOrder = int(request.form["DaySinceLastOrder"])
        CashbackAmount = float(request.form["CashbackAmount"])
        NumberOfAddress = int(request.form["NumberOfAddress"])
        MaritalStatus = request.form["MaritalStatus"]
        Complain = int(request.form["Complain"])
        PreferedOrderCat = request.form["PreferedOrderCat"]
        SatisfactionScore = int(request.form["SatisfactionScore"])

        # Prepare data for prediction
        df_customer = pd.DataFrame({
            'Tenure': [Tenure],
            'WarehouseToHome': [WarehouseToHome],
            'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
            'MaritalStatus': [MaritalStatus],
            'PreferedOrderCat': [PreferedOrderCat],
            'SatisfactionScore': [SatisfactionScore],
            'NumberOfAddress': [NumberOfAddress],
            'CashbackAmount': [CashbackAmount],
            'Complain': [Complain],
            'DaySinceLastOrder': [DaySinceLastOrder]
        })

        # Make prediction
        kelas = model_loaded.predict(df_customer)[0]
        probabilities = model_loaded.predict_proba(df_customer)

        # Prepare results
        result = "CHURN" if kelas == 1 else "STAY"
        probability = probabilities[0][1] if kelas == 1 else probabilities[0][0]
        
        return render_template("index.html", result=result, probability=f"{probability:.2f}", df_customer=df_customer.to_html())
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
