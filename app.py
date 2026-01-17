from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open("loan_model(2).pkl", "rb") as file:
    data = pickle.load(file)

rf = data["model"]
scaler = data["scaler"]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():

    no_of_dependents = int(request.form["no_of_dependents"])
    education = request.form["education"]
    self_employed = request.form["self_employed"]
    income_annum = float(request.form["income_annum"])
    loan_amount = float(request.form["loan_amount"])
    loan_term = int(request.form["loan_term"])
    cibil_score = int(request.form["cibil_score"])
    residential_assets_value = float(request.form["residential_assets_value"])
    commercial_assets_value = float(request.form["commercial_assets_value"])
    luxury_assets_value = float(request.form["luxury_assets_value"])
    bank_asset_value = float(request.form["bank_asset_value"])

    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    input_data = np.array([[  
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    input_data_scaled = scaler.transform(input_data)

    prediction = rf.predict(input_data_scaled)[0]
    probability = rf.predict_proba(input_data_scaled)[0][1] * 100

    result = "Approved ✅" if prediction == 1 else "Rejected ❌"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
