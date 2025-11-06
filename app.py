from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from time import sleep

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Dummy model
class DummyModel:
    def predict(self, X):
        return [1 if X["SMOKING"][0] == 1 else 0}

model = DummyModel()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username and password:
            session["user"] = username
            sleep(1)
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Please fill in all fields.")
    return render_template("login.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        form = request.form
        encode = lambda v: 1 if v == "Yes" else 0
        GENDER = 0 if form["GENDER"] == "Male" else 1

        input_data = pd.DataFrame({
            'GENDER': [GENDER],
            'AGE': [int(form["AGE"])],
            'SMOKING': [encode(form["SMOKING"])],
            'YELLOW_FINGERS': [encode(form["YELLOW_FINGERS"])],
            'ANXIETY': [encode(form["ANXIETY"])],
            'PEER_PRESSURE': [encode(form["PEER_PRESSURE"])],
            'CHRONIC_DISEASE': [encode(form["CHRONIC_DISEASE"])],
            'FATIGUE': [encode(form["FATIGUE"])],
            'ALLERGY': [encode(form["ALLERGY"])],
            'WHEEZING': [encode(form["WHEEZING"])],
            'ALCOHOL_CONSUMING': [encode(form["ALCOHOL_CONSUMING"])],
            'COUGHING': [encode(form["COUGHING"])],
            'SHORTNESS_OF_BREATH': [encode(form["SHORTNESS_OF_BREATH"])],
            'SWALLOWING_DIFFICULTY': [encode(form["SWALLOWING_DIFFICULTY"])],
            'CHEST_PAIN': [encode(form["CHEST_PAIN"])]
        })

        pred = model.predict(input_data)[0]
        session["result"] = "⚠️ High Risk of Lung Cancer Detected" if pred == 1 else "✅ No Lung Cancer Risk Detected"
        return redirect(url_for("thankyou"))

    return render_template("predict.html", user=session["user"])


@app.route("/thankyou")
def thankyou():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("thankyou.html", result=session["result"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
