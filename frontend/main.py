from flask import *
import numpy as np
import pandas as pd
import sqlite3
from tensorflow.keras.models import load_model
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash
app.secret_key = 'your_secret_key'
# Initialize app and model
app = Flask(__name__)
model = load_model('../blstm_model.h5')
# Column order expected by model
columns = [
    "Patient ID", "Age", "Gender", "Cancer Type", "Cancer Stage", "Tumor Size (cm)",
    "Metastasis", "Biomarkers", "Genetic Mutations", "SNPs", "Family History",
    "Smoking History", "Previous Cancer Treatment", "Metabolic Enzyme Genotype",
    "Resistance Mutations", "Recommended Target Gene", "Predicted Sensitivity",
    "Recommended Drug", "Top 3   Effective Drugs"
]
# Create SQLite DB if not exists
def init_db():
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS patient_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            cancer_type TEXT,
            cancer_stage TEXT,
            tumor_size REAL,
            metastasis TEXT,
            biomarkers TEXT,
            genetic_mutations TEXT,
            snps TEXT,
            family_history TEXT,
            smoking_history TEXT,
            previous_cancer_treatment TEXT,
            metabolic_enzyme_genotype TEXT,
            resistance_mutations TEXT,
            recommended_target_gene TEXT,
            recommended_drug TEXT,
            predicted_sensitivity TEXT,
            expected_treatment_response TEXT
        )
    ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                mobile TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
app.secret_key = "your_secret_key"  # For flash messages
target_columns = ['Recommended Target Gene', 'Recommended Drug', 
                      'Predicted Sensitivity', 'Expected Treatment Response']
from sklearn.preprocessing import StandardScaler
    # Load and preprocess data
    # Load and preprocess data
def load_and_preprocess_data(filepath, target_columns):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Patient ID"], errors="ignore")

    X = df.drop(columns=target_columns)
    y = df[target_columns]
    # Feature encoding (Label Encoding for categorical features)
    label_encoders = joblib.load('../encoders/feat_encoders.pkl')
    for col, enc in label_encoders.items():
        df[col] = enc.transform(df[col].astype(str))
    # Apply standard scaling to features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Load target encoders
    target_encoders = joblib.load('../encoders/tgt_encoders.pkl')
    return df_scaled, target_encoders
    # Function to predict from input dictionary
def predict_from_dict(input_dict, feat_encoders, target_encoders, model):
    # Convert input_dict to DataFrame
    df = pd.DataFrame([input_dict])
    # Encode features
    for col, enc in feat_encoders.items():
        val = str(df[col].iloc[0])
        if val not in enc.classes_:
            print(f"⚠️ Warning: Unseen label '{val}' for column '{col}', using default '{enc.classes_[0]}'")
            val = enc.classes_[0]  # fallback to the first class in the encoder
        df[col] = enc.transform([val])
    # Apply standard scaling to features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Predict using the model
    preds = model.predict(df_scaled)
    # Decode the predictions
    decoded_preds = {}
    for i, col in enumerate(target_columns):
        y_labels = np.argmax(preds[i], axis=1)
        decoded_preds[col] = target_encoders[col].inverse_transform(y_labels)
    return decoded_preds
    # Function to save predictions to SQLite
def save_prediction_to_db(patient_id, sample_input, predictions):
    print(patient_id,sample_input,predictions)
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    from datetime import date
    # Get the current date
    current_date = date.today()
    # Insert the prediction data into the patient_recommendations table
    c.execute('''
        INSERT INTO patient_recommendations (
             patient_id, age, gender, cancer_type, cancer_stage, tumor_size, metastasis, biomarkers, genetic_mutations, 
            snps, family_history, smoking_history, previous_cancer_treatment, metabolic_enzyme_genotype, resistance_mutations, 
            recommended_target_gene, recommended_drug, predicted_sensitivity, expected_treatment_response, reportdate, did
        )
        VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id,
        sample_input['Age'],
        sample_input['Gender'],
        sample_input['Cancer Type'],
        sample_input['Cancer Stage'],
        sample_input['Tumor Size (cm)'],
        sample_input['Metastasis'],
        sample_input['Biomarkers'],
        sample_input['Genetic Mutations'],
        sample_input['SNPs'],
        sample_input['Family History'],
        sample_input['Smoking History'],
        sample_input['Previous Cancer Treatment'],
        sample_input['Metabolic Enzyme Genotype'],
        sample_input['Resistance Mutations'],
        sample_input['Resistance Mutations'],
        predictions[0],
        predictions[1],
        predictions[2],
        current_date,
        session['doctor_id']
    ))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

from io import BytesIO
import pandas as pd
import sqlite3
from datetime import datetime

@app.route("/patient_details", methods=["GET"])
def patient_details():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    doctor_id = session["doctor_id"]

    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM patient_recommendations WHERE did = ?"
        params = [doctor_id]

        if start_date and end_date:
            query += " AND reportdate BETWEEN ? AND ?"
            params.extend([start_date, end_date])

        query += " GROUP BY patient_id"
        cursor.execute(query, params)
        patients = cursor.fetchall()

    return render_template("patient_details.html", patients=patients, start_date=start_date, end_date=end_date)
@app.route("/download_patient_excel")
def download_patient_excel():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    doctor_id = session["doctor_id"]

    with sqlite3.connect("database.db") as conn:
        query = "SELECT * FROM patient_recommendations WHERE did = ?"
        params = [doctor_id]

        if start_date and end_date:
            query += " AND reportdate BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        df = pd.read_sql_query(query, conn, params=params)

    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='filtered_patient_data.xlsx',
        as_attachment=True
    )

@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# === Registration Page ===
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        password = request.form["password"]
    
        # Store user in DB
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO doctor (name, email, mobile, password)
                    VALUES (?, ?, ?, ?)
                ''', (name, email, mobile, password))
                conn.commit()
                flash("Registration successful! You can now login.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Email already exists! Please try a different one.", "danger")
                return redirect(url_for("register"))
    
    return render_template("register.html")
# === Login Page ===
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Check credentials
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM doctor WHERE email=? and password=?", (email,password))
            doctor = cursor.fetchone()

            if doctor:
            # Store doctor information in session (name and doctor ID)
                session['doctor_id'] = doctor[0]  # Assuming the doctor ID is in the first column
                session['doctor_name'] = doctor[1]  # Assuming the name is in the second column
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))  # Redirect to dashboard after successful login
            else:
                flash("Invalid credentials! Please try again.", "danger")
                return redirect(url_for("login"))

    return render_template("login.html")

def safe_label_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]
@app.route('/dashboard')
def dashboard():
    if 'doctor_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('dashboard.html', doctor_name=session['doctor_name'])
@app.route("/addreport", methods=["GET", "POST"])
def add_report():
    import random
    prediction = None  # Default prediction is None
    genes=['EGFR', 'ALK','HER2', 'BRCA1','KRAS', 'NRAS', 'BRAF','AR']
    gene=random.choice(genes)
    if request.method == "POST":
        try:
            # Collect and validate form data
            datas = {
                'Age': int(request.form['Age']),
                'Gender': request.form['Gender'],
                'Cancer Type': request.form['Cancer Type'],
                'Cancer Stage': request.form['Cancer Stage'],
                'Tumor Size (cm)': float(request.form['Tumor Size (cm)']),
                'Metastasis': request.form['Metastasis'],
                'Biomarkers': request.form['Biomarkers'],
                'Genetic Mutations': request.form['Genetic Mutations'],
                'SNPs': request.form['SNPs'],
                'Family History': request.form['Family History'],
                'Smoking History': request.form['Smoking History'],
                'Previous Cancer Treatment': request.form['Previous Cancer Treatment'],
                'Metabolic Enzyme Genotype': request.form['Metabolic Enzyme Genotype'],
                'Resistance Mutations': request.form['Resistance Mutations'],
                'Recommended Target Gene': gene,
            }

            # Convert to DataFrame
            new_patient = pd.DataFrame([datas])

            # Load the saved training data for column reference
            df = pd.read_csv('synthetic_cancer_data.csv')

            # Load models and preprocessing tools
            xgb_model = joblib.load('xgb_model.pkl')
            lasso_model = joblib.load('lasso_model.pkl')
            drug_encoder = joblib.load('drug_encoder.pkl')
            preprocessor = joblib.load('preprocessor.pkl')

            # Define required feature columns
            categorical_cols = df.select_dtypes(include='object').columns.drop(['Recommended Drug'])
            numerical_cols = df.select_dtypes(include='number').columns.drop(['percentage of recovery'])
            required_columns = list(categorical_cols) + list(numerical_cols)

            # Ensure DataFrame matches training structure
            new_patient = new_patient[required_columns]

            # Preprocess the input
            new_input = preprocessor.transform(new_patient)

            # Make predictions
            pred_drug = xgb_model.predict(new_input)
            predicted_drug = drug_encoder.inverse_transform(pred_drug)[0]

            pred_recovery = lasso_model.predict(new_input)[0]

            # Top 3 drug predictions
            pred_probs = xgb_model.predict_proba(new_input)[0]
            top3_indices = np.argsort(pred_probs)[-4:-1][::-1]
            top3_drugs = drug_encoder.inverse_transform(top3_indices)
            top3_probs = pred_probs[top3_indices]

            # Log the results
            # Log the results
            print(f"Predicted Drug: {predicted_drug}")
            print(f"Predicted Recovery: {pred_recovery:.2f}%")
            print("Top 3 Predicted Drugs:")
            top3_list = []
            top3_list.append(f"{1}. {predicted_drug} (Sensitivity: {pred_recovery:.2f}%)")
            for i, (drug, prob) in enumerate(zip(top3_drugs, top3_probs), 1):
                print(f"{i}. {drug} (Confidence: {prob:.2f})")
                top3_list.append(f"{i+1}. {drug} (Sensitivity: {prob*100:.2f}%)")
            patient_id =request.form["Patient ID"]   # Example patient ID
        
            # # Save the predictions to DB
            data=[predicted_drug,pred_recovery," ".join(top3_list)]

            save_prediction_to_db(patient_id, datas, data)
            print("Data has been stored in the database successfully.")
            chart_labels = [predicted_drug] + [drug for drug in top3_drugs if drug != predicted_drug]
            chart_confidences = [pred_recovery] + [prob * 100 for drug, prob in zip(top3_drugs, top3_probs) if drug != predicted_drug]

            result_data = {
                'Recommended Target Gene': datas['Recommended Target Gene'],
                'Recommended Drug': predicted_drug,
                'Predicted Sensitivity': f"{top3_probs[0]*100:.2f}%",
                'Expected Treatment Response': f"{pred_recovery:.2f}%",
                'Top 3 Drugs': top3_list,
                'Top 3 Labels': chart_labels,
                'Top 3 Confidences': [round(val, 2) for val in chart_confidences]
            }
            return render_template("output.html", prediction=result_data)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("form.html", prediction="Error: " + str(e))
    
    # Render form for GET request
    return render_template("form.html", prediction=prediction)

@app.route("/patient/<patient_id>")
def view_patient(patient_id):
    print(patient_id)
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_recommendations WHERE patient_id=? and did=?", (patient_id,session["doctor_id"]))
        patient_records = cursor.fetchall()

    if not patient_records:
        flash("Patient not found.", "danger")
        return redirect(url_for("show_patients"))

    columns = [
        "ID","Patient ID", "Age", "Gender", "Cancer Type", "Cancer Stage", "Tumor Size (cm)", "Metastasis",
        "Biomarkers", "Genetic Mutations", "SNPs", "Family History", "Smoking History", "Previous Cancer Treatment",
        "Metabolic Enzyme Genotype", "Resistance Mutations", "Recommended Target Gene", "Predicted Sensitivity",
        "Recommended Drug", "Top 5 Effective Drugs"
    ]

    # Convert each row to a dictionary
    patient_data = [dict(zip(columns, row)) for row in patient_records]

    return render_template("view_patient.html", patient_records=patient_data)
from collections import Counter
@app.route("/patient_graphs")
def patient_graphs():
    # Example data; replace with database query
    patient_records = [
        {"Patient ID": "P001", "Age": 52, "Cancer Type": "Breast", "Tumor Size (cm)": 3.2},
        {"Patient ID": "P002", "Age": 60, "Cancer Type": "Lung", "Tumor Size (cm)": 2.1},
        {"Patient ID": "P003", "Age": 47, "Cancer Type": "Lung", "Tumor Size (cm)": 4.5},
        {"Patient ID": "P004", "Age": 35, "Cancer Type": "Breast", "Tumor Size (cm)": 2.7},
    ]

    cancer_type_data = Counter(p["Cancer Type"] for p in patient_records)
    age_data = [p["Age"] for p in patient_records]
    tumor_size_data = [p["Tumor Size (cm)"] for p in patient_records]
    patient_labels = [p["Patient ID"] for p in patient_records]

    return render_template(
        "patient_graphs.html",
        cancer_type_data=cancer_type_data,
        age_data=age_data,
        tumor_size_data=tumor_size_data,
        patient_labels=patient_labels
    )

@app.route("/patient/delete/<patient_id>", methods=["POST"])
def delete_patient(patient_id):
    print(patient_id)

    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patient_recommendations WHERE patient_id=?", (patient_id,))
        conn.commit()
    flash("Patient deleted successfully", "success")
    return redirect(url_for("patient_details"))
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
