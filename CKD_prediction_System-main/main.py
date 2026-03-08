import os
import pickle
import numpy as np
import pandas as pd
import mysql.connector 
from flask import Flask, render_template, send_from_directory, request, jsonify


try:
    from Backend.preprocessing.preprocessing import load_and_preprocess_data
except ImportError:
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Backend.preprocessing.preprocessing import load_and_preprocess_data

app = Flask(__name__)


model = None
scaler = None
target_encoder = None
feature_columns = None


db_config = {
    'user': 'root',
    'password': '1234567890', 
    'host': '127.0.0.1',
    'database': 'ckd_db', 
    'auth_plugin': 'mysql_native_password' 
}



def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)

        cursor = conn.cursor(dictionary=True) 
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None, None

def init_db():
    """Creates the database table if it doesn't exist."""


    try:
        temp_conn = mysql.connector.connect(
            user='root', 
            password=db_config['password'], 
            host='localhost'
        )
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        temp_cursor.execute(f"USE {db_config['database']}")
        temp_conn.close()
    except mysql.connector.Error as err:
        print(f"Error creating database: {err}")
        return

 
    conn, cursor = get_db_connection()
    if not conn: return

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age FLOAT,
            bp FLOAT,
            sg FLOAT,
            al INT,
            su INT,
            rbc VARCHAR(20),
            pc VARCHAR(20),
            pcc VARCHAR(20),
            ba VARCHAR(20),
            bgr FLOAT,
            bu FLOAT,
            sc FLOAT,
            sod FLOAT,
            pot FLOAT,
            hemo FLOAT,
            pcv FLOAT,
            wbcc FLOAT,
            rbcc FLOAT,
            htn VARCHAR(10),
            dm VARCHAR(10),
            cad VARCHAR(10),
            appet VARCHAR(10),
            pe VARCHAR(10),
            ane VARCHAR(10),
            classification VARCHAR(20)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database table checked/created.")



def load_ml_components():
    """Loads the pre-trained model and preprocessing artifacts."""
    global model, scaler, target_encoder, feature_columns
    
    print("Loading preprocessing artifacts...")

    try:
        X_train, X_test, y_train, y_test, scaler_obj, encoder_obj, cols = load_and_preprocess_data()
        scaler = scaler_obj
        target_encoder = encoder_obj
        feature_columns = cols
        print("Preprocessing artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading preprocessing: {e}")
        raise



    model_path = "Backend/models/ckd_model.pkl"
    if not os.path.exists(model_path):
        model_path = "ckd_model.pkl"
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print("Error: Model file not found.")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

@app.route('/concept')
@app.route('/concept.html')
def concept():
    return render_template('concept.html')

@app.route('/contact')
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        

        input_df = pd.DataFrame([data])


        input_df = input_df.reindex(columns=feature_columns)


        cat_mapping_normal = {
            'abnormal': 0, 'normal': 1,
            'notpresent': 0, 'present': 1, 
            'no': 0, 'yes': 1,
            'good': 0, 'poor': 1
        }

        cat_cols = input_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            input_df[col] = input_df[col].astype(str).str.lower().str.strip()
            input_df[col] = input_df[col].map(cat_mapping_normal).fillna(0)

        input_df = input_df.astype(float)


        input_df = input_df.fillna(0)


        input_scaled = scaler.transform(input_df)


        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]


        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        if prediction == 1:
            prob_percent = round(probability[1] * 100, 2)
        else:
            prob_percent = round(probability[0] * 100, 2)


        try:
            conn, cursor = get_db_connection()
            if conn:

                columns = [
                    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 
                    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
                ]
                

                values = [data.get(col) for col in columns]
                

                values.append(predicted_label)
                

                placeholders = ', '.join(['%s'] * (len(columns) + 1))
                col_names = ', '.join(columns + ['classification'])
                query = f"INSERT INTO patients ({col_names}) VALUES ({placeholders})"
                
                cursor.execute(query, values)
                conn.commit()
                conn.close()
                print("Prediction saved to MySQL Database.")
        except Exception as db_err:
            print(f"Database logging failed: {db_err}")

        return jsonify({
            'prediction': predicted_label,
            'probability': prob_percent
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/styles.css')
def serve_css():
    return send_from_directory('static', 'styles.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('static', 'script.js')

if __name__ == '__main__':

    init_db()
    

    load_ml_components()
    
    app.run(debug=True, port=5000)