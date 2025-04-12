from flask import Flask, render_template, request, send_file, url_for
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def compute_pk_parameters(df):
    """
    Computes pharmacokinetic (PK) parameters for each patient:
    - Cmax, Tmax, AUC, Half-life, Clearance
    """
    results = []
    for patient, group in df.groupby('PatientID'):
        group = group.sort_values('Time').dropna(subset=['Time', 'Concentration'])
        times = group['Time'].values
        conc = group['Concentration'].values

        if len(times) < 2:
            continue  # Skip if not enough data points

        # PK Calculations
        Cmax = np.max(conc)
        Tmax = times[np.argmax(conc)]
        AUC = np.trapz(conc, times)
        
        if conc[-1] > 0 and conc[-2] > 0:
            kel = (np.log(conc[-2]) - np.log(conc[-1])) / (times[-1] - times[-2])
            HalfLife = np.log(2) / kel if kel > 0 else np.nan
        else:
            HalfLife = np.nan
        
        Clearance = Cmax / AUC if AUC > 0 else np.nan
        
        results.append({
            'PatientID': patient,
            'Tmax': round(Tmax, 2),
            'Cmax': round(Cmax, 2),
            'AUC': round(AUC, 2),
            'Half-life': round(HalfLife, 2) if not np.isnan(HalfLife) else 'N/A',
            'Clearance': round(Clearance, 2) if not np.isnan(Clearance) else 'N/A'
        })
    return pd.DataFrame(results)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    if not {'PatientID', 'Time', 'Concentration'}.issubset(df.columns):
        return "Missing required columns: PatientID, Time, Concentration", 400
    
    df.dropna(subset=['PatientID', 'Time', 'Concentration'], inplace=True)
    results_df = compute_pk_parameters(df)
    results_csv_path = os.path.join(RESULTS_FOLDER, 'pk_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    return render_template('index.html', 
                           table=results_df.to_html(classes='table table-bordered', index=False),
                           csv_url=url_for('download_file', filename='pk_results.csv'))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')