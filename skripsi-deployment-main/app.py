# pip install flask transformers pandas
# pip3 install torch 
# torchvision torchaudio
# labeling ipynb : https://drive.google.com/drive/u/0/folders/16SWta0z2NsFVLwYNLcV1dKukvYnhDxAv
from flask import Flask, request, jsonify, render_template, send_file
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import io

app = Flask(__name__)

# Load the sentiment analysis model
classifier = pipeline("sentiment-analysis", model="tegaranggana/f_80_20")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = classifier(text)
        return jsonify(result)
    
@app.route('/predict-file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        # Read the file into a Pandas DataFrame
        df = pd.read_excel(file, engine='openpyxl')
        
        df_inf_content = df['user_comment'].str.lower().tolist()

        # Create an empty list to store the inference results
        inf_result = []

        # Use tqdm to wrap the iterable
        for comment in tqdm(df_inf_content, desc="Processing comments"):
            # Apply the classifier to each comment and store the result
            inf_result.append(classifier(comment)[0])
            
        df_inf_result = pd.DataFrame(inf_result)    
        df_with_inf = pd.concat([df, df_inf_result], axis=1)
        
        # Filter out NaN values
        df_with_inf = df_with_inf.fillna("")
        
        # Save the processed DataFrame to a new Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_with_inf.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)

        # Send the file as a response
        return send_file(output, download_name='result.xlsx', as_attachment=True)
        
        # return jsonify(), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

@app.route('/download-contoh')
def download_file():
    path = 'format.xlsx'  # Replace with your file path
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
