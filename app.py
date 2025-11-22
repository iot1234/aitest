import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

# Import local modules
import config
import s3_storage as storage
from gemini_service import GeminiService
from model_trainer import AdvancedModelTrainer, PredictionService, preprocess_tan1_data
from explainable_ai import ExplainablePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(config.get_config())

# Initialize Services
gemini_service = GeminiService()
prediction_service = PredictionService()
# Try to load the latest model on startup
try:
    model_folder = app.config['MODEL_FOLDER']
    if os.path.exists(model_folder):
        models = [f for f in os.listdir(model_folder) if f.endswith('.joblib')]
        if models:
            latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(model_folder, x)))
            prediction_service.trainer.load_model(os.path.join(model_folder, latest_model))
            logger.info(f"✅ Loaded latest model: {latest_model}")
except Exception as e:
    logger.warning(f"⚠️ Could not load latest model on startup: {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page')
def main_page():
    return render_template('main_page.html')

@app.route('/curriculum')
def curriculum_page():
    try:
        courses_json = json.dumps(app.config.get('COURSES_DATA', []))
        terms_json = json.dumps(app.config.get('ALL_TERMS_DATA', []))
        grades_json = json.dumps(app.config.get('DATA_CONFIG', {}).get('grade_mapping', {}))
        return render_template('curriculum_prediction_form.html', 
                             coursesData=courses_json, 
                             allTermsData=terms_json, 
                             gradeMapping=grades_json)
    except Exception as e:
        logger.error(f"Error rendering curriculum page: {e}")
        return render_template('curriculum_prediction_form.html', coursesData='[]', allTermsData='[]', gradeMapping='{}')

@app.route('/models')
def models_page():
    return render_template('model_management.html')

@app.route('/predict-batch')
def predict_batch_page():
    return render_template('index.html') # Reusing index for batch prediction as per original

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preview data
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Check if it needs preprocessing (TAN1 format)
            if 'STUDENT_ID' in df.columns and 'COURSE_ID' in df.columns:
                df_wide, _, _ = preprocess_tan1_data(filepath)
                preview = df_wide.head().to_dict(orient='records')
                columns = df_wide.columns.tolist()
            else:
                preview = df.head().to_dict(orient='records')
                columns = df.columns.tolist()
                
            return jsonify({
                'success': True, 
                'filename': filename,
                'preview': preview,
                'columns': columns
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f"Error processing file: {str(e)}"})
            
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        filename = data.get('filename')
        enable_gemini = data.get('enable_gemini_analysis', False)
        gemini_goal = data.get('gemini_analysis_goal', 'Analyze data quality and balance')
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
            
        # Load and preprocess
        if filename.endswith('.csv'):
            df_raw = pd.read_csv(filepath)
        else:
            df_raw = pd.read_excel(filepath)
            
        # If TAN1 format, preprocess for training
        # Note: AdvancedFeatureEngineer handles raw transcript data, but let's ensure we pass what it expects.
        # If the file is already wide format, we might need to adjust.
        # Assuming TAN1 format (Long) is what we have.
        
        # Train
        trainer = AdvancedModelTrainer()
        results = trainer.train(df_raw)
        
        # Save model
        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        trainer.save_model(model_path)
        
        # Gemini Analysis
        gemini_analysis = None
        if enable_gemini and gemini_service.is_configured:
            # Create a summary for Gemini
            summary = {
                "total_rows": len(df_raw),
                "columns": list(df_raw.columns),
                "sample_data": df_raw.head(5).to_dict(orient='records')
            }
            gemini_analysis = gemini_service.analyze_dataset(summary, gemini_goal)
            
        return jsonify({
            'success': True,
            'results': {k: {'accuracy': v['accuracy'], 'f1': v['f1']} for k, v in results.items()},
            'best_model_score': trainer.best_score,
            'model_filename': model_filename,
            'gemini_analysis': gemini_analysis
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gemini/predict', methods=['POST'])
def gemini_predict():
    try:
        data = request.json
        student_data = data.get('student_data') # Expecting dict of grades or student profile
        goal = data.get('analysis_goal', 'Predict graduation and provide advice')
        
        if not student_data:
            return jsonify({'success': False, 'error': 'No student data provided'})
            
        # 1. Get ML Prediction first (for context)
        ml_prediction = prediction_service.predict(student_data.get('grades', {}))
        
        # 2. Add ML result to Gemini payload
        payload = {
            "student_data": student_data,
            "ml_prediction": ml_prediction
        }
        
        # 3. Call Gemini
        gemini_result = gemini_service.predict_student_outcome(payload, goal)
        
        return jsonify({
            'success': True,
            'ml_prediction': ml_prediction,
            'gemini_result': gemini_result
        })
        
    except Exception as e:
        logger.error(f"Gemini prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze_curriculum', methods=['POST'])
def analyze_curriculum():
    """Legacy/Standard endpoint for prediction"""
    try:
        data = request.json
        current_grades = data.get('current_grades', {})
        
        # Use PredictionService
        result = prediction_service.predict(current_grades)
        
        return jsonify({
            'success': True,
            'prediction_result': {
                'prob_pass': result['probability_graduate'],
                'confidence': result['confidence'],
                'will_graduate': result['will_graduate']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# --- Legacy API Endpoints (kept for compatibility) ---
# These might need to be updated to use the new service or logic
# For now, I'll implement basic versions or placeholders if logic is complex to port immediately

@app.route('/api/three-line-chart', methods=['POST'])
def get_three_line_chart_data():
    # Placeholder: Implement actual logic if needed, or return dummy data
    return jsonify({'success': True, 'chart_data': []})

@app.route('/api/graduation-analysis', methods=['POST'])
def get_graduation_analysis():
    return jsonify({'success': True, 'analysis': {}})

@app.route('/api/next-term-prediction', methods=['POST'])
def get_next_term_prediction():
    return jsonify({'success': True, 'prediction_table': []})

@app.route('/api/comprehensive-analysis', methods=['POST'])
def get_comprehensive_analysis():
    # This aggregates everything
    try:
        data = request.json
        current_grades = data.get('current_grades', {})
        
        # ML Prediction
        pred = prediction_service.predict(current_grades)
        
        return jsonify({
            'success': True,
            'comprehensive_summary': {
                'current_status': {
                    'graduation_probability': pred['probability_graduate'] * 100,
                    'risk_level': 'High' if pred['probability_not_graduate'] > 0.5 else 'Low'
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create directories
    for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
        
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=app.config['DEBUG'])
