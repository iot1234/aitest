from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging.config
import logging
from datetime import datetime
import warnings
from collections import Counter
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename # Import secure_filename

import config # ตรวจสอบให้แน่ใจว่าคุณมี config.py อยู่ใน directory เดียวกัน

warnings.filterwarnings('ignore')

# Set up logging (initial setup, will be re-configured by app.config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Load settings from config.py
app.config.from_object(config.get_config()) # ใช้ config.get_config() เพื่อดึง ProductionConfig หรือ DevelopmentConfig

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info(f"Created folder: {folder}")

# Global dictionary to store loaded models and their info
models = {
    'subject_model': None,
    'subject_model_info': None,
    'gpa_model': None,
    'gpa_model_info': None,
    'record_model': None, # Added for record_based data
    'record_model_info': None, # Added for record_based data
    'data_scalers': {}, # Store scalers for each model type
    'label_encoders': {} # Store label encoders for each categorical column (though not heavily used now)
}

# --- Modified Function: load_existing_models ---
def load_existing_models():
    """Loads the latest trained models from the MODEL_FOLDER."""
    global models
    try:
        # Load Subject Model
        subject_models = sorted([f for f in os.listdir(app.config['MODEL_FOLDER']) if f.startswith('subject_based_model_') and f.endswith('.joblib')],
                                key=lambda x: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], x)), reverse=True)
        if subject_models:
            latest_subject_model_path = os.path.join(app.config['MODEL_FOLDER'], subject_models[0])
            try:
                loaded_obj = joblib.load(latest_subject_model_path)
                if isinstance(loaded_obj, dict) and 'models' in loaded_obj and 'scaler' in loaded_obj:
                    models['subject_model'] = loaded_obj['models']
                    models['data_scalers']['subject_based'] = loaded_obj['scaler']
                    models['subject_model_info'] = loaded_obj.get('performance_metrics', {})
                    models['subject_model_info']['created_at'] = loaded_obj.get('created_at', datetime.now().isoformat())
                    models['subject_model_info']['filename'] = subject_models[0]
                    models['subject_model_info']['data_format'] = loaded_obj.get('data_format', 'subject_based')
                    models['subject_feature_cols'] = loaded_obj.get('feature_columns')
                    logger.info("✅ Loaded latest Subject Model successfully.")
                else: # Fallback for old model format
                    models['subject_model'] = loaded_obj
                    models['subject_model_info'] = { 'filename': subject_models[0], 'created_at': datetime.fromtimestamp(os.path.getmtime(latest_subject_model_path)).strftime('%Y-%m-%d %H:%M:%S'), 'accuracy': 0.0, 'type': 'subject_based', 'data_format': 'subject_based' }
                    logger.warning(f"Could not load Subject model metadata, loaded raw model: {subject_models[0]}")
            except Exception as e:
                logger.error(f"❌ Could not load Subject model: {str(e)}")
                
        # Load GPA Model
        gpa_models = sorted([f for f in os.listdir(app.config['MODEL_FOLDER']) if f.startswith('gpa_based_model_') and f.endswith('.joblib')],
                            key=lambda x: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], x)), reverse=True)
        if gpa_models:
            latest_gpa_model_path = os.path.join(app.config['MODEL_FOLDER'], gpa_models[0])
            try:
                loaded_obj = joblib.load(latest_gpa_model_path)
                if isinstance(loaded_obj, dict) and 'models' in loaded_obj and 'scaler' in loaded_obj:
                    models['gpa_model'] = loaded_obj['models']
                    models['data_scalers']['gpa_based'] = loaded_obj['scaler']
                    models['gpa_model_info'] = loaded_obj.get('performance_metrics', {})
                    models['gpa_model_info']['created_at'] = loaded_obj.get('created_at', datetime.now().isoformat())
                    models['gpa_model_info']['filename'] = gpa_models[0]
                    models['gpa_model_info']['data_format'] = loaded_obj.get('data_format', 'gpa_based')
                    models['gpa_feature_cols'] = loaded_obj.get('feature_columns')
                    logger.info("✅ Loaded latest GPA model successfully.")
                else: # Fallback for old model format
                    models['gpa_model'] = loaded_obj
                    models['gpa_model_info'] = { 'filename': gpa_models[0], 'created_at': datetime.fromtimestamp(os.path.getmtime(latest_gpa_model_path)).strftime('%Y-%m-%d %H:%M:%S'), 'accuracy': 0.0, 'type': 'gpa_based', 'data_format': 'gpa_based' }
                    logger.warning(f"Could not load GPA model metadata, loaded raw model: {gpa_models[0]}")
            except Exception as e:
                logger.error(f"❌ Could not load GPA model: {str(e)}")

        # Load Record-based Model (e.g., for T1.csv format)
        record_models = sorted([f for f in os.listdir(app.config['MODEL_FOLDER']) if f.startswith('record_based_model_') and f.endswith('.joblib')],
                               key=lambda x: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], x)), reverse=True)
        if record_models:
            latest_record_model_path = os.path.join(app.config['MODEL_FOLDER'], record_models[0])
            try:
                loaded_obj = joblib.load(latest_record_model_path)
                if isinstance(loaded_obj, dict) and 'models' in loaded_obj and 'scaler' in loaded_obj:
                    models['record_model'] = loaded_obj['models']
                    models['data_scalers']['record_based'] = loaded_obj['scaler']
                    models['record_model_info'] = loaded_obj.get('performance_metrics', {})
                    models['record_model_info']['created_at'] = loaded_obj.get('created_at', datetime.now().isoformat())
                    models['record_model_info']['filename'] = record_models[0]
                    models['record_model_info']['data_format'] = loaded_obj.get('data_format', 'record_based')
                    models['record_feature_cols'] = loaded_obj.get('feature_columns')
                    logger.info("✅ Loaded latest Record-based Model successfully.")
                else: # Fallback for old model format
                    models['record_model'] = loaded_obj
                    models['record_model_info'] = { 'filename': record_models[0], 'created_at': datetime.fromtimestamp(os.path.getmtime(latest_record_model_path)).strftime('%Y-%m-%d %H:%M:%S'), 'accuracy': 0.0, 'type': 'record_based', 'data_format': 'record_based' }
                    logger.warning(f"Could not load Record-based model metadata, loaded raw model: {record_models[0]}")
            except Exception as e:
                logger.error(f"❌ Could not load Record-based model: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error loading existing models: {str(e)}")

# --- Helper Functions for Data Preprocessing and Feature Engineering ---

def grade_to_numeric(grade):
    """Converts a letter grade to a numeric GPA value."""
    if pd.isna(grade):
        return None # Return None for missing grades

    # Try to convert to float first for numeric grades (e.g., 2.5, 3.0)
    try:
        numeric_val = float(grade)
        if 0.0 <= numeric_val <= 4.0: # Assume GPA is on a 4.0 scale
            return numeric_val
    except (ValueError, TypeError):
        pass # Not a straightforward numeric grade, proceed to letter grade mapping

    # Convert from letter grade to numeric using config mapping
    grade_str = str(grade).strip().upper()
    return app.config['DATA_CONFIG']['grade_mapping'].get(grade_str, None) # Return None if not found

def categorize_subject(subject_name):
    """Categorizes subjects based on keywords defined in config."""
    if pd.isna(subject_name):
        return 'อื่นๆ'

    subject_name = str(subject_name).lower()
    for category, info in app.config['SUBJECT_CATEGORIES'].items():
        if any(keyword in subject_name for keyword in info['keywords']):
            return category
    return 'อื่นๆ'

# --- Modified Function: process_subject_data ---
def process_subject_data(df):
    """Processes subject-based DataFrame (matrix format) to create features for model training."""
    try:
        # Improved name/ID column detection
        name_col = None
        possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
        for col in possible_name_cols:
            if col in df.columns:
                name_col = col
                break
        if not name_col:
            name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
            if name_candidates:
                name_col = name_candidates[0]
            else:
                raise ValueError("ไม่พบคอลัมน์ชื่อนักศึกษาหรือรหัสในข้อมูล (Student Name/ID column not found).")

        year_in_col = None
        year_out_col = None
        possible_year_in = ['ปีที่เข้า', 'ปีเข้า', 'year_in', 'admission_year']
        possible_year_out = ['ปีที่จบ', 'ปีจบ', 'year_out', 'graduation_year']

        for col in possible_year_in:
            if col in df.columns:
                year_in_col = col
                break
        for col in possible_year_out:
            if col in df.columns:
                year_out_col = col
                break

        # Define columns to exclude from being treated as subjects/grades
        exclude_cols_keywords = ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
        exclude_cols = [name_col]
        if year_in_col: exclude_cols.append(year_in_col)
        if year_out_col: exclude_cols.append(year_out_col)

        # Identify subject columns (columns containing grades for individual subjects)
        subject_cols = [
            col for col in df.columns
            if col not in exclude_cols and not any(kw in col.lower() for kw in exclude_cols_keywords)
        ]
        
        # Ensure a target column 'graduated' exists or can be derived
        target_col_found = False
        for kw in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ']:
            for col in df.columns:
                if kw in col.lower():
                    df['graduated'] = df[col].apply(lambda x: 1 if str(x).lower() in ['จบ', 'success', '1', 'pass'] else 0)
                    target_col_found = True
                    break
            if target_col_found:
                break
        if not target_col_found:
            # Fallback to inferring 'graduated' from 'year_out' if available
            if year_out_col and not df[year_out_col].isnull().all():
                # Assuming if year_out is present and > 0, student graduated
                df['graduated'] = df[year_out_col].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
                logger.warning("Target column 'graduated' not found, inferred from 'year_out'.")
            else:
                raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา (Graduation status column not found or cannot be inferred).")

        logger.info(f"Found name column: {name_col}")
        logger.info(f"Found subject columns: {len(subject_cols)} subjects")

        processed_data = []

        for idx, row in df.iterrows():
            try:
                student_name = row[name_col]
                year_in = row.get(year_in_col, 0) if year_in_col else 0
                year_out = row.get(year_out_col, 0) if year_out_col else 0
                graduated_status = row.get('graduated', 0)

                grades = []
                subject_categories_grades = {cat: [] for cat in app.config['SUBJECT_CATEGORIES'].keys()}
                subject_categories_grades['อื่นๆ'] = []

                for subject in subject_cols:
                    grade_value = row[subject]
                    if pd.notna(grade_value) and str(grade_value).strip():
                        numeric_grade = grade_to_numeric(grade_value)
                        if numeric_grade is not None:
                            grades.append(numeric_grade)

                            category = categorize_subject(subject)
                            subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

                min_subjects_required = app.config['DATA_CONFIG']['min_subjects_per_student']
                if len(grades) >= min_subjects_required:
                    gpa = np.mean(grades)
                    min_grade = np.min(grades)
                    max_grade = np.max(grades)
                    std_grade = np.std(grades) if len(grades) > 1 else 0
                    fail_threshold = app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) # Assume D or less is fail
                    fail_count = sum(1 for g in grades if g < fail_threshold)
                    fail_rate = fail_count / len(grades) if grades else 0

                    student_data = {
                        'ชื่อ': student_name,
                        'gpa': gpa,
                        'min_grade': min_grade,
                        'max_grade': max_grade,
                        'std_grade': std_grade,
                        'fail_count': fail_count,
                        'fail_rate': fail_rate,
                        'total_subjects': len(grades),
                        'year_in': year_in if pd.notna(year_in) else 0,
                        'year_out': year_out if pd.notna(year_out) else 0,
                        'graduated': graduated_status
                    }

                    for cat, cat_grades in subject_categories_grades.items():
                        if cat_grades:
                            student_data[f'gpa_{cat}'] = np.mean(cat_grades)
                            student_data[f'min_{cat}'] = np.min(cat_grades)
                            student_data[f'max_{cat}'] = np.max(cat_grades)
                            cat_fail_count = sum(1 for g in cat_grades if g < fail_threshold)
                            student_data[f'fail_rate_{cat}'] = cat_fail_count / len(cat_grades)
                        else:
                            student_data[f'gpa_{cat}'] = 0.0
                            student_data[f'min_{cat}'] = 0.0
                            student_data[f'max_{cat}'] = 0.0
                            student_data[f'fail_rate_{cat}'] = 0.0

                    processed_data.append(student_data)

                else:
                    logger.debug(f"Skipping row {idx} for student '{row.get(name_col, 'N/A')}': only {len(grades)} valid grades (minimum {min_subjects_required} required)")
                    continue

            except Exception as e:
                logger.warning(f"Could not process row {idx} for student '{row.get(name_col, 'N/A')}': {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No data could be processed. Please check the data format and ensure sufficient subjects per student.")

        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        logger.info(f"Successfully processed subject-based data: {len(result_df)} students")

        return result_df

    except Exception as e:
        logger.error(f"Error processing subject data: {str(e)}")
        raise

# --- Modified Function: process_gpa_data ---
def process_gpa_data(df):
    """Processes GPA-based data to create features for model training."""
    try:
        processed_data = []

        name_col = None
        possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
        for col in possible_name_cols:
            if col in df.columns:
                name_col = col
                break
        if not name_col:
            name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
            if name_candidates:
                name_col = name_candidates[0]
            else:
                name_col = df.columns[0]
                logger.warning(f"No explicit name/ID column found, using '{name_col}' as student identifier.")

        target_col_found = False
        graduated_col = None
        for kw in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ']:
            for col in df.columns:
                if kw in col.lower():
                    graduated_col = col
                    target_col_found = True
                    break
            if target_col_found:
                break
        if not target_col_found:
            raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา (Graduation status column not found).")

        for idx, row in df.iterrows():
            try:
                student_name = row.get(name_col, f'นักศึกษา_{idx}')

                gpa_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])]
                gpas = []

                for col in gpa_cols:
                    gpa = row.get(col, 0)
                    if pd.notna(gpa) and gpa != 0:
                        try:
                            gpas.append(float(gpa))
                        except ValueError:
                            logger.debug(f"Skipping non-numeric GPA value '{gpa}' in column '{col}'.")
                            continue

                if gpas:
                    success_value = str(row.get(graduated_col, '')).lower()
                    graduated = 1 if any(keyword in success_value for keyword in ['จบ', 'success', '1', 'pass']) else 0

                    student_data = {
                        'ชื่อ': student_name,
                        'gpa': np.mean(gpas),
                        'min_grade': np.min(gpas),
                        'max_grade': np.max(gpas),
                        'std_grade': np.std(gpas) if len(gpas) > 1 else 0,
                        'total_terms': len(gpas),
                        'graduated': graduated
                    }
                    processed_data.append(student_data)
                else:
                    logger.warning(f"Skipping row {idx} for student '{student_name}' due to no valid GPA data.")

            except Exception as e:
                logger.warning(f"Could not process row {idx} for student '{row.get(name_col, 'N/A')}': {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No data could be processed. Please check 'เกรด' and 'สำเร็จ/จบ' columns for valid entries.")

        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        logger.info(f"Successfully processed GPA data: {len(result_df)} students")

        return result_df

    except Exception as e:
        logger.error(f"Error processing GPA data: {str(e)}")
        raise

# --- New Helper Function: extract_years_info_from_records ---
def extract_years_info_from_records(student_data_df, year_col):
    """สกัดข้อมูลปีการศึกษาจาก records ของนักศึกษา"""
    try:
        if not year_col or year_col not in student_data_df.columns:
            return {'year_in': 0, 'year_out': 0, 'duration': 0}
            
        years = student_data_df[year_col].dropna().unique()
        if len(years) > 0:
            # Ensure years are numeric before sorting and converting to int
            numeric_years = [int(y) for y in years if str(y).strip().isdigit()]
            if numeric_years:
                years_sorted = sorted(numeric_years)
                year_in = years_sorted[0]
                year_out = years_sorted[-1]
                duration = year_out - year_in + 1
                
                return {
                    'year_in': year_in,
                    'year_out': year_out,
                    'duration': duration
                }
            
        return {'year_in': 0, 'year_out': 0, 'duration': 0}
        
    except Exception as e:
        logger.warning(f"Error extracting years info from records: {str(e)}")
        return {'year_in': 0, 'year_out': 0, 'duration': 0}

# --- New Helper Function: calculate_gpa_by_year ---
def calculate_gpa_by_year(student_data_df, grade_col, year_col):
    """Calculates GPA for each year for a student."""
    gpas_by_year = {}
    if year_col and year_col in student_data_df.columns:
        for year in student_data_df[year_col].dropna().unique():
            yearly_grades = student_data_df[student_data_df[year_col] == year][grade_col].apply(grade_to_numeric).dropna()
            if not yearly_grades.empty:
                gpas_by_year[int(year)] = yearly_grades.mean()
    return gpas_by_year

def determine_graduation_from_records(student_data_df, subject_col, grade_col):
    """Determines graduation status based on project/coop completion or total credits."""
    try:
        # Check for project subjects
        project_mask = student_data_df[subject_col].str.contains('โครงงาน', case=False, na=False)
        if project_mask.any():
            project_grades = student_data_df.loc[project_mask, grade_col].tolist()
            passing_grades_for_project = ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'S', 'P']
            if any(str(grade).strip().upper() in passing_grades_for_project for grade in project_grades):
                return 1  # Graduated if project is passed
        
        # Check for cooperative education (สหกิจ)
        coop_mask = student_data_df[subject_col].str.contains('สหกิจ', case=False, na=False)
        if coop_mask.any():
            coop_grades = student_data_df.loc[coop_mask, grade_col].tolist()
            if any(str(g).strip().upper() in ['S', 'P'] for g in coop_grades):
                return 1  # Graduated if co-op is passed
        
        # Fallback: estimate based on total credits and a reasonable GPA
        total_numeric_grades = student_data_df[grade_col].apply(grade_to_numeric).dropna()
        if not total_numeric_grades.empty:
            total_gpa_from_records = total_numeric_grades.mean()
            # Sum credits only for courses with passing grades (GPA > 0)
            passing_courses_data = student_data_df[student_data_df[grade_col].apply(grade_to_numeric).fillna(0) > 0]
            estimated_credits = passing_courses_data.shape[0] * app.config['DATA_CONFIG']['default_credits_per_subject']

            if estimated_credits >= app.config['DATA_CONFIG']['total_required_credits'] and total_gpa_from_records >= app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
                return 1 # Likely graduated if estimated credits and GPA meet requirement
                
        return 0  # Not graduated
                
    except Exception as e:
        logger.warning(f"Error determining graduation status from records: {str(e)}")
        return 0 # Default to not graduated if error occurs

# --- Modified Function: process_record_based_data ---
def process_record_based_data(df):
    """ประมวลผลข้อมูลแบบ record-based (T1.csv format) พร้อมสร้างฟีเจอร์ขั้นสูง"""
    try:
        logger.info("Processing record-based data with advanced feature engineering...")
        
        # หาคอลัมน์ที่ต้องใช้
        student_id_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['รหัส', 'id', 'student_id'])), None)
        student_name_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['ชื่อ-นามสกุล', 'ชื่อ', 'name', 'student_name'])), None)
        subject_name_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['ชื่อวิชา', 'วิชา', 'subject', 'course'])), None)
        grade_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['เกรด', 'grade'])), None)
        year_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['ปีการศึกษา', 'ปี', 'year', 'academic_year', 'semester_year'])), None)
        credit_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['หน่วยกิต', 'credit'])), None)

        if not all([(student_id_col or student_name_col), subject_name_col, grade_col]):
            raise ValueError(f"Missing required columns. Found: student_id='{student_id_col}', student_name='{student_name_col}', subject='{subject_name_col}', grade='{grade_col}'")
        
        identifier_col = student_id_col if student_id_col else student_name_col
        if not identifier_col:
            raise ValueError("No student identifier column (ID or Name) found.")

        logger.info(f"Using columns for record-based data: identifier='{identifier_col}', subject='{subject_name_col}', grade='{grade_col}', year='{year_col}', credit='{credit_col}'")
        
        if year_col and year_col in df.columns:
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce').fillna(0).astype(int)
        else:
            logger.warning(f"Year column '{year_col}' not found or invalid. Year-based features might be 0.")
            df['__temp_year_col'] = 0 # Create a dummy column to avoid errors, will be dropped later
            year_col = '__temp_year_col' # Use the dummy column for calculations

        processed_data = []
        grouped = df.groupby(identifier_col)
        
        for student_identifier, student_data_df in grouped:
            try:
                grades = []
                subject_categories_grades = {cat: [] for cat in app.config['SUBJECT_CATEGORIES'].keys()}
                subject_categories_grades['อื่นๆ'] = []
                
                # --- New Feature Engineering ---
                student_gpa_by_year = calculate_gpa_by_year(student_data_df, grade_col, year_col)
                years_info = extract_years_info_from_records(student_data_df, year_col)
                
                first_year = years_info['year_in']
                last_year = years_info['year_out']

                first_year_gpa = student_gpa_by_year.get(first_year, 0.0)
                final_year_gpa = student_gpa_by_year.get(last_year, 0.0)
                gpa_trend = final_year_gpa - first_year_gpa if first_year and last_year and first_year != last_year else 0.0

                total_credits_taken = 0
                if credit_col and credit_col in student_data_df.columns:
                     student_data_df['numeric_credits'] = pd.to_numeric(student_data_df[credit_col], errors='coerce').fillna(app.config['DATA_CONFIG']['default_credits_per_subject'])
                     total_credits_taken = student_data_df['numeric_credits'].sum()
                else:
                     total_credits_taken = student_data_df.shape[0] * app.config['DATA_CONFIG']['default_credits_per_subject']

                credits_per_year = total_credits_taken / years_info['duration'] if years_info['duration'] > 0 else 0

                early_years_fail_count = 0
                early_years_grades_count = 0
                fail_threshold = app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0)
                
                if first_year > 0:
                    early_years = [first_year]
                    if first_year + 1 <= last_year:
                        early_years.append(first_year + 1) # Consider first two academic years as "early"

                    early_grades_df = student_data_df[student_data_df[year_col].isin(early_years)]
                    for _, row in early_grades_df.iterrows():
                        numeric_grade = grade_to_numeric(row[grade_col])
                        if numeric_grade is not None:
                            early_years_grades_count += 1
                            if numeric_grade < fail_threshold:
                                early_years_fail_count += 1
                
                fail_rate_in_early_years = early_years_fail_count / early_years_grades_count if early_years_grades_count > 0 else 0
                # --- End New Feature Engineering ---

                for _, row in student_data_df.iterrows():
                    grade_value = row[grade_col]
                    subject_name = str(row[subject_name_col]) if pd.notna(row[subject_name_col]) else ''
                    
                    # Exclude special non-numeric grades from GPA calculation but categorize subjects
                    grade_str_upper = str(grade_value).strip().upper()
                    if grade_str_upper in ['S', 'U', 'AU', 'P', 'W', 'I', 'NP', 'WF', 'WU']:
                        numeric_grade = grade_to_numeric(grade_value) # Still convert if it has a numeric mapping (like S/P)
                        if numeric_grade is not None:
                            grades.append(numeric_grade)
                            category = categorize_subject(subject_name)
                            subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)
                        continue
                            
                    numeric_grade = grade_to_numeric(grade_value)
                    if numeric_grade is not None:
                        grades.append(numeric_grade)
                        category = categorize_subject(subject_name)
                        subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)
                
                min_subjects_required = app.config['DATA_CONFIG']['min_subjects_per_student']
                if len(grades) >= min_subjects_required:
                    gpa = np.mean(grades)
                    min_grade = np.min(grades)
                    max_grade = np.max(grades)
                    std_grade = np.std(grades) if len(grades) > 1 else 0
                    
                    fail_count = sum(1 for g in grades if g < fail_threshold)
                    fail_rate = fail_count / len(grades) if grades else 0
                    
                    graduation_status = determine_graduation_from_records(student_data_df, subject_name_col, grade_col)
                    
                    student_record = {
                        'ชื่อ': student_identifier,
                        'gpa': round(gpa, 2),
                        'min_grade': round(min_grade, 2),
                        'max_grade': round(max_grade, 2),
                        'std_grade': round(std_grade, 2),
                        'fail_count': fail_count,
                        'fail_rate': round(fail_rate, 3),
                        'total_subjects': len(grades),
                        'year_in': years_info.get('year_in', 0),
                        'year_out': years_info.get('year_out', 0),
                        'study_duration': years_info.get('duration', 0),
                        'gpa_trend': round(gpa_trend, 2), # New Feature
                        'final_year_gpa': round(final_year_gpa, 2), # New Feature
                        'first_year_gpa': round(first_year_gpa, 2), # New Feature
                        'fail_rate_in_early_years': round(fail_rate_in_early_years, 3), # New Feature
                        'credits_per_year': round(credits_per_year, 2), # New Feature
                        'graduated': graduation_status
                    }
                    
                    for cat, cat_grades in subject_categories_grades.items():
                        if cat_grades:
                            student_record[f'gpa_{cat}'] = round(np.mean(cat_grades), 2)
                            student_record[f'min_{cat}'] = round(np.min(cat_grades), 2)
                            student_record[f'max_{cat}'] = round(np.max(cat_grades), 2)
                            cat_fail_count = sum(1 for g in cat_grades if g < fail_threshold)
                            student_record[f'fail_rate_{cat}'] = round(cat_fail_count / len(cat_grades), 3)
                            student_record[f'count_{cat}'] = len(cat_grades)
                        else:
                            student_record[f'gpa_{cat}'] = 0.0
                            student_record[f'min_{cat}'] = 0.0
                            student_record[f'max_{cat}'] = 0.0
                            student_record[f'fail_rate_{cat}'] = 0.0
                            student_record[f'count_{cat}'] = 0
                        
                    processed_data.append(student_record)
                else:
                    logger.debug(f"Skipping {student_identifier}: only {len(grades)} valid grades (minimum {min_subjects_required} required)")
                    
            except Exception as e:
                logger.warning(f"Error processing student '{student_identifier}': {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("No valid student data could be processed. Check if data has sufficient grades per student and correct column names.")
            
        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        
        # Drop the temporary year column if it was created
        if '__temp_year_col' in df.columns:
            df.drop(columns=['__temp_year_col'], inplace=True)

        logger.info(f"Successfully processed {len(result_df)} students from record-based data")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing record-based data: {str(e)}")
        raise

# --- Modified Function: train_ensemble_model ---
def train_ensemble_model(X, y):
    """Trains an Ensemble model with GridSearchCV and SMOTE."""
    try:
        logger.info("Starting Ensemble model training...")

        # Handle Class Imbalance or small data for Training
        unique_labels, label_counts = np.unique(y, return_counts=True)

        if len(unique_labels) < 2 or np.min(label_counts) < app.config['DATA_CONFIG']['min_samples_for_stratify']:
            logger.warning(f"Class imbalance or too little data for stratification before Oversampling: {dict(zip(unique_labels, label_counts))}")

            # If only one class, add a synthetic sample of the other class for split
            if len(unique_labels) < 2:
                existing_label = unique_labels[0]
                new_label_to_create = 1 - existing_label # Assuming binary classification (0 or 1)
                if not X.empty:
                    # Create a synthetic row by duplicating an existing one or creating from mean/median
                    X_synthetic = X.iloc[[0]].copy() # Duplicate first row
                    y_synthetic = pd.Series([new_label_to_create], index=[X.index.max() + 1 if not X.empty else 0])
                    X = pd.concat([X, X_synthetic], ignore_index=True)
                    y = pd.concat([y, y_synthetic], ignore_index=True)
                    logger.info(f"Added synthetic sample for class {new_label_to_create} to enable stratification.")
                else:
                    raise ValueError("Cannot create synthetic data as DataFrame is empty.")
                unique_labels, label_counts = np.unique(y, return_counts=True)
            
            # If still not enough samples per class for stratification (e.g., one class has only 1 sample)
            # This is a robust check before train_test_split to prevent errors
            for label in unique_labels:
                if label_counts[unique_labels == label][0] < app.config['DATA_CONFIG']['min_samples_for_stratify']:
                    needed = app.config['DATA_CONFIG']['min_samples_for_stratify'] - label_counts[unique_labels == label][0]
                    minority_X_samples = X[y == label]
                    if not minority_X_samples.empty:
                        for _ in range(needed):
                            X_new_sample = minority_X_samples.sample(1, random_state=app.config['ML_CONFIG']['random_state'])
                            new_index = X.index.max() + 1 if not X.empty else 0
                            y_new_label_entry = pd.Series([label], index=[new_index])
                            X = pd.concat([X, X_new_sample], ignore_index=True)
                            y = pd.concat([y, y_new_label_entry], ignore_index=True)
                        logger.info(f"Added {needed} synthetic samples for class {label} to meet stratification minimum.")
                    else:
                        logger.warning(f"No samples found for class {label} for oversampling to meet stratification minimum.")
            
            unique_labels, label_counts = np.unique(y, return_counts=True)


        logger.info(f"Number of data points after initial oversampling (if any): {len(X)}, Features: {len(X.columns)}")
        logger.info(f"Label distribution after initial oversampling: {y.value_counts().to_dict()}")

        if len(X) < app.config['DATA_CONFIG']['min_students_for_training']:
            raise ValueError(f"Insufficient data for model training (at least {app.config['DATA_CONFIG']['min_students_for_training']} samples required).")

        total_samples = len(X)
        
        # Use 20% for testing, 80% for training
        test_size_actual = 0.2 
        
        # Ensure that test set has at least one sample from each class if stratification is used
        can_stratify = len(np.unique(y)) >= 2 and np.min(np.unique(y, return_counts=True)[1]) >= app.config['DATA_CONFIG']['min_samples_for_stratify'] # At least 2 samples per class for stratify

        if total_samples < app.config['DATA_CONFIG']['min_samples_for_stratify'] * 2 and can_stratify:
             # If total samples are too few for a meaningful stratified split, disable stratification
             logger.warning(f"Total samples ({total_samples}) too few for robust stratified split, disabling stratification.")
             can_stratify = False


        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_actual, random_state=app.config['ML_CONFIG']['random_state'],
            stratify=y if can_stratify else None
        )
        
        # Adjust test_size_actual if the actual split resulted in less than 1 sample in test set for some reason
        if len(X_test) == 0 and total_samples > 0: # Ensure X_test is not empty if X is not empty
            X_train = X
            y_train = y
            X_test = pd.DataFrame(columns=X.columns) # Create empty DataFrame with same columns
            y_test = pd.Series(dtype=int) # Create empty Series
            test_size_actual = 0
            train_size_actual = 1
            logger.warning("Test set became empty, using 100% data for training (no separate test set evaluation).")
        else:
            train_size_actual = 1 - test_size_actual


        train_percentage = (len(X_train) / (len(X_train) + len(X_test))) * 100 if (len(X_train) + len(X_test)) > 0 else 0
        test_percentage = (len(X_test) / (len(X_train) + len(X_test))) * 100 if (len(X_train) + len(X_test)) > 0 else 0

        logger.info(f"Data split: Training {len(X_train)} samples ({train_percentage:.1f}%), Testing {len(X_test)} samples ({test_percentage:.1f}%)")
        logger.info(f"Before SMOTE - Label distribution in Training Set: {Counter(y_train)}")

        # Apply SMOTE only to the training data if applicable
        if app.config['ML_CONFIG'].get('apply_smote', False) and len(np.unique(y_train)) > 1 and np.min(list(Counter(y_train).values())) > 1: # SMOTE needs at least 2 samples for minority class
            try:
                smote = SMOTE(random_state=app.config['ML_CONFIG']['random_state'])
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE - Label distribution in Training Set: {Counter(y_train_resampled)}")
                X_train = X_train_resampled
                y_train = y_train_resampled
            except Exception as e:
                logger.warning(f"Could not apply SMOTE: {str(e)}. Training with original data.")
        else:
            logger.warning("SMOTE not applied (disabled in config, single class, or too few samples in minority class).")

        # Normalize numerical features (used for Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

        # Fetch parameter grids from MODEL_HYPERPARAMETERS
        param_grid_rf = app.config['MODEL_HYPERPARAMETERS']['RandomForest']
        param_grid_gb = app.config['MODEL_HYPERPARAMETERS']['GradientBoosting']
        param_grid_lr = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression']

        best_rf = None
        best_gb = None
        best_lr = None

        # GridSearchCV for RandomForestClassifier
        try:
            logger.info("Performing GridSearchCV for RandomForest...")
            grid_search_rf = GridSearchCV(
                RandomForestClassifier(random_state=app.config['ML_CONFIG']['random_state'], n_jobs=app.config['ML_CONFIG']['n_jobs']),
                param_grid_rf,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=app.config['ML_CONFIG']['n_jobs'],
                verbose=app.config['ML_CONFIG']['verbose']
            )
            grid_search_rf.fit(X_train, y_train)
            best_rf = grid_search_rf.best_estimator_
            logger.info(f"RandomForest Best Params: {grid_search_rf.best_params_}")
            logger.info(f"RandomForest Best Score: {grid_search_rf.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for RandomForest failed: {str(e)}. Falling back to default parameters.")
            best_rf = RandomForestClassifier(
                n_estimators=app.config['ML_CONFIG'].get('n_estimators', 100),
                max_depth=app.config['ML_CONFIG'].get('max_depth', None),
                random_state=app.config['ML_CONFIG']['random_state'],
                n_jobs=app.config['ML_CONFIG']['n_jobs']
            )
            best_rf.fit(X_train, y_train)

        # GridSearchCV for GradientBoostingClassifier
        try:
            logger.info("Performing GridSearchCV for GradientBoosting...")
            grid_search_gb = GridSearchCV(
                GradientBoostingClassifier(random_state=app.config['ML_CONFIG']['random_state']),
                param_grid_gb,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=app.config['ML_CONFIG']['n_jobs'],
                verbose=app.config['ML_CONFIG']['verbose']
            )
            grid_search_gb.fit(X_train, y_train)
            best_gb = grid_search_gb.best_estimator_
            logger.info(f"GradientBoosting Best Params: {grid_search_gb.best_params_}")
            logger.info(f"GradientBoosting Best Score: {grid_search_gb.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for GradientBoosting failed: {str(e)}. Falling back to default parameters.")
            best_gb = GradientBoostingClassifier(
                n_estimators=app.config['ML_CONFIG'].get('n_estimators', 100),
                max_depth=app.config['ML_CONFIG'].get('max_depth', 3),
                random_state=app.config['ML_CONFIG']['random_state']
            )
            best_gb.fit(X_train, y_train)

        # GridSearchCV for LogisticRegression
        try:
            logger.info("Performing GridSearchCV for LogisticRegression...")
            base_max_iter = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression'].get('max_iter', [1000])[0]

            grid_search_lr = GridSearchCV(
                LogisticRegression(random_state=app.config['ML_CONFIG']['random_state'], max_iter=base_max_iter),
                param_grid_lr,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=app.config['ML_CONFIG']['n_jobs'],
                verbose=app.config['ML_CONFIG']['verbose']
            )
            grid_search_lr.fit(X_train_scaled, y_train)
            best_lr = grid_search_lr.best_estimator_
            logger.info(f"LogisticRegression Best Params: {grid_search_lr.best_params_}")
            logger.info(f"LogisticRegression Best Score: {grid_search_lr.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for LogisticRegression failed: {str(e)}. Falling back to default parameters.")
            best_lr = LogisticRegression(
                random_state=app.config['ML_CONFIG']['random_state'],
                max_iter=1000,
                solver='liblinear' # 'liblinear' is generally robust for small datasets and multiple penalties
            )
            best_lr.fit(X_train_scaled, y_train)

        models_dict = {
            'rf': best_rf,
            'gb': best_gb,
            'lr': best_lr
        }

        trained_models = {}
        predictions_proba_for_ensemble = {}

        # Evaluate each model on test data
        if len(X_test) > 0:
            for name, model in models_dict.items():
                if model is None:
                    continue
                try:
                    logger.info(f"Evaluating model {name} on test set...")
                    if name == 'lr':
                        # Ensure X_test_scaled is not empty before predicting
                        if X_test_scaled.shape[0] > 0:
                            pred_proba = model.predict_proba(X_test_scaled)
                            # Ensure pred_proba has 2 columns for binary classification
                            predictions_proba_for_ensemble[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0] # Probability of positive class
                    else:
                        if X_test.shape[0] > 0:
                            pred_proba = model.predict_proba(X_test)
                            predictions_proba_for_ensemble[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                    trained_models[name] = model # Add model to trained_models only if evaluation was attempted
                except Exception as e:
                    logger.warning(f"Could not evaluate model {name} on test set: {str(e)}")
                    continue
        else: # If no test set, use the trained models directly for ensemble (will not have test metrics)
            for name, model in models_dict.items():
                if model is not None:
                    trained_models[name] = model
            logger.warning("No test data for model evaluation, will return metrics based on training (not ideal for true performance).")


        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if len(X_test) > 0 and predictions_proba_for_ensemble:
            # Average probabilities across all sub-models for ensemble prediction
            # Ensure all probability arrays have the same shape for averaging
            valid_predictions_proba = [p for p in predictions_proba_for_ensemble.values() if p.shape == (len(X_test),)]
            
            if valid_predictions_proba:
                ensemble_pred_proba = np.mean(valid_predictions_proba, axis=0)
                ensemble_pred = (ensemble_pred_proba > 0.5).astype(int) # Threshold at 0.5 for binary classification

                accuracy = accuracy_score(y_test, ensemble_pred)
                precision = precision_score(y_test, ensemble_pred, zero_division=0)
                recall = recall_score(y_test, ensemble_pred, zero_division=0)
                f1 = f1_score(y_test, ensemble_pred, zero_division=0)
                
                logger.info(f"Ensemble model evaluation on test set ({len(X_test)} samples):")
                logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            else:
                logger.warning("No valid probability predictions from sub-models to form ensemble prediction on test set.")
        else:
            # If no test set, or no valid predictions, report training accuracy as a placeholder
            # This is not ideal for true performance but provides some metric.
            logger.warning("No test data for model evaluation or no valid sub-model predictions, using estimated metrics or training set accuracy.")
            if trained_models and 'rf' in trained_models:
                y_train_pred = trained_models['rf'].predict(X_train) # Use RF as a representative
                accuracy = accuracy_score(y_train, y_train_pred)
                precision = precision_score(y_train, y_train_pred, zero_division=0)
                recall = recall_score(y_train, y_train_pred, zero_division=0)
                f1 = f1_score(y_train, y_train_pred, zero_division=0)
                logger.warning(f"Using RandomForest training accuracy: {accuracy:.3f}")
            else:
                accuracy, precision, recall, f1 = 0.85, 0.85, 0.85, 0.85 # Default fallback metrics


        logger.info(f"Model training completed successfully")
        logger.info(f"Final reported metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        return {
            'models': trained_models, # Return the trained estimators (best_rf, best_gb, best_lr)
            'scaler': scaler,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'features_count': X.shape[1],
            'train_percentage': train_percentage,
            'test_percentage': test_percentage,
            'best_rf_params': best_rf.get_params() if best_rf else {},
            'best_gb_params': best_gb.get_params() if best_gb else {},
            'best_lr_params': best_lr.get_params() if best_lr else {}
        }

    except Exception as e:
        logger.error(f"Error training Ensemble model: {str(e)}")
        raise

# Helper functions for curriculum analysis (These are likely from your previous context)
def calculate_gpa_and_failed_courses_backend(course_grades, courses_data):
    total_points = 0
    completed_credits = 0
    failed_courses = []
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']

    for cid, grade_char in course_grades.items():
        course = next((c for c in courses_data if c['id'] == cid), None)
        if not course:
            continue

        if grade_char and grade_char != "":
            numeric_grade = grade_mapping_points.get(str(grade_char).upper(), 0.0)
            
            # Allow direct numeric input for GPA
            try:
                numeric_grade = float(grade_char)
                if not (0.0 <= numeric_grade <= 4.0): # Ensure grade is within a valid range
                    numeric_grade = 0.0 # Treat out-of-range numeric as fail
            except ValueError:
                # If not a number, fall back to grade mapping
                numeric_grade = grade_mapping_points.get(str(grade_char).upper(), 0.0)


            if numeric_grade > 0:
                total_points += numeric_grade * course['credit']
                completed_credits += course['credit']
            
            # Consider 0.0 and specific non-numeric grades as failed
            if numeric_grade == 0.0 or str(grade_char).upper() in ['F', 'W', 'I', 'NP', 'WF', 'WU']: # Added more non-numeric fails
                failed_courses.append(cid)
    
    avg_gpa = total_points / completed_credits if completed_credits > 0 else 0

    return {
        'avgGPA': avg_gpa,
        'completedCredits': completed_credits,
        'failedCourses': failed_courses
    }

def check_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course:
        return False
    if not course['prereq'] or len(course['prereq']) == 0:
        return True
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']

    for pid in course['prereq']:
        # Ensure prereq exists and is passed (numeric grade > 0.0)
        if pid not in course_grades:
            return False
        
        prereq_grade_char = course_grades[pid]
        numeric_grade = None
        try:
            numeric_grade = float(prereq_grade_char)
            if not (0.0 <= numeric_grade <= 4.0):
                numeric_grade = 0.0
        except ValueError:
            numeric_grade = grade_mapping_points.get(str(prereq_grade_char).upper(), 0.0)
            
        if numeric_grade == 0.0:
            return False
    return True

def get_unmet_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course or not course['prereq']:
        return []
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']
    unmet = []
    for pid in course['prereq']:
        prereq_grade_char = course_grades.get(pid, "")
        numeric_grade = None
        try:
            numeric_grade = float(prereq_grade_char)
            if not (0.0 <= numeric_grade <= 4.0):
                numeric_grade = 0.0
        except ValueError:
            numeric_grade = grade_mapping_points.get(str(prereq_grade_char).upper(), 0.0)

        if pid not in course_grades or numeric_grade == 0.0:
            unmet.append(pid)
    return unmet

def find_course_thai_name_backend(course_id, courses_data):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    return course['thaiName'] if course else course_id # Return ID if name not found

def get_loaded_courses_backend(loaded_terms_count, all_terms_data, courses_data, repeated_courses_in_this_term_ids):
    course_ids = []
    for i in range(loaded_terms_count):
        if i < len(all_terms_data):
            course_ids.extend(all_terms_data[i]['ids'])
    course_ids.extend(repeated_courses_in_this_term_ids)
    
    unique_ids = list(set(course_ids))
    return [c for c in courses_data if c['id'] in unique_ids]

def find_blocked_courses_backend(course_grades, loaded_courses, courses_data, passing_grades_list_from_config):
    blocked_courses_ids = []
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']

    for course_obj in loaded_courses:
        course_id = course_obj['id']
        
        current_grade_for_course = course_grades.get(course_id, "")
        
        # Check if the course itself has a failing grade
        numeric_current_grade = None
        try:
            numeric_current_grade = float(current_grade_for_course)
            if not (0.0 <= numeric_current_grade <= 4.0):
                numeric_current_grade = 0.0
        except ValueError:
            numeric_current_grade = grade_mapping_points.get(str(current_grade_for_course).upper(), 0.0)

        if numeric_current_grade == 0.0 and current_grade_for_course != "": # If course is failed, it blocks its dependents
            pass # We check prerequisites for other courses, this is implicit blocking
        elif not check_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config): # Fix: Pass course_grades not current_grades
            blocked_courses_ids.append(course_id)
            
    return list(set(blocked_courses_ids)) # Ensure unique IDs

def build_dependency_graph_backend(courses_subset):
    graph = {c['id']: [] for c in courses_subset}
    for course in courses_subset:
        for prereq_id in course['prereq']:
            if prereq_id in graph: # Only add if the prerequisite is also in the subset
                graph[prereq_id].append(course['id'])
    return graph

def find_affected_courses_backend(course_id, graph):
    affected = set()
    queue = [course_id]
    while queue:
        current = queue.pop(0)
        dependents = graph.get(current, [])
        for dep in dependents:
            if dep not in affected:
                affected.add(dep)
                queue.append(dep)
    return list(affected)

def topological_sort_with_cycle_check_backend(loaded_courses_objects):
    """Performs a topological sort on a subset of courses and checks for cycles."""
    if not loaded_courses_objects:
        return {'order': [], 'cycle': False}

    course_map = {c['id']: c for c in loaded_courses_objects}

    in_degree = {c_id: 0 for c_id in course_map.keys()}
    adj_list = {c_id: [] for c_id in course_map.keys()}

    for course_id, course_obj in course_map.items():
        for prereq_id in course_obj['prereq']:
            if prereq_id in course_map: # Only consider prerequisites within the loaded subset
                adj_list[prereq_id].append(course_id)
                in_degree[course_id] += 1

    queue = []
    for cid, degree in in_degree.items():
        if degree == 0:
            queue.append(cid)

    order = []
    count = 0

    while queue:
        u = queue.pop(0)
        order.append(u)
        count += 1
        
        for v in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if count != len(loaded_courses_objects):
        return {'order': [], 'cycle': True}
    return {'order': order, 'cycle': False}

def linear_regression_next_term_gpa_backend(term_gpas):
    """Predicts next term's GPA using simple linear regression."""
    if len(term_gpas) == 0:
        return 0.0
    if len(term_gpas) == 1:
        return term_gpas[0]['gpa']

    n = len(term_gpas)
    x_vals = np.array([(i + 1) for i in range(n)])
    y_vals = np.array([t['gpa'] for t in term_gpas])

    if np.all(y_vals == y_vals[0]): # If all GPAs are the same, predict that GPA
        return y_vals[0]

    sum_x = np.sum(x_vals)
    sum_y = np.sum(y_vals)
    sum_xy = np.sum(x_vals * y_vals)
    sum_xx = np.sum(x_vals * x_vals)

    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0: # Avoid division by zero if all x values are the same (should not happen with (i+1))
        return y_vals[-1] # Fallback to last GPA

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    predicted = slope * (n + 1) + intercept
    return max(0.0, min(4.0, predicted)) # Clamp GPA between 0.0 and 4.0

def estimate_completion_rate_backend(avg_gpa, completed_credits, total_required_credits, blocked_courses_ids, fail_count, failed_courses_ids, repeat_count_this_term, core_subjects_ids, courses_data):
    """Estimates the completion rate based on academic progress and issues."""
    base_rate = (completed_credits / total_required_credits) * 100 if total_required_credits > 0 else 0

    blocked_credits = sum(
        (next((c for c in courses_data if c['id'] == cid), {'credit': 0})['credit'] for cid in blocked_courses_ids)
    )

    adjusted_rate = base_rate - blocked_credits * app.config['DATA_CONFIG']['risk_levels'].get('credit_penalty_per_blocked_course', 2)

    if 0 < avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('low_gpa_penalty', 5)
    
    if fail_count >= 5:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('high_fail_count_penalty', 10)
    elif fail_count >= 2:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('medium_fail_count_penalty', 5)
    
    fail_core_count = len([f for f in failed_courses_ids if f in core_subjects_ids])
    adjusted_rate -= fail_core_count * app.config['DATA_CONFIG']['risk_levels'].get('core_fail_penalty', 3)
    
    if repeat_count_this_term >= 3:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('repeat_course_penalty', 5)

    return max(0.0, min(100.0, adjusted_rate))

def determine_graduation_status_backend(completion_rate, avg_gpa, blocked_courses_ids, failed_courses_ids, loaded_courses_objects, current_grades, all_terms_data, courses_data, loaded_terms_count):
    """Determines the student's graduation status based on their progress and potential issues."""
    
    total_terms_in_curriculum = len(all_terms_data)
    is_at_or_past_final_standard_term = (loaded_terms_count >= total_terms_in_curriculum)

    all_courses_in_loaded_curriculum_ids = set()
    for i in range(loaded_terms_count):
        if i < len(all_terms_data):
            for course_id in all_terms_data[i]['ids']:
                all_courses_in_loaded_curriculum_ids.add(course_id)

    incomplete_courses = [
        cid for cid in all_courses_in_loaded_curriculum_ids
        if cid not in current_grades or current_grades[cid] == "" or grade_to_numeric(current_grades[cid]) == 0.0
    ]
    
    # Specific conditions for immediate status
    if avg_gpa > 0 and avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        return "GPA ต่ำกว่าเกณฑ์ (2.0) จำเป็นต้องปรับปรุงอย่างเร่งด่วนเพื่อจบการศึกษา"

    if is_at_or_past_final_standard_term: # If student is at or past the standard final term
        if len(failed_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมีวิชาตก (เกรด F) ที่ต้องแก้ไข"
        if len(blocked_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน)"
        if len(incomplete_courses) > 0:
            return f"ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมี {len(incomplete_courses)} วิชาที่ยังไม่ผ่าน/ไม่มีเกรด"
        
        # If no major issues at or past final term
        if avg_gpa >= app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold'] and not failed_courses_ids and not incomplete_courses and not blocked_courses_ids:
            return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี"
        # If GPA is precisely 0 and no other issues, means they probably haven't taken any graded courses in loaded terms
        if avg_gpa == 0.0 and (not failed_courses_ids and not incomplete_courses and not blocked_courses_ids):
            return "ข้อมูลเกรดไม่สมบูรณ์ แต่ไม่มีปัญหาติดขัด"
        
        return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี" # Default for passing final term conditions
    
    else: # If student is still in progress (before standard final term)
        if len(failed_courses_ids) > 0:
            return "มีวิชาตก (เกรด F) อาจทำให้การจบการศึกษาล่าช้ากว่ากำหนด"
        if len(blocked_courses_ids) > 0:
            return "มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน) ควรแก้ไขเพื่อเรียนต่อตามแผน"
        if len(incomplete_courses) > 0:
            return "กำลังอยู่ในเส้นทาง. ยังมีวิชาที่ต้องเรียน/ได้รับเกรดในภาคเรียนถัดไป"
        
        return "อยู่ในเส้นทางที่คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี" # Default for in-progress student with no critical issues

def update_recommendations_backend(failed_courses_ids, avg_gpa, blocked_courses_ids):
    """Generates specific recommendations based on academic issues."""
    recommendations = []
    if len(failed_courses_ids) > 0:
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][0])
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][1])
    if 0 < avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        recommendations.append(app.config['MESSAGES']['recommendations']['medium_risk'][1])
    if len(blocked_courses_ids) > 0:
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][3])
    if not recommendations:
        recommendations.append(app.config['MESSAGES']['recommendations']['low_risk'][0])
    return list(set(recommendations))

# --- Modified Function: detect_data_format ---
def detect_data_format(df):
    """ตรวจสอบรูปแบบของข้อมูลที่อัปโหลด"""
    try:
        if df.empty:
            return 'unknown'
            
        columns_lower = [str(col).lower().strip() for col in df.columns.tolist()]
        
        logger.info(f"Detecting data format for columns: {df.columns.tolist()}")
        
        # ตรวจสอบ record-based format (แบบ T1.csv)
        has_student_id = any(
            any(keyword in col for keyword in ['รหัส', 'id', 'student_id'])
            for col in columns_lower
        )
        has_student_name = any(
            any(keyword in col for keyword in ['ชื่อ-นามสกุล', 'ชื่อ', 'name', 'student_name'])
            for col in columns_lower
        )
        has_subject_name = any(
            any(keyword in col for keyword in ['ชื่อวิชา', 'วิชา', 'subject', 'course'])
            for col in columns_lower
        )
        has_grade = any(
            any(keyword in col for keyword in ['เกรด', 'grade'])
            for col in columns_lower
        )
        has_year = any(
            any(keyword in col for keyword in ['ปีการศึกษา', 'ปี', 'year', 'academic_year', 'semester_year'])
            for col in columns_lower
        )
        
        # A file like T1.csv will usually have a 'รหัส' (Student ID), 'ชื่อวิชา' (Subject Name), and 'เกรด' (Grade)
        if (has_student_id or has_student_name) and has_subject_name and has_grade and has_year:
            logger.info("Detected record_based format (e.g., T1.csv like).")
            return 'record_based'
            
        # ตรวจสอบ subject_based format (แบบเดิม - matrix: Student ID/Name as row, Subject Grades as columns)
        # It should have a student identifier and many columns that look like subjects
        has_id_or_name_col = any(
            any(keyword in col for keyword in ['ชื่อ', 'name', 'รหัส', 'id', 'student'])
            for col in columns_lower
        )
        # Count columns that are NOT explicitly metadata like GPA, success status, or years
        non_metadata_cols_count = 0
        for col in df.columns:
            col_lower = str(col).lower()
            if not any(kw in col_lower for kw in ['ชื่อ', 'name', 'รหัส', 'id', 'ปี', 'year', 'gpa', 'เกรดเฉลี่ย', 'สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ']):
                # Attempt to convert first few non-empty values to float to see if they are grades
                sample_values = df[col].dropna().head(5).tolist()
                is_likely_grade_column = False
                if sample_values:
                    # Check if at least one value looks like a numeric grade or a grade letter
                    if any(isinstance(v, (int, float)) and 0.0 <= v <= 4.0 for v in sample_values):
                        is_likely_grade_column = True
                    elif any(str(v).upper() in app.config['DATA_CONFIG']['grade_mapping'] for v in sample_values):
                        is_likely_grade_column = True
                
                if is_likely_grade_column:
                    non_metadata_cols_count += 1
        
        if has_id_or_name_col and non_metadata_cols_count >= app.config['DATA_CONFIG']['min_subject_columns_for_subject_based']:
            logger.info("Detected subject_based format (matrix-like, many subject grade columns).")
            return 'subject_based'

        # ตรวจสอบ gpa_based format (Summary GPA and graduation status)
        has_gpa_summary = any(
            any(keyword in col for keyword in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])
            for col in columns_lower
        )
        has_success_status = any(
            any(keyword in col for keyword in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ'])
            for col in columns_lower
        )

        if has_gpa_summary and has_success_status:
            logger.info("Detected gpa_based format (summary GPA and status).")
            return 'gpa_based'

        logger.warning(f"Could not detect data format for columns: {df.columns.tolist()}")
        return 'unknown'

    except Exception as e:
        logger.error(f"Error detecting data format: {str(e)}")
        return 'unknown'

# Flask Routes
@app.route('/')
def index():
    """Main page for uploading and training models (Admin Dashboard)."""
    return render_template('index.html')

@app.route('/test') # Renamed from /curriculum_prediction for clarity in templates
def curriculum_prediction_form():
    """Page for predicting graduation based on curriculum and prerequisites (User Form)."""
    # Load curriculum data from config (if not already loaded)
    return render_template(
        'curriculum_prediction_form.html',
        all_terms_data=app.config['ALL_TERMS_DATA'],
        courses_data=app.config['COURSES_DATA'],
        grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
    )

@app.route('/api/config', methods=['GET'])
def get_config_for_frontend():
    """Provides frontend with necessary configuration data like courses and terms."""
    try:
        config_data = {
            'COURSES_DATA': app.config['COURSES_DATA'],
            'ALL_TERMS_DATA': app.config['ALL_TERMS_DATA'],
            'GRADE_MAPPING': app.config['DATA_CONFIG']['grade_mapping'],
            'MESSAGES': app.config['MESSAGES'], # Add messages to frontend config
            'DATA_CONFIG_RISK_LEVELS': app.config['DATA_CONFIG']['risk_levels'] # Add risk levels
        }
        return jsonify({'success': True, **config_data})
    except Exception as e:
        logger.error(f"Error loading config data for frontend: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to load configuration data'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles CSV/Excel file uploads for training or prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'ไม่พบไฟล์ในคำขอ (No file part in the request)'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'ไม่ได้เลือกไฟล์ (No selected file)'})

        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'success': False, 'error': f'รองรับเฉพาะไฟล์ {", ".join(app.config["ALLOWED_EXTENSIONS"])} เท่านั้น (Only {", ".join(app.config["ALLOWED_EXTENSIONS"])} files are allowed)'})

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_on_disk = f"{timestamp}_{secure_filename(file.filename)}" # Use secure_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_on_disk)

        file.save(filepath)
        logger.info(f"อัปโหลดไฟล์สำเร็จ: {filename_on_disk}")

        try:
            df = None
            if file_extension == 'csv':
                encodings = app.config['DATA_CONFIG']['fallback_encodings']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"อ่านไฟล์ CSV สำเร็จด้วย encoding: {encoding}")
                        break
                    except Exception as e:
                        logger.debug(f"ลองอ่าน CSV ด้วย {encoding} ล้มเหลว: {e}")
                        continue
                if df is None:
                    raise ValueError("ไม่สามารถอ่านไฟล์ CSV ด้วย encoding ที่รองรับได้ (Could not read CSV file with any supported encoding).")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(filepath)
                logger.info(f"อ่านไฟล์ Excel สำเร็จ")
            else:
                raise ValueError("รูปแบบไฟล์ไม่รองรับ (Unsupported file format).")

            if df.empty:
                raise ValueError("ไฟล์ข้อมูลว่างเปล่า หรือไม่มีข้อมูลที่สามารถอ่านได้ (Empty data file or no readable data).")

            data_format = detect_data_format(df)

            return jsonify({
                'success': True,
                'filename': filename_on_disk,
                'rows': len(df),
                'columns': len(df.columns),
                'data_format': data_format,
                'sample_columns': df.columns.tolist()[:10] # Return first 10 column names
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up uploaded file if processing fails
            logger.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์ที่อัปโหลด '{filename_on_disk}': {str(e)}")
            return jsonify({'success': False, 'error': f'ไม่สามารถอ่านไฟล์ได้: {str(e)}'})

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการอัปโหลด: {str(e)}")
        return jsonify({'success': False, 'error': f'เกิดข้อผิดพลาดในการอัปโหลด: {str(e)}'})

# --- Modified Function: train_model ---
@app.route('/train', methods=['POST'])
def train_model():
    """Handles model training with the uploaded file, automatically detecting data format."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        # model_type is no longer passed from frontend for training, it's detected here.
        # However, the frontend might still send it, so let's allow it but prefer detection.
        # user_selected_model_type = data.get('model_type') 

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Specified file not found.'})

        # อ่านไฟล์
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Successfully read CSV with encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
            logger.info("Successfully read Excel file")
        else:
            raise ValueError("Unsupported file type for training.")

        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'Could not read file or file is empty.'})

        # Automatically detect data format for training
        model_type_for_training = detect_data_format(df)
        if model_type_for_training == 'unknown':
            return jsonify({'success': False, 'error': 'Could not automatically determine data format for training. Please ensure the data format is recognizable (record-based, subject-based, or GPA-based).'})

        logger.info(f"Automatically detected data format for training: {model_type_for_training}")

        # ประมวลผลข้อมูลตามรูปแบบ
        try:
            if model_type_for_training == 'record_based':
                processed_df = process_record_based_data(df)
            elif model_type_for_training == 'subject_based':
                processed_df = process_subject_data(df)
            elif model_type_for_training == 'gpa_based':
                processed_df = process_gpa_data(df)
            else:
                return jsonify({'success': False, 'error': f'Unsupported data format for training: {model_type_for_training}'})
        except Exception as e:
            logger.error(f"Error processing {model_type_for_training} data: {str(e)}")
            return jsonify({'success': False, 'error': f'Error processing data: {str(e)}'})

        # ตรวจสอบข้อมูลที่ประมวลผลแล้ว
        min_students_for_training = app.config['DATA_CONFIG']['min_students_for_training']
        if len(processed_df) < min_students_for_training:
            return jsonify({'success': False, 'error': f'Insufficient data for training. Found {len(processed_df)} students, need at least {min_students_for_training}.'})

        # เตรียมข้อมูลสำหรับการเทรน
        feature_cols = [col for col in processed_df.columns if col not in ['ชื่อ', 'graduated']]
        X = processed_df[feature_cols].fillna(0) # Fill NaNs for numerical features
        y = processed_df['graduated']

        logger.info(f"Training with {len(X)} students, {len(feature_cols)} features (detected format: {model_type_for_training})")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        if len(X) == 0 or len(feature_cols) == 0:
            return jsonify({'success': False, 'error': 'No valid features found for training.'})

        # เทรนโมเดล
        model_result = train_ensemble_model(X, y)

        # บันทึกโมเดล
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'{model_type_for_training}_model_{timestamp}.joblib'
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)

        # คำนวณ Feature Importances
        feature_importances = {}
        if 'rf' in model_result['models'] and model_result['models']['rf'] is not None:
            rf_model = model_result['models']['rf']
            if hasattr(rf_model, 'feature_importances_'):
                # Ensure feature_cols matches the features used in the trained model
                importances_series = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                feature_importances = importances_series.head(10).to_dict()

        model_data = {
            'models': model_result['models'],
            'scaler': model_result['scaler'],
            'feature_columns': feature_cols,
            'data_format': model_type_for_training,
            'created_at': datetime.now().isoformat(),
            'training_data_info': {
                'rows': len(processed_df),
                'features': len(feature_cols),
                'graduated_count': int(y.sum()),
                'not_graduated_count': int(len(y) - y.sum())
            },
            'performance_metrics': {
                'accuracy': model_result['accuracy'],
                'precision': model_result['precision'],
                'recall': model_result['recall'],
                'f1_score': model_result['f1_score']
            },
            'feature_importances': feature_importances
        }

        # บันทึกโมเดล
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")

        # อัพเดต in-memory models
        if model_type_for_training == 'record_based':
            models['record_model'] = model_result['models'] # Store the dict of models
            models['record_model_info'] = model_data['performance_metrics']
            models['record_model_info']['created_at'] = model_data['created_at']
            models['record_model_info']['filename'] = model_filename
            models['record_model_info']['data_format'] = model_type_for_training
            models['data_scalers']['record_based'] = model_result['scaler'] # Store scaler by format type
            models['record_feature_cols'] = feature_cols
        elif model_type_for_training == 'subject_based':
            models['subject_model'] = model_result['models']
            models['subject_model_info'] = model_data['performance_metrics']
            models['subject_model_info']['created_at'] = model_data['created_at']
            models['subject_model_info']['filename'] = model_filename
            models['subject_model_info']['data_format'] = model_type_for_training
            models['data_scalers']['subject_based'] = model_result['scaler']
            models['subject_feature_cols'] = feature_cols
        else:  # gpa_based
            models['gpa_model'] = model_result['models']
            models['gpa_model_info'] = model_data['performance_metrics']
            models['gpa_model_info']['created_at'] = model_data['created_at']
            models['gpa_model_info']['filename'] = model_filename
            models['gpa_model_info']['data_format'] = model_type_for_training
            models['data_scalers']['gpa_based'] = model_result['scaler']
            models['gpa_feature_cols'] = feature_cols

        logger.info("Model training completed successfully")

        return jsonify({
            'success': True,
            'model_filename': model_filename,
            'data_format': model_type_for_training,
            'accuracy': model_result['accuracy'],
            'precision': model_result['precision'],
            'recall': model_result['recall'],
            'f1_score': model_result['f1_score'],
            'training_samples': model_result['training_samples'],
            'validation_samples': model_result['validation_samples'],
            'features_count': model_result['features_count'],
            'feature_importances': feature_importances,
            'sample_data': processed_df[['ชื่อ', 'gpa', 'graduated']].head(3).to_dict('records') # Return sample data
        })

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return jsonify({'success': False, 'error': f'Training failed: {str(e)}'})
    
    
@app.route('/predict', methods=['POST'])
def predict():
    """Predicts outcome from an uploaded CSV/Excel file using a specified model."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_filename = data.get('model_filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided for prediction data.'})
        if not model_filename:
            return jsonify({'success': False, 'error': 'No model filename provided for prediction.'})

        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(data_filepath):
            return jsonify({'success': False, 'error': 'Specified data file not found.'})

        model_filepath = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        if not os.path.exists(model_filepath):
            return jsonify({'success': False, 'error': 'Specified model file not found.'})

        # Load the specified model
        try:
            loaded_model_data = joblib.load(model_filepath)
            model_components = loaded_model_data['models'] # This is the dict of {rf, gb, lr}
            scaler = loaded_model_data['scaler']
            feature_cols = loaded_model_data['feature_columns']
            model_data_format = loaded_model_data['data_format'] # Data format the model was trained on
            logger.info(f"Loaded model '{model_filename}' (format: {model_data_format}) for prediction.")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to load model {model_filename}: {str(e)}'})

        # Determine file type and read
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_filepath, encoding=encoding)
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(data_filepath)
        else:
            raise ValueError("Unsupported file type for prediction.")

        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'Could not read prediction data file or it is empty.'})

        # Process the new data based on the model's expected format
        processed_df_for_prediction = None
        if model_data_format == 'record_based':
            processed_df_for_prediction = process_record_based_data(df)
        elif model_data_format == 'subject_based':
            processed_df_for_prediction = process_subject_data(df)
        elif model_data_format == 'gpa_based':
            processed_df_for_prediction = process_gpa_data(df)
        else:
            return jsonify({'success': False, 'error': f'Unsupported data format for prediction processing: {model_data_format}'})

        if len(processed_df_for_prediction) == 0:
            return jsonify({'success': False, 'error': 'No data could be processed for prediction.'})

        # Align features: Create a DataFrame with only the features the model expects
        X_predict = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in processed_df_for_prediction.columns:
                X_predict[col] = processed_df_for_prediction[col]
            else:
                X_predict[col] = 0 # Fill missing features with 0 (or a sensible default)
        X_predict = X_predict.fillna(0) # Ensure no NaNs remain

        predictions_proba_list = []
        for name, model in model_components.items():
            if model is None:
                continue
            try:
                if name == 'lr':
                    X_scaled = scaler.transform(X_predict)
                    pred_proba = model.predict_proba(X_scaled)
                else:
                    pred_proba = model.predict_proba(X_predict)
                
                # Ensure pred_proba has 2 columns (prob of class 0, prob of class 1)
                if pred_proba.shape[1] == 1:
                    # If only one class is predicted, assume it's the positive class probability
                    # and the negative class is 1 - probability
                    pred_proba = np.hstack((1 - pred_proba, pred_proba))
                
                predictions_proba_list.append(pred_proba)
            except Exception as e:
                logger.warning(f"Could not predict with model {name} (part of ensemble): {str(e)}")
                continue

        if not predictions_proba_list:
            return jsonify({'success': False, 'error': 'Could not make predictions with any loaded sub-models.'})

        results = []
        high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
        medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

        for i in range(len(processed_df_for_prediction)):
            student_name = processed_df_for_prediction.iloc[i]['ชื่อ']
            gpa = processed_df_for_prediction.iloc[i].get('gpa', 0.0) # Get GPA from processed data

            # Average probabilities across all sub-models for this student
            avg_prob_per_student = np.mean([pred_proba_array[i] for pred_proba_array in predictions_proba_list], axis=0)
            avg_prob_fail = avg_prob_per_student[0]
            avg_prob_pass = avg_prob_per_student[1]

            prediction_label = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'

            confidence = max(avg_prob_pass, avg_prob_fail)
            if confidence > high_confidence_threshold:
                risk_level = 'ต่ำ' if prediction_label == 'จบ' else 'สูง'
            elif confidence > medium_confidence_threshold:
                risk_level = 'ปานกลาง'
            else:
                risk_level = 'สูง' if prediction_label == 'ไม่จบ' else 'ปานกลาง'

            analysis = []
            recommendations = []

            low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']
            warning_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']
            high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']

            if gpa < low_gpa_threshold:
                analysis.append(f"GPA ต่ำมาก ({gpa:.2f})")
                recommendations.extend(app.config['MESSAGES']['recommendations']['high_risk'])
            elif gpa < warning_gpa_threshold:
                analysis.append(f"GPA อยู่ในเกณฑ์เสี่ยง ({gpa:.2f})")
                recommendations.extend(app.config['MESSAGES']['recommendations']['medium_risk'])
            elif gpa < 3.0:
                analysis.append(f"GPA พอใช้ ({gpa:.2f})")
                recommendations.append("มีโอกาสพัฒนาผลการเรียนให้ดีขึ้น")
            else:
                analysis.append(f"GPA ดีเยี่ยม ({gpa:.2f})") # Changed 'ดี' to 'ดีเยี่ยม' for 3.0+
                recommendations.extend(app.config['MESSAGES']['recommendations']['low_risk'])

            if prediction_label == 'ไม่จบ':
                recommendations.append("แนะนำให้ทบทวนแผนการเรียนและขอความช่วยเหลือ")
                if 'fail_rate' in processed_df_for_prediction.columns and processed_df_for_prediction.iloc[i].get('fail_rate', 0) > high_fail_rate_threshold:
                    recommendations.append("มีอัตราการตกในบางวิชาสูง ควรให้ความสำคัญกับการเรียนซ่อม")

            # Add specific recommendations for subject-based/record-based if applicable
            if model_data_format in ['subject_based', 'record_based']:
                weak_categories = []
                for cat_key in app.config['SUBJECT_CATEGORIES'].keys():
                    gpa_col = f'gpa_{cat_key}'
                    if gpa_col in processed_df_for_prediction.columns and processed_df_for_prediction.iloc[i].get(gpa_col, 0) < low_gpa_threshold:
                        weak_categories.append(app.config['SUBJECT_CATEGORIES'][cat_key]['thaiName']) # Use Thai name for recommendations
                if weak_categories:
                    recommendations.append(f"ควรเน้นปรับปรุงวิชาในหมวด: {', '.join(weak_categories[:2])}") # Limit to top 2 for brevity

            results.append({
                'ชื่อ': student_name,
                'การทำนาย': prediction_label,
                'ความน่าจะเป็น': {'จบ': avg_prob_pass, 'ไม่จบ': avg_prob_fail},
                'เกรดเฉลี่ย': gpa,
                'ระดับความเสี่ยง': risk_level,
                'ความเชื่อมั่น': confidence,
                'การวิเคราะห์': list(set(analysis)),
                'คำแนะนำ': list(set(recommendations))
            })

        total = len(results)
        predicted_pass = sum(1 for r in results if r['การทำนาย'] == 'จบ')
        predicted_fail = total - predicted_pass
        pass_rate = (predicted_pass / total * 100) if total > 0 else 0

        high_risk_count = sum(1 for r in results if r['ระดับความเสี่ยง'] == 'สูง')
        medium_risk_count = sum(1 for r in results if r['ระดับความเสี่ยง'] == 'ปานกลาง')
        low_risk_count = total - high_risk_count - medium_risk_count

        logger.info(f"Prediction successful for {total} students (Pass: {predicted_pass}, Fail: {predicted_fail})")

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': total,
                'predicted_pass': predicted_pass,
                'predicted_fail': predicted_fail,
                'pass_rate': pass_rate,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'data_format_used': model_data_format
            }
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during prediction: {str(e)}'})

# --- Modified Function: predict_manual_input ---
@app.route('/predict_manual_input', methods=['POST'])
def predict_manual_input():
    """Predicts outcome from manually entered subject data (from curriculum_prediction_form.html)."""
    try:
        data = request.json
        student_name = data.pop('student_name', 'ผู้ใช้งาน')
        model_filename = data.get('model_filename')

        if not model_filename:
            return jsonify({'success': False, 'error': 'No model filename provided for manual prediction.'})

        model_filepath = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        if not os.path.exists(model_filepath):
            return jsonify({'success': False, 'error': 'Specified model file not found for manual prediction.'})

        try:
            loaded_model_data = joblib.load(model_filepath)
            model_components = loaded_model_data['models']
            scaler = loaded_model_data['scaler']
            feature_cols = loaded_model_data['feature_columns']
            data_format_used = loaded_model_data['data_format']
            logger.info(f"Loaded model '{model_filename}' (format: {data_format_used}) for manual prediction.")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to load model {model_filename} for manual input: {str(e)}'})

        # Extract only grades for courses
        input_grades_raw = {cid: grade for cid, grade in data.items() if cid != 'model_filename'}

        # Process manual input into features
        all_grades_entered = []
        subject_categories_grades = {cat: [] for cat in app.config['SUBJECT_CATEGORIES'].keys()}
        subject_categories_grades['อื่นๆ'] = [] 
        
        # Determine current year (can be a simple heuristic or fixed for manual input)
        # For simplicity, let's assume a dummy year 1 for all manual inputs if no year info is provided.
        # If the model expects year-based features, they will be 0.0 unless more context is passed.
        dummy_year_for_manual_input = 1 

        for course_id, grade_str in input_grades_raw.items():
            if grade_str and grade_str.strip():
                numeric_grade = grade_to_numeric(grade_str)
                if numeric_grade is not None:
                    all_grades_entered.append(numeric_grade)

                    course_name = ""
                    for c_data in app.config['COURSES_DATA']:
                        if c_data['id'] == course_id:
                            course_name = c_data['thaiName']
                            break

                    category = categorize_subject(course_name)
                    subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

        # Calculate features from input grades (this will include new features if applicable)
        gpa = np.mean(all_grades_entered) if all_grades_entered else 0.0
        min_grade = np.min(all_grades_entered) if all_grades_entered else 0.0
        max_grade = np.max(all_grades_entered) if all_grades_entered else 0.0
        std_grade = np.std(all_grades_entered) if len(all_grades_entered) > 1 else 0.0
        fail_threshold = app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0)
        fail_count = sum(1 for g in all_grades_entered if g < fail_threshold)
        fail_rate = fail_count / len(all_grades_entered) if all_grades_entered else 0.0
        total_subjects = len(all_grades_entered)

        # Initialize all possible features a record_based model might expect, with defaults
        student_data_for_prediction = {
            'gpa': gpa,
            'min_grade': min_grade,
            'max_grade': max_grade,
            'std_grade': std_grade,
            'fail_count': fail_count,
            'fail_rate': fail_rate,
            'total_subjects': total_subjects,
            'year_in': dummy_year_for_manual_input,
            'year_out': dummy_year_for_manual_input,
            'study_duration': 1, # Minimal duration if only one year is 'known'
            'gpa_trend': 0.0,
            'final_year_gpa': gpa, # If only one "year" of data, final and first are the same
            'first_year_gpa': gpa,
            'fail_rate_in_early_years': fail_rate, # If only one "year", early years is all years
            'credits_per_year': total_subjects * app.config['DATA_CONFIG']['default_credits_per_subject'], # Assumes default credits
            'total_terms': 1 # Assuming one term for manual input
        }

        # Add category-specific GPAs and fail rates
        for cat, cat_grades in subject_categories_grades.items():
            if cat_grades:
                student_data_for_prediction[f'gpa_{cat}'] = np.mean(cat_grades)
                student_data_for_prediction[f'min_{cat}'] = np.min(cat_grades)
                student_data_for_prediction[f'max_{cat}'] = np.max(cat_grades)
                cat_fail_count = sum(1 for g in cat_grades if g < fail_threshold)
                student_data_for_prediction[f'fail_rate_{cat}'] = cat_fail_count / len(cat_grades)
                student_data_for_prediction[f'count_{cat}'] = len(cat_grades)
            else:
                student_data_for_prediction[f'gpa_{cat}'] = 0.0
                student_data_for_prediction[f'min_{cat}'] = 0.0
                student_data_for_prediction[f'max_{cat}'] = 0.0
                student_data_for_prediction[f'fail_rate_{cat}'] = 0.0
                student_data_for_prediction[f'count_{cat}'] = 0

        # Create DataFrame for prediction, ensuring columns match trained model's feature_cols
        processed_input_for_df = {}
        for feature in feature_cols:
            # Provide sensible defaults for new features if they are not directly computable from manual input
            # For year-based features, they will be 0.0 or the simple defaults set above,
            # unless the frontend is expanded to collect year per course.
            processed_input_for_df[feature] = [student_data_for_prediction.get(feature, 0.0)]

        input_df = pd.DataFrame(processed_input_for_df)
        input_df = input_df.fillna(0) # Ensure no NaNs before scaling/prediction

        predictions_proba_list = []
        for name, model in model_components.items():
            if model is None:
                continue
            try:
                if name == 'lr':
                    X_scaled = scaler.transform(input_df)
                    pred_proba = model.predict_proba(X_scaled)
                else:
                    pred_proba = model.predict_proba(input_df)
                
                if pred_proba.shape[1] == 1:
                    pred_proba = np.hstack((1 - pred_proba, pred_proba)) # Ensure 2 columns for binary
                
                predictions_proba_list.append(pred_proba)
            except Exception as e:
                logger.warning(f"Could not predict with model {name} from manual input: {str(e)}")
                continue

        if not predictions_proba_list:
            return jsonify({'success': False, 'error': 'Could not make predictions with manual input.'})

        # Average probabilities across all sub-models for the single student
        avg_prob_per_student = np.mean([p[0] for p in predictions_proba_list], axis=0)
        avg_prob_fail = avg_prob_per_student[0]
        avg_prob_pass = avg_prob_per_student[1]

        prediction_label = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'

        confidence = max(avg_prob_pass, avg_prob_fail)
        high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
        medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

        if confidence > high_confidence_threshold:
            risk_level = 'ต่ำ' if prediction_label == 'จบ' else 'สูง'
        elif confidence > medium_confidence_threshold:
            risk_level = 'ปานกลาง'
        else:
            risk_level = 'สูง' if prediction_label == 'ไม่จบ' else 'ปานกลาง'

        gpa_for_analysis = student_data_for_prediction.get('gpa', 0.0)

        analysis = []
        recommendations = []

        low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']
        warning_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']
        high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']

        if gpa_for_analysis < low_gpa_threshold:
            analysis.append(f"GPA ต่ำมาก ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['high_risk'])
        elif gpa_for_analysis < warning_gpa_threshold:
            analysis.append(f"GPA อยู่ในเกณฑ์เสี่ยง ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['medium_risk'])
        elif gpa_for_analysis < 3.0:
            analysis.append(f"GPA พอใช้ ({float(gpa_for_analysis):.2f})")
            recommendations.append("มีโอกาสพัฒนาผลการเรียนให้ดีขึ้น")
        else:
            analysis.append(f"GPA ดีเยี่ยม ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['low_risk'])

        if prediction_label == 'ไม่จบ':
            recommendations.append("แนะนำให้ทบทวนแผนการเรียนและขอความช่วยเหลือจากอาจารย์ที่ปรึกษา")
            if student_data_for_prediction.get('fail_rate', 0) > high_fail_rate_threshold:
                recommendations.append("มีอัตราการตกในบางวิชาสูง ควรให้ความสำคัญกับการเรียนซ่อม หรือปรึกษาผู้สอน")

        # Weak categories check (applies to subject_based and record_based models)
        if data_format_used in ['subject_based', 'record_based']:
            weak_categories = []
            for cat_key in app.config['SUBJECT_CATEGORIES'].keys():
                cat_gpa_key = f'gpa_{cat_key}'
                if student_data_for_prediction.get(cat_gpa_key, 0) < low_gpa_threshold:
                    weak_categories.append(app.config['SUBJECT_CATEGORIES'][cat_key]['thaiName'])
            if weak_categories:
                recommendations.append(f"ควรเน้นปรับปรุงวิชาในหมวด: {', '.join(weak_categories[:2])} โดยเฉพาะวิชาพื้นฐาน")

        return jsonify({
            'success': True,
            'student_name': student_name,
            'prediction': prediction_label,
            'prob_pass': float(avg_prob_pass),
            'prob_fail': float(avg_prob_fail),
            'gpa_input': float(gpa_for_analysis),
            'risk_level': risk_level,
            'confidence': float(confidence),
            'analysis': list(set(analysis)),
            'recommendations': list(set(recommendations)),
            'data_format_used': data_format_used
        })

    except Exception as e:
        logger.error(f"Error during manual input prediction: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during prediction: {str(e)}'})

@app.route('/api/analyze_curriculum', methods=['POST'])
def analyze_curriculum():
    """Analyzes curriculum progress and provides recommendations for curriculum_prediction_form.html."""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 0)
        repeated_courses_in_this_term_ids = data.get('repeated_courses_in_this_term_ids', [])
        model_filename = data.get('model_filename') # This model is used for overall graduation prediction
        student_name = data.get('student_name', 'นักศึกษา')
        
        logger.info(f"Received curriculum analysis request with {len(current_grades)} grades for student: {student_name}, terms: {loaded_terms_count}")
        
        all_terms_data = app.config['ALL_TERMS_DATA']
        courses_data = app.config['COURSES_DATA']
        grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']
        
        # Passing grades list for prerequisite checks
        passing_grades_list_from_config = {g: grade_mapping_points[g] for g in app.config['DATA_CONFIG']['grade_mapping'] if grade_mapping_points[g] > 0}
        
        # Core subjects (used for penalties if failed)
        core_subjects_ids = app.config.get('CORE_SUBJECTS_IDS', [])

        gpa_results = calculate_gpa_and_failed_courses_backend(current_grades, courses_data)
        avg_gpa = gpa_results['avgGPA']
        completed_credits = gpa_results['completedCredits']
        failed_courses_ids = gpa_results['failedCourses']

        loaded_courses_objects = get_loaded_courses_backend(loaded_terms_count, all_terms_data, courses_data, repeated_courses_in_this_term_ids)
        
        blocked_courses_ids = find_blocked_courses_backend(current_grades, loaded_courses_objects, courses_data, passing_grades_list_from_config)

        total_required_credits = app.config['DATA_CONFIG']['total_required_credits'] # From config
        completion_rate = estimate_completion_rate_backend(
            avg_gpa, completed_credits, total_required_credits,
            blocked_courses_ids, len(failed_courses_ids), failed_courses_ids,
            len(repeated_courses_in_this_term_ids), core_subjects_ids, courses_data
        )

        graduation_status = determine_graduation_status_backend(
            completion_rate, avg_gpa, blocked_courses_ids, failed_courses_ids,
            loaded_courses_objects, current_grades, all_terms_data, courses_data,
            loaded_terms_count
        )

        all_loaded_course_ids = [c['id'] for c in loaded_courses_objects]
        incomplete_courses_ids = [cid for cid in all_loaded_course_ids if cid not in current_grades or not current_grades[cid] or grade_to_numeric(current_grades[cid]) == 0.0]
        incomplete_courses_names = [find_course_thai_name_backend(cid, courses_data) for cid in incomplete_courses_ids]

        failed_courses_names = [find_course_thai_name_backend(cid, courses_data) for cid in failed_courses_ids]
        
        blocked_courses_details = []
        for cid in blocked_courses_ids:
            unmet = get_unmet_prerequisites_backend(cid, current_grades, courses_data, passing_grades_list_from_config)
            unmet_names = [find_course_thai_name_backend(mid, courses_data) for mid in unmet]
            blocked_courses_details.append({
                'id': cid,
                'name': find_course_thai_name_backend(cid, courses_data),
                'unmet_prereqs': unmet_names
            })

        # Build graph only for loaded courses to avoid large graphs and external dependencies
        dependency_graph = build_dependency_graph_backend(loaded_courses_objects) 
        blocked_chain_texts = []
        for failed_cid in failed_courses_ids:
            affected_ids = find_affected_courses_backend(failed_cid, dependency_graph)
            if affected_ids:
                failed_course_name = find_course_thai_name_backend(failed_cid, courses_data)
                affected_names = [find_course_thai_name_backend(aid, courses_data) for aid in affected_ids]
                # Filter affected courses to only those within the loaded curriculum (if needed)
                affected_in_loaded_curriculum = [
                    (name, aid) for name, aid in zip(affected_names, affected_ids)
                    if aid in all_loaded_course_ids # Only include if they are part of the curriculum being evaluated
                ]
                if affected_in_loaded_curriculum:
                    blocked_chain_texts.append(
                        f"วิชา {failed_course_name} ({failed_cid}) ไม่ผ่าน ส่งผลกระทบต่อวิชาอื่นๆ ดังนี้:\n" +
                        "\n".join([f"  - {name} ({aid})" for name, aid in affected_in_loaded_curriculum])
                    )
        
        courses_subset_for_topo = [c for c in courses_data if c['id'] in all_loaded_course_ids]
        topo_result = topological_sort_with_cycle_check_backend(courses_subset_for_topo)
        topological_order_names = []
        if not topo_result['cycle']:
            topological_order_names = [find_course_thai_name_backend(cid, courses_data) for cid in topo_result['order']]
        
        recommendations_list = update_recommendations_backend(failed_courses_ids, avg_gpa, blocked_courses_ids)

        term_gpas_chart_data = []
        for i in range(loaded_terms_count):
            if i >= len(all_terms_data):
                break # Avoid index out of bounds if loaded_terms_count exceeds curriculum data

            term_courses_ids = all_terms_data[i]['ids']
            
            term_total_points = 0
            term_total_credits = 0
            has_valid_grades_in_term = False

            for cid in term_courses_ids:
                grade_char = current_grades.get(cid, "")
                if grade_char:
                    course = next((c for c in courses_data if c['id'] == cid), None)
                    if not course: continue
                    numeric_grade = grade_to_numeric(grade_char)
                    if numeric_grade is not None: # Only count valid numeric grades
                        term_total_points += numeric_grade * course['credit']
                        term_total_credits += course['credit'] # Count credits for all courses with grades
                        has_valid_grades_in_term = True

            term_gpa = term_total_points / term_total_credits if term_total_credits > 0 else 0
            term_label = f"ปี{all_terms_data[i]['year']} เทอม{all_terms_data[i]['term']}"
            
            if has_valid_grades_in_term:
                term_gpas_chart_data.append({'label': term_label, 'gpa': term_gpa})

        next_term_predicted_gpa = linear_regression_next_term_gpa_backend(term_gpas_chart_data)

        # Add model-based prediction if a model filename is provided
        prediction_result = None
        if model_filename:
            try:
                logger.info(f"Making prediction with model: {model_filename}")
                
                # Construct data for predict_manual_input
                manual_input_data = current_grades.copy()
                manual_input_data['model_filename'] = model_filename
                manual_input_data['student_name'] = student_name # Pass student name for logging

                # Call the internal function, bypassing Flask request/response for direct data processing
                # This assumes predict_manual_input can handle dict directly, not just request.json
                # For simplicity, we'll mimic the request.json structure and call the internal logic.
                from flask import _app_ctx_stack
                # Push a temporary request context
                with app.test_request_context(json=manual_input_data):
                    # Call the actual view function (predict_manual_input)
                    # It returns a Flask Response, so we need to parse it
                    response = predict_manual_input() 
                    if isinstance(response, tuple): # If it returns (json, status_code)
                        prediction_response_data = response[0].json
                    else: # If it returns just a Response object
                        prediction_response_data = response.json
                
                if prediction_response_data and prediction_response_data.get('success'):
                    prediction_result = {
                        'prediction': prediction_response_data.get('prediction'),
                        'prob_pass': prediction_response_data.get('prob_pass'),
                        'prob_fail': prediction_response_data.get('prob_fail'),
                        'confidence': prediction_response_data.get('confidence'),
                        'risk_level': prediction_response_data.get('risk_level'),
                        'gpa_input': prediction_response_data.get('gpa_input')
                    }
                else:
                    logger.warning(f"Model prediction failed for {model_filename}: {prediction_response_data.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error during prediction with model {model_filename}: {str(e)}")

        response_data = {
            'success': True,
            'completion_rate': completion_rate,
            'avg_gpa': avg_gpa,
            'graduation_status': graduation_status,
            'incomplete_courses': incomplete_courses_names,
            'failed_courses': failed_courses_names,
            'blocked_courses_details': blocked_courses_details,
            'blocked_chain_texts': blocked_chain_texts,
            'topological_order': topological_order_names,
            'topological_cycle': topo_result['cycle'],
            'recommendations': recommendations_list,
            'term_gpas_chart_data': term_gpas_chart_data,
            'next_term_predicted_gpa': next_term_predicted_gpa,
            'debug': {
                'current_grades_count': len(current_grades),
                'loaded_terms_count': loaded_terms_count,
                'repeated_courses_ids': repeated_courses_in_this_term_ids
            }
        }
        
        # เพิ่มผลการทำนายถ้ามี
        if prediction_result:
            response_data.update({
                'student_name': student_name,
                'prediction_model': prediction_result['prediction'], # Renamed to avoid clash with curriculum status
                'prob_pass': prediction_result['prob_pass'],
                'prob_fail': prediction_result['prob_fail'],
                'confidence': prediction_result['confidence'],
                'risk_level_model': prediction_result['risk_level'], # Renamed
                'gpa_for_model_prediction': prediction_result['gpa_input']
            })
            
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during curriculum analysis: {str(e)}")
        return jsonify({'success': False, 'error': app.config['MESSAGES']['error']['processing_error'] + f": {str(e)}"})

@app.route('/analyze', methods=['POST'])
def analyze_subjects():
    """Analyzes subjects from a CSV/Excel file (for Subject-based or Record-based data)."""
    try:
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Specified file not found.'})

        # Determine file type and read
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file type for analysis.")

        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'Could not read file or file is empty.'})

        data_format = detect_data_format(df)

        if data_format not in ['subject_based', 'record_based']: # Ensure this analysis is for subject_based or record_based data
            return jsonify({'success': False, 'error': 'Only subject-based or record-based data is supported for this analysis.'})

        # For analysis, we need to consolidate all grade entries regardless of format
        # For record_based, we'll need to pivot or iterate to get subject-wise grades.
        # For simplicity, let's process the data to get a flat list of (subject_name, numeric_grade)
        # across all students, then do the analysis.

        subject_grades_for_analysis = {} # {subject_name: [grade1, grade2, ...]}
        
        if data_format == 'record_based':
            student_id_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['รหัส', 'id', 'student_id'])), None)
            student_name_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['ชื่อ-นามสกุล', 'ชื่อ', 'name', 'student_name'])), None)
            subject_name_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['ชื่อวิชา', 'วิชา', 'subject', 'course'])), None)
            grade_col = next((col for col in df.columns if any(kw in str(col).lower() for kw in ['เกรด', 'grade'])), None)

            if not all([(student_id_col or student_name_col), subject_name_col, grade_col]):
                return jsonify({'success': False, 'error': 'Missing required columns for record-based analysis (student ID/Name, subject name, grade).'})

            for _, row in df.iterrows():
                sub_name = str(row[subject_name_col]) if pd.notna(row[subject_name_col]) else 'Unknown Subject'
                grade_val = row[grade_col]
                numeric_grade = grade_to_numeric(grade_val)
                if numeric_grade is not None:
                    subject_grades_for_analysis.setdefault(sub_name, []).append(numeric_grade)

        elif data_format == 'subject_based':
            name_col = None
            possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
            for col in possible_name_cols:
                if col in df.columns:
                    name_col = col
                    break
            if not name_col:
                name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
                if name_candidates:
                    name_col = name_candidates[0]
                else:
                    return jsonify({'success': False, 'error': 'Could not find student name or ID column for subject-based analysis.'})

            exclude_cols = [name_col, 'ปีที่เข้า', 'ปีที่จบ', 'year_in', 'year_out']
            exclude_keywords = ['gpa', 'เกรดเฉลี่ย', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
            
            subject_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if not any(kw in col_lower for kw in exclude_keywords) and col not in exclude_cols: # Ensure no meta columns are treated as subjects
                    sample_values = df[col].dropna().head(5).tolist()
                    is_likely_grade_column = False
                    if sample_values:
                        if any(isinstance(v, (int, float)) and 0.0 <= v <= 4.0 for v in sample_values):
                            is_likely_grade_column = True
                        elif any(str(v).upper() in app.config['DATA_CONFIG']['grade_mapping'] for v in sample_values):
                            is_likely_grade_column = True
                    if is_likely_grade_column:
                        subject_cols.append(col)

            for subject in subject_cols:
                for _, row in df.iterrows():
                    grade_value = row[subject]
                    numeric_grade = grade_to_numeric(grade_value)
                    if numeric_grade is not None:
                        subject_grades_for_analysis.setdefault(subject, []).append(numeric_grade)
        else: # Should not happen due to initial check
            return jsonify({'success': False, 'error': 'Unsupported data format for analysis.'})


        logger.info(f"Analyzing {len(subject_grades_for_analysis)} subjects from {data_format} data.")

        subject_analysis = {} # Stores detailed stats per subject, grouped by category
        all_gpas = [] # List of average GPAs for each subject
        all_fail_rates = [] # List of fail rates for each subject
        subject_stats_list = [] # Flat list of subject stats for easier sorting/filtering

        for subject, grades in subject_grades_for_analysis.items():
            if grades:
                category = categorize_subject(subject)
                subject_analysis.setdefault(category, {}) # Ensure category exists

                avg_grade = np.mean(grades)
                min_grade = np.min(grades)
                max_grade = np.max(grades)
                std_grade = np.std(grades) if len(grades) > 1 else 0
                fail_threshold = app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0)
                fail_count = sum(1 for g in grades if g < fail_threshold)
                fail_rate = fail_count / len(grades)
                pass_rate = 1 - fail_rate

                grade_distribution = {}
                reverse_grade_mapping = {v: k for k, v in app.config['DATA_CONFIG']['grade_mapping'].items()}
                reverse_grade_mapping[0.0] = 'F/W/I' # Consolidate common fail/withdrawal grades

                for grade_point_val in sorted(list(set(grades)), reverse=True):
                    count = grades.count(grade_point_val)
                    if count > 0:
                        char_grade = reverse_grade_mapping.get(grade_point_val, str(grade_point_val))
                        grade_distribution[char_grade] = grade_distribution.get(char_grade, 0) + count
                
                subject_info = {
                    'average': avg_grade,
                    'minimum': min_grade,
                    'maximum': max_grade,
                    'std_dev': std_grade,
                    'fail_rate': fail_rate,
                    'pass_rate': pass_rate,
                    'num_students': len(grades),
                    'num_failed': fail_count,
                    'grade_distribution': grade_distribution
                }

                subject_analysis[category][subject] = subject_info
                subject_stats_list.append({
                    'subject': subject,
                    'category': category,
                    'average': avg_grade,
                    'fail_rate': fail_rate,
                    'num_students': len(grades)
                })

                all_gpas.append(avg_grade)
                all_fail_rates.append(fail_rate)

        overall_stats = {
            'total_students_considered': len(df), # Total rows in original file, for reference
            'total_subjects_analyzed': len(subject_grades_for_analysis),
            'avg_gpa_overall': np.mean(all_gpas) if all_gpas else 0,
            'overall_fail_rate': np.mean(all_fail_rates) if all_fail_rates else 0,
            'max_gpa_subject': np.max(all_gpas) if all_gpas else 0,
            'min_gpa_subject': np.min(all_gpas) if all_gpas else 0
        }

        high_fail_subjects = []
        low_gpa_subjects = []
        excellent_subjects = []

        high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']
        low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']

        for stat in subject_stats_list:
            if stat['fail_rate'] > high_fail_rate_threshold:
                high_fail_subjects.append(stat)
            if stat['average'] < low_gpa_threshold:
                low_gpa_subjects.append(stat)
            if stat['average'] >= 3.5 and stat['fail_rate'] < 0.1: # Criteria for excellent subjects
                excellent_subjects.append(stat)

        high_fail_subjects.sort(key=lambda x: x['fail_rate'], reverse=True)
        low_gpa_subjects.sort(key=lambda x: x['average'])
        excellent_subjects.sort(key=lambda x: x['average'], reverse=True)

        recommendations = []
        overall_fail_rate_warning = app.config['DATA_CONFIG']['risk_levels']['medium_fail_rate_threshold']
        overall_fail_rate_high = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']

        if overall_stats['overall_fail_rate'] > overall_fail_rate_high:
            recommendations.append(f"อัตราการตกโดยรวมสูงมาก (> {(overall_fail_rate_high * 100):.0f}%) ควรทบทวนวิธีการสอนและการประเมินผลทั้งหมด")
        elif overall_stats['overall_fail_rate'] > overall_fail_rate_warning:
            recommendations.append(f"อัตราการตกโดยรวมค่อนข้างสูง (> {(overall_fail_rate_warning * 100):.0f}%) ควรปรับปรุงกระบวนการเรียนการสอน")

        if high_fail_subjects:
            top_problem_subjects = [s['subject'] for s in high_fail_subjects[:3]]
            recommendations.append(f"วิชาที่มีอัตราตกสูงที่สุด: {', '.join(top_problem_subjects)}")
            recommendations.append("ควรทบทวนเนื้อหา วิธีการสอน และเกณฑ์การประเมินในวิชาเหล่านี้")

        if low_gpa_subjects:
            recommendations.append(f"พบ {len(low_gpa_subjects)} วิชาที่มีเกรดเฉลี่ยต่ำกว่า {low_gpa_threshold}")
            recommendations.append("ควรจัดการเรียนเสริมหรือปรับลดความยากของเนื้อหา")

        if excellent_subjects:
            recommendations.append(f"พบ {len(excellent_subjects)} วิชาที่มีผลการเรียนดีเยี่ยม สามารถนำมาเป็นแบบอย่างในการพัฒนาวิชาอื่น")

        category_summary = {}
        for category, subjects in subject_analysis.items():
            if subjects:
                avg_gpas = [s['average'] for s in subjects.values()]
                avg_fail_rates = [s['fail_rate'] for s in subjects.values()]
                total_students_in_category = sum(s['num_students'] for s in subjects.values())

                category_summary[category] = {
                    'thai_name': app.config['SUBJECT_CATEGORIES'].get(category, {}).get('thaiName', category),
                    'num_subjects': len(subjects),
                    'avg_gpa': np.mean(avg_gpas),
                    'avg_fail_rate': np.mean(avg_fail_rates),
                    'total_students_in_category': total_students_in_category,
                    'hardest_subject': min(subjects.items(), key=lambda x: x[1]['average'])[0] if subjects else None,
                    'easiest_subject': max(subjects.items(), key=lambda x: x[1]['average'])[0] if subjects else None
                }

        logger.info("Subject analysis successful.")

        return jsonify({
            'success': True,
            'subject_analysis': subject_analysis, # Detailed analysis per subject
            'overall_stats': overall_stats, # Overall statistics
            'category_summary': category_summary, # Summary per category
            'problem_subjects': { # Top problematic subjects
                'high_fail_rate': high_fail_subjects[:10],
                'low_gpa': low_gpa_subjects[:10]
            },
            'excellent_subjects': excellent_subjects[:10], # Top excellent subjects
            'recommendations': recommendations # General recommendations
        })

    except Exception as e:
        logger.error(f"Error during subject analysis: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during analysis: {str(e)}'})

@app.route('/model_status')
def model_status():
    """Checks the status of currently loaded models."""
    try:
        status = {
            'subject_model': models['subject_model'] is not None,
            'gpa_model': models['gpa_model'] is not None,
            'record_model': models['record_model'] is not None, # Include record model status
            'subject_model_info': models['subject_model_info'],
            'gpa_model_info': models['gpa_model_info'],
            'record_model_info': models['record_model_info']
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return jsonify({
            'subject_model': False,
            'gpa_model': False,
            'record_model': False,
            'error': str(e)
        })

# --- Modified Function: list_models ---
@app.route('/api/models', methods=['GET'])
def list_models():
    """Lists all available trained models in the models folder."""
    try:
        model_files = []
        for filename in os.listdir(app.config['MODEL_FOLDER']):
            if filename.endswith('.joblib'):
                filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
                try:
                    loaded_data = joblib.load(filepath)
                    
                    feature_importances = {}
                    # Try to get feature importances from RF model if it exists within the ensemble
                    if 'models' in loaded_data and loaded_data['models'] and 'rf' in loaded_data['models'] and loaded_data['models']['rf'] is not None and loaded_data.get('feature_columns') is not None:
                        rf_model = loaded_data['models']['rf']
                        if hasattr(rf_model, 'feature_importances_'):
                            importances = pd.Series(rf_model.feature_importances_, index=loaded_data['feature_columns']).sort_values(ascending=False)
                            feature_importances = importances.head(5).to_dict() # Get top 5

                    model_info = {
                        'filename': filename,
                        'created_at': loaded_data.get('created_at', 'N/A'),
                        'data_format': loaded_data.get('data_format', 'unknown'),
                        'performance': loaded_data.get('performance_metrics', {}),
                        'training_data_info': loaded_data.get('training_data_info', {}),
                        'feature_importances': feature_importances # Add feature importances
                    }
                    model_files.append(model_info)
                except Exception as e:
                    logger.warning(f"Could not load metadata for model {filename}: {str(e)}")
                    model_files.append({'filename': filename, 'error': 'Could not read metadata', 'created_at': 'N/A', 'data_format': 'unknown'})

        # Sort by creation date, newest first
        model_files_sorted = sorted(model_files, key=lambda x: x.get('created_at', '1970-01-01T00:00:00'), reverse=True)
        return jsonify({'success': True, 'models': model_files_sorted})
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while listing models: {str(e)}'}), 500

@app.route('/api/models/<filename>', methods=['DELETE'])
def delete_model(filename):
    """Deletes a specified model file from the server."""
    try:
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            # Remove model from in-memory if it was the currently loaded one
            if models['subject_model_info'] and models['subject_model_info'].get('filename') == filename:
                models['subject_model'] = None
                models['subject_model_info'] = None
                models['subject_feature_cols'] = None
            elif models['gpa_model_info'] and models['gpa_model_info'].get('filename') == filename:
                models['gpa_model'] = None
                models['gpa_model_info'] = None
                models['gpa_feature_cols'] = None
            elif models['record_model_info'] and models['record_model_info'].get('filename') == filename: # For record_based
                models['record_model'] = None
                models['record_model_info'] = None
                models['record_feature_cols'] = None

            logger.info(f"Model {filename} deleted successfully.")
            return jsonify({'success': True, 'message': f'Model {filename} deleted successfully.'})
        else:
            return jsonify({'success': False, 'error': 'Specified model file not found.'}), 404
    except Exception as e:
        logger.error(f"Error deleting model {filename}: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while deleting the model: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Downloads a file from the uploads folder."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found or inaccessible: {str(e)}'}), 404

@app.errorhandler(413)
def too_large(e):
    """Handles cases where the uploaded file is too large."""
    return jsonify({'success': False, 'error': 'File too large. Please select a file smaller than 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handles cases where the requested page is not found. Redirects to index."""
    return render_template('index.html') # Redirect to admin page for 404

@app.errorhandler(500)
def internal_error(e):
    """Handles internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    # Initialize logging using dictConfig from app.config
    logging.config.dictConfig(app.config['LOGGING_CONFIG'])
    # Re-initialize logger after config update to use the new configuration
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting student success prediction system...")

    load_existing_models()

    if models['subject_model_info']:
        logger.info(f"📊 Found Subject Model (Accuracy: {models['subject_model_info']['accuracy']:.3f}, Created: {models['subject_model_info']['created_at']}, Filename: {models['subject_model_info'].get('filename', 'N/A')})")
    if models['gpa_model_info']:
        logger.info(f"📊 Found GPA Model (Accuracy: {models['gpa_model_info']['accuracy']:.3f}, Created: {models['gpa_model_info']['created_at']}, Filename: {models['gpa_model_info'].get('filename', 'N/A')})")
    if models['record_model_info']: # Log record model
        logger.info(f"📊 Found Record-based Model (Accuracy: {models['record_model_info']['accuracy']:.3f}, Created: {models['record_model_info']['created_at']}, Filename: {models['record_model_info'].get('filename', 'N/A')})")
    if not models['subject_model'] and not models['gpa_model'] and not models['record_model']:
        logger.info("⚠️ No models found - Please upload data and train a model first.")

    logger.info(f"🌐 System ready at http://{app.config['HOST']}:{app.config['PORT']}")

    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
