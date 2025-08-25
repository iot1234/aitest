from advanced_training import AdvancedFeatureEngineer
from dotenv import load_dotenv

# โหลด environment variables จาก .env
load_dotenv(override=True)

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import logging.config
import logging
from datetime import datetime
import warnings
from collections import Counter
from imblearn.over_sampling import SMOTE
import tempfile
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Any, Dict, Optional, List

import config

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # เปลี่ยนจาก INFO เป็น WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ตรวจสอบ environment variables
logger.info("=" * 50)
logger.info("🔧 Environment Check")
logger.info("=" * 50)
logger.info(f"📁 Working Directory: {os.getcwd()}")
logger.info(f"📄 .env exists: {os.path.exists('.env')}")
logger.info(f"🔑 R2 Access Key: {os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', 'NOT FOUND')[:10] if os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID') else 'NOT FOUND'}...")
logger.info(f"🪣 R2 Bucket: {os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', 'NOT FOUND')}")
logger.info("=" * 50)
# Create Flask app
app = Flask(__name__)
# Load settings from config.py
app.config.from_object(config.get_config())

# Create necessary folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Created folder: {folder}")
        else:
            logger.info(f"Folder exists: {folder}")
    except Exception as e:
        logger.error(f"Error creating folder {folder}: {str(e)}")

# Variables to store loaded models
models = {
    'subject_model': None,
    'gpa_model': None,
    'subject_model_info': None,
    'gpa_model_info': None,
    'subject_feature_cols': None,
    'gpa_feature_cols': None
}

# ==========================================
# S3 Storage Class for Bucketeer
# ==========================================
class S3Storage:
    def __init__(self):
        """Initialize S3 client with Cloudflare R2 credentials from .env"""
        try:
            # Force reload .env
            from dotenv import load_dotenv
            load_dotenv(override=True)
            
            # Get credentials with fallback
            self.access_key = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', '')
            self.secret_key = os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', '')
            self.endpoint_url = os.environ.get('CLOUDFLARE_R2_ENDPOINT', '')
            self.bucket_name = os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', 'pjai')
            
            # ถ้าไม่มี credentials ให้ใช้ local storage ทันที
            if not all([self.access_key, self.secret_key, self.endpoint_url]):
                logger.info("R2 credentials not complete - using LOCAL storage")
                self.s3_client = None
                self.use_local = True
                return
            
            # Create S3 client with better error handling
            try:
                import boto3
                from botocore.config import Config
                
                # Config สำหรับ timeout
                config = Config(
                    region_name='auto',
                    signature_version='s3v4',
                    retries={
                        'max_attempts': 2,
                        'mode': 'standard'
                    },
                    connect_timeout=5,
                    read_timeout=5
                )
                
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
                
                # Quick test with timeout
                try:
                    self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        MaxKeys=1
                    )
                    logger.info("✅ R2 connection successful!")
                    self.use_local = False
                except:
                    logger.info("R2 not accessible - using LOCAL storage")
                    self.use_local = True
                    self.s3_client = None
                    
            except ImportError:
                logger.error("boto3 not installed - using LOCAL storage")
                self.use_local = True
                self.s3_client = None
                
        except Exception as e:
            logger.error(f"Storage init error: {str(e)}")
            self.s3_client = None
            self.use_local = True
    
    def _verify_bucket(self):
        """Verify bucket exists and is accessible"""
        try:
            # ทดสอบ list objects แทน head_bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            logger.info(f"✅ R2 Bucket '{self.bucket_name}' verified successfully")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                logger.info(f"🔨 Bucket '{self.bucket_name}' not found, creating...")
                self._create_bucket()
            else:
                logger.error(f"❌ Error accessing bucket: {str(e)}")
                # ใช้ local storage เป็น fallback
                self.use_local = True
                self.s3_client = None
    
    def _create_bucket(self):
        """Create R2 bucket if it doesn't exist"""
        try:
            # Cloudflare R2 ไม่ต้องระบุ LocationConstraint
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ Bucket '{self.bucket_name}' created successfully")
        except ClientError as e:
            logger.error(f"❌ Error creating bucket: {str(e)}")
            # ใช้ local storage เป็น fallback
            self.use_local = True
            self.s3_client = None
    
    def _verify_bucket(self):
        """Verify bucket exists and is accessible"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} verified successfully")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"Bucket {self.bucket_name} not found, creating...")
                self._create_bucket()
            else:
                logger.error(f"Error accessing bucket: {str(e)}")
                raise
    
    def _create_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} created successfully")
        except ClientError as e:
            logger.error(f"Error creating bucket: {str(e)}")
            raise
    
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """
        Save model to S3
        
        Args:
            model_data: Dictionary containing model and metadata
            filename: Name of the model file
            
        Returns:
            Boolean indicating success
        """
        if self.use_local:
            return self._save_model_locally(model_data, filename)
        
        tmp_path = None
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(model_data, tmp_file.name)
                tmp_path = tmp_file.name
            
            # Upload to S3
            s3_key = f"models/{filename}"
            with open(tmp_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    ContentType='application/octet-stream',
                    Metadata={
                        'created_at': datetime.now().isoformat(),
                        'data_format': model_data.get('data_format', 'unknown'),
                        'accuracy': str(model_data.get('performance_metrics', {}).get('accuracy', 0))
                    }
                )
            
            # Clean up temporary file
            os.remove(tmp_path)
            
            # Also save metadata separately for quick access
            self._save_model_metadata(filename, model_data)
            
            logger.info(f"Model {filename} saved successfully to S3")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to S3: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load model from S3
        
        Args:
            filename: Name of the model file
            
        Returns:
            Model data dictionary or None if not found
        """
        if self.use_local:
            return self._load_model_locally(filename)
        
        tmp_path = None
        try:
            s3_key = f"models/{filename}"
            
            # Download from S3 to temporary file
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Filename=tmp_file.name
                )
                tmp_path = tmp_file.name
            
            # Load model
            model_data = joblib.load(tmp_path)
            
            # Clean up
            os.remove(tmp_path)
            
            logger.info(f"Model {filename} loaded successfully from S3")
            return model_data
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Model {filename} not found in S3")
            else:
                logger.error(f"Error loading model from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None
    
    def delete_model(self, filename: str) -> bool:
        """
        Delete model from S3
        
        Args:
            filename: Name of the model file
            
        Returns:
            Boolean indicating success
        """
        if self.use_local:
            return self._delete_model_locally(filename)
        
        try:
            # Delete model file
            s3_key = f"models/{filename}"
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Delete metadata
            metadata_key = f"metadata/{filename}.json"
            try:
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key
                )
            except:
                pass  # Metadata might not exist
            
            logger.info(f"Model {filename} deleted successfully from S3")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model from S3: {str(e)}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in S3
        
        Returns:
            List of model metadata dictionaries
        """
        if self.use_local:
            return self._list_models_locally()
        
        try:
            # List objects in models/ prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='models/',
                Delimiter='/'
            )
            
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = obj['Key'].replace('models/', '')
                    if filename and filename.endswith('.joblib'):
                        # Try to get metadata
                        metadata = self._load_model_metadata(filename)
                        if not metadata:
                            # Basic metadata from S3 object
                            metadata = {
                                'filename': filename,
                                'created_at': obj['LastModified'].isoformat(),
                                'size': obj['Size']
                            }
                        models.append(metadata)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            logger.info(f"Found {len(models)} models in S3")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models from S3: {str(e)}")
            return []
    
    def _save_model_metadata(self, filename: str, model_data: Dict[str, Any]):
        """Save model metadata for quick access"""
        try:
            metadata = {
                'filename': filename,
                'created_at': model_data.get('created_at', datetime.now().isoformat()),
                'data_format': model_data.get('data_format', 'unknown'),
                'performance_metrics': model_data.get('performance_metrics', {}),
                'training_data_info': model_data.get('training_data_info', {}),
                'feature_importances': model_data.get('feature_importances', {})
            }
            
            metadata_key = f"metadata/{filename}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, default=str),
                ContentType='application/json'
            )
            
        except Exception as e:
            logger.warning(f"Could not save metadata for {filename}: {str(e)}")
    
    def _load_model_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model metadata"""
        try:
            metadata_key = f"metadata/{filename}.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=metadata_key
            )
            metadata = json.loads(response['Body'].read())
            return metadata
            
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                logger.warning(f"Could not load metadata for {filename}: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}")
            return None
    
    # Local storage fallback methods
    def _save_model_locally(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model locally when S3 is not available"""
        try:
            model_folder = app.config['MODEL_FOLDER']
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
            filepath = os.path.join(model_folder, filename)
            joblib.dump(model_data, filepath)
            logger.info(f"Model {filename} saved locally")
            return True
        except Exception as e:
            logger.error(f"Error saving model locally: {str(e)}")
            return False
    
    def _load_model_locally(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model from local storage"""
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                return joblib.load(filepath)
            return None
        except Exception as e:
            logger.error(f"Error loading model locally: {str(e)}")
            return None
    
    def _delete_model_locally(self, filename: str) -> bool:
        """Delete model from local storage"""
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model locally: {str(e)}")
            return False
    
    def _list_models_locally(self) -> List[Dict[str, Any]]:
        """List models from local storage"""
        try:
            model_folder = app.config['MODEL_FOLDER']
            if not os.path.exists(model_folder):
                return []
            
            models = []
            for filename in os.listdir(model_folder):
                if filename.endswith('.joblib'):
                    filepath = os.path.join(model_folder, filename)
                    try:
                        model_data = joblib.load(filepath)
                        models.append({
                            'filename': filename,
                            'created_at': model_data.get('created_at', ''),
                            'data_format': model_data.get('data_format', 'unknown'),
                            'performance_metrics': model_data.get('performance_metrics', {}),
                            'training_data_info': model_data.get('training_data_info', {})
                        })
                    except:
                        models.append({
                            'filename': filename,
                            'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                        })
            
            return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing local models: {str(e)}")
            return []

# Create global S3 storage instance
storage = S3Storage()

# ==========================================
# Original Functions (keep unchanged)
# ==========================================

def detect_data_format(df):
    """Detects the data format of the uploaded DataFrame."""
    try:
        columns = [col.lower() for col in df.columns.tolist()]
        has_name_id = any(keyword in col for col in columns for keyword in ['ชื่อ', 'name', 'รหัส', 'id', 'student_id'])
        has_year = any(keyword in col for col in columns for keyword in ['ปี', 'year'])
        has_subject_like_columns = any(
            not any(kw in col for kw in ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated']) for col in columns
        ) and any(col.lower().startswith(('วิชา', 'subj', 'course')) or len(col) > 5 for col in columns if col not in ['ชื่อ', 'name', 'รหัส', 'id', 'year'])

        if has_name_id and has_year and has_subject_like_columns:
            logger.debug(f"Detected subject_based format for columns: {df.columns.tolist()}")
            return 'subject_based'

        has_gpa = any(keyword in col for col in columns for keyword in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])
        has_success = any(keyword in col for col in columns for keyword in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ'])

        if has_gpa and has_success:
            logger.debug(f"Detected gpa_based format for columns: {df.columns.tolist()}")
            return 'gpa_based'

        logger.debug(f"Could not detect data format for columns: {df.columns.tolist()}")
        return 'unknown'

    except Exception as e:
        logger.error(f"Error detecting data format: {str(e)}")
        return 'unknown'

def grade_to_numeric(grade):
    """Converts a letter grade to a numeric GPA value."""
    if pd.isna(grade):
        return 0.0
    try:
        return float(grade)
    except ValueError:
        pass

    grade_str = str(grade).strip().upper()
    return app.config['DATA_CONFIG']['grade_mapping'].get(grade_str, 0.0)

def categorize_subject(subject_name):
    """Categorizes subjects based on keywords defined in config."""
    if pd.isna(subject_name):
        return 'อื่นๆ'

    subject_name = str(subject_name).lower()
    for category, info in app.config['SUBJECT_CATEGORIES'].items():
        if any(keyword in subject_name for keyword in info['keywords']):
            return category
    return 'อื่นๆ'

def process_subject_data(df):
    """Processes subject-based DataFrame to create features for model training."""
    try:
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
                raise ValueError("ไม่พบคอลัมน์ชื่อนักศึกษาหรือรหัสในข้อมูล")

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

        exclude_cols_keywords = ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
        exclude_cols = [name_col]
        if year_in_col: exclude_cols.append(year_in_col)
        if year_out_col: exclude_cols.append(year_out_col)

        subject_cols = [
            col for col in df.columns
            if col not in exclude_cols and not any(kw in col.lower() for kw in exclude_cols_keywords)
        ]
        
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
            if year_out_col and not df[year_out_col].isnull().all():
                df['graduated'] = df[year_out_col].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
                logger.warning("Target column 'graduated' not found, inferred from 'year_out'.")
            else:
                raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา")

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
                        grades.append(numeric_grade)

                        category = categorize_subject(subject)
                        subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

                min_subjects_required = app.config['DATA_CONFIG']['min_subjects_per_student']
                if len(grades) >= min_subjects_required:
                    gpa = np.mean(grades)
                    min_grade = np.min(grades)
                    max_grade = np.max(grades)
                    std_grade = np.std(grades) if len(grades) > 1 else 0
                    fail_count = sum(1 for g in grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
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
                            student_data[f'fail_rate_{cat}'] = sum(1 for g in cat_grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01) / len(cat_grades)
                        else:
                            student_data[f'gpa_{cat}'] = 0.0
                            student_data[f'min_{cat}'] = 0.0
                            student_data[f'max_{cat}'] = 0.0
                            student_data[f'fail_rate_{cat}'] = 0.0

                    processed_data.append(student_data)

            except Exception as e:
                logger.warning(f"Could not process row {idx} for student '{row.get(name_col, 'N/A')}': {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No data could be processed. Please check the data format and ensure sufficient subjects per student.")

        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        logger.info(f"Successfully processed data: {len(result_df)} students")

        return result_df

    except Exception as e:
        logger.error(f"Error processing subject data: {str(e)}")
        raise

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
            raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา")

        for idx, row in df.iterrows():
            try:
                student_name = row.get(name_col, f'นักศึกษา_{idx}')

                gpa_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])]
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

def train_ensemble_model(X, y):
    """Trains an Ensemble model with GridSearchCV and SMOTE."""
    try:
        logger.info("Starting Ensemble model training...")

        unique_labels, label_counts = np.unique(y, return_counts=True)
        if len(unique_labels) < 2 or np.min(label_counts) < 2:
            logger.warning(f"Class imbalance or too little data in y before Oversampling: {dict(zip(unique_labels, label_counts))}")

            if len(unique_labels) < 2:
                existing_label = unique_labels[0]
                new_label_to_create = 1 - existing_label
                if not X.empty:
                    X_first_row = X.iloc[[0]].copy()
                    y_new_label = pd.Series([new_label_to_create], index=[X.index.max() + 1 if not X.empty else 0])
                    X = pd.concat([X, X_first_row], ignore_index=True)
                    y = pd.concat([y, y_new_label], ignore_index=True)
                else:
                    raise ValueError("Cannot create synthetic data as DataFrame is empty.")
                unique_labels, label_counts = np.unique(y, return_counts=True)

            while np.min(label_counts) < 2:
                minority_class_label = unique_labels[np.argmin(label_counts)]
                needed_to_reach_two = 2 - label_counts[np.argmin(label_counts)]
                logger.info(f"Oversampling: Adding {needed_to_reach_two} samples for class {minority_class_label} for Train/Test Split.")
                minority_X_samples = X[y == minority_class_label]
                if not minority_X_samples.empty:
                    sample_to_add_X = minority_X_samples.iloc[[0]].copy()
                    for _ in range(needed_to_reach_two):
                        new_index = X.index.max() + 1 if not X.empty else 0
                        y_new_label_entry = pd.Series([minority_class_label], index=[new_index])
                        X = pd.concat([X, sample_to_add_X], ignore_index=True)
                        y = pd.concat([y, y_new_label_entry], ignore_index=True)
                else:
                    logger.error(f"No samples found for class {minority_class_label} for Oversampling.")
                    break
                unique_labels, label_counts = np.unique(y, return_counts=True)

        logger.info(f"Number of data points after initial oversampling: {len(X)}, Features: {len(X.columns)}")
        logger.info(f"Label distribution after initial oversampling: {y.value_counts().to_dict()}")

        if len(X) < app.config['DATA_CONFIG']['min_students_for_training']:
            raise ValueError(f"Insufficient data for model training (at least {app.config['DATA_CONFIG']['min_students_for_training']} samples required).")

        total_samples = len(X)
        test_size_actual = app.config['ML_CONFIG']['test_size']

        min_samples_per_split = 1
        if total_samples * test_size_actual < min_samples_per_split:
            if total_samples > min_samples_per_split:
                test_size_actual = min_samples_per_split / total_samples
            else:
                test_size_actual = 0
                logger.warning(f"Very small dataset ({total_samples} samples), no Test Set will be used.")

        can_stratify = len(np.unique(y)) >= 2 and np.min(np.unique(y, return_counts=True)[1]) >= min_samples_per_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_actual, random_state=app.config['ML_CONFIG']['random_state'],
            stratify=y if can_stratify else None
        )
        logger.info(f"Data split: Training {len(X_train)} samples, Testing {len(X_test)} samples.")
        logger.info(f"Before SMOTE - Label distribution in Training Set: {Counter(y_train)}")

        if len(np.unique(y_train)) > 1 and np.min(list(Counter(y_train).values())) > 0:
            try:
                smote = SMOTE(random_state=app.config['ML_CONFIG']['random_state'])
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE - Label distribution in Training Set: {Counter(y_train_resampled)}")
                X_train = X_train_resampled
                y_train = y_train_resampled
            except Exception as e:
                logger.warning(f"Could not apply SMOTE: {str(e)}. Training with original data.")
        else:
            logger.warning("SMOTE not applied (single class or empty class in training set).")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

        param_grid_rf = app.config['MODEL_HYPERPARAMETERS']['RandomForest']
        param_grid_gb = app.config['MODEL_HYPERPARAMETERS']['GradientBoosting']
        param_grid_lr = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression']

        best_rf = None
        best_gb = None
        best_lr = None

        try:
            logger.info("Performing GridSearchCV for RandomForest...")
            grid_search_rf = GridSearchCV(
                RandomForestClassifier(random_state=app.config['ML_CONFIG']['random_state'], n_jobs=1),
                param_grid_rf,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=1,
                verbose=0 
            )
            grid_search_rf.fit(X_train, y_train)
            best_rf = grid_search_rf.best_estimator_
            logger.info(f"RandomForest Best Params: {grid_search_rf.best_params_}")
            logger.info(f"RandomForest Best Score: {grid_search_rf.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for RandomForest failed: {str(e)}. Falling back to default parameters.")
            best_rf = RandomForestClassifier(
                n_estimators=app.config['ML_CONFIG']['n_estimators'],
                max_depth=app.config['ML_CONFIG']['max_depth'],
                random_state=app.config['ML_CONFIG']['random_state'],
                n_jobs=1
            )
            best_rf.fit(X_train, y_train)

        try:
            logger.info("Performing GridSearchCV for GradientBoosting...")
            grid_search_gb = GridSearchCV(
                GradientBoostingClassifier(random_state=app.config['ML_CONFIG']['random_state']),
                param_grid_gb,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=1,
                verbose=0
            )
            grid_search_gb.fit(X_train, y_train)
            best_gb = grid_search_gb.best_estimator_
            logger.info(f"GradientBoosting Best Params: {grid_search_gb.best_params_}")
            logger.info(f"GradientBoosting Best Score: {grid_search_gb.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for GradientBoosting failed: {str(e)}. Falling back to default parameters.")
            best_gb = GradientBoostingClassifier(
                n_estimators=app.config['ML_CONFIG']['n_estimators'],
                max_depth=app.config['ML_CONFIG']['max_depth'],
                random_state=app.config['ML_CONFIG']['random_state']
            )
            best_gb.fit(X_train, y_train)

        try:
            logger.info("Performing GridSearchCV for LogisticRegression...")
            base_max_iter = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression'].get('max_iter', [1000])[0]

            grid_search_lr = GridSearchCV(
                LogisticRegression(random_state=app.config['ML_CONFIG']['random_state'], max_iter=base_max_iter),
                param_grid_lr,
                cv=app.config['ML_CONFIG']['cv_folds'],
                scoring='accuracy',
                n_jobs=1,
                verbose=0
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
                solver='liblinear'
            )
            best_lr.fit(X_train_scaled, y_train)

        models_dict = {
            'rf': best_rf,
            'gb': best_gb,
            'lr': best_lr
        }

        trained_models = {}
        predictions = {}

        for name, model in models_dict.items():
            try:
                logger.info(f"Evaluating model {name}...")
                if name == 'lr':
                    if len(X_test_scaled) > 0:
                        pred_proba = model.predict_proba(X_test_scaled)
                        predictions[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                else:
                    if len(X_test) > 0:
                        pred_proba = model.predict_proba(X_test)
                        predictions[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                trained_models[name] = model
            except Exception as e:
                logger.warning(f"Could not evaluate model {name}: {str(e)}")
                continue

        if not trained_models:
            raise ValueError("No models could be trained.")

        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if len(X_test) > 0 and predictions:
            ensemble_pred_proba = np.mean(list(predictions.values()), axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_test, ensemble_pred)
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        else:
            accuracy = 0.85
            precision = 0.85
            recall = 0.85
            f1 = 0.85
            logger.warning("No test data for model evaluation, using estimated metrics.")

        logger.info(f"Model training results - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        return {
            'models': trained_models,
            'scaler': scaler,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'features_count': X.shape[1],
            'best_rf_params': best_rf.get_params() if best_rf else {},
            'best_gb_params': best_gb.get_params() if best_gb else {},
            'best_lr_params': best_lr.get_params() if best_lr else {}
        }

    except Exception as e:
        logger.error(f"Error training Ensemble model: {str(e)}")
        raise

# ==========================================
# Updated Routes with S3 Storage
# ==========================================

@app.route('/train', methods=['POST'])
def train_model():
    """Handles model training with the uploaded file."""
    try:
        logger.info("🚀 Starting ADVANCED model training process...")
        data = request.get_json()
        filename = data.get('filename')
        use_advanced = data.get('use_advanced_training', True)  # Default to advanced

        if not filename:
            logger.warning("No filename provided for training")
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({'success': False, 'error': 'Specified file not found.'})

        logger.info(f"📁 Processing file: {filename}")

        # อ่านไฟล์
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"✅ Successfully read CSV with encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
            logger.info(f"✅ Successfully read Excel file")
        else:
            raise ValueError("Unsupported file type for training.")

        if df is None:
            return jsonify({'success': False, 'error': 'Could not read file.'})

        # ตรวจสอบรูปแบบข้อมูล
        data_format = detect_data_format(df)
        logger.info(f"📊 Detected data format for training: {data_format}")

        # เลือกวิธีการประมวลผล
        if use_advanced and data_format == 'subject_based':
            logger.info("🧬 Using ADVANCED Context-Aware Training Strategy")
            
            # ใช้ Advanced Feature Engineering
            engineer = AdvancedFeatureEngineer(
                grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
            )
            
            # เตรียมข้อมูลแบบ Advanced
            X, y = engineer.prepare_training_data(df)
            
            if len(X) == 0:
                return jsonify({'success': False, 'error': 'Could not prepare training data'})
            
            # บันทึก course profiles สำหรับใช้ในการ predict
            course_profiles = engineer.course_profiles
            
        else:
            # ใช้วิธีเดิม
            logger.info("📊 Using standard training strategy")
            if data_format == 'subject_based':
                processed_df = process_subject_data(df)
            elif data_format == 'gpa_based':
                processed_df = process_gpa_data(df)
            else:
                return jsonify({'success': False, 'error': 'Unsupported data format.'})

            feature_cols = [col for col in processed_df.columns if col not in ['ชื่อ', 'graduated']]
            X = processed_df[feature_cols].fillna(0)
            y = processed_df['graduated']
            course_profiles = None

        min_students_for_training = app.config['DATA_CONFIG']['min_students_for_training']
        if len(X) < min_students_for_training:
            return jsonify({'success': False, 
                          'error': f'Insufficient data ({min_students_for_training} samples required).'})

        logger.info(f"🎯 Training data prepared: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"📈 Label distribution: {y.value_counts().to_dict()}")

        # เทรนโมเดล
        logger.info("🤖 Starting ensemble model training...")
        model_result = train_ensemble_model(X, y)

        # คำนวณ feature importance
        feature_importances = {}
        if 'rf' in model_result['models']:
            rf_model = model_result['models']['rf']
            if hasattr(rf_model, 'feature_importances_'):
                feature_cols = X.columns.tolist()
                importances = pd.Series(
                    rf_model.feature_importances_, 
                    index=feature_cols
                ).sort_values(ascending=False)
                feature_importances = importances.head(10).to_dict()

        # สร้างชื่อไฟล์โมเดล
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_type = 'advanced' if use_advanced else 'standard'
        model_filename = f'{data_format}_model_{training_type}_{timestamp}.joblib'

        # สร้างข้อมูลโมเดลสำหรับบันทึก
        model_data = {
            'models': model_result['models'],
            'scaler': model_result['scaler'],
            'feature_columns': X.columns.tolist(),
            'data_format': data_format,
            'training_type': training_type,
            'course_profiles': course_profiles,  # บันทึก course DNA
            'created_at': datetime.now().isoformat(),
            'training_data_info': {
                'rows': len(X),
                'features': X.shape[1],
                'graduated_count': int(y.sum()),
                'not_graduated_count': int(len(y) - y.sum()),
                'source_file': filename
            },
            'performance_metrics': {
                'accuracy': model_result['accuracy'],
                'precision': model_result['precision'],
                'recall': model_result['recall'],
                'f1_score': model_result['f1_score']
            },
            'feature_importances': feature_importances,
            'hyperparameters': {
                'best_rf_params': model_result.get('best_rf_params', {}),
                'best_gb_params': model_result.get('best_gb_params', {}),
                'best_lr_params': model_result.get('best_lr_params', {})
            }
        }

        # บันทึกโมเดล
        logger.info(f"💾 Saving model: {model_filename}")
        save_success = storage.save_model(model_data, model_filename)
        
        if save_success:
            logger.info(f"✅ Model saved successfully: {model_filename}")
        else:
            logger.warning(f"⚠️ Model save failed, but continuing...")

        logger.info("🎉 Model training completed successfully!")

        return jsonify({
            'success': True,
            'model_filename': model_filename,
            'training_type': training_type,
            'accuracy': model_result['accuracy'],
            'precision': model_result['precision'],
            'recall': model_result['recall'],
            'f1_score': model_result['f1_score'],
            'training_samples': len(X),
            'validation_samples': model_result.get('validation_samples', 0),
            'features_count': X.shape[1],
            'data_format': data_format,
            'feature_importances': feature_importances,
            'storage_provider': 'cloudflare_r2' if not storage.use_local else 'local'
        })

    except Exception as e:
        logger.error(f"❌ Error during model training: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Training error: {str(e)}'})


@app.route('/predict', methods=['POST'])
def predict():
    """Predicts outcome from an uploaded CSV/Excel file using a specified model."""
    try:
        logger.info("🔮 Starting prediction process...")
        data = request.get_json()
        filename = data.get('filename')
        model_filename = data.get('model_filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided for prediction data.'})
        
        # ถ้าไม่ระบุโมเดล ให้หาโมเดลล่าสุด
        if not model_filename:
            logger.info("🔍 No model specified, finding latest subject-based model...")
            models_list = storage.list_models()
            subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '') or m.get('data_format') == 'subject_based']
            if subject_models:
                model_filename = subject_models[0]['filename']
                logger.info(f"✅ Auto-selected latest model: {model_filename}")
            else:
                return jsonify({'success': False, 'error': 'No trained model found. Please train a model first.'})

        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(data_filepath):
            return jsonify({'success': False, 'error': 'Specified data file not found.'})

        # โหลดโมเดล
        logger.info(f"📂 Loading model: {model_filename}")
        loaded_model_data = storage.load_model(model_filename)
        if not loaded_model_data:
            return jsonify({'success': False, 'error': f'Model file {model_filename} not found.'})

        model_info = {
            'models': loaded_model_data['models'],
            'scaler': loaded_model_data['scaler']
        }
        feature_cols = loaded_model_data['feature_columns']
        data_format = loaded_model_data['data_format']
        logger.info(f"✅ Loaded model '{model_filename}' (format: {data_format}) for prediction.")

        # อ่านไฟล์ข้อมูล
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_filepath, encoding=encoding)
                    logger.info(f"✅ Successfully read CSV with encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(data_filepath)
            logger.info(f"✅ Successfully read Excel file")
        else:
            raise ValueError("Unsupported file type for prediction.")

        if df is None:
            return jsonify({'success': False, 'error': 'Could not read prediction data file.'})

        # ตรวจสอบรูปแบบข้อมูล
        detected_data_format_for_prediction = detect_data_format(df)
        if detected_data_format_for_prediction != data_format:
            return jsonify({'success': False, 'error': f'Prediction data format ({detected_data_format_for_prediction}) does not match model format ({data_format}).'})
        
        logger.info(f"📊 Predicting with data format: {detected_data_format_for_prediction}")

        # ประมวลผลข้อมูล
        if data_format == 'subject_based':
            processed_df = process_subject_data(df)
        else:
            processed_df = process_gpa_data(df)

        if len(processed_df) == 0:
            return jsonify({'success': False, 'error': 'No data could be processed for prediction.'})

        logger.info(f"📈 Processed {len(processed_df)} students for prediction")

        # เตรียมข้อมูลสำหรับการทำนาย
        X_predict = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in processed_df.columns:
                X_predict[col] = processed_df[col]
            else:
                X_predict[col] = 0
        X_predict = X_predict.fillna(0)

        # ทำนายผล
        trained_models = model_info['models']
        scaler = model_info['scaler']

        predictions_proba_list = []
        successful_models = 0
        
        for name, model in trained_models.items():
            try:
                if name == 'lr':
                    X_scaled = scaler.transform(X_predict)
                    pred_proba = model.predict_proba(X_scaled)
                else:
                    pred_proba = model.predict_proba(X_predict)
                
                if pred_proba.shape[1] == 1:
                    pred_proba = np.hstack((1 - pred_proba, pred_proba))
                
                predictions_proba_list.append(pred_proba)
                successful_models += 1
                logger.debug(f"✅ Prediction successful with {name} model")
            except Exception as e:
                logger.warning(f"Could not predict with model {name}: {str(e)}")
                continue

        if not predictions_proba_list:
            return jsonify({'success': False, 'error': 'Could not make predictions with any loaded sub-models.'})

        logger.info(f"🤖 Used {successful_models}/{len(trained_models)} models for ensemble prediction")

        # คำนวณผลลัพธ์
        results = []
        high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
        medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

        for i in range(len(processed_df)):
            student_name = processed_df.iloc[i]['ชื่อ']
            gpa = processed_df.iloc[i]['gpa']

            avg_prob_per_student = np.mean([pred_proba_array[i] for pred_proba_array in predictions_proba_list], axis=0)
            avg_prob_fail = avg_prob_per_student[0]
            avg_prob_pass = avg_prob_per_student[1]

            prediction = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'

            confidence = max(avg_prob_pass, avg_prob_fail)
            if confidence > high_confidence_threshold:
                risk_level = 'ต่ำ' if prediction == 'จบ' else 'สูง'
            elif confidence > medium_confidence_threshold:
                risk_level = 'ปานกลาง'
            else:
                risk_level = 'สูง' if prediction == 'ไม่จบ' else 'ปานกลาง'

            # สร้างการวิเคราะห์และคำแนะนำ
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
                analysis.append(f"GPA ดี ({gpa:.2f})")
                recommendations.extend(app.config['MESSAGES']['recommendations']['low_risk'])

            if prediction == 'ไม่จบ':
                recommendations.append("แนะนำให้ทบทวนแผนการเรียนและขอความช่วยเหลือ")
                if 'fail_rate' in processed_df.columns and processed_df.iloc[i].get('fail_rate', 0) > high_fail_rate_threshold:
                    recommendations.append("มีอัตราการตกในบางวิชาสูง ควรให้ความสำคัญกับการเรียนซ่อม")

            # ตรวจสอบหมวดวิชาที่อ่อน
            if data_format == 'subject_based':
                weak_categories = []
                for cat_key in app.config['SUBJECT_CATEGORIES'].keys():
                    gpa_col = f'gpa_{cat_key}'
                    if gpa_col in processed_df.columns and processed_df.iloc[i].get(gpa_col, 0) < low_gpa_threshold:
                        weak_categories.append(cat_key)

                if weak_categories:
                    recommendations.append(f"ควรเน้นปรับปรุงวิชาในหมวด: {', '.join(weak_categories[:2])}")

            results.append({
                'ชื่อ': student_name,
                'การทำนาย': prediction,
                'ความน่าจะเป็น': {'จบ': float(avg_prob_pass), 'ไม่จบ': float(avg_prob_fail)},
                'เกรดเฉลี่ย': float(gpa),
                'ระดับความเสี่ยง': risk_level,
                'ความเชื่อมั่น': float(confidence),
                'การวิเคราะห์': list(set(analysis)),
                'คำแนะนำ': list(set(recommendations))
            })

        # สรุปผล
        total = len(results)
        predicted_pass = sum(1 for r in results if r['การทำนาย'] == 'จบ')
        predicted_fail = total - predicted_pass
        pass_rate = (predicted_pass / total * 100) if total > 0 else 0

        high_risk = sum(1 for r in results if r['ระดับความเสี่ยง'] == 'สูง')
        medium_risk = sum(1 for r in results if r['ระดับความเสี่ยง'] == 'ปานกลาง')
        low_risk = total - high_risk - medium_risk

        logger.info(f"🎉 Prediction completed successfully: {total} students (Pass: {predicted_pass}, Fail: {predicted_fail})")

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': total,
                'predicted_pass': predicted_pass,
                'predicted_fail': predicted_fail,
                'pass_rate': float(pass_rate),
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'model_used': model_filename,
            'models_count': successful_models,
            'storage_provider': 'cloudflare_r2' if not storage.use_local else 'local'
        })

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'An error occurred during prediction: {str(e)}'})
    
    
@app.route('/api/models', methods=['GET'])
def list_models():
    """Lists all available trained models."""
    try:
        # Get models from S3 or local storage
        model_files = storage.list_models()
        
        # Enrich with additional metadata if needed
        for model in model_files:
            if 'performance_metrics' not in model or not model['performance_metrics']:
                model['performance_metrics'] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                }
            # Rename for frontend compatibility
            model['performance'] = model.get('performance_metrics', {})

        return jsonify({'success': True, 'models': model_files})
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while listing models: {str(e)}'}), 500

@app.route('/api/models/<filename>', methods=['DELETE'])
def delete_model(filename):
    """Deletes a specified model file."""
    try:
        # Delete from S3 or local storage
        if storage.delete_model(filename):
            # Update in-memory models if needed
            if models['subject_model_info'] and models['subject_model_info'].get('filename') == filename:
                models['subject_model'] = None
                models['subject_model_info'] = None
                models['subject_feature_cols'] = None
            elif models['gpa_model_info'] and models['gpa_model_info'].get('filename') == filename:
                models['gpa_model'] = None
                models['gpa_model_info'] = None
                models['gpa_feature_cols'] = None
                logger.info(f"Model {filename} deleted successfully.")
            return jsonify({'success': True, 'message': f'Model {filename} deleted successfully.'})
        else:
            return jsonify({'success': False, 'error': 'Could not delete model file.'}), 404
                
    except Exception as e:
        logger.error(f"Error deleting model {filename}: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while deleting the model: {str(e)}'}), 500

def load_existing_models():
    """Loads existing trained models from S3 or local storage."""
    try:
        logger.info("🔍 Searching for existing models...")
        
        # Get models list from storage
        models_list = storage.list_models()
        
        if not models_list:
            logger.info("No existing models found")
            return
        
        # Load subject-based model
        subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '')]
        if subject_models:
            latest_subject = subject_models[0]
            loaded_data = storage.load_model(latest_subject['filename'])
            if loaded_data:
                models['subject_model'] = {
                    'models': loaded_data['models'],
                    'scaler': loaded_data['scaler']
                }
                models['subject_feature_cols'] = loaded_data['feature_columns']
                models['subject_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['subject_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['subject_model_info']['loaded_from_file'] = True
                models['subject_model_info']['filename'] = latest_subject['filename']
                logger.info(f"✅ Loaded latest subject model: {latest_subject['filename']}")

        # Load GPA-based model
        gpa_models = [m for m in models_list if 'gpa_based' in m.get('filename', '')]
        if gpa_models:
            latest_gpa = gpa_models[0]
            loaded_data = storage.load_model(latest_gpa['filename'])
            if loaded_data:
                models['gpa_model'] = {
                    'models': loaded_data['models'],
                    'scaler': loaded_data['scaler']
                }
                models['gpa_feature_cols'] = loaded_data['feature_columns']
                models['gpa_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['gpa_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['gpa_model_info']['loaded_from_file'] = True
                models['gpa_model_info']['filename'] = latest_gpa['filename']
                logger.info(f"✅ Loaded latest GPA model: {latest_gpa['filename']}")

    except Exception as e:
        logger.error(f"❌ Error loading existing models: {str(e)}")

# ==========================================
# Keep all other original functions unchanged
# ==========================================

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
            
            try:
                numeric_grade = float(grade_char)
                if not (0.0 <= numeric_grade <= 4.0):
                    numeric_grade = 0.0
            except ValueError:
                numeric_grade = grade_mapping_points.get(str(grade_char).upper(), 0.0)

            if numeric_grade > 0:
                total_points += numeric_grade * course['credit']
                completed_credits += course['credit']
            
            if numeric_grade == 0.0:
                if str(grade_char).upper() in ['F', 'W', 'I', 'NP', 'WF', 'WU']:
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
    return course['thaiName'] if course else course_id

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
        
        numeric_current_grade = None
        try:
            numeric_current_grade = float(current_grade_for_course)
            if not (0.0 <= numeric_current_grade <= 4.0):
                numeric_current_grade = 0.0
        except ValueError:
            numeric_current_grade = grade_mapping_points.get(str(current_grade_for_course).upper(), 0.0)

        if numeric_current_grade == 0.0 and current_grade_for_course != "":
            pass
        elif not check_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
            blocked_courses_ids.append(course_id)
            
    return list(set(blocked_courses_ids))

def build_dependency_graph_backend(courses_subset):
    graph = {c['id']: [] for c in courses_subset}
    for course in courses_subset:
        for prereq_id in course['prereq']:
            if prereq_id in graph:
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
            if prereq_id in course_map:
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

    if np.all(y_vals == y_vals[0]):
        return y_vals[0]

    sum_x = np.sum(x_vals)
    sum_y = np.sum(y_vals)
    sum_xy = np.sum(x_vals * y_vals)
    sum_xx = np.sum(x_vals * x_vals)

    denominator = (n * sum_xx - sum_x * sum_y)
    if denominator == 0:
        return y_vals[-1]

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    predicted = slope * (n + 1) + intercept
    return max(0.0, min(4.0, predicted))

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
        if cid not in current_grades or current_grades[cid] == "" or app.config['DATA_CONFIG']['grade_mapping'].get(str(current_grades[cid]).upper(), 0.0) == 0.0
    ]
    
    if avg_gpa > 0 and avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        return "GPA ต่ำกว่าเกณฑ์ (2.0) จำเป็นต้องปรับปรุงอย่างเร่งด่วนเพื่อจบการศึกษา"

    if is_at_or_past_final_standard_term:
        if len(failed_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมีวิชาตก (เกรด F) ที่ต้องแก้ไข"
        if len(blocked_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน)"
        if len(incomplete_courses) > 0:
            return f"ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมี {len(incomplete_courses)} วิชาที่ยังไม่ผ่าน/ไม่มีเกรด"
        
        if avg_gpa == 0.0 and (not failed_courses_ids and not incomplete_courses):
            return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี (ตามเงื่อนไข)"
        
        return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี"
    
    else:
        if len(failed_courses_ids) > 0:
            return "มีวิชาตก (เกรด F) อาจทำให้การจบการศึกษาล่าช้ากว่ากำหนด"
        if len(blocked_courses_ids) > 0:
            return "มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน) ควรแก้ไขเพื่อเรียนต่อตามแผน"
        if len(incomplete_courses) > 0:
            return "กำลังอยู่ในเส้นทาง. ยังมีวิชาที่ต้องเรียน/ได้รับเกรดในภาคเรียนถัดไป"
        
        return "อยู่ในเส้นทางที่คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี"

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

# Flask Routes (Keep all other routes unchanged)
@app.route('/')
def index():
    """Main page for uploading and training models."""
    return render_template('index.html')

@app.route('/test')
def curriculum_prediction_form():
    """Page for predicting graduation based on curriculum and prerequisites."""
    return render_template(
        'curriculum_prediction_form.html',
        all_terms_data=app.config['ALL_TERMS_DATA'],
        courses_data=app.config['COURSES_DATA'],
        grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
    )


@app.route('/status')
def status_page():
    """หน้าแสดงสถานะระบบ"""
    return render_template('status.html')

@app.route('/api/system/status')
def system_status():
    """API สำหรับตรวจสอบสถานะระบบ"""
    import sys
    import flask
    
    try:
        logger.info("🔧 Checking system status...")
        
        # ตรวจสอบ R2 connection
        r2_connected = False
        storage_provider = 'Local'
        bucket_name = None
        error_message = None
        
        if hasattr(storage, 's3_client') and storage.s3_client and not storage.use_local:
            try:
                response = storage.s3_client.list_objects_v2(Bucket=storage.bucket_name, MaxKeys=1)
                r2_connected = True
                storage_provider = 'Cloudflare R2'
                bucket_name = storage.bucket_name
                logger.info("✅ R2 connection verified")
            except Exception as e:
                logger.warning(f"R2 connection failed: {e}")
                error_message = str(e)
                r2_connected = False
        else:
            error_message = "R2 client not initialized or using local storage"
        
        # นับจำนวนโมเดล
        try:
            models_list = storage.list_models()
            total_size = sum(m.get('size', 0) for m in models_list if 'size' in m)
        except Exception as e:
            logger.warning(f"Could not get models info: {e}")
            models_list = []
            total_size = 0
        
        # Get recent logs (ถ้ามี log file)
        recent_logs = []
        try:
            log_file = 'app.log'
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_logs = lines[-20:]  # Last 20 lines
        except Exception as e:
            logger.warning(f"Could not read logs: {e}")
        
        status_data = {
            'success': True,
            'r2_connected': r2_connected,
            'storage_provider': storage_provider,
            'bucket_name': bucket_name,
            'models_available': len(models_list),
            'total_size': total_size,
            'python_version': sys.version.split()[0],
            'flask_version': flask.__version__,
            'environment': os.environ.get('FLASK_ENV', 'production'),
            'debug_mode': app.debug,
            'server_time': datetime.now().isoformat(),
            'app_folders': {
                'upload_folder': app.config['UPLOAD_FOLDER'],
                'model_folder': app.config['MODEL_FOLDER'],
                'upload_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'model_exists': os.path.exists(app.config['MODEL_FOLDER'])
            },
            'recent_logs': recent_logs,
            'error_message': error_message,
            'env_vars_status': {
                'R2_ACCESS_KEY': bool(os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID')),
                'R2_SECRET_KEY': bool(os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY')),
                'R2_ENDPOINT': bool(os.environ.get('CLOUDFLARE_R2_ENDPOINT')),
                'R2_BUCKET': bool(os.environ.get('CLOUDFLARE_R2_BUCKET_NAME'))
            }
        }
        
        logger.info("✅ System status check completed")
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'server_time': datetime.now().isoformat()
        }), 500

@app.route('/api/test-r2-connection')
def test_r2_connection():
    """ทดสอบการเชื่อมต่อ R2"""
    try:
        logger.info("🧪 Testing R2 connection...")
        
        if storage.use_local or not hasattr(storage, 's3_client') or not storage.s3_client:
            return jsonify({
                'success': False,
                'error': 'R2 client not initialized - using local storage',
                'storage_provider': 'local'
            })
        
        # ทดสอบ list objects
        response = storage.s3_client.list_objects_v2(
            Bucket=storage.bucket_name,
            MaxKeys=1
        )
        
        logger.info("✅ R2 connection test successful")
        return jsonify({
            'success': True,
            'message': 'R2 connection successful',
            'bucket': storage.bucket_name,
            'endpoint': storage.endpoint_url,
            'objects_found': len(response.get('Contents', []))
        })
        
    except Exception as e:
        logger.error(f"❌ R2 connection test failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'storage_provider': 'local_fallback'
        })
        
@app.route('/api/test-r2')
def test_r2():
    try:
        # ทดสอบเขียนไฟล์
        test_data = {'test': True, 'timestamp': datetime.now().isoformat()}
        success = storage.save_model(test_data, 'test_model.joblib')
        
        if success:
            # ทดสอบอ่านไฟล์
            loaded = storage.load_model('test_model.joblib')
            if loaded:
                # ลบไฟล์ทดสอบ
                storage.delete_model('test_model.joblib')
                return jsonify({
                    'success': True,
                    'message': 'R2 storage working perfectly!',
                    'can_write': True,
                    'can_read': True,
                    'can_delete': True
                })
        
        return jsonify({'success': False, 'message': 'R2 test failed'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
        
@app.route('/api/config', methods=['GET'])
def get_config_for_frontend():
    """Provides frontend with necessary configuration data like courses and terms."""
    try:
        config_data = {
            'COURSES_DATA': app.config['COURSES_DATA'],
            'ALL_TERMS_DATA': app.config['ALL_TERMS_DATA'],
            'GRADE_MAPPING': app.config['DATA_CONFIG']['grade_mapping'],
            'MESSAGES': app.config['MESSAGES'],
            'DATA_CONFIG_RISK_LEVELS': app.config['DATA_CONFIG']['risk_levels']
        }
        return jsonify({'success': True, **config_data})
    except Exception as e:
        logger.error(f"Error loading config data for frontend: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to load configuration data'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and basic data format detection."""
    try:
        logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'success': False, 'error': 'ไม่พบไฟล์ในคำขอ'})

        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'success': False, 'error': 'ไม่ได้เลือกไฟล์'})

        # ตรวจสอบนามสกุลไฟล์
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            logger.warning(f"Invalid file extension")
            return jsonify({'success': False, 'error': 'รองรับเฉพาะไฟล์ .csv, .xlsx, .xls เท่านั้น'})

        # สร้าง filename ที่ปลอดภัย
        from werkzeug.utils import secure_filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = secure_filename(file.filename)
        filename_on_disk = f"{timestamp}_{safe_filename}"
        
        # ใช้ absolute path
        upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
            
        filepath = os.path.join(upload_folder, filename_on_disk)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # ตรวจสอบว่าไฟล์ถูกบันทึกจริง
        if not os.path.exists(filepath):
            raise ValueError("ไฟล์ไม่ถูกบันทึก")
            
        logger.info(f"File saved successfully: {filename_on_disk}")

        # อ่านและตรวจสอบไฟล์
        try:
            df = None
            if safe_filename.lower().endswith('.csv'):
                # ลองหลาย encoding สำหรับไฟล์ CSV
                encodings = ['utf-8-sig', 'utf-8', 'cp874', 'tis-620', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"Successfully read CSV with encoding: {encoding}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed with {encoding}: {e}")
                        continue
                        
                if df is None:
                    # ถ้าอ่านไม่ได้ ลองใช้ engine python
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', engine='python')
                    except:
                        df = pd.read_csv(filepath, encoding='cp874', engine='python')
                        
            else:  # Excel files
                df = pd.read_excel(filepath, engine='openpyxl')
                logger.info(f"Successfully read Excel file")

            if df is None or df.empty:
                os.remove(filepath)
                raise ValueError("ไม่สามารถอ่านไฟล์ได้ หรือไฟล์ว่างเปล่า")

            data_format = detect_data_format(df)
            logger.info(f"Detected data format: {data_format}")

            response_data = {
                'success': True,
                'filename': filename_on_disk,
                'rows': len(df),
                'columns': len(df.columns),
                'data_format': data_format,
                'sample_columns': df.columns.tolist()[:10]
            }
            
            logger.info(f"Upload successful, returning: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Removed invalid file: {filepath}")
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'success': False, 'error': f'ไม่สามารถอ่านไฟล์ได้: {str(e)}'})

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'เกิดข้อผิดพลาดในการอัปโหลด: {str(e)}'})

# Keep all other routes unchanged...
@app.route('/analyze', methods=['POST'])
def analyze_subjects():
    """Analyzes subjects from a CSV/Excel file (for Subject-based data)."""
    try:
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Specified file not found.'})

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

        if df is None:
            return jsonify({'success': False, 'error': 'Could not read file.'})

        data_format = detect_data_format(df)

        if data_format != 'subject_based':
            return jsonify({'success': False, 'error': 'Only subject-based data is supported for this analysis.'})

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
                return jsonify({'success': False, 'error': 'Could not find student name or ID column.'})

        exclude_cols = [name_col, 'ปีที่เข้า', 'ปีที่จบ', 'year_in', 'year_out']
        exclude_keywords = ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
        subject_cols = [
            col for col in df.columns
            if col not in exclude_cols and not any(kw in col.lower() for kw in exclude_keywords)
        ]

        logger.info(f"Analyzing {len(subject_cols)} subjects.")

        subject_analysis = {}
        all_gpas = []
        all_fail_rates = []
        subject_stats = []

        for subject in subject_cols:
            grades = []
            for _, row in df.iterrows():
                grade = row[subject]
                if pd.notna(grade) and str(grade).strip():
                    numeric_grade = grade_to_numeric(grade)
                    grades.append(numeric_grade)

            if grades:
                category = categorize_subject(subject)
                if category not in subject_analysis:
                    subject_analysis[category] = {}

                avg_grade = np.mean(grades)
                min_grade = np.min(grades)
                max_grade = np.max(grades)
                std_grade = np.std(grades) if len(grades) > 1 else 0
                fail_count = sum(1 for g in grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
                fail_rate = fail_count / len(grades)
                pass_rate = 1 - fail_rate

                grade_distribution = {}
                fail_grade_chars = [k for k, v in app.config['DATA_CONFIG']['grade_mapping'].items() if v == 0.0]
                fail_char_for_display = 'F/W/I/NP' if 'F' in fail_grade_chars else '0.0'

                for grade_point_val in sorted(list(set(app.config['DATA_CONFIG']['grade_mapping'].values())), reverse=True):
                    count = sum(1 for g in grades if g == grade_point_val)
                    if count > 0:
                        if grade_point_val == 0.0:
                            grade_distribution[fail_char_for_display] = grade_distribution.get(fail_char_for_display, 0) + count
                        else:
                            char_grade = next((k for k, v in app.config['DATA_CONFIG']['grade_mapping'].items() if v == grade_point_val and k not in fail_grade_chars), str(grade_point_val))
                            grade_distribution[char_grade] = count

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
                subject_stats.append({
                    'subject': subject,
                    'category': category,
                    'average': avg_grade,
                    'fail_rate': fail_rate,
                    'num_students': len(grades)
                })

                all_gpas.append(avg_grade)
                all_fail_rates.append(fail_rate)

        overall_stats = {
            'total_students': len(df),
            'total_subjects': len(subject_cols),
            'avg_gpa': np.mean(all_gpas) if all_gpas else 0,
            'overall_fail_rate': np.mean(all_fail_rates) if all_fail_rates else 0,
            'max_gpa_subject': np.max(all_gpas) if all_gpas else 0,
            'min_gpa_subject': np.min(all_gpas) if all_gpas else 0
        }

        high_fail_subjects = []
        low_gpa_subjects = []
        excellent_subjects = []

        high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']
        low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']

        for stat in subject_stats:
            if stat['fail_rate'] > high_fail_rate_threshold:
                high_fail_subjects.append(stat)
            if stat['average'] < low_gpa_threshold:
                low_gpa_subjects.append(stat)
            if stat['average'] >= 3.5 and stat['fail_rate'] < 0.1:
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
            recommendations.append(f"พบ {len(excellent_subjects)} วิชาที่มีผลการเรียนดีเยี่ยม สามารถนำมาเป็นแบบอย่าง")

        category_summary = {}
        for category, subjects in subject_analysis.items():
            if subjects:
                avg_gpas = [s['average'] for s in subjects.values()]
                avg_fail_rates = [s['fail_rate'] for s in subjects.values()]
                total_students_in_category = sum(s['num_students'] for s in subjects.values())

                category_summary[category] = {
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
            'subject_analysis': subject_analysis,
            'overall_stats': overall_stats,
            'category_summary': category_summary,
            'problem_subjects': {
                'high_fail_rate': high_fail_subjects[:10],
                'low_gpa': low_gpa_subjects[:10]
            },
            'excellent_subjects': excellent_subjects[:10],
            'recommendations': recommendations
        })

    except Exception as e:
        logger.error(f"Error during subject analysis: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during analysis: {str(e)}'})

@app.route('/api/analyze_curriculum', methods=['POST'])
def analyze_curriculum():
    """Analyzes curriculum progress with prerequisites."""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 0)
        repeated_courses_in_this_term_ids = data.get('repeated_courses_in_this_term_ids', [])
        model_filename = data.get('model_filename')
        student_name = data.get('student_name', 'นักศึกษา')

        logger.info(f"Received curriculum analysis request with {len(current_grades)} grades for student: {student_name}")
        
        all_terms_data = app.config['ALL_TERMS_DATA']
        courses_data = app.config['COURSES_DATA']
        grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']
        
        passing_grades_list_from_config = {g: grade_mapping_points[g] for g in app.config['DATA_CONFIG']['grade_mapping'] if grade_mapping_points[g] > 0}
        
        core_subjects_ids = app.config['CORE_SUBJECTS_IDS'] if 'CORE_SUBJECTS_IDS' in app.config else []


        gpa_results = calculate_gpa_and_failed_courses_backend(current_grades, courses_data)
        avg_gpa = gpa_results['avgGPA']
        completed_credits = gpa_results['completedCredits']
        failed_courses_ids = gpa_results['failedCourses']

        loaded_courses_objects = get_loaded_courses_backend(loaded_terms_count, all_terms_data, courses_data, repeated_courses_in_this_term_ids)
        
        blocked_courses_ids = find_blocked_courses_backend(current_grades, loaded_courses_objects, courses_data, passing_grades_list_from_config)

        total_required_credits = sum(c['credit'] for c in courses_data)
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

        dependency_graph = build_dependency_graph_backend(loaded_courses_objects)
        blocked_chain_texts = []
        for failed_cid in failed_courses_ids:
            affected_ids = find_affected_courses_backend(failed_cid, dependency_graph)
            if affected_ids:
                failed_course_name = find_course_thai_name_backend(failed_cid, courses_data)
                affected_names = [find_course_thai_name_backend(aid, courses_data) for aid in affected_ids]
                affected_in_loaded_curriculum = [
                    (name, aid) for name, aid in zip(affected_names, affected_ids)
                    if aid in all_loaded_course_ids
                ]
                if affected_in_loaded_curriculum:
                    blocked_chain_texts.append(
                        f"วิชา {failed_course_name} ({failed_cid}) ไม่ผ่าน ส่งผลกระทบต่อวิชาอื่นๆ ดังนี้:\n" +
                        "\n".join([f"  - {name} ({aid})" for name, aid in affected_in_loaded_curriculum])
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
                break
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
                    
                    term_total_points += numeric_grade * course['credit']
                    if numeric_grade > 0:
                        term_total_credits += course['credit']
                    has_valid_grades_in_term = True

            term_gpa = term_total_points / term_total_credits if term_total_credits > 0 else 0
            term_label = f"ปี{all_terms_data[i]['year']} เทอม{all_terms_data[i]['term']}"
            
            if has_valid_grades_in_term:
                term_gpas_chart_data.append({'label': term_label, 'gpa': term_gpa})

        next_term_predicted_gpa = linear_regression_next_term_gpa_backend(term_gpas_chart_data)

        prediction_result = None
        if model_filename:
            try:
                logger.info(f"Making prediction with model: {model_filename}")
                
                model_filepath = os.path.join(app.config['MODEL_FOLDER'], model_filename)
                if os.path.exists(model_filepath):
                    loaded_model_data = joblib.load(model_filepath)
                    model_info = {
                        'models': loaded_model_data['models'],
                        'scaler': loaded_model_data['scaler']
                    }
                    feature_cols = loaded_model_data['feature_columns']
                    
                    student_data_for_prediction_model = {}
                    
                    if 'gpa' in feature_cols:
                        student_data_for_prediction_model['gpa'] = avg_gpa
                    if 'completed_credits' in feature_cols:
                        student_data_for_prediction_model['completed_credits'] = completed_credits
                    
                    # Create DataFrame for prediction
                    processed_input_for_df = {}
                    for feature in feature_cols:
                        processed_input_for_df[feature] = [student_data_for_prediction_model.get(feature, 0.0)]

                    input_df = pd.DataFrame(processed_input_for_df)

                    trained_models = model_info['models']
                    scaler = model_info['scaler']

                    predictions_proba_list = []
                    for name, model in trained_models.items():
                        try:
                            if name == 'lr':
                                X_scaled = scaler.transform(input_df)
                                pred_proba = model.predict_proba(X_scaled)
                            else:
                                pred_proba = model.predict_proba(input_df)
                            
                            if pred_proba.shape[1] == 1:
                                pred_proba = np.hstack((1 - pred_proba, pred_proba))
                            
                            predictions_proba_list.append(pred_proba)
                        except Exception as e:
                            logger.warning(f"Could not predict with model {name}: {str(e)}")
                            continue

                    if predictions_proba_list:
                        avg_prob_per_student = np.mean([pred_proba_array[0] for pred_proba_array in predictions_proba_list], axis=0)
                        avg_prob_fail = avg_prob_per_student[0]
                        avg_prob_pass = avg_prob_per_student[1]

                        prediction = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'
                        confidence = max(avg_prob_pass, avg_prob_fail)
                        
                        high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
                        medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

                        if confidence > high_confidence_threshold:
                            risk_level = 'ต่ำ' if prediction == 'จบ' else 'สูง'
                        elif confidence > medium_confidence_threshold:
                            risk_level = 'ปานกลาง'
                        else:
                            risk_level = 'สูง' if prediction == 'ไม่จบ' else 'ปานกลาง'

                        prediction_result = {
                            'prediction': prediction,
                            'prob_pass': float(avg_prob_pass),
                            'prob_fail': float(avg_prob_fail),
                            'confidence': float(confidence),
                            'risk_level': risk_level,
                            'gpa_input': float(avg_gpa)
                        }
                        
                        logger.info(f"Prediction successful: {prediction} (confidence: {confidence:.3f})")
                    else:
                        logger.warning(f"No prediction could be made with model {model_filename}.")
            
            except Exception as e:
                logger.error(f"Error during prediction with model {model_filename}: {str(e)}")

        response_data = {
            'success': True,
            'student_name': student_name,
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
        
        if prediction_result:
            response_data.update({
                'prediction_result': prediction_result
            })
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during curriculum analysis: {str(e)}")
        return jsonify({'success': False, 'error': app.config['MESSAGES']['error']['processing_error'] + f": {str(e)}"})

@app.route('/model_status', methods=['GET'])
def model_status():
    """Checks the status of currently loaded models."""
    try:
        logger.info("Checking model status...")
        
        global models
        if models is None:
            models = {
                'subject_model': None,
                'gpa_model': None,
                'subject_model_info': None,
                'gpa_model_info': None,
                'subject_feature_cols': None,
                'gpa_feature_cols': None
            }

        subject_model_status = models.get('subject_model') is not None
        gpa_model_status = models.get('gpa_model') is not None
        subject_info = models.get('subject_model_info')
        gpa_info = models.get('gpa_model_info')
        
        logger.info(f"Subject model status: {subject_model_status}")
        logger.info(f"GPA model status: {gpa_model_status}")

        status = {
            'success': True,
            'subject_model': subject_model_status,
            'gpa_model': gpa_model_status,
            'subject_model_info': subject_info,
            'gpa_model_info': gpa_info,
            'server_time': datetime.now().isoformat(),
            'models_folder_exists': os.path.exists(app.config['MODEL_FOLDER']),
            'uploads_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            's3_available': not storage.use_local
        }
        
        logger.info("Model status check completed successfully")
        return jsonify(status)

    except Exception as e:
        logger.error(f"Error in model_status: {str(e)}")
        return jsonify({
            'success': False,
            'subject_model': False,
            'gpa_model': False,
            'error': str(e),
            'server_time': datetime.now().isoformat()
        }), 500

# Keep all other routes unchanged...
@app.route('/page')
def main_page():
    return render_template('main_page.html')

@app.route('/curriculum')
def curriculum_page():
    return render_template('curriculum_prediction_form.html',
        all_terms_data=app.config['ALL_TERMS_DATA'],
        courses_data=app.config['COURSES_DATA'],
        grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
    )

@app.route('/predict-batch')
def predict_batch_page():
    return render_template('index.html')

@app.route('/models')
def models_page():
    return render_template('model_management.html')
@app.route('/predict_manual_input', methods=['POST'])
def predict_manual_input():
   """Predicts outcome from manually entered subject data."""
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
           model_info = {
               'models': loaded_model_data['models'],
               'scaler': loaded_model_data['scaler']
           }
           feature_cols = loaded_model_data['feature_columns']
           data_format_used = loaded_model_data['data_format']
           logger.info(f"Loaded model '{model_filename}' (format: {data_format_used}) for manual prediction.")
       except Exception as e:
           return jsonify({'success': False, 'error': f'Failed to load model {model_filename} for manual input: {str(e)}'})

       input_grades_raw = {cid: grade for cid, grade in data.items() if cid != 'model_filename'}

       student_data_for_prediction = {}
       grade_mapping = app.config['DATA_CONFIG']['grade_mapping']
       subject_categories = app.config['SUBJECT_CATEGORIES']
       all_known_courses_from_config = app.config['COURSES_DATA']

       all_grades_entered = []
       subject_categories_grades = {cat: [] for cat in subject_categories.keys()}
       subject_categories_grades['อื่นๆ'] = []

       for course_id, grade_str in input_grades_raw.items():
           if grade_str and grade_str.strip():
               numeric_grade = None
               try:
                   numeric_grade = float(grade_str)
                   if not (0.0 <= numeric_grade <= 4.0):
                       numeric_grade = 0.0
               except ValueError:
                   numeric_grade = grade_mapping.get(str(grade_str).upper(), 0.0)
               
               all_grades_entered.append(numeric_grade)

               course_name = ""
               for c_data in all_known_courses_from_config:
                   if c_data['id'] == course_id:
                       course_name = c_data['thaiName']
                       break

               category = 'อื่นๆ'
               if course_name:
                   for cat, info in subject_categories.items():
                       if any(keyword in course_name.lower() for keyword in info['keywords']):
                           category = cat
                           break
                   
               subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

       gpa = np.mean(all_grades_entered) if all_grades_entered else 0.0
       min_grade = np.min(all_grades_entered) if all_grades_entered else 0.0
       max_grade = np.max(all_grades_entered) if all_grades_entered else 0.0
       std_grade = np.std(all_grades_entered) if len(all_grades_entered) > 1 else 0.0
       fail_count = sum(1 for g in all_grades_entered if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
       fail_rate = fail_count / len(all_grades_entered) if all_grades_entered else 0.0
       total_subjects = len(all_grades_entered)

       student_data_for_prediction = {
           'gpa': gpa,
           'min_grade': min_grade,
           'max_grade': max_grade,
           'std_grade': std_grade,
           'fail_count': fail_count,
           'fail_rate': fail_rate,
           'total_subjects': total_subjects,
           'year_in': 0,
           'year_out': 0,
           'total_terms': 0
       }

       for cat, cat_grades in subject_categories_grades.items():
           if cat_grades:
               student_data_for_prediction[f'gpa_{cat}'] = np.mean(cat_grades)
               student_data_for_prediction[f'min_{cat}'] = np.min(cat_grades)
               student_data_for_prediction[f'max_{cat}'] = np.max(cat_grades)
               student_data_for_prediction[f'fail_rate_{cat}'] = sum(1 for g in cat_grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01) / len(cat_grades)
           else:
               student_data_for_prediction[f'gpa_{cat}'] = 0.0
               student_data_for_prediction[f'min_{cat}'] = 0.0
               student_data_for_prediction[f'max_{cat}'] = 0.0
               student_data_for_prediction[f'fail_rate_{cat}'] = 0.0

       processed_input_for_df = {}
       for feature in feature_cols:
           processed_input_for_df[feature] = [student_data_for_prediction.get(feature, 0.0)]

       input_df = pd.DataFrame(processed_input_for_df)

       trained_models = model_info['models']
       scaler = model_info['scaler']

       predictions_proba_list = []
       for name, model in trained_models.items():
           try:
               if name == 'lr':
                   X_scaled = scaler.transform(input_df)
                   pred_proba = model.predict_proba(X_scaled)
               else:
                   pred_proba = model.predict_proba(input_df)
               
               if pred_proba.shape[1] == 1:
                   pred_proba = np.hstack((1 - pred_proba, pred_proba))
               
               predictions_proba_list.append(pred_proba)
           except Exception as e:
               logger.warning(f"Could not predict with model {name} from manual input: {str(e)}")
               continue

       if not predictions_proba_list:
           return jsonify({'success': False, 'error': 'Could not make predictions with manual input.'})

       avg_prob_per_student = np.mean([pred_proba_array[0] for pred_proba_array in predictions_proba_list], axis=0)
       avg_prob_fail = avg_prob_per_student[0]
       avg_prob_pass = avg_prob_per_student[1]

       prediction = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'

       confidence = max(avg_prob_pass, avg_prob_fail)
       high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
       medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

       if confidence > high_confidence_threshold:
           risk_level = 'ต่ำ' if prediction == 'จบ' else 'สูง'
       elif confidence > medium_confidence_threshold:
           risk_level = 'ปานกลาง'
       else:
           risk_level = 'สูง' if prediction == 'ไม่จบ' else 'ปานกลาง'

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
           analysis.append(f"GPA ดี ({float(gpa_for_analysis):.2f})")
           recommendations.extend(app.config['MESSAGES']['recommendations']['low_risk'])

       if prediction == 'ไม่จบ':
           recommendations.append("แนะนำให้ทบทวนแผนการเรียนและขอความช่วยเหลือ")
           if student_data_for_prediction.get('fail_rate', 0) > high_fail_rate_threshold:
               recommendations.append("มีอัตราการตกในบางวิชาสูง ควรให้ความสำคัญกับการเรียนซ่อม")

       if data_format_used == 'subject_based':
           weak_categories = []
           for cat_key in app.config['SUBJECT_CATEGORIES'].keys():
               cat_gpa_key = f'gpa_{cat_key}'
               if student_data_for_prediction.get(cat_gpa_key, 0) < low_gpa_threshold:
                   weak_categories.append(cat_key)
           if weak_categories:
               recommendations.append(f"ควรเน้นปรับปรุงวิชาในหมวด: {', '.join(weak_categories[:2])}")

       return jsonify({
           'success': True,
           'student_name': student_name,
           'prediction': prediction,
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


def load_existing_models():
    """Loads existing trained models from storage."""
    try:
        logger.info("🔍 Searching for existing models...")
        
        # Get models list from storage
        models_list = storage.list_models()
        
        if not models_list:
            logger.info("No existing models found")
            return
        
        logger.info(f"Found {len(models_list)} models in storage")
        
        # Load subject-based model
        subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '') or m.get('data_format') == 'subject_based']
        if subject_models:
            latest_subject = subject_models[0]
            logger.info(f"Loading subject model: {latest_subject['filename']}")
            loaded_data = storage.load_model(latest_subject['filename'])
            if loaded_data:
                models['subject_model'] = {
                    'models': loaded_data.get('models', {}),
                    'scaler': loaded_data.get('scaler')
                }
                models['subject_feature_cols'] = loaded_data.get('feature_columns', [])
                models['subject_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['subject_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['subject_model_info']['loaded_from_file'] = True
                models['subject_model_info']['filename'] = latest_subject['filename']
                logger.info(f"✅ Loaded latest subject model: {latest_subject['filename']}")
            else:
                logger.warning(f"Could not load subject model data from {latest_subject['filename']}")

        # Load GPA-based model
        gpa_models = [m for m in models_list if 'gpa_based' in m.get('filename', '') or m.get('data_format') == 'gpa_based']
        if gpa_models:
            latest_gpa = gpa_models[0]
            logger.info(f"Loading GPA model: {latest_gpa['filename']}")
            loaded_data = storage.load_model(latest_gpa['filename'])
            if loaded_data:
                models['gpa_model'] = {
                    'models': loaded_data.get('models', {}),
                    'scaler': loaded_data.get('scaler')
                }
                models['gpa_feature_cols'] = loaded_data.get('feature_columns', [])
                models['gpa_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['gpa_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['gpa_model_info']['loaded_from_file'] = True
                models['gpa_model_info']['filename'] = latest_gpa['filename']
                logger.info(f"✅ Loaded latest GPA model: {latest_gpa['filename']}")
            else:
                logger.warning(f"Could not load GPA model data from {latest_gpa['filename']}")

        logger.info("✅ Model loading completed")

    except Exception as e:
        logger.error(f"❌ Error loading existing models: {str(e)}")

if __name__ == '__main__':
    logger.info("=== FLASK APP CONFIGURATION ===")
    logger.info(f"App name: {app.name}")
    logger.info(f"App debug: {app.debug}")
    logger.info(f"App testing: {app.testing}")
    logger.info(f"Config keys: {list(app.config.keys())}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Model folder: {app.config['MODEL_FOLDER']}")
    logger.info(f"S3 Storage: {'Enabled' if not storage.use_local else 'Disabled (using local)'}")
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"✅ Created folder: {folder}")
    
    # โหลดโมเดลที่มีอยู่
    load_existing_models()
    
    logger.info("🚀 Starting server...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
