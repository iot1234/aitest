# แก้ไขส่วนบนของ app.py - TIMEOUT OPTIMIZED VERSION
from advanced_training_optimized import AdvancedFeatureEngineer, train_ensemble_model  # ใช้ไฟล์ที่ปรับปรุงแล้ว
from dotenv import load_dotenv
import sys
import signal
import time

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
from sklearn.model_selection import train_test_split
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
from werkzeug.utils import secure_filename

import config

warnings.filterwarnings('ignore')

# Set up logging with timeout information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# TIMEOUT CONFIGURATION
# ==========================================
TRAINING_TIMEOUT = int(os.environ.get('TRAINING_TIMEOUT', '25'))  # 25 seconds default
WORKER_TIMEOUT = int(os.environ.get('WORKER_TIMEOUT', '30'))      # 30 seconds default

logger.info("=" * 70)
logger.info("⏰ TIMEOUT CONFIGURATION")
logger.info("=" * 70)
logger.info(f"✅ Training timeout: {TRAINING_TIMEOUT} seconds")
logger.info(f"✅ Worker timeout: {WORKER_TIMEOUT} seconds")
logger.info("=" * 70)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                logger.error(f"❌ Function {func.__name__} timed out after {timeout_seconds} seconds")
                raise
            finally:
                # Restore old handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

# ==========================================
# ตรวจสอบ Environment Variables แบบละเอียด
# ==========================================
logger.info("=" * 70)
logger.info("🔧 CLOUDFLARE R2 CONFIGURATION CHECK")
logger.info("=" * 70)

def check_env_variables():
    """ตรวจสอบและแสดงสถานะ environment variables แบบละเอียด"""
    env_status = {
        'all_present': True,
        'missing': [],
        'details': {}
    }
    
    required_vars = {
        'CLOUDFLARE_R2_ACCESS_KEY_ID': 'Access Key for R2 authentication',
        'CLOUDFLARE_R2_SECRET_ACCESS_KEY': 'Secret Key for R2 authentication',
        'CLOUDFLARE_R2_ENDPOINT': 'R2 endpoint URL',
        'CLOUDFLARE_R2_BUCKET_NAME': 'R2 bucket name'
    }
    
    for var_name, description in required_vars.items():
        value = os.environ.get(var_name, '')
        
        if value:
            # แสดงแค่ส่วนแรกของค่าเพื่อความปลอดภัย
            if 'SECRET' in var_name or 'KEY' in var_name:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            
            logger.info(f"✅ {var_name}: {display_value}")
            env_status['details'][var_name] = {
                'present': True,
                'length': len(value),
                'description': description
            }
        else:
            logger.error(f"❌ {var_name}: NOT FOUND - {description}")
            env_status['missing'].append(var_name)
            env_status['all_present'] = False
            env_status['details'][var_name] = {
                'present': False,
                'description': description
            }
    
    # ตรวจสอบไฟล์ .env
    if os.path.exists('.env'):
        logger.info(f"📄 .env file: EXISTS (size: {os.path.getsize('.env')} bytes)")
        
        # อ่านและตรวจสอบเนื้อหา .env
        try:
            with open('.env', 'r') as f:
                env_lines = f.readlines()
                env_vars_in_file = {}
                for line in env_lines:
                    if '=' in line and not line.strip().startswith('#'):
                        key = line.split('=')[0].strip()
                        env_vars_in_file[key] = True
            
            logger.info(f"📋 Variables in .env file: {list(env_vars_in_file.keys())}")
            
            # เช็คว่าตัวแปรใน .env ถูกโหลดหรือไม่
            for var in required_vars.keys():
                if var in env_vars_in_file and not os.environ.get(var):
                    logger.warning(f"⚠️ {var} exists in .env but not loaded into environment!")
                    
        except Exception as e:
            logger.error(f"❌ Error reading .env file: {str(e)}")
    else:
        logger.warning("⚠️ .env file: NOT FOUND")
    
    return env_status

env_check_result = check_env_variables()

if not env_check_result['all_present']:
    logger.error("=" * 70)
    logger.error("⚠️ CLOUDFLARE R2 CONFIGURATION INCOMPLETE!")
    logger.error("Missing variables: " + ", ".join(env_check_result['missing']))
    logger.error("R2 storage will be DISABLED. Using LOCAL storage instead.")
    logger.error("=" * 70)

# Create Flask app
app = Flask(__name__)
app.config.from_object(config.get_config())

# ==========================================
# สร้างโฟลเดอร์ที่จำเป็นทันที
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

logger.info("=" * 70)
logger.info("📁 DIRECTORY SETUP")
logger.info("=" * 70)

for folder_name, folder_path in [('uploads', UPLOAD_FOLDER), ('models', MODEL_FOLDER)]:
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"✅ Created {folder_name} folder: {folder_path}")
            
            # สร้างไฟล์ .gitkeep เพื่อให้ git track โฟลเดอร์ว่าง
            gitkeep_path = os.path.join(folder_path, '.gitkeep')
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, 'w') as f:
                    f.write('')
                logger.info(f"   Created .gitkeep in {folder_name}")
        else:
            logger.info(f"✅ {folder_name} folder exists: {folder_path}")
            
        # ตรวจสอบ permissions
        if os.access(folder_path, os.W_OK):
            logger.info(f"   ✅ Write permission: OK")
        else:
            logger.error(f"   ❌ Write permission: DENIED")
            
        # แสดงไฟล์ที่มีอยู่
        files = os.listdir(folder_path)
        if files:
            logger.info(f"   Files in {folder_name}: {len(files)} file(s)")
            for file in files[:5]:  # แสดงแค่ 5 ไฟล์แรก
                file_path = os.path.join(folder_path, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                logger.info(f"     - {file} ({file_size} bytes)")
                
    except Exception as e:
        logger.error(f"❌ Error with {folder_name} folder: {str(e)}")
        logger.error(f"   Current working directory: {os.getcwd()}")
        logger.error(f"   Python executable: {sys.executable}")

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
# Enhanced S3 Storage Class with Timeout Protection
# ==========================================
class S3Storage:
    def __init__(self):
        """Initialize S3 client with timeout protection"""
        self.connection_errors = []
        self.use_local = False
        self.s3_client = None
        
        logger.info("=" * 70)
        logger.info("🚀 INITIALIZING CLOUDFLARE R2 STORAGE")
        logger.info("=" * 70)
        
        try:
            # ตรวจสอบ credentials
            self.access_key = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', '')
            self.secret_key = os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', '')
            self.endpoint_url = os.environ.get('CLOUDFLARE_R2_ENDPOINT', '')
            self.bucket_name = os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', 'pjai')
            
            # ตรวจสอบแต่ละ credential
            missing_creds = []
            if not self.access_key:
                missing_creds.append('CLOUDFLARE_R2_ACCESS_KEY_ID')
            if not self.secret_key:
                missing_creds.append('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
            if not self.endpoint_url:
                missing_creds.append('CLOUDFLARE_R2_ENDPOINT')
                
            if missing_creds:
                error_msg = f"Missing R2 credentials: {', '.join(missing_creds)}"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
                logger.info("📂 Fallback: Using LOCAL storage")
                return
            
            # ตรวจสอบ boto3
            try:
                import boto3
                from botocore.config import Config
                from botocore.exceptions import NoCredentialsError, ClientError
            except ImportError as e:
                error_msg = "boto3 library not installed. Run: pip install boto3"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
                logger.info("📂 Fallback: Using LOCAL storage")
                return
            
            # พยายามสร้าง S3 client with timeout
            logger.info("🔗 Attempting to connect to Cloudflare R2...")
            logger.info(f"   Endpoint: {self.endpoint_url}")
            logger.info(f"   Bucket: {self.bucket_name}")
            
            try:
                config = Config(
                    region_name='auto',
                    signature_version='s3v4',
                    retries={'max_attempts': 2, 'mode': 'standard'},  # ลด retries
                    connect_timeout=5,   # ลด timeout
                    read_timeout=5       # ลด timeout
                )
                
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
                
                # ทดสอบการเชื่อมต่อ with timeout
                logger.info("🧪 Testing R2 connection...")
                test_response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    MaxKeys=1
                )
                
                logger.info("✅ R2 CONNECTION SUCCESSFUL!")
                logger.info(f"   Objects in bucket: {len(test_response.get('Contents', []))}")
                self.use_local = False
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                if error_code == 'NoSuchBucket':
                    error_msg = f"Bucket '{self.bucket_name}' does not exist"
                    logger.error(f"❌ {error_msg}")
                    logger.info("🔨 Attempting to create bucket...")
                    
                    try:
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                        logger.info(f"✅ Bucket '{self.bucket_name}' created successfully!")
                        self.use_local = False
                    except Exception as create_error:
                        error_msg = f"Failed to create bucket: {str(create_error)}"
                        logger.error(f"❌ {error_msg}")
                        self.connection_errors.append(error_msg)
                        self.use_local = True
                        
                elif error_code == 'InvalidAccessKeyId':
                    error_msg = "Invalid Access Key ID - Check your CLOUDFLARE_R2_ACCESS_KEY_ID"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
                elif error_code == 'SignatureDoesNotMatch':
                    error_msg = "Invalid Secret Key - Check your CLOUDFLARE_R2_SECRET_ACCESS_KEY"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
                else:
                    error_msg = f"R2 Error [{error_code}]: {error_message}"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
            
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.connection_errors.append(error_msg)
            self.use_local = True
        
        # สรุปผล
        logger.info("=" * 70)
        if self.use_local:
            logger.warning("⚠️ R2 STORAGE: DISABLED")
            logger.info("📂 Using LOCAL FILE STORAGE")
            if self.connection_errors:
                logger.error("Connection errors:")
                for error in self.connection_errors:
                    logger.error(f"  • {error}")
        else:
            logger.info("✅ R2 STORAGE: ENABLED AND READY")
        logger.info("=" * 70)
    
    def get_connection_status(self):
        """Get detailed connection status for API"""
        return {
            'connected': not self.use_local,
            'storage_type': 'local' if self.use_local else 'cloudflare_r2',
            'bucket_name': self.bucket_name if not self.use_local else None,
            'endpoint': self.endpoint_url if not self.use_local else None,
            'errors': self.connection_errors
        }
    
    @with_timeout(10)  # 10 second timeout for save operations
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model with timeout protection"""
        if self.use_local:
            return self._save_model_locally(model_data, filename)
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(model_data, tmp_file.name)
                tmp_path = tmp_file.name
            
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
            
            os.remove(tmp_path)
            logger.info(f"✅ Model {filename} saved to R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ R2 save error: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Fallback to local
            logger.info("📂 Trying local save as fallback...")
            return self._save_model_locally(model_data, filename)
    
    def _save_model_locally(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model locally"""
        try:
            model_folder = app.config['MODEL_FOLDER']
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
            filepath = os.path.join(model_folder, filename)
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Model {filename} saved locally")
            return True
        except Exception as e:
            logger.error(f"❌ Local save error: {str(e)}")
            return False
    
    @with_timeout(10)  # 10 second timeout for load operations
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model with timeout protection"""
        if self.use_local:
            return self._load_model_locally(filename)
        
        tmp_path = None
        try:
            s3_key = f"models/{filename}"
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Filename=tmp_file.name
                )
                tmp_path = tmp_file.name
            
            model_data = joblib.load(tmp_path)
            os.remove(tmp_path)
            logger.info(f"✅ Model {filename} loaded from R2")
            return model_data
            
        except Exception as e:
            logger.error(f"❌ R2 load error: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Fallback to local
            logger.info("📂 Trying local load as fallback...")
            return self._load_model_locally(filename)
    
    def _load_model_locally(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model locally"""
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                logger.info(f"💾 Model {filename} loaded locally")
                return model_data
            else:
                logger.warning(f"⚠️ Local model file not found: {filename}")
                return None
        except Exception as e:
            logger.error(f"❌ Local load error: {str(e)}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models with timeout protection"""
        models_list = []
        
        # List from R2
        if not self.use_local:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix='models/'
                )
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.joblib'):
                        filename = obj['Key'].replace('models/', '')
                        models_list.append({
                            'filename': filename,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'storage': 'r2'
                        })
                        
                logger.info(f"📊 Found {len(models_list)} models in R2")
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not list R2 models: {e}")
        
        # List from local
        try:
            model_folder = app.config['MODEL_FOLDER']
            if os.path.exists(model_folder):
                local_count = 0
                for filename in os.listdir(model_folder):
                    if filename.endswith('.joblib'):
                        # Check if not already in list from R2
                        if not any(m['filename'] == filename for m in models_list):
                            filepath = os.path.join(model_folder, filename)
                            models_list.append({
                                'filename': filename,
                                'size': os.path.getsize(filepath),
                                'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
                                'storage': 'local'
                            })
                            local_count += 1
                            
                logger.info(f"📂 Found {local_count} models locally")
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not list local models: {e}")
        
        logger.info(f"Found {len(models_list)} models total")
        return models_list
    
    def delete_model(self, filename: str) -> bool:
        """Delete model with timeout protection"""
        success = False
        
        # Delete from R2
        if not self.use_local:
            try:
                s3_key = f"models/{filename}"
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.info(f"✅ Model {filename} deleted from R2")
                success = True
            except Exception as e:
                logger.warning(f"⚠️ Could not delete from R2: {e}")
        
        # Delete from local
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"💾 Model {filename} deleted locally")
                success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not delete locally: {e}")
        
        return success

# Initialize storage
storage = S3Storage()

# ==========================================
# TIMEOUT-PROTECTED TRAINING FUNCTION
# ==========================================
@with_timeout(TRAINING_TIMEOUT)
def train_model_with_timeout(df, data_format):
    """Train model with timeout protection"""
    logger.info(f"🚀 Starting model training with {TRAINING_TIMEOUT}s timeout...")
    start_time = time.time()
    
    try:
        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer(
            grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
        )
        
        # Prepare training data
        logger.info("🔧 Preparing training data...")
        X, y = feature_engineer.prepare_training_data(df)
        
        # Train ensemble model with remaining time
        elapsed = time.time() - start_time
        remaining_time = max(5, TRAINING_TIMEOUT - elapsed - 2)  # Leave 2 seconds buffer
        
        logger.info(f"⏰ Training models with {remaining_time:.1f}s remaining...")
        training_result = train_ensemble_model(X, y, max_training_time=int(remaining_time))
        
        # Prepare model data for saving
        model_data = {
            'models': training_result['models'],
            'scaler': training_result['scaler'],
            'feature_columns': training_result['feature_names'],
            'data_format': data_format,
            'created_at': datetime.now().isoformat(),
            'performance_metrics': {
                'accuracy': training_result['accuracy'],
                'precision': training_result['precision'],
                'recall': training_result['recall'],
                'f1_score': training_result['f1_score']
            },
            'training_info': {
                'training_samples': training_result['training_samples'],
                'validation_samples': training_result['validation_samples'],
                'training_time': training_result.get('training_time', 0)
            }
        }
        
        total_time = time.time() - start_time
        logger.info(f"✅ Model training completed in {total_time:.1f}s")
        
        return model_data, training_result['performance_metrics']
        
    except TimeoutError:
        logger.error(f"❌ Model training timed out after {TRAINING_TIMEOUT}s")
        raise
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise

# ==========================================
# REST OF THE APP.PY CODE CONTINUES HERE...
# ==========================================

# [เพิ่มส่วนที่เหลือของ app.py ที่ไม่เปลี่ยนแปลง]
# ฟังก์ชันอื่นๆ เช่น detect_data_format, grade_to_numeric, categorize_subject, etc.
# และ routes ต่างๆ

def detect_data_format(df):
    """Detect the format of uploaded data"""
    columns = [col.lower() for col in df.columns]
    
    # Check for transcript format (student_id, course_id, grade, term, year)
    transcript_indicators = ['student_id', 'course_id', 'grade', 'term', 'year']
    if all(indicator in ' '.join(columns) for indicator in transcript_indicators[:3]):
        return 'transcript'
    
    # Check for subject-based format (student columns + subject grade columns)
    if len(df.columns) > 10:  # Likely many subject columns
        return 'subject_based'
    
    # Default to subject-based
    return 'subject_based'

def grade_to_numeric(grade):
    """Convert grade to numeric value"""
    grade_mapping = app.config['DATA_CONFIG']['grade_mapping']
    
    if isinstance(grade, (int, float)):
        return float(grade)
    
    grade_str = str(grade).strip().upper()
    return grade_mapping.get(grade_str, 0.0)

def categorize_subject(subject_name):
    """Categorize subject into predefined categories"""
    subject_categories = app.config['SUBJECT_CATEGORIES']
    
    subject_lower = subject_name.lower()
    for category, info in subject_categories.items():
        if any(keyword in subject_lower for keyword in info['keywords']):
            return category
    
    return 'อื่นๆ'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train model with timeout protection"""
    try:
        logger.info("🚀 Starting model training request...")
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided.'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found.'})
        
        # Load data
        logger.info(f"📂 Loading data from {filename}...")
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            return jsonify({'success': False, 'error': f'Error loading file: {str(e)}'})
        
        # Detect data format
        data_format = detect_data_format(df)
        logger.info(f"📊 Detected data format: {data_format}")
        
        # Train model with timeout protection
        try:
            model_data, performance_metrics = train_model_with_timeout(df, data_format)
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{data_format}_model_{timestamp}.joblib"
            
            logger.info(f"💾 Saving model as {model_filename}...")
            save_success = storage.save_model(model_data, model_filename)
            
            if save_success:
                # Update global models variable
                if data_format == 'subject_based':
                    models['subject_model'] = {
                        'models': model_data['models'],
                        'scaler': model_data['scaler']
                    }
                    models['subject_feature_cols'] = model_data['feature_columns']
                    models['subject_model_info'] = performance_metrics
                    models['subject_model_info']['filename'] = model_filename
                
                logger.info("✅ Model training and saving completed successfully!")
                
                return jsonify({
                    'success': True,
                    'message': 'Model trained and saved successfully!',
                    'model_filename': model_filename,
                    'data_format': data_format,
                    'performance_metrics': performance_metrics,
                    'training_info': model_data.get('training_info', {})
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Model training completed but saving failed.'
                })
                
        except TimeoutError:
            return jsonify({
                'success': False,
                'error': f'Model training timed out after {TRAINING_TIMEOUT} seconds. Please try with a smaller dataset or contact support.'
            })
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Training failed: {str(e)}'
            })
    
    except Exception as e:
        logger.error(f"❌ Request error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Request failed: {str(e)}'
        })

# Add other routes here...
@app.route('/status')
def status_page():
    """หน้าแสดงสถานะระบบ"""
    return render_template('status.html')

@app.route('/api/system/status')
def system_status():
    """API สำหรับตรวจสอบสถานะระบบ"""
    try:
        logger.info("🔧 Checking system status...")
        
        storage_status = storage.get_connection_status()
        
        # นับจำนวนโมเดล
        try:
            models_list = storage.list_models()
            total_size = sum(m.get('size', 0) for m in models_list if 'size' in m)
        except Exception as e:
            logger.warning(f"Could not get models info: {e}")
            models_list = []
            total_size = 0
        
        status_data = {
            'success': True,
            'r2_connected': storage_status['connected'],
            'storage_provider': storage_status['storage_type'],
            'bucket_name': storage_status['bucket_name'],
            'models_available': len(models_list),
            'total_size': total_size,
            'training_timeout': TRAINING_TIMEOUT,
            'worker_timeout': WORKER_TIMEOUT,
            'server_time': datetime.now().isoformat(),
            'app_folders': {
                'upload_folder': app.config['UPLOAD_FOLDER'],
                'model_folder': app.config['MODEL_FOLDER'],
                'upload_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'model_exists': os.path.exists(app.config['MODEL_FOLDER'])
            }
        }
        
        logger.info("✅ System status check completed")
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'server_time': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("=== FLASK APP CONFIGURATION ===")
    logger.info(f"App name: {app.name}")
    logger.info(f"Training timeout: {TRAINING_TIMEOUT}s")
    logger.info(f"Worker timeout: {WORKER_TIMEOUT}s")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Model folder: {app.config['MODEL_FOLDER']}")
    logger.info(f"S3 Storage: {'Enabled' if not storage.use_local else 'Disabled (using local)'}")
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"✅ Created folder: {folder}")
    
    logger.info("🚀 Starting server...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

