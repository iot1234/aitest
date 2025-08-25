# แก้ไขส่วนบนของ app.py
from advanced_training import AdvancedFeatureEngineer
from dotenv import load_dotenv
import sys

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
from typing import Any, Dict, Optional, List
from werkzeug.utils import secure_filename

import config

warnings.filterwarnings('ignore')

# Set up logging
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
# Enhanced S3 Storage Class with Detailed Error Reporting
# ==========================================
class S3Storage:
    def __init__(self):
        """Initialize S3 client with detailed error reporting"""
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
            
            # พยายามสร้าง S3 client
            logger.info("🔗 Attempting to connect to Cloudflare R2...")
            logger.info(f"   Endpoint: {self.endpoint_url}")
            logger.info(f"   Bucket: {self.bucket_name}")
            
            try:
                config = Config(
                    region_name='auto',
                    signature_version='s3v4',
                    retries={'max_attempts': 3, 'mode': 'standard'},
                    connect_timeout=10,
                    read_timeout=10
                )
                
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
                
                # ทดสอบการเชื่อมต่อ
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
    
    # [คง methods อื่นๆ เหมือนเดิม แต่เพิ่ม error handling]
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model with detailed error reporting"""
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
    
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model"""
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
            logger.warning(f"R2 load failed: {str(e)}, trying local...")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return self._load_model_locally(filename)
    
    def _load_model_locally(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model from local storage"""
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                logger.info(f"📂 Model {filename} loaded from local")
                return model_data
            return None
        except Exception as e:
            logger.error(f"❌ Local load error: {str(e)}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models"""
        models = []
        
        # Try R2 first
        if not self.use_local and self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix='models/',
                    Delimiter='/'
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        filename = obj['Key'].replace('models/', '')
                        if filename and filename.endswith('.joblib'):
                            models.append({
                                'filename': filename,
                                'created_at': obj['LastModified'].isoformat(),
                                'size': obj['Size'],
                                'storage': 'r2',
                                'performance_metrics': {},
                                'data_format': 'unknown'
                            })
                logger.info(f"📊 Found {len(models)} models in R2")
            except Exception as e:
                logger.warning(f"R2 list failed: {str(e)}")
        
        # Also check local
        try:
            model_folder = app.config['MODEL_FOLDER']
            if os.path.exists(model_folder):
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
                                'storage': 'local'
                            })
                        except:
                            models.append({
                                'filename': filename,
                                'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                                'storage': 'local',
                                'performance_metrics': {},
                                'data_format': 'unknown'
                            })
                logger.info(f"📂 Found {len([m for m in models if m['storage'] == 'local'])} models locally")
        except Exception as e:
            logger.error(f"Local list error: {str(e)}")
        
        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_model(self, filename: str) -> bool:
        """Delete model"""
        deleted = False
        
        # Try R2
        if not self.use_local and self.s3_client:
            try:
                s3_key = f"models/{filename}"
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.info(f"✅ Model {filename} deleted from R2")
                deleted = True
            except Exception as e:
                logger.warning(f"R2 delete failed: {str(e)}")
        
        # Also try local
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"🗑️ Model {filename} deleted from local")
                deleted = True
        except Exception as e:
            logger.warning(f"Local delete failed: {str(e)}")
        
        return deleted

# Create global storage instance
storage = S3Storage()

# เพิ่ม API endpoint สำหรับตรวจสอบสถานะ
@app.route('/api/storage/status')
def get_storage_status():
    """Get detailed storage status"""
    status = storage.get_connection_status()
    status['env_check'] = env_check_result
    return jsonify(status)

# [เพิ่มส่วนที่เหลือของ code เดิมต่อจากนี้...]
