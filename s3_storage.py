import os
import boto3
import joblib
import tempfile
import logging
from typing import Any, Dict, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CloudflareR2Storage:
    """Manager for Cloudflare R2 storage operations using S3-compatible API"""
    
    def __init__(self):
        """Initialize R2 client with Cloudflare R2 credentials"""
        try:
            # Get Cloudflare R2 credentials from environment
            self.access_key = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID')
            self.secret_key = os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
            self.endpoint_url = os.environ.get('CLOUDFLARE_R2_ENDPOINT')
            self.bucket_name = os.environ.get('CLOUDFLARE_R2_BUCKET_NAME')
            
            if not all([self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]):
                logger.warning("Cloudflare R2 credentials not found. Using local storage fallback.")
                self.s3_client = None
                self.use_local = True
                return
            
            # Initialize R2 client with Cloudflare endpoint
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='auto'  # R2 uses 'auto' region
            )
            self.use_local = False
            logger.info(f"✅ Cloudflare R2 Storage initialized successfully")
            logger.info(f"R2 Endpoint: {self.endpoint_url}")
            logger.info(f"R2 Bucket: {self.bucket_name}")
            
            # Test connection
            self._test_connection()
                
        except Exception as e:
            logger.error(f"❌ Error initializing Cloudflare R2 storage: {str(e)}")
            self.s3_client = None
            self.use_local = True
    
    def _test_connection(self):
        """Test connection to R2"""
        try:
            # Try to list objects to test connection
            self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            logger.info(f"✅ R2 connection test successful")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                logger.warning(f"⚠️ R2 Bucket '{self.bucket_name}' not found, will try to create when needed")
            else:
                logger.warning(f"⚠️ R2 connection test failed: {e}, falling back to local storage")
                self.use_local = True
                self.s3_client = None
        except Exception as e:
            logger.warning(f"⚠️ R2 connection test failed: {e}, falling back to local storage")
            self.use_local = True
            self.s3_client = None
    
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model to Cloudflare R2 or local storage"""
        if self.use_local:
            return self._save_model_locally(model_data, filename)
        
        tmp_path = None
        try:
            # Create bucket if it doesn't exist
            self._ensure_bucket_exists()
            
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(model_data, tmp_file.name)
                tmp_path = tmp_file.name
            
            # Upload to R2
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
                        'accuracy': str(model_data.get('performance_metrics', {}).get('accuracy', 0)),
                        'storage_provider': 'cloudflare_r2'
                    }
                )
            
            # Clean up temporary file
            os.remove(tmp_path)
            
            logger.info(f"✅ Model '{filename}' saved successfully to Cloudflare R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model to Cloudflare R2: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Try local fallback
            logger.info("🔄 Trying local storage fallback...")
            return self._save_model_locally(model_data, filename)
    
    def _ensure_bucket_exists(self):
        """Ensure bucket exists, create if not"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"✅ R2 Bucket '{self.bucket_name}' created successfully")
                except Exception as create_error:
                    logger.error(f"❌ Could not create bucket: {create_error}")
                    raise
            else:
                logger.error(f"❌ Bucket access error: {e}")
                raise
    
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model from Cloudflare R2 or local storage"""
        if self.use_local:
            return self._load_model_locally(filename)
        
        tmp_path = None
        try:
            s3_key = f"models/{filename}"
            
            # Download from R2 to temporary file
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
            
            logger.info(f"✅ Model '{filename}' loaded successfully from Cloudflare R2")
            return model_data
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Model '{filename}' not found in Cloudflare R2, trying local storage")
                return self._load_model_locally(filename)
            else:
                logger.error(f"Error loading model from Cloudflare R2: {str(e)}")
                return self._load_model_locally(filename)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return self._load_model_locally(filename)
    
    def delete_model(self, filename: str) -> bool:
        """Delete model from Cloudflare R2 or local storage"""
        if self.use_local:
            return self._delete_model_locally(filename)
        
        try:
            s3_key = f"models/{filename}"
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"✅ Model '{filename}' deleted successfully from Cloudflare R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting model from Cloudflare R2: {str(e)}")
            # Try local fallback
            return self._delete_model_locally(filename)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in Cloudflare R2 or local storage"""
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
                        metadata = {
                            'filename': filename,
                            'created_at': obj['LastModified'].isoformat(),
                            'size': obj['Size'],
                            'storage_provider': 'cloudflare_r2',
                            'performance_metrics': {},
                            'data_format': 'unknown'
                        }
                        models.append(metadata)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            logger.info(f"📊 Found {len(models)} models in Cloudflare R2")
            
            # Also include local models if any
            local_models = self._list_models_locally()
            models.extend(local_models)
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Error listing models from Cloudflare R2: {str(e)}")
            return self._list_models_locally()
    
    # Local storage fallback methods
    def _save_model_locally(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model locally when R2 is not available"""
        try:
            model_folder = 'models'
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
            filepath = os.path.join(model_folder, filename)
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Model '{filename}' saved locally (R2 fallback)")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model locally: {str(e)}")
            return False
    
    def _load_model_locally(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model from local storage"""
        try:
            filepath = os.path.join('models', filename)
            if os.path.exists(filepath):
                logger.info(f"📂 Loading model '{filename}' from local storage")
                return joblib.load(filepath)
            return None
        except Exception as e:
            logger.error(f"❌ Error loading model locally: {str(e)}")
            return None
    
    def _delete_model_locally(self, filename: str) -> bool:
        """Delete model from local storage"""
        try:
            filepath = os.path.join('models', filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"🗑️ Model '{filename}' deleted from local storage")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error deleting model locally: {str(e)}")
            return False
    
    def _list_models_locally(self) -> List[Dict[str, Any]]:
        """List models from local storage"""
        try:
            model_folder = 'models'
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
                            'training_data_info': model_data.get('training_data_info', {}),
                            'storage_provider': 'local'
                        })
                    except:
                        models.append({
                            'filename': filename,
                            'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                            'storage_provider': 'local',
                            'performance_metrics': {},
                            'data_format': 'unknown'
                        })
            
            logger.info(f"📂 Found {len(models)} models in local storage")
            return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error listing local models: {str(e)}")
            return []
