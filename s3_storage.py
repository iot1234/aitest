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
            # Get Cloudflare R2 credentials from environment or use provided values
            self.access_key = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', '2f097b76d5be26634ea77723b6b55f23')
            self.secret_key = os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', 'a8c8c99ef7e1224ad0a451f832d030c41e448b7204bf58ef141af8eed22f3730')
            self.endpoint_url = os.environ.get('CLOUDFLARE_R2_ENDPOINT', 'https://e7dfcd2d210b2a0e8d8158449f38ab2e.r2.cloudflarestorage.com')
            self.bucket_name = os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', 'pjai')
            
            # For local development, fallback to regular AWS credentials (for testing)
            if not self.access_key and not self.secret_key:
                self.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
                self.secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
                self.bucket_name = os.environ.get('S3_BUCKET_NAME', 'student-predictor-models')
                self.endpoint_url = None
            
            if not all([self.access_key, self.secret_key, self.endpoint_url]):
                logger.warning("Cloudflare R2 credentials not found. Using local storage fallback.")
                self.s3_client = None
                self.use_local = True
            else:
                # Initialize R2 client with Cloudflare endpoint
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name='auto'  # R2 uses 'auto' region
                )
                self.use_local = False
                logger.info(f"Cloudflare R2 Storage initialized successfully")
                logger.info(f"R2 Endpoint: {self.endpoint_url}")
                logger.info(f"R2 Bucket: {self.bucket_name}")
                
                # Ensure bucket exists and is accessible
                self._verify_bucket()
                
        except Exception as e:
            logger.error(f"Error initializing Cloudflare R2 storage: {str(e)}")
            self.s3_client = None
            self.use_local = True
    
    def _verify_bucket(self):
        """Verify bucket exists and is accessible"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ R2 Bucket '{self.bucket_name}' verified successfully")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"R2 Bucket '{self.bucket_name}' not found, creating...")
                self._create_bucket()
            else:
                logger.error(f"Error accessing R2 bucket: {str(e)}")
                # Don't raise, fall back to local storage
                self.use_local = True
                self.s3_client = None
    
    def _create_bucket(self):
        """Create R2 bucket if it doesn't exist"""
        try:
            # For R2, we don't need to specify CreateBucketConfiguration
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ R2 Bucket '{self.bucket_name}' created successfully")
        except ClientError as e:
            logger.error(f"Error creating R2 bucket: {str(e)}")
            # Fall back to local storage instead of raising
            self.use_local = True
            self.s3_client = None
    
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """
        Save model to Cloudflare R2
        
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
            
            # Also save metadata separately for quick access
            self._save_model_metadata(filename, model_data)
            
            logger.info(f"✅ Model '{filename}' saved successfully to Cloudflare R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model to Cloudflare R2: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load model from Cloudflare R2
        
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
                logger.warning(f"Model '{filename}' not found in Cloudflare R2")
            else:
                logger.error(f"Error loading model from Cloudflare R2: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None
    
    def delete_model(self, filename: str) -> bool:
        """
        Delete model from Cloudflare R2
        
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
            
            logger.info(f"✅ Model '{filename}' deleted successfully from Cloudflare R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting model from Cloudflare R2: {str(e)}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in Cloudflare R2
        
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
                            # Basic metadata from R2 object
                            metadata = {
                                'filename': filename,
                                'created_at': obj['LastModified'].isoformat(),
                                'size': obj['Size'],
                                'storage_provider': 'cloudflare_r2'
                            }
                        models.append(metadata)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            logger.info(f"📊 Found {len(models)} models in Cloudflare R2")
            return models
            
        except Exception as e:
            logger.error(f"❌ Error listing models from Cloudflare R2: {str(e)}")
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
                'feature_importances': model_data.get('feature_importances', {}),
                'storage_provider': 'cloudflare_r2'
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
    
    # Local storage fallback methods (unchanged)
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
                            'storage_provider': 'local'
                        })
            
            logger.info(f"📂 Found {len(models)} models in local storage")
            return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error listing local models: {str(e)}")
            return []

# Create global storage instance
storage = CloudflareR2Storage()
