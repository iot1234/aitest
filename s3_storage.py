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

class S3Storage:
    """Manager for S3 storage operations using Bucketeer"""
    
    def __init__(self):
        """Initialize S3 client with Bucketeer credentials"""
        try:
            # Get Bucketeer credentials from environment
            self.access_key = os.environ.get('BUCKETEER_AWS_ACCESS_KEY_ID')
            self.secret_key = os.environ.get('BUCKETEER_AWS_SECRET_ACCESS_KEY')
            self.bucket_name = os.environ.get('BUCKETEER_BUCKET_NAME')
            
            # For local development, fallback to regular AWS credentials
            if not self.access_key:
                self.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
                self.secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
                self.bucket_name = os.environ.get('S3_BUCKET_NAME', 'student-predictor-models')
            
            if not all([self.access_key, self.secret_key, self.bucket_name]):
                logger.warning("S3 credentials not found. Using local storage fallback.")
                self.s3_client = None
                self.use_local = True
            else:
                # Initialize S3 client
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name='us-east-1'
                )
                self.use_local = False
                logger.info(f"S3 Storage initialized with bucket: {self.bucket_name}")
                
                # Ensure bucket exists and is accessible
                self._verify_bucket()
                
        except Exception as e:
            logger.error(f"Error initializing S3 storage: {str(e)}")
            self.s3_client = None
            self.use_local = True
    
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
            if os.path.exists(tmp_path):
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
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=metadata_key
            )
            
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
            model_folder = 'models'
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
            filepath = os.path.join('models', filename)
            if os.path.exists(filepath):
                return joblib.load(filepath)
            return None
        except Exception as e:
            logger.error(f"Error loading model locally: {str(e)}")
            return None
    
    def _delete_model_locally(self, filename: str) -> bool:
        """Delete model from local storage"""
        try:
            filepath = os.path.join('models', filename)
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

# Create global instance
storage = S3Storage()
