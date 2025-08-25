import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class CloudflareR2Storage:
    def __init__(self):
        self.access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT")
        bucket_name = os.getenv("CLOUDFLARE_R2_BUCKET_NAME")

        if not all([self.access_key, self.secret_key, endpoint, bucket_name]):
            raise ValueError("Missing Cloudflare R2 configuration in environment variables")

        # ถ้า endpoint มี /bucket_name ติดมาจาก Dashboard -> ตัดออก
        if endpoint.endswith(f"/{bucket_name}"):
            endpoint = endpoint[:-(len(bucket_name) + 1)]

        self.bucket = bucket_name

        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        )

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="auto",   # Cloudflare R2 ใช้ 'auto'
            config=config
        )

    def upload_file(self, file_path: str, object_name: str = None):
        """อัปโหลดไฟล์ไปยัง R2"""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket, object_name)
            print(f"✅ Uploaded {file_path} to {self.bucket}/{object_name}")
        except ClientError as e:
            print(f"❌ Upload failed: {e}")
            raise

    def download_file(self, object_name: str, file_path: str):
        """ดาวน์โหลดไฟล์จาก R2"""
        try:
            self.s3_client.download_file(self.bucket, object_name, file_path)
            print(f"✅ Downloaded {object_name} to {file_path}")
        except ClientError as e:
            print(f"❌ Download failed: {e}")
            raise

    def list_files(self, prefix: str = ""):
        """ดึงรายชื่อไฟล์ใน bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            print(f"❌ List files failed: {e}")
            raise

    def delete_file(self, object_name: str):
        """ลบไฟล์จาก bucket"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            print(f"🗑️ Deleted {object_name} from {self.bucket}")
        except ClientError as e:
            print(f"❌ Delete failed: {e}")
            raise
