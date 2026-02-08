# 🔐 แนวทางความปลอดภัย (Security Best Practices)

**โครงการ**: ระบบทำนายการจบการศึกษาของนักศึกษา  
**วันที่**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0  
**ระดับความเข้มงวด**: Medium-High

---

## 📋 สารบัญ

1. [ภาพรวมความปลอดภัย](#ภาพรวมความปลอดภัย)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [API Security](#api-security)
5. [Input Validation](#input-validation)
6. [Session Management](#session-management)
7. [File Upload Security](#file-upload-security)
8. [Database Security](#database-security)
9. [Deployment Security](#deployment-security)
10. [Monitoring & Incident Response](#monitoring--incident-response)

---

## 🎯 ภาพรวมความปลอดภัย

### ภัยคุกคามหลัก (Threat Model)

| ภัยคุกคาม | ระดับความเสี่ยง | การป้องกัน |
|-----------|-----------------|------------|
| SQL Injection | 🟡 Medium | Parameterized queries |
| XSS (Cross-Site Scripting) | 🟡 Medium | Input sanitization, CSP |
| CSRF (Cross-Site Request Forgery) | 🔴 High | **ยังไม่มี** - ต้องเพิ่ม |
| Session Hijacking | 🟡 Medium | Secure cookies, HTTPS |
| Unauthorized Access | 🟠 Medium-High | Authentication, authorization |
| File Upload Attacks | 🟡 Medium | File type validation |
| API Key Exposure | 🔴 High | Environment variables |
| Data Breach | 🟠 Medium-High | Encryption, access control |

### OWASP Top 10 Compliance

| OWASP Item | Status | Notes |
|------------|--------|-------|
| A01: Broken Access Control | ⚠️ Partial | Session-based auth implemented |
| A02: Cryptographic Failures | ⚠️ Partial | Password hashing, need HTTPS |
| A03: Injection | ✅ Good | Using parameterized queries |
| A04: Insecure Design | ⚠️ Partial | Need security reviews |
| A05: Security Misconfiguration | ⚠️ Partial | Debug mode off in prod |
| A06: Vulnerable Components | ✅ Good | Regular updates needed |
| A07: Authentication Failures | ⚠️ Partial | Need MFA, rate limiting |
| A08: Data Integrity Failures | ❌ Missing | No integrity checks |
| A09: Logging Failures | ⚠️ Partial | Basic logging implemented |
| A10: SSRF | ✅ Good | Not applicable |

---

## 🔑 Authentication & Authorization

### Current Implementation

```python
# Session-based authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check credentials
        if check_password_hash(stored_hash, password):
            session['username'] = username
            return redirect('/admin')
```

### ⚠️ Current Vulnerabilities

1. **ไม่มี Rate Limiting**
   - Brute force attacks ได้ง่าย
   - ไม่มีการ lock account

2. **ไม่มี Multi-Factor Authentication (MFA)**
   - พึ่งพา password อย่างเดียว

3. **Session Fixation**
   - ไม่ regenerate session ID หลัง login

### ✅ Recommendations

#### 1. เพิ่ม Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # login logic
    pass
```

#### 2. Session Regeneration

```python
from flask import session
import secrets

def regenerate_session():
    """Regenerate session ID to prevent fixation."""
    old_data = dict(session)
    session.clear()
    session.update(old_data)
    session.modified = True

@app.route('/login', methods=['POST'])
def login():
    if authenticate(username, password):
        regenerate_session()  # ป้องกัน session fixation
        session['username'] = username
```

#### 3. Account Lockout

```python
# ใน grade_form_db.py
def track_failed_login(username):
    """Track failed login attempts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Increment failed count
    cursor.execute('''
        UPDATE users 
        SET failed_attempts = failed_attempts + 1,
            last_failed_attempt = CURRENT_TIMESTAMP
        WHERE username = ?
    ''', (username,))
    
    # Lock if too many attempts
    cursor.execute('''
        UPDATE users 
        SET locked = 1
        WHERE username = ? AND failed_attempts >= 5
    ''', (username,))
    
    conn.commit()
    conn.close()
```

#### 4. Password Policy

```python
import re

def validate_password(password):
    """
    Validate password strength.
    - At least 8 characters
    - At least 1 uppercase
    - At least 1 lowercase
    - At least 1 digit
    - At least 1 special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special character"
    
    return True, "Password is strong"
```

---

## 🛡️ Data Protection

### 1. HTTPS (Transport Layer Security)

**❌ Current State**: HTTP only

**✅ Required**:

```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Redirect HTTP to HTTPS
    if ($scheme != "https") {
        return 301 https://$server_name$request_uri;
    }
}
```

### 2. Sensitive Data Storage

**❌ Current Issues**:
- API keys in .env (better than hardcoded, but still risky)
- No encryption at rest

**✅ Recommendations**:

```python
# Use environment variables (current - OK)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Better: Use secrets management (AWS Secrets Manager, Azure Key Vault)
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    """Get secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        logger.error(f"Failed to get secret: {e}")
        raise
```

### 3. Data Encryption at Rest

```python
from cryptography.fernet import Fernet

class DataEncryption:
    """Encrypt sensitive data before storing."""
    
    def __init__(self):
        # Load key from secure location
        self.key = os.getenv('ENCRYPTION_KEY').encode()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Usage
encryptor = DataEncryption()

# Encrypt before storing
encrypted_email = encryptor.encrypt(user_email)
save_to_db(encrypted_email)

# Decrypt when retrieving
decrypted_email = encryptor.decrypt(stored_email)
```

---

## 🔐 API Security

### 1. Gemini API Key Protection

**✅ Current (Good)**:
```python
# Stored in .env
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Not in version control
# .gitignore includes .env
```

**✅ Additional Protection**:

```python
# Validate API key before use
def validate_api_key(api_key):
    """Validate API key format and presence."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")
    
    if len(api_key) < 30:
        raise ValueError("Invalid GEMINI_API_KEY format")
    
    return True

# Never log API keys
logger.info(f"Using Gemini model: {model_name}")  # ✅ OK
logger.debug(f"API Key: {api_key}")  # ❌ NEVER DO THIS
```

### 2. Rate Limiting for External APIs

```python
from functools import wraps
import time
from collections import deque

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            # Check limit
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.time_window}s")
            
            # Add call
            self.calls.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper

# Usage
gemini_limiter = RateLimiter(max_calls=10, time_window=60)

@gemini_limiter
def call_gemini_api(prompt):
    # API call
    pass
```

---

## ✅ Input Validation

### 1. Grade Validation

**✅ Current (Good)**:
```python
grade_mapping = {
    'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
    'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'S': None
}

def validate_grade(grade):
    if grade not in grade_mapping:
        raise ValueError(f"Invalid grade: {grade}")
```

### 2. File Upload Validation

**⚠️ Current (Partial)**:
```python
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

**✅ Improved**:
```python
import magic  # python-magic library

def validate_uploaded_file(file):
    """Comprehensive file validation."""
    
    # 1. Check filename
    if not file or not file.filename:
        raise ValueError("No file provided")
    
    # 2. Secure filename
    filename = secure_filename(file.filename)
    
    # 3. Check extension
    if not allowed_file(filename):
        raise ValueError(f"File type not allowed: {filename}")
    
    # 4. Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes")
    
    # 5. Check MIME type (magic bytes)
    mime = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)
    
    allowed_mimes = ['text/csv', 'application/vnd.ms-excel', 
                     'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
    
    if mime not in allowed_mimes:
        raise ValueError(f"Invalid file type: {mime}")
    
    return filename
```

### 3. SQL Injection Prevention

**✅ Current (Good)**:
```python
# Using parameterized queries
cursor.execute(
    "SELECT * FROM users WHERE username = ?",
    (username,)
)
```

**❌ Never Do**:
```python
# String concatenation - VULNERABLE
cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")
```

---

## 🍪 Session Management

### Current Configuration

```python
# config.py
PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
SESSION_COOKIE_SECURE = False  # ⚠️ Should be True with HTTPS
SESSION_COOKIE_HTTPONLY = True  # ✅ Good
SESSION_COOKIE_SAMESITE = 'Lax'  # ✅ Good
```

### ✅ Recommendations

```python
# Production configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY'),  # Strong random key
    SESSION_COOKIE_SECURE=True,          # HTTPS only
    SESSION_COOKIE_HTTPONLY=True,        # No JavaScript access
    SESSION_COOKIE_SAMESITE='Strict',    # CSRF protection
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),  # Shorter timeout
    SESSION_REFRESH_EACH_REQUEST=True    # Auto-extend on activity
)

# Generate strong secret key
import secrets
SECRET_KEY = secrets.token_hex(32)
```

---

## 📁 File Upload Security

### Current Risks

1. ✅ Extension check (Good)
2. ⚠️ No content validation
3. ⚠️ No virus scanning
4. ⚠️ Stored in web-accessible directory

### ✅ Secure Implementation

```python
import os
import hashlib
from werkzeug.utils import secure_filename

def secure_file_upload(file, upload_dir):
    """Secure file upload handler."""
    
    # 1. Validate file
    filename = validate_uploaded_file(file)
    
    # 2. Generate unique filename (prevent overwrites)
    hash_suffix = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:8]
    safe_filename = f"{hash_suffix}_{filename}"
    
    # 3. Create non-web-accessible directory
    upload_path = os.path.join(upload_dir, safe_filename)
    os.makedirs(upload_dir, exist_ok=True)
    
    # 4. Save with restricted permissions
    file.save(upload_path)
    os.chmod(upload_path, 0o640)  # rw-r-----
    
    # 5. Scan for viruses (optional, requires ClamAV)
    # scan_file_for_viruses(upload_path)
    
    return safe_filename

# Usage
UPLOAD_FOLDER = '/var/app/uploads'  # Outside web root
safe_name = secure_file_upload(request.files['file'], UPLOAD_FOLDER)
```

---

## 🗄️ Database Security

### 1. Connection Security

```python
# SQLite permissions
os.chmod('data/grade_forms.db', 0o640)  # rw-r-----

# PostgreSQL (if migrating)
DATABASE_URL = os.getenv('DATABASE_URL')  # Never hardcode
conn = psycopg2.connect(
    DATABASE_URL,
    sslmode='require',  # Enforce SSL
    connect_timeout=10
)
```

### 2. Prepared Statements

```python
# ✅ Always use parameterized queries
cursor.execute(
    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
    (username, password_hash)
)

# ✅ For multiple values
cursor.executemany(
    "INSERT INTO grades (student_id, course_id, grade) VALUES (?, ?, ?)",
    [(sid, cid, grade) for ...]
)
```

### 3. Least Privilege

```sql
-- Create read-only user for reports
CREATE USER report_user WITH PASSWORD 'strong_password';
GRANT SELECT ON grade_forms TO report_user;

-- Create app user with limited permissions
CREATE USER app_user WITH PASSWORD 'strong_password';
GRANT SELECT, INSERT, UPDATE ON grade_forms TO app_user;
GRANT SELECT ON users TO app_user;
```

---

## 🚀 Deployment Security

### 1. Environment Separation

```bash
# Development
export FLASK_ENV=development
export DEBUG=True

# Production
export FLASK_ENV=production
export DEBUG=False
export SECRET_KEY=$(openssl rand -hex 32)
```

### 2. Secure Headers

```python
from flask_talisman import Talisman

# Add security headers
Talisman(app, 
    force_https=True,
    strict_transport_security=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", 'cdn.jsdelivr.net'],
        'style-src': ["'self'", "'unsafe-inline'", 'cdn.jsdelivr.net'],
    }
)
```

### 3. Firewall Rules

```bash
# Allow only necessary ports
ufw allow 80/tcp   # HTTP (redirect to HTTPS)
ufw allow 443/tcp  # HTTPS
ufw deny 5000/tcp  # Block direct Flask access
ufw enable
```

---

## 📊 Monitoring & Incident Response

### 1. Logging Security Events

```python
import logging
from logging.handlers import RotatingFileHandler

# Security event logger
security_logger = logging.getLogger('security')
security_handler = RotatingFileHandler(
    'logs/security.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10
)
security_logger.addHandler(security_handler)

# Log security events
def log_security_event(event_type, details):
    """Log security-related events."""
    security_logger.warning(f"[{event_type}] {details}")

# Usage
log_security_event('FAILED_LOGIN', f"User: {username}, IP: {request.remote_addr}")
log_security_event('UNAUTHORIZED_ACCESS', f"Path: {request.path}, User: {session.get('username')}")
```

### 2. Intrusion Detection

```python
from datetime import datetime, timedelta
from collections import defaultdict

class IntrusionDetector:
    """Detect suspicious activity."""
    
    def __init__(self):
        self.failed_logins = defaultdict(list)
        self.suspicious_ips = set()
    
    def record_failed_login(self, ip, username):
        """Record failed login attempt."""
        now = datetime.now()
        self.failed_logins[ip].append(now)
        
        # Clean old entries
        self.failed_logins[ip] = [
            t for t in self.failed_logins[ip]
            if now - t < timedelta(minutes=15)
        ]
        
        # Check threshold
        if len(self.failed_logins[ip]) > 10:
            self.suspicious_ips.add(ip)
            log_security_event('BRUTE_FORCE_DETECTED', f"IP: {ip}")
            return True
        
        return False
```

---

## 📝 Security Checklist

### Before Production Deployment

- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Set DEBUG=False
- [ ] Enable HTTPS (SSL/TLS)
- [ ] Set secure cookie flags
- [ ] Implement rate limiting
- [ ] Add CSRF protection
- [ ] Validate all inputs
- [ ] Scan dependencies for vulnerabilities
- [ ] Set up logging and monitoring
- [ ] Configure firewall rules
- [ ] Restrict file permissions
- [ ] Backup database regularly
- [ ] Test disaster recovery
- [ ] Document security procedures
- [ ] Train team on security

---

**จัดทำโดย**: Security Team  
**อัพเดทล่าสุด**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0  
**ทบทวนถัดไป**: 8 พฤษภาคม 2026
