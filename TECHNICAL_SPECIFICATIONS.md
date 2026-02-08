# 📋 ข้อกำหนดทางเทคนิค (Technical Specifications)

**โครงการ**: ระบบทำนายการจบการศึกษาของนักศึกษา  
**วันที่**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0

---

## 🎯 ภาพรวม (Overview)

ระบบทำนายการจบการศึกษาของนักศึกษาด้วย Machine Learning และ Gemini AI เพื่อช่วยในการระบุนักศึกษาที่มีความเสี่ยง และให้คำแนะนำที่เหมาะสม

---

## 💻 ความต้องการระบบ (System Requirements)

### 1. Hardware Requirements

#### Minimum (Development)
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Storage**: 10 GB
- **Network**: Broadband internet (for Gemini API)

#### Recommended (Production)
- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Storage**: 50+ GB SSD
- **Network**: High-speed internet with low latency

### 2. Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04+ (recommended)
- **Windows**: 10/11 or Windows Server 2019+
- **macOS**: 11.0+ (Big Sur or later)

#### Python
- **Version**: 3.8+ (tested on 3.9, 3.10, 3.11)
- **Virtual Environment**: venv or conda recommended

#### Database
- **SQLite**: 3.35+ (included with Python)
- **Optional**: PostgreSQL 13+ or MySQL 8+ for production

#### Web Server (Production)
- **Gunicorn**: 21.2.0+
- **Nginx**: 1.18+ (reverse proxy)

---

## 📦 Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 2.3.3 | Web framework |
| pandas | 2.0.3+ | Data processing |
| numpy | 1.26.0+ | Numerical computing |
| scikit-learn | 1.5.2+ | Machine learning |
| joblib | 1.3.2+ | Model serialization |
| Werkzeug | 2.3.7 | WSGI utilities |

### Machine Learning

| Package | Version | Purpose |
|---------|---------|---------|
| imbalanced-learn | 0.12.4+ | SMOTE |
| networkx | 3.1+ | Graph analysis |
| matplotlib | 3.7.2+ | Plotting |
| seaborn | 0.12.2+ | Statistical viz |

### AI Integration

| Package | Version | Purpose |
|---------|---------|---------|
| google-generativeai | 0.8.5+ | Gemini AI |

### Storage & Files

| Package | Version | Purpose |
|---------|---------|---------|
| boto3 | 1.28.57+ | S3-compatible storage |
| botocore | 1.31.57+ | AWS core library |
| openpyxl | 3.1.2+ | Excel support |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| python-dotenv | 1.0.0+ | Environment variables |
| gunicorn | 21.2.0+ | Production server |

---

## 🔧 Configuration Parameters

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=production              # development|production
SECRET_KEY=<random-secret-key>    # Session encryption key
HOST=0.0.0.0                      # Listen address
PORT=5000                         # Listen port
DEBUG=False                       # Debug mode (True|False)

# Gemini AI Configuration
GEMINI_API_KEY=<your-api-key>     # Required for Gemini features
GEMINI_MODEL_NAME=gemini-3-flash-preview  # Model name
GEMINI_MAX_FILE_SIZE_MB=5         # Max file size for Gemini

# Cloudflare R2 Configuration (Optional)
CLOUDFLARE_R2_ACCESS_KEY_ID=<key-id>
CLOUDFLARE_R2_SECRET_ACCESS_KEY=<secret-key>
CLOUDFLARE_R2_ENDPOINT_URL=<endpoint-url>
CLOUDFLARE_R2_BUCKET_NAME=<bucket-name>

# Admin Authentication
ADMIN_USERNAME=admin              # Admin username
ADMIN_PASSWORD=admin123           # Admin password (hashed in DB)
```

### Application Configuration (config.py)

```python
# File Upload
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Session
PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
SESSION_COOKIE_SECURE = False  # Set True for HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# Data Processing
min_students_for_training = 5
min_subjects_per_student = 3
default_encoding = 'utf-8-sig'

# Grade Mapping
grade_mapping = {
    'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
    'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'S': None
}

# Risk Thresholds
high_confidence_threshold = 0.8
medium_confidence_threshold = 0.6
high_fail_rate_threshold = 0.3
low_gpa_threshold = 2.0
```

---

## 🗄️ Database Specifications

### SQLite Schema

#### Table: users
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
```

#### Table: grade_forms
```sql
CREATE TABLE grade_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    form_data TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_grade_forms_student_id ON grade_forms(student_id);
CREATE INDEX idx_grade_forms_status ON grade_forms(status);
```

### Migration to PostgreSQL (Optional for Production)

```sql
-- users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- grade_forms table
CREATE TABLE grade_forms (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(255) NOT NULL,
    form_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_grade_forms_student_id ON grade_forms(student_id);
CREATE INDEX idx_grade_forms_status ON grade_forms(status);
CREATE INDEX idx_grade_forms_form_data ON grade_forms USING GIN(form_data);
```

---

## 📡 API Specifications

### HTTP Endpoints

#### 1. Authentication

**POST /login**
```json
Request:
{
  "username": "admin",
  "password": "admin123"
}

Response (Success):
{
  "status": "success",
  "redirect": "/admin"
}

Response (Error):
{
  "status": "error",
  "message": "Invalid credentials"
}
```

**GET /logout**
```
Response: Redirect to /login
```

#### 2. Model Training

**POST /train**
```
Request:
Content-Type: multipart/form-data
- file: CSV file
- model_name: "my_model" (optional)
- use_gemini: true|false (optional)
- gemini_goal: "analysis goal" (optional)

Response:
{
  "status": "success",
  "model_name": "my_model_20240208.pkl",
  "accuracy": 0.85,
  "training_time": 120.5,
  "models_trained": ["RandomForest", "GradientBoosting", ...],
  "best_model": "RandomForest",
  "gemini_analysis": {...} (if use_gemini=true)
}
```

#### 3. Prediction

**POST /predict**
```json
Request:
{
  "model_name": "my_model_20240208.pkl",
  "grades": {
    "CPE101": "A",
    "MAT101": "B+",
    "PHY101": "B"
  }
}

Response:
{
  "status": "success",
  "prediction": {
    "will_graduate": true,
    "probability": 0.85,
    "confidence": 0.85,
    "risk_level": "low"
  },
  "explanation": {
    "key_factors": [...],
    "strengths": [...],
    "obstacles": [...],
    "recommendations": [...]
  },
  "charts": {
    "feature_importance": {...},
    "gpa_trend": {...}
  }
}
```

**POST /predict_gemini**
```json
Request:
{
  "model_name": "my_model_20240208.pkl",
  "grades": {
    "CPE101": "A",
    "MAT101": "B+"
  },
  "gemini_goal": "Provide detailed analysis" (optional)
}

Response:
{
  "status": "success",
  "ml_prediction": {...},
  "gemini_analysis": {
    "analysis_markdown": "...",
    "risk_level": "low",
    "outcome_summary": {...},
    "key_metrics": [...],
    "recommendations": [...]
  }
}
```

#### 4. Model Management

**GET /api/models**
```json
Response:
{
  "local_models": [
    {
      "name": "model1.pkl",
      "size": 1024000,
      "created": "2024-02-08T10:00:00"
    }
  ],
  "r2_models": [...]
}
```

**POST /api/models/upload**
```
Request:
Content-Type: multipart/form-data
- file: .pkl file

Response:
{
  "status": "success",
  "filename": "uploaded_model.pkl"
}
```

**DELETE /api/models/{model_name}**
```json
Response:
{
  "status": "success",
  "message": "Model deleted"
}
```

---

## 📊 Data Formats

### 1. Input CSV Format (Long Format)

```csv
STUDENT_ID,COURSE_ID,GRADE,CREDIT,SEMESTER,YEAR
60001,CPE101,A,3,1,2020
60001,MAT101,B+,3,1,2020
60001,PHY101,C,3,1,2020
60002,CPE101,B,3,1,2020
```

**Required Columns:**
- STUDENT_ID: Unique student identifier
- COURSE_ID: Course code
- GRADE: Letter grade (A, B+, B, C+, C, D+, D, F, W, S)
- CREDIT: Course credits (integer)

**Optional Columns:**
- SEMESTER: Semester number
- YEAR: Academic year

### 2. Processed Format (Wide Format)

```csv
STUDENT_ID,CPE101,MAT101,PHY101,...,RESULT
60001,A,B+,C,...,1
60002,B,A,B+,...,1
60003,C,C,D,...,0
```

### 3. Model File Format

**File Extension:** .pkl (joblib)

**Contents:**
```python
{
    'model': <trained_model>,
    'scaler': <StandardScaler>,
    'grade_mapping': {...},
    'course_profiles': {...},
    'best_score': 0.85
}
```

---

## 🔐 Security Specifications

### 1. Authentication

**Method:** Session-based authentication
**Password Storage:** Werkzeug password hash (PBKDF2-SHA256)
**Session Duration:** 24 hours
**Cookie Security:**
- HttpOnly: True
- SameSite: Lax
- Secure: False (set True for HTTPS)

### 2. Authorization

**Roles:**
- `admin`: Full access
- `user`: Limited access (view only)

**Protected Routes:**
- `/admin/*`: Requires admin role
- `/train`: Requires admin role
- `/model_management`: Requires admin role

### 3. Input Validation

**File Uploads:**
- Extensions: csv, xlsx, xls only
- Max size: 100MB
- Filename sanitization: `secure_filename()`

**Grade Validation:**
- Must match grade_mapping keys
- Invalid grades rejected

### 4. API Security

**Gemini API Key:**
- Stored in .env file
- Never committed to version control
- Loaded via python-dotenv

**R2 Credentials:**
- Stored in .env file
- Never exposed in responses

---

## ⚡ Performance Specifications

### Response Time Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Page Load | < 2s | First contentful paint |
| Login | < 500ms | Database query |
| Prediction (Normal) | < 1s | ML inference |
| Prediction (Gemini) | < 5s | API call latency |
| Model Training | 1-5 min | Depends on data size |
| File Upload | < 10s | For 20MB file |

### Scalability Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Concurrent Users | 100+ | With proper server |
| Requests/sec | 50+ | Normal operations |
| Database Size | 10GB+ | SQLite limit |
| Model Size | 100MB+ | Per model file |

### Resource Usage

| Resource | Development | Production |
|----------|-------------|------------|
| CPU | 10-20% | 30-50% |
| RAM | 500MB | 2-4GB |
| Disk I/O | Low | Medium |
| Network | Minimal | High (Gemini) |

---

## 🧪 Testing Specifications

### Unit Tests
- Coverage target: > 70%
- Framework: pytest
- Mocking: unittest.mock

### Integration Tests
- API endpoint testing
- Database operations
- File upload/download

### End-to-End Tests
- Browser automation: Selenium/Playwright
- User workflows
- UI validation

### Performance Tests
- Load testing: Locust/JMeter
- Stress testing
- Endurance testing

---

## 🚀 Deployment Specifications

### Development Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your values

# Initialize database
python grade_form_db.py

# Run development server
python app.py
```

### Production Deployment (Gunicorn)

```bash
# Install production dependencies
pip install -r requirements.txt gunicorn

# Set up environment
export FLASK_ENV=production
export SECRET_KEY="your-production-secret"

# Run with Gunicorn
gunicorn app:app \
  --bind 0.0.0.0:5000 \
  --workers 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Production Deployment (Nginx + Gunicorn)

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 100M;
    }

    location /static {
        alias /path/to/static;
        expires 30d;
    }
}
```

### Docker Deployment (Optional)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

---

## 📈 Monitoring & Logging

### Logging Levels

```python
# Production
logging.basicConfig(level=logging.INFO)

# Development
logging.basicConfig(level=logging.DEBUG)
```

### Log Format

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Metrics to Monitor

- Request rate
- Error rate
- Response time (p50, p95, p99)
- CPU/Memory usage
- Database connections
- Gemini API calls
- Model prediction latency

### Recommended Tools

- **Logging**: Python logging module
- **Monitoring**: Prometheus + Grafana
- **Error Tracking**: Sentry
- **APM**: New Relic or DataDog

---

## 🔄 Backup & Recovery

### Backup Strategy

**Database:**
- Frequency: Daily
- Retention: 30 days
- Method: SQLite file copy

**Models:**
- Frequency: After each training
- Retention: Keep last 10 versions
- Method: Cloud storage (R2)

**Configuration:**
- Frequency: On change
- Retention: Version control
- Method: Git

### Recovery Procedures

**Database Corruption:**
1. Stop application
2. Restore from latest backup
3. Verify data integrity
4. Restart application

**Model Loss:**
1. Download from R2 storage
2. Verify model integrity
3. Update local cache

---

## 📝 Compliance & Standards

### Code Standards
- **Style Guide**: PEP 8
- **Documentation**: NumPy docstring format
- **Type Hints**: Python 3.8+ type annotations

### Security Standards
- OWASP Top 10 awareness
- Secure password storage
- Input validation
- API key protection

### Data Privacy
- No personal data in logs
- Secure data transmission
- Data retention policies

---

**เอกสารจัดทำโดย**: Technical Team  
**อัพเดทล่าสุด**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0
