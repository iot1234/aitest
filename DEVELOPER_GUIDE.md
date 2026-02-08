# 👨‍💻 คู่มือสำหรับนักพัฒนา (Developer Guide)

**โครงการ**: ระบบทำนายการจบการศึกษาของนักศึกษา  
**วันที่**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0

---

## 📚 สารบัญ

1. [การติดตั้งสภาพแวดล้อมการพัฒนา](#การติดตั้งสภาพแวดล้อมการพัฒนา)
2. [โครงสร้างโค้ด](#โครงสร้างโค้ด)
3. [การพัฒนาฟีเจอร์ใหม่](#การพัฒนาฟีเจอร์ใหม่)
4. [การทดสอบ](#การทดสอบ)
5. [Debugging Tips](#debugging-tips)
6. [Best Practices](#best-practices)
7. [Common Issues](#common-issues)

---

## 🔧 การติดตั้งสภาพแวดล้อมการพัฒนา

### 1. Prerequisites

```bash
# ตรวจสอบเวอร์ชัน Python
python --version  # ควรเป็น 3.8+

# ติดตั้ง virtualenv (ถ้ายังไม่มี)
pip install virtualenv
```

### 2. Clone Repository

```bash
git clone https://github.com/iot1234/aitest.git
cd aitest
```

### 3. สร้าง Virtual Environment

```bash
# สร้าง venv
python -m venv venv

# เปิดใช้งาน venv
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. ติดตั้ง Dependencies

```bash
# ติดตั้ง packages
pip install -r requirements.txt

# ตรวจสอบการติดตั้ง
pip list
```

### 5. ตั้งค่า Environment Variables

```bash
# คัดลอก .env.example
cp .env.example .env

# แก้ไข .env
# ใส่ค่าจริงแทน placeholders
nano .env  # หรือ vim, code, etc.
```

### 6. Initialize Database

```bash
# รัน script สร้าง database
python -c "import grade_form_db; grade_form_db.init_db()"

# หรือเปิด Python shell
python
>>> import grade_form_db
>>> grade_form_db.init_db()
>>> exit()
```

### 7. รันแอปพลิเคชัน

```bash
# รันในโหมด development
python app.py

# หรือใช้ Flask CLI
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

### 8. ตรวจสอบการติดตั้ง

```bash
# เปิด browser ไปที่
http://localhost:5000

# ควรเห็นหน้าแรกของระบบ
```

---

## 📁 โครงสร้างโค้ด

### ไฟล์และโฟลเดอร์สำคัญ

```
aitest/
│
├── 📄 app.py                    # Flask app หลัก (8,541 บรรทัด)
├── 📄 config.py                 # Configuration settings
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env.example              # ตัวอย่าง environment variables
│
├── 📂 Service Modules
│   ├── model_trainer.py         # ML model training
│   ├── advanced_training.py     # Feature engineering
│   ├── explainable_ai.py        # Explainable AI
│   ├── gemini_service.py        # Gemini API service
│   ├── grade_form_db.py         # Database operations
│   └── s3_storage.py            # Cloud storage
│
├── 📂 templates/                # HTML templates
│   ├── index.html               # หน้าแรก
│   ├── login.html               # หน้า login
│   ├── curriculum_prediction_form.html
│   ├── model_management.html
│   └── ...
│
├── 📂 data/                     # Data directory
│   └── grade_forms.db           # SQLite database
│
├── 📂 models/                   # ML models
│   └── *.pkl                    # Trained models
│
└── 📂 uploads/                  # Temporary uploads
```

### โมดูลหลัก (Core Modules)

#### 1. app.py
```python
# หน้าที่หลัก
- Flask application setup
- Route handlers
- Business logic
- Data preprocessing
- Model training orchestration
- Prediction logic

# ฟังก์ชันสำคัญ
- preprocess_tan1_data()      # ประมวลผล CSV
- train_model()                # เทรนโมเดล
- predict_student()            # ทำนายผล
- predict_with_gemini()        # ทำนายด้วย Gemini
```

#### 2. model_trainer.py
```python
# Classes
class AdvancedModelTrainer:
    def __init__(self)         # Initialize models
    def train(df)              # Train models
    def save_model(filepath)   # Save to disk
    def load_model(filepath)   # Load from disk

class PredictionService:
    def __init__(model_path)   # Load model
    def predict(student_grades) # Make prediction
```

#### 3. advanced_training.py
```python
# Classes
class AdvancedFeatureEngineer:
    def prepare_training_data(df)  # Feature engineering
    def create_course_dna()         # Course analysis
    def calculate_metrics()         # Student metrics

class ContextAwarePredictor:
    def predict(student_data)       # Context-aware prediction
    def analyze_pathway()           # Analyze learning path
```

#### 4. explainable_ai.py
```python
# Classes
class ExplainablePredictor:
    def explain_prediction()        # Explain results
    def analyze_key_factors()       # Key factors
    def generate_recommendations()  # Recommendations
    def suggest_graduation_path()   # Suggest path
```

#### 5. gemini_service.py
```python
# Classes
class GeminiService:
    def __init__()                      # Initialize Gemini
    def call_gemini_structured()        # Call API
    def analyze_dataset()               # Analyze training data
    def predict_student_outcome()       # Predict with Gemini
```

---

## 🆕 การพัฒนาฟีเจอร์ใหม่

### ขั้นตอนการเพิ่มฟีเจอร์

#### 1. วางแผนและออกแบบ

```markdown
# ตัวอย่าง: เพิ่มฟีเจอร์ส่ง Email แจ้งเตือน

## Requirements
- ส่ง email เมื่อทำนายว่านักศึกษามีความเสี่ยง
- รองรับ SMTP server
- Template สำหรับ email

## Design
- สร้างไฟล์ email_service.py
- เพิ่ม email templates
- เพิ่ม config ใน config.py
- เพิ่ม route /send_alert
```

#### 2. เขียนโค้ด

**สร้างไฟล์ใหม่: email_service.py**

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, smtp_server, smtp_port, username, password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert(self, to_email, student_data, prediction):
        """Send risk alert email to student or advisor."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = 'การแจ้งเตือน: นักศึกษามีความเสี่ยง'
            
            # Create email body
            body = self._create_email_body(student_data, prediction)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_email_body(self, student_data, prediction):
        """Create HTML email body."""
        return f"""
        <html>
        <body>
            <h2>การแจ้งเตือนนักศึกษาเสี่ยง</h2>
            <p>รหัสนักศึกษา: {student_data.get('student_id')}</p>
            <p>ความเสี่ยง: {prediction.get('risk_level')}</p>
            <p>โอกาสจบการศึกษา: {prediction.get('probability')*100:.1f}%</p>
        </body>
        </html>
        """
```

**เพิ่มใน config.py**

```python
# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'username': os.getenv('EMAIL_USERNAME'),
    'password': os.getenv('EMAIL_PASSWORD')
}
```

**เพิ่ม route ใน app.py**

```python
from email_service import EmailService

# Initialize email service
email_service = EmailService(
    config.EMAIL_CONFIG['smtp_server'],
    config.EMAIL_CONFIG['smtp_port'],
    config.EMAIL_CONFIG['username'],
    config.EMAIL_CONFIG['password']
)

@app.route('/send_alert', methods=['POST'])
def send_alert():
    """Send alert email for at-risk student."""
    try:
        data = request.get_json()
        student_data = data.get('student_data')
        prediction = data.get('prediction')
        to_email = data.get('email')
        
        success = email_service.send_alert(to_email, student_data, prediction)
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send email'}), 500
            
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
```

#### 3. เพิ่ม Dependencies

```bash
# เพิ่มใน requirements.txt (ถ้าจำเป็น)
# สำหรับ email ใช้ built-in smtplib ได้

# อัพเดท .env.example
echo "SMTP_SERVER=smtp.gmail.com" >> .env.example
echo "SMTP_PORT=587" >> .env.example
echo "EMAIL_USERNAME=your_email@gmail.com" >> .env.example
echo "EMAIL_PASSWORD=your_app_password" >> .env.example
```

#### 4. เทสต์

```python
# สร้างไฟล์ test_email_service.py
import unittest
from email_service import EmailService

class TestEmailService(unittest.TestCase):
    def setUp(self):
        self.service = EmailService(
            'smtp.gmail.com', 587, 
            'test@gmail.com', 'password'
        )
    
    def test_create_email_body(self):
        student_data = {'student_id': '60001'}
        prediction = {'risk_level': 'high', 'probability': 0.3}
        
        body = self.service._create_email_body(student_data, prediction)
        
        self.assertIn('60001', body)
        self.assertIn('high', body)
        self.assertIn('30.0%', body)

if __name__ == '__main__':
    unittest.main()
```

#### 5. Documentation

```markdown
# เพิ่มใน README หรือ DEVELOPER_GUIDE

## Email Alert Feature

### Configuration
Set the following in `.env`:
- SMTP_SERVER
- SMTP_PORT
- EMAIL_USERNAME
- EMAIL_PASSWORD

### Usage
POST /send_alert
{
  "student_data": {...},
  "prediction": {...},
  "email": "student@example.com"
}
```

---

## 🧪 การทดสอบ

### Unit Testing

**ติดตั้ง pytest**

```bash
pip install pytest pytest-cov
```

**สร้างโครงสร้างทดสอบ**

```
tests/
├── __init__.py
├── test_model_trainer.py
├── test_gemini_service.py
├── test_explainable_ai.py
└── test_app.py
```

**ตัวอย่าง Unit Test**

```python
# tests/test_model_trainer.py
import pytest
from model_trainer import AdvancedModelTrainer
import pandas as pd

class TestAdvancedModelTrainer:
    def setup_method(self):
        """Setup before each test."""
        self.trainer = AdvancedModelTrainer()
    
    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.models is not None
        assert 'RandomForest' in self.trainer.models
    
    def test_train_with_valid_data(self):
        """Test training with valid dataset."""
        # Create sample data
        df = pd.DataFrame({
            'CPE101': ['A', 'B', 'C'],
            'MAT101': ['B', 'C', 'D'],
            'RESULT': [1, 1, 0]
        })
        
        results = self.trainer.train(df)
        
        assert results is not None
        assert 'RandomForest' in results
        assert results['RandomForest']['accuracy'] > 0
    
    def test_save_and_load_model(self, tmp_path):
        """Test model save/load functionality."""
        # Train a simple model
        df = pd.DataFrame({
            'CPE101': ['A', 'B', 'C', 'D', 'F'],
            'RESULT': [1, 1, 1, 0, 0]
        })
        self.trainer.train(df)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        self.trainer.save_model(str(model_path))
        
        # Load model
        new_trainer = AdvancedModelTrainer()
        new_trainer.load_model(str(model_path))
        
        assert new_trainer.best_model is not None

# รัน tests
# pytest tests/ -v --cov=.
```

### Integration Testing

```python
# tests/test_app.py
import pytest
from app import app

@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test index page loads."""
    response = client.get('/')
    assert response.status_code == 200

def test_login_required(client):
    """Test protected routes require login."""
    response = client.get('/admin')
    assert response.status_code == 302  # Redirect to login

def test_prediction_endpoint(client):
    """Test prediction endpoint."""
    # Mock data
    data = {
        'model_name': 'test_model.pkl',
        'grades': {
            'CPE101': 'A',
            'MAT101': 'B+'
        }
    }
    
    response = client.post('/predict', json=data)
    
    # Should return 200 or error if model doesn't exist
    assert response.status_code in [200, 400, 404]
```

### Running Tests

```bash
# รัน all tests
pytest

# รัน with coverage
pytest --cov=. --cov-report=html

# รัน specific test file
pytest tests/test_model_trainer.py

# รัน specific test
pytest tests/test_model_trainer.py::TestAdvancedModelTrainer::test_initialization

# รัน with verbose output
pytest -v
```

---

## 🐛 Debugging Tips

### 1. Enable Debug Mode

```python
# ใน app.py
app.config['DEBUG'] = True

# หรือใน .env
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### 2. Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ใช้ใน code
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 3. Python Debugger (pdb)

```python
# เพิ่มใน code ที่ต้องการ debug
import pdb; pdb.set_trace()

# หรือใช้ breakpoint (Python 3.7+)
breakpoint()

# คำสั่ง pdb
# n - next line
# s - step into
# c - continue
# p variable - print variable
# l - list code
# q - quit
```

### 4. Flask Debug Toolbar

```bash
# ติดตั้ง
pip install flask-debugtoolbar

# เพิ่มใน app.py
from flask_debugtoolbar import DebugToolbarExtension

app.config['SECRET_KEY'] = 'secret'
toolbar = DebugToolbarExtension(app)
```

### 5. Debugging Gemini API Calls

```python
# ใน gemini_service.py
def call_gemini_structured(self, task_name, payload):
    # Log request
    logger.debug(f"Gemini request: {task_name}")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = self.model.generate_content(prompt)
        
        # Log response
        logger.debug(f"Gemini response: {response.text}")
        
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
```

---

## ✅ Best Practices

### 1. Code Style

```python
# ใช้ PEP 8
# ติดตั้ง linter
pip install flake8 black

# รัน linter
flake8 app.py

# Auto-format
black app.py
```

### 2. Type Hints

```python
from typing import Dict, List, Optional, Any

def predict_student(
    grades: Dict[str, str],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict student graduation.
    
    Args:
        grades: Dictionary of course_code: grade
        model_name: Optional model name to use
    
    Returns:
        Prediction result dictionary
    """
    pass
```

### 3. Error Handling

```python
# ใช้ try-except อย่างเหมาะสม
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Expected error: {e}")
    # Handle gracefully
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    # Re-raise or handle
    raise
```

### 4. Configuration Management

```python
# ไม่ hard-code values
# ❌ Bad
api_key = "AIzaSyAjQ9n0NbvXBwxtyjJK5JSXP5fCknNWw1w"

# ✅ Good
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")
```

### 5. Database Operations

```python
# ใช้ context manager
with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
# connection auto-closed

# ใช้ parameterized queries
cursor.execute(
    "SELECT * FROM users WHERE username = ?",
    (username,)
)
```

---

## ⚠️ Common Issues

### Issue 1: Import Errors

```bash
# Problem
ModuleNotFoundError: No module named 'flask'

# Solution
pip install -r requirements.txt
# ตรวจสอบว่าใช้ virtual environment
```

### Issue 2: Gemini API Error

```bash
# Problem
Gemini API error: 404 models/gemini-1.5-flash is not found

# Solution
# แก้ไข GEMINI_MODEL_NAME ใน .env
GEMINI_MODEL_NAME=gemini-3-flash-preview
```

### Issue 3: Database Locked

```bash
# Problem
sqlite3.OperationalError: database is locked

# Solution
# ปิดการเชื่อมต่อทั้งหมด
# ใช้ WAL mode
conn = sqlite3.connect('database.db')
conn.execute('PRAGMA journal_mode=WAL')
```

### Issue 4: Model File Not Found

```bash
# Problem
FileNotFoundError: Model file not found

# Solution
# ตรวจสอบ path
import os
print(os.path.abspath('models/'))
# เทรนโมเดลใหม่
```

### Issue 5: Memory Error

```bash
# Problem
MemoryError: Unable to allocate array

# Solution
# ลดขนาดข้อมูล
# ใช้ chunking
df = pd.read_csv('large_file.csv', chunksize=1000)
for chunk in df:
    process(chunk)
```

---

## 📞 Getting Help

### Resources
- **Documentation**: อ่าน docs ในโปรเจกต์
- **GitHub Issues**: https://github.com/iot1234/aitest/issues
- **Stack Overflow**: ค้นหาปัญหาที่คล้ายกัน
- **Flask Docs**: https://flask.palletsprojects.com/
- **scikit-learn Docs**: https://scikit-learn.org/

### Contact
- **Email**: developer@example.com
- **Slack**: #aitest-dev
- **Teams**: AI Development Team

---

**จัดทำโดย**: Development Team  
**อัพเดทล่าสุด**: 8 กุมภาพันธ์ 2026  
**เวอร์ชัน**: 1.0
