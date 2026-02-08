# 📊 การวิเคราะห์ระบบอย่างละเอียด (Comprehensive System Analysis)

**วันที่วิเคราะห์**: 8 กุมภาพันธ์ 2026  
**เวอร์ชันระบบ**: 1.0 - Gemini AI Integration  
**ผู้วิเคราะห์**: AI System Analyzer  

---

## 🎯 บทสรุปผู้บริหาร (Executive Summary)

ระบบนี้เป็น **ระบบทำนายการจบการศึกษาของนักศึกษา** (Student Graduation Prediction System) ที่พัฒนาด้วย Python Flask Framework และผสานรวมกับ Google Gemini AI เพื่อให้การวิเคราะห์ที่ลึกซึ้งและคำแนะนำที่เข้าใจได้ง่าย

### ✅ จุดเด่นของระบบ
- ✨ ใช้ Machine Learning หลายโมเดลเพื่อทำนายผลการศึกษา
- 🤖 ผสานรวม Gemini AI สำหรับการวิเคราะห์เชิงลึก
- 📊 มีระบบอธิบายการทำนายแบบ Explainable AI
- 🎓 รองรับการจัดการข้อมูลนักศึกษาและหลักสูตร
- 📈 มีการวิเคราะห์แบบกราฟและภาพต่างๆ
- 🔐 มีระบบ Authentication สำหรับผู้ดูแลระบบ

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

### 1. โครงสร้างหลัก (Core Structure)

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                        │
│                    (Flask Templates)                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                  Flask Application                      │
│                     (app.py)                           │
└───┬────────┬────────┬────────┬────────┬────────┬───────┘
    │        │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Config│ │Model │ │Gemini│ │Expl. │ │Adv.  │ │Grade │
│      │ │Train │ │Serv. │ │AI    │ │Train │ │DB    │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
    │        │        │        │        │        │
    └────────┴────────┴────────┴────────┴────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────┐           ┌──────▼──────┐
    │ Database  │           │  File       │
    │ (SQLite)  │           │  Storage    │
    └───────────┘           └─────────────┘
```

### 2. ไฟล์หลักและหน้าที่ (Core Files & Functions)

#### **app.py** (8,541 บรรทัด)
- 🎯 **หน้าที่**: ไฟล์หลักของแอปพลิเคชัน Flask
- **ความสามารถ**:
  - Route handling สำหรับ UI ทั้งหมด
  - การประมวลผลข้อมูล TAN1 format (Long to Wide format)
  - การเทรนโมเดล ML (Random Forest, Gradient Boosting, etc.)
  - การทำนายผลการศึกษา (Normal & Gemini modes)
  - การจัดการโมเดล (upload, download, delete)
  - การสร้างกราฟและการวิเคราะห์
  - Session management และ authentication

#### **config.py** (656 บรรทัด)
- 🎯 **หน้าที่**: การตั้งค่าระบบทั้งหมด
- **การตั้งค่าสำคัญ**:
  - `SECRET_KEY`: สำหรับ session security
  - `MAX_CONTENT_LENGTH`: 100MB สำหรับอัปโหลดไฟล์
  - `grade_mapping`: การแปลงเกรดเป็นตัวเลข (A=4.0, B+=3.5, ...)
  - `SUBJECT_CATEGORIES`: หมวดวิชา (คณิต, โปรแกรม, วิศวกรรม, ระบบฯ, ศึกษาทั่วไป)
  - `risk_levels`: ค่า threshold สำหรับการประเมินความเสี่ยง

#### **gemini_service.py** (136 บรรทัด)
- 🎯 **หน้าที่**: บริการ Gemini AI
- **ฟังก์ชันหลัก**:
  - `call_gemini_structured()`: เรียก Gemini API พร้อม structured prompt
  - `analyze_dataset()`: วิเคราะห์ชุดข้อมูลเทรนนิ่ง
  - `predict_student_outcome()`: ทำนายผลการศึกษาด้วย Gemini
- **การตั้งค่า**:
  - Model: `gemini-3-flash-preview` (configurable)
  - Response format: JSON
  - Safety settings: All harm categories blocked at BLOCK_NONE

#### **model_trainer.py** (254 บรรทัด)
- 🎯 **หน้าที่**: เทรนและจัดการโมเดล ML
- **โมเดลที่รองรับ**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
- **คุณสมบัติ**:
  - Feature engineering ด้วย `AdvancedFeatureEngineer`
  - SMOTE สำหรับจัดการข้อมูลไม่สมดุล
  - Automatic model selection (เลือกโมเดลที่ดีที่สุด)
  - Model persistence (บันทึกและโหลดโมเดล)

#### **advanced_training.py** (2,468 บรรทัด)
- 🎯 **หน้าที่**: Feature engineering ขั้นสูง
- **คลาสหลัก**:
  - `AdvancedFeatureEngineer`: สร้าง features ที่ซับซ้อน
  - `ContextAwarePredictor`: ทำนายแบบมีบริบท
- **Features ที่สร้าง**:
  - Course DNA (ความยาก, failure rate)
  - Student trajectory (เส้นทางการเรียน)
  - GPA trends
  - Subject category performance
  - Prerequisite course analysis

#### **explainable_ai.py** (928 บรรทัด)
- 🎯 **หน้าที่**: อธิบายการทำนายของ AI
- **ความสามารถ**:
  - Feature importance analysis
  - คำอธิบายว่าทำไมทำนายแบบนี้
  - ระบุอุปสรรคและจุดแข็ง
  - แนะนำแนวทางปรับปรุง
  - ทำนายสถานการณ์ในอนาคต
  - แนะนำเส้นทางการจบการศึกษา

#### **grade_form_db.py** (317 บรรทัด)
- 🎯 **หน้าที่**: จัดการฐานข้อมูล SQLite
- **ตารางข้อมูล**:
  - `users`: ข้อมูลผู้ใช้และ authentication
  - `grade_forms`: ฟอร์มกรอกเกรดของนักศึกษา
- **ฟังก์ชัน**:
  - CRUD operations
  - User authentication
  - Form status management

#### **s3_storage.py** (74 บรรทัด)
- 🎯 **หน้าที่**: จัดการ Cloudflare R2 storage
- **ความสามารถ**:
  - Upload/download models to R2
  - List available models
  - Delete models from cloud

---

## 💾 โครงสร้างข้อมูล (Data Structure)

### 1. รูปแบบข้อมูลนักศึกษา (Student Data Format)

#### **Long Format** (TAN1.csv)
```csv
STUDENT_ID,COURSE_ID,GRADE,CREDIT,SEMESTER,YEAR
60001,CPE101,A,3,1,2020
60001,MAT101,B+,3,1,2020
60002,CPE101,B,3,1,2020
```

#### **Wide Format** (หลังจาก preprocessing)
```csv
STUDENT_ID,CPE101,MAT101,PHY101,...,RESULT
60001,A,B+,C,...,1
60002,B,A,B+,...,1
```

### 2. เกณฑ์การจบการศึกษา (Graduation Criteria)

```python
RESULT = 1 if (GPA >= 2.00 AND TOTAL_CREDITS >= 136) else 0
```

### 3. Grade Mapping

| เกรด | คะแนน | หมายเหตุ |
|------|-------|---------|
| A    | 4.0   | ดีเยี่ยม |
| B+   | 3.5   | ดีมาก |
| B    | 3.0   | ดี |
| C+   | 2.5   | ค่อนข้างดี |
| C    | 2.0   | พอใช้ |
| D+   | 1.5   | อ่อน |
| D    | 1.0   | อ่อนมาก |
| F    | 0.0   | ตก |
| W    | 0.0   | ถอน |
| S    | -     | ไม่คิดเกรด |

---

## 🔄 กระบวนการทำงาน (Workflows)

### 1. การเทรนโมเดล (Model Training Workflow)

```
1. อัปโหลดไฟล์ CSV
   ↓
2. Preprocess Data (Long → Wide format)
   ↓
3. Feature Engineering
   - Course DNA analysis
   - Student trajectory
   - GPA calculation
   ↓
4. Train Multiple Models
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - SVM
   ↓
5. Select Best Model (highest accuracy)
   ↓
6. Save Model (with scaler & feature engineer)
   ↓
7. Optional: Gemini Analysis
   - Dataset quality
   - Recommendations
```

### 2. การทำนาย (Prediction Workflow)

#### **โหมดปกติ (Normal AI)**
```
1. กรอกเกรดนักศึกษา
   ↓
2. Feature Extraction
   ↓
3. ML Model Prediction
   ↓
4. Explainable AI Analysis
   - ปัจจัยหลัก
   - คำแนะนำ
   - เส้นทางการจบ
   ↓
5. แสดงผลพร้อมกราฟ
```

#### **โหมด Gemini**
```
1. กรอกเกรดนักศึกษา
   ↓
2. Feature Extraction
   ↓
3. ML Model Prediction (baseline)
   ↓
4. Gemini AI Analysis
   - Deep analysis
   - Context-aware recommendations
   - Natural language explanations
   ↓
5. Combine ML + Gemini insights
   ↓
6. แสดงผลพร้อมกราฟและคำแนะนำ
```

---

## 🌟 คุณสมบัติหลัก (Key Features)

### 1. **การทำนายแบบ Dual-Mode**
- **Normal AI**: รวดเร็ว, ใช้ ML model ที่เทรนไว้
- **Gemini AI**: ละเอียด, วิเคราะห์เชิงลึก, ให้คำแนะนำที่เข้าใจง่าย

### 2. **Explainable AI**
- อธิบายปัจจัยที่มีผลต่อการทำนาย
- แสดง Feature Importance
- แนะนำแนวทางแก้ไข
- ทำนายสถานการณ์ในอนาคต

### 3. **Advanced Feature Engineering**
- Course DNA: วิเคราะห์ความยากของวิชา
- Student Trajectory: ติดตามพัฒนาการเรียน
- Category Performance: ประสิทธิภาพตามหมวดวิชา
- Prerequisite Analysis: วิเคราะห์วิชาบังคับก่อน

### 4. **Multi-Model Ensemble**
- รองรับโมเดล ML หลายประเภท
- เลือกโมเดลที่ดีที่สุดอัตโนมัติ
- SMOTE สำหรับจัดการข้อมูลไม่สมดุล

### 5. **Visualization & Reporting**
- กราฟแสดงผล (Chart.js)
- Feature importance charts
- GPA trend analysis
- Risk assessment visualization

### 6. **Data Management**
- อัปโหลด CSV/Excel
- ระบบ Grade Form (SQLite)
- Cloud storage support (R2)
- Model versioning

---

## 🔐 ความปลอดภัย (Security)

### 1. **Authentication & Authorization**
```python
# Session-based authentication
@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
```

### 2. **File Upload Security**
- ตรวจสอบ file extension: `{'csv', 'xlsx', 'xls'}`
- จำกัดขนาดไฟล์: 100MB
- ใช้ `secure_filename()` สำหรับชื่อไฟล์

### 3. **API Key Management**
- Gemini API Key เก็บใน `.env` file
- ไม่ commit API keys ใน git
- Environment variable override support

### 4. **Database Security**
- Password hashing ด้วย `werkzeug.security`
- SQLite database สำหรับ production เล็กๆ
- Session cookie security:
  - `HTTPONLY=True`
  - `SAMESITE='Lax'`

### 5. **Input Validation**
- ตรวจสอบ required columns
- Grade validation ตาม grade_mapping
- Credit และ semester validation

---

## 📦 Dependencies & Technology Stack

### **Backend**
- **Flask 2.3.3**: Web framework
- **Python 3.x**: Core language
- **Pandas 2.0.3+**: Data processing
- **NumPy 1.26.0+**: Numerical computing
- **scikit-learn 1.5.2+**: Machine learning
- **imbalanced-learn 0.12.4+**: SMOTE
- **joblib 1.3.2+**: Model persistence

### **AI & ML**
- **google-generativeai 0.8.5+**: Gemini AI
- **NetworkX 3.1+**: Graph analysis
- **Matplotlib 3.7.2+**: Plotting
- **Seaborn 0.12.2+**: Statistical visualization

### **Storage & Database**
- **SQLite**: Database (via grade_form_db.py)
- **boto3 1.28.57+**: AWS S3 compatible storage
- **openpyxl 3.1.2+**: Excel file support

### **Frontend**
- **Chart.js**: Data visualization
- **Bootstrap**: UI framework
- **jQuery**: DOM manipulation

### **Deployment**
- **Gunicorn 21.2.0+**: Production server
- **Vercel**: Deployment platform (vercel.json)
- **Heroku**: Alternative (Procfile)

---

## 🚀 การติดตั้งและใช้งาน (Installation & Usage)

### 1. **ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **ตั้งค่า Environment Variables**
```bash
# คัดลอก .env.example เป็น .env
cp .env.example .env

# แก้ไข .env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-3-flash-preview
SECRET_KEY=your_secret_key
```

### 3. **รันแอปพลิเคชัน**
```bash
# Development
python app.py

# Production
gunicorn app:app --bind 0.0.0.0:5000
```

### 4. **เข้าถึงระบบ**
- URL: `http://localhost:5000`
- Admin Login: ตามที่ตั้งใน `.env`

---

## 📊 การประเมินประสิทธิภาพ (Performance Metrics)

### 1. **Model Performance**
- Accuracy: ≥ 80% (ตามโมเดลที่เลือก)
- F1 Score: ใช้สำหรับข้อมูลไม่สมดุล
- Precision & Recall: ติดตามทั้งคู่

### 2. **Response Time**
- การทำนายปกติ: < 1 วินาที
- การทำนายด้วย Gemini: 2-5 วินาที (ขึ้นกับ API)
- การเทรนโมเดล: 1-2 นาที (ขึ้นกับขนาดข้อมูล)

### 3. **Data Processing**
- รองรับไฟล์ CSV ขนาดใหญ่ (tested: 20MB+)
- Preprocessing: < 10 วินาที
- Feature engineering: ขึ้นกับจำนวนนักศึกษา

---

## ⚠️ ข้อจำกัดและความเสี่ยง (Limitations & Risks)

### 1. **ข้อจำกัดทางเทคนิค**
- ❌ ต้องการข้อมูลขั้นต่ำ 5 นักศึกษาสำหรับ training
- ❌ ต้องการ Gemini API key สำหรับโหมด Gemini
- ❌ SQLite ไม่เหมาะกับ concurrent writes มาก
- ❌ ไม่มี real-time collaboration

### 2. **ความเสี่ยงด้านความปลอดภัย**
- ⚠️ Session-based auth (ไม่ใช่ JWT)
- ⚠️ ไม่มี rate limiting สำหรับ API
- ⚠️ ไม่มี CSRF protection
- ⚠️ Admin password เก็บใน .env (plain text hash)

### 3. **ข้อจำกัดด้าน ML**
- ⚠️ ความแม่นยำขึ้นกับคุณภาพข้อมูล training
- ⚠️ อาจทำนายผิดสำหรับกรณีพิเศษ
- ⚠️ Feature engineering ต้องการข้อมูลครบถ้วน

### 4. **การพึ่งพา Third-Party Services**
- ⚠️ Gemini API อาจมี downtime
- ⚠️ Cost สำหรับ API calls
- ⚠️ Rate limits ของ Gemini

---

## 💡 คำแนะนำการปรับปรุง (Recommendations)

### 1. **Security Enhancements** (ระดับความสำคัญ: สูง)
- ✅ เพิ่ม CSRF protection
- ✅ ใช้ JWT แทน session-based auth
- ✅ เพิ่ม rate limiting (Flask-Limiter)
- ✅ Implement API key rotation
- ✅ เพิ่ม input sanitization
- ✅ ใช้ HTTPS ใน production

### 2. **Performance Optimization** (ระดับความสำคัญ: กลาง)
- ✅ เพิ่ม caching (Redis) สำหรับผลลัพธ์
- ✅ Async processing สำหรับ Gemini calls
- ✅ Database indexing
- ✅ Lazy loading สำหรับ templates
- ✅ CDN สำหรับ static files
- ✅ Connection pooling

### 3. **Feature Additions** (ระดับความสำคัญ: กลาง)
- 📌 Real-time notifications
- 📌 Multi-language support (Thai/English)
- 📌 Export reports (PDF/Excel)
- 📌 Email notifications สำหรับนักศึกษาเสี่ยง
- 📌 Dashboard สำหรับผู้บริหาร
- 📌 Historical prediction tracking

### 4. **Testing & Quality** (ระดับความสำคัญ: สูง)
- ✅ เพิ่ม unit tests (pytest)
- ✅ Integration tests
- ✅ End-to-end tests (Selenium)
- ✅ Code coverage (> 80%)
- ✅ Automated testing pipeline (CI/CD)
- ✅ Performance testing

### 5. **Documentation** (ระดับความสำคัญ: กลาง)
- ✅ API documentation (Swagger/OpenAPI)
- ✅ User manual (Thai/English)
- ✅ Video tutorials
- ✅ Deployment guide
- ✅ Troubleshooting guide
- ✅ Architecture diagrams

### 6. **Scalability** (ระดับความสำคัญ: ต่ำ-กลาง)
- 📌 Migrate to PostgreSQL/MySQL
- 📌 Implement message queue (Celery)
- 📌 Horizontal scaling with load balancer
- 📌 Microservices architecture
- 📌 Kubernetes deployment

### 7. **AI/ML Improvements** (ระดับความสำคัญ: กลาง)
- 📌 Model retraining pipeline (automatic)
- 📌 A/B testing สำหรับโมเดล
- 📌 Ensemble methods ที่ซับซ้อนขึ้น
- 📌 Deep learning models (Neural Networks)
- 📌 Time series analysis สำหรับ trends
- 📌 Recommendation system สำหรับวิชา

---

## 📈 การวัดผลและ KPIs (Metrics & KPIs)

### 1. **Technical KPIs**
- **Model Accuracy**: > 85%
- **API Response Time**: < 500ms (normal), < 5s (Gemini)
- **System Uptime**: > 99.5%
- **Error Rate**: < 1%

### 2. **Business KPIs**
- **Prediction Accuracy**: จำนวนนักศึกษาที่ทำนายถูก
- **Early Warning Success**: นักศึกษาที่ได้รับการช่วยเหลือทันเวลา
- **User Adoption**: จำนวนผู้ใช้งานต่อเดือน
- **Data Coverage**: % ของนักศึกษาที่อยู่ในระบบ

### 3. **User Experience KPIs**
- **Page Load Time**: < 2s
- **User Satisfaction**: > 4/5 stars
- **Feature Usage**: % ของฟีเจอร์ที่ถูกใช้
- **Support Tickets**: จำนวนปัญหาที่รายงาน

---

## 🎓 Use Cases

### 1. **สำหรับนักศึกษา**
- ตรวจสอบโอกาสจบการศึกษา
- รับคำแนะนำการปรับปรุง
- วางแผนการเรียนในอนาคต

### 2. **สำหรับอาจารย์ที่ปรึกษา**
- ระบุนักศึกษาเสี่ยง
- ให้คำปรึกษาที่มีข้อมูลสนับสนุน
- ติดตามพัฒนาการนักศึกษา

### 3. **สำหรับผู้บริหารคณะ**
- วิเคราะห์ trends ของหลักสูตร
- ปรับปรุงหลักสูตร
- วางแผนทรัพยากร

### 4. **สำหรับงานทะเบียน**
- จัดการข้อมูลนักศึกษา
- สร้างรายงาน
- ติดตามสถานะนักศึกษา

---

## 🔄 กระบวนการ DevOps

### 1. **Version Control**
- Git repository
- Branch strategy: main, develop, feature branches
- Pull request reviews

### 2. **Deployment**
- **Development**: Local (python app.py)
- **Production**: Gunicorn + Nginx
- **Cloud**: Vercel or Heroku
- **Storage**: Local or Cloudflare R2

### 3. **Monitoring** (แนะนำ)
- Application logs
- Error tracking (Sentry)
- Performance monitoring (New Relic)
- Uptime monitoring (Pingdom)

### 4. **Backup Strategy** (แนะนำ)
- Daily database backups
- Model versioning in cloud
- Configuration backups

---

## 🌍 ความพร้อมใช้งาน (Production Readiness)

### ✅ **พร้อมแล้ว (Ready)**
- Core functionality ทำงานได้
- Gemini integration สมบูรณ์
- Basic authentication
- File upload/download
- Model training & prediction

### ⚠️ **ต้องปรับปรุงก่อน Production (Needs Improvement)**
- Security hardening (CSRF, rate limiting)
- Comprehensive testing
- Error handling & logging
- Performance optimization
- Documentation

### ❌ **ยังไม่พร้อม (Not Ready)**
- Scalability สำหรับ users จำนวนมาก
- Real-time features
- Advanced monitoring
- Disaster recovery plan

---

## 📞 การสนับสนุนและบำรุงรักษา (Support & Maintenance)

### 1. **Regular Maintenance**
- Model retraining: ทุกๆ ภาคการศึกษา
- Database cleanup: monthly
- Dependency updates: quarterly
- Security patches: as needed

### 2. **Support Channels** (แนะนำ)
- Email support
- Issue tracker (GitHub)
- User documentation
- FAQ section

### 3. **Change Management**
- Version control
- Release notes
- Rollback procedures
- User communication

---

## 🏁 สรุป (Conclusion)

ระบบทำนายการจบการศึกษานี้เป็นระบบที่มีศักยภาพสูง มีการผสมผสานระหว่าง Machine Learning แบบดั้งเดิมกับ Gemini AI ที่ทันสมัย ทำให้สามารถให้ข้อมูลที่ลึกซึ้งและเข้าใจง่ายแก่ผู้ใช้

### **จุดแข็ง (Strengths)**
- ✨ AI ที่หลากหลายและทรงพลัง
- 📊 การวิเคราะห์ที่ครอบคลุม
- 🎯 Explainable AI ที่อธิบายได้ชัดเจน
- 🔧 Code ที่จัดระเบียบดี

### **โอกาสพัฒนา (Opportunities)**
- 🔐 Security enhancements
- ⚡ Performance optimization
- 📈 Scalability improvements
- 🧪 Testing coverage

### **คำแนะนำสุดท้าย**
ระบบนี้เหมาะสำหรับการใช้งานในระดับ pilot หรือสถาบันขนาดเล็กถึงกลาง หากต้องการใช้ในระดับองค์กรขนาดใหญ่ ควรปรับปรุงด้าน security, scalability และ testing ก่อน

---

**จัดทำโดย**: AI System Analyzer  
**วันที่**: 8 กุมภาพันธ์ 2026  
**สถานะ**: ✅ เอกสารฉบับสมบูรณ์

---

## 📚 เอกสารอ้างอิง (References)

1. **START_HERE.md** - คู่มือเริ่มต้นใช้งาน
2. **GEMINI_INTEGRATION_GUIDE.md** - คู่มือการใช้ Gemini
3. **README_FIX_TH.md** - สรุปการแก้ไขภาษาไทย
4. **SYSTEM_FIX_SUMMARY.md** - สรุปการแก้ไขทางเทคนิค
5. **EXPLAINABLE_AI_IMPLEMENTATION.md** - รายละเอียด Explainable AI

---

## 📋 Appendix

### A. ตัวอย่างคำสั่ง CLI

```bash
# เทรนโมเดลใหม่
curl -X POST http://localhost:5000/train \
  -F "file=@data.csv" \
  -F "use_gemini=true"

# ทำนายผล
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"grades": {"CPE101": "A", "MAT101": "B"}}'
```

### B. Database Schema

```sql
-- users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- grade_forms table
CREATE TABLE grade_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    form_data TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### C. API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | หน้าแรก |
| `/login` | GET/POST | Login page |
| `/logout` | GET | Logout |
| `/train` | POST | เทรนโมเดล |
| `/predict` | POST | ทำนายผล |
| `/predict_gemini` | POST | ทำนายด้วย Gemini |
| `/model_management` | GET | จัดการโมเดล |
| `/admin/grade_forms` | GET | จัดการ grade forms |

---

**🎉 สิ้นสุดการวิเคราะห์ระบบ**
