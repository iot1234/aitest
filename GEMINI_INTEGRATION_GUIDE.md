# คู่มือการใช้งาน Gemini Integration

## ภาพรวม

ระบบทำนายการจบการศึกษามีการผสานรวม Google Gemini AI เพื่อเพิ่มความสามารถในการวิเคราะห์ข้อมูลแบบลึกซึ้งและให้คำแนะนำที่เป็นภาษาธรรมชาติ

## 🎯 ฟีเจอร์ที่รองรับ

### 1. การเทรนโมเดลพร้อม Gemini Analysis (Training)
เมื่อเทรนโมเดล Gemini จะช่วยวิเคราะห์ไฟล์ข้อมูลเทรนและให้ insights สำคัญ

**วิธีใช้งาน:**
1. ไปที่หน้า "จัดการโมเดล" (Model Management)
2. อัปโหลดไฟล์ CSV ข้อมูลนักศึกษา
3. เลือก ✅ "เรียกใช้ Gemini วิเคราะห์ไฟล์ระหว่างฝึก"
4. (ไม่บังคับ) ระบุเป้าหมายการวิเคราะห์ เช่น "โฟกัสที่คุณภาพข้อมูลและความสมดุล"
5. กดปุ่ม "เทรนโมเดล"

**ผลลัพธ์:**
- Gemini จะวิเคราะห์:
  - ✅ คุณภาพข้อมูล (Data Quality)
  - ✅ ความสมดุลของข้อมูล (Class Balance)
  - ✅ ปัญหาที่อาจเกิดขึ้น (Potential Issues)
  - ✅ คำแนะนำการปรับปรุง (Recommendations)
  - ✅ ความเสี่ยง (Risk Assessment)
- ผลการวิเคราะห์จะถูกบันทึกไว้กับโมเดล

### 2. การทำนายพร้อม Gemini Analysis (Prediction)
Gemini จะวิเคราะห์เกรดของนักศึกษาและให้คำแนะนำแบบเจาะลึก

**วิธีใช้งาน:**
1. ไปที่หน้า "ทำนายหลักสูตร" (Curriculum Prediction)
2. เลือก **วิธีการทำนาย**:
   - 🤖 **ปกติ (Normal AI)**: ใช้โมเดล Machine Learning ที่เทรนไว้
   - ✨ **Gemini**: ใช้โมเดล ML + การวิเคราะห์เชิงลึกจาก Gemini
3. กรอกข้อมูลเกรดของนักศึกษา
4. (สำหรับ Gemini Mode) ระบุเป้าหมายการวิเคราะห์
5. กดปุ่ม "วิเคราะห์และทำนาย"

**ผลลัพธ์ (Gemini Mode):**
- การทำนายจากโมเดล ML (ความน่าจะเป็นการจบ, ความเชื่อมั่น)
- การวิเคราะห์เชิงลึกจาก Gemini:
  - 📊 **แนวโน้มการจบการศึกษา** (Graduation Probability)
  - ⚠️ **ระดับความเสี่ยง** (Risk Level: Very Low → Very High)
  - 🎯 **ปัจจัยสำคัญ** (Key Factors)
  - 💡 **คำแนะนำเชิงปฏิบัติ** (Actionable Recommendations)
  - 📈 **ตัวชี้วัดสำคัญ** (Key Metrics)

### 3. การทดสอบด้วยการเลือกวิธีการ (Testing)
ผู้ใช้สามารถเปรียบเทียบผลลัพธ์ระหว่างโมเดล ML แบบปกติกับ Gemini

**วิธีทดสอบ:**
1. เตรียมข้อมูลเกรดทดสอบ
2. ทดสอบด้วย **โหมดปกติ** → บันทึกผลลัพธ์
3. ทดสอบด้วย **โหมด Gemini** → เปรียบเทียบผลลัพธ์
4. วิเคราะห์ความแตกต่าง:
   - ความแม่นยำ (Accuracy)
   - ความละเอียดของคำอธิบาย (Explanation Depth)
   - ความเป็นประโยชน์ของคำแนะนำ (Usefulness)

---

## ⚙️ การตั้งค่า (Configuration)

### 1. ตั้งค่า Gemini API Key

**ผ่าน Environment Variable:**
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
export GEMINI_MODEL_NAME="gemini-1.5-flash"  # Optional (ค่าเริ่มต้น)
```

**ผ่าน .env file:**
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL_NAME=gemini-1.5-flash
```

### 2. รับ API Key
1. ไปที่ [Google AI Studio](https://makersuite.google.com/app/apikey)
2. สร้าง API Key ใหม่
3. คัดลอกและบันทึก API Key

### 3. ตรวจสอบสถานะ
ระบบจะแสดงสถานะ Gemini ที่มุมขวาบน:
- ✅ **พร้อมใช้งาน** (เขียว): Gemini พร้อมใช้งาน
- ❌ **ปิดอยู่** (แดง): ยังไม่ได้ตั้งค่า API Key

---

## 🔧 การทำงานภายใน (Technical Details)

### API Endpoints

#### 1. Training Analysis
```
POST /train
Content-Type: application/json

{
  "filename": "data.csv",
  "use_advanced_training": true,
  "enable_gemini_analysis": true,
  "gemini_analysis_goal": "วิเคราะห์คุณภาพข้อมูล"
}
```

**Response:**
```json
{
  "success": true,
  "gemini_training_analysis": {
    "analysis_goal": "...",
    "gemini": {
      "analysis_markdown": "...",
      "risk_level": "low",
      "key_metrics": [...],
      "recommendations": [...]
    }
  }
}
```

#### 2. Prediction with Gemini
```
POST /api/gemini/predict
Content-Type: application/json

{
  "student_name": "นักศึกษาทดสอบ",
  "course_grades": {
    "CS101": "A",
    "MATH201": "B+",
    ...
  },
  "loaded_terms_count": 3,
  "analysis_goal": "วิเคราะห์แนวโน้มการจบ"
}
```

**Response:**
```json
{
  "success": true,
  "gemini_result": {
    "analysis_markdown": "สรุปการวิเคราะห์...",
    "risk_level": "low",
    "outcome_summary": {
      "status": "มีแนวโน้มจบการศึกษา",
      "confidence": 0.85,
      "description": "..."
    },
    "key_metrics": [...],
    "recommendations": [...]
  }
}
```

### Core Functions

#### 1. `run_gemini_training_analysis(df, analysis_goal, training_context)`
**Location:** `app.py` (search for "run_gemini_training_analysis")

**Purpose:** วิเคราะห์ไฟล์เทรนด้วย Gemini

**Parameters:**
- `df`: DataFrame ของข้อมูลเทรน
- `analysis_goal`: เป้าหมายการวิเคราะห์ (optional)
- `training_context`: บริบทการเทรน (data format, label distribution, etc.)

**Returns:**
```python
{
  'analysis_goal': str,
  'dataset_summary': dict,
  'sample_rows': list,
  'training_context': dict,
  'gemini': dict,  # Gemini analysis result
  'generated_at': str,
  'analysis_type': str
}
```

#### 2. `gemini_predict_route()`
**Location:** `app.py` (search for "@app.route('/api/gemini/predict')")

**Purpose:** ทำนายด้วย Gemini

**Request Body:**
```python
{
  'student_name': str,
  'course_grades': dict,  # {course_id: grade}
  'loaded_terms_count': int,
  'analysis_goal': str (optional),
  'model_filename': str (optional)
}
```

#### 3. `call_gemini_structured(task_name, payload, schema_key)`
**Location:** `app.py` (search for "def call_gemini_structured")

**Purpose:** Core function สำหรับเรียก Gemini API

**Parameters:**
- `task_name`: ชื่อ task (เช่น 'training_dataset_analysis')
- `payload`: ข้อมูลที่ส่งให้ Gemini (dict)
- `schema_key`: คีย์สำหรับ response schema (default: 'insights')

**Features:**
- รองรับ structured JSON responses
- Auto-retry on API errors
- Detailed logging
- Error handling

### Response Schemas

#### Insights Schema
```python
{
  "type": "object",
  "properties": {
    "analysis_markdown": {"type": "string"},  # การวิเคราะห์แบบละเอียด
    "risk_level": {"type": "string"},  # very_low, low, moderate, high, very_high
    "outcome_summary": {
      "type": "object",
      "properties": {
        "status": {"type": "string"},  # สถานะการทำนาย
        "confidence": {"type": "number"},  # ความเชื่อมั่น 0-1
        "description": {"type": "string"}  # คำอธิบาย
      }
    },
    "key_metrics": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "label": {"type": "string"},
          "value": {"type": "string"},
          "trend": {"type": "string"}  # up, down, stable
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

---

## 📋 ตัวอย่างการใช้งาน (Examples)

### Example 1: Training with Gemini Analysis

```python
# Upload and train with Gemini
import requests

# Upload file
files = {'file': open('student_data.csv', 'rb')}
upload_response = requests.post('http://localhost:5000/upload', files=files)
filename = upload_response.json()['filename']

# Train with Gemini analysis
train_data = {
    'filename': filename,
    'use_advanced_training': True,
    'enable_gemini_analysis': True,
    'gemini_analysis_goal': 'วิเคราะห์คุณภาพข้อมูลและความสมดุล'
}
train_response = requests.post('http://localhost:5000/train', json=train_data)
result = train_response.json()

# Access Gemini insights
if result['success']:
    gemini_analysis = result['gemini_training_analysis']
    print("Analysis:", gemini_analysis['gemini']['analysis_markdown'])
    print("Risk Level:", gemini_analysis['gemini']['risk_level'])
    print("Recommendations:", gemini_analysis['gemini']['recommendations'])
```

### Example 2: Prediction with Gemini

```python
# Predict with Gemini
predict_data = {
    'student_name': 'นักศึกษา A',
    'course_grades': {
        'CS101': 'A',
        'MATH201': 'B+',
        'ENG102': 'A',
        'PHY201': 'B',
        'CS202': 'B+'
    },
    'loaded_terms_count': 2,
    'analysis_goal': 'วิเคราะห์แนวโน้มการจบการศึกษา'
}

predict_response = requests.post(
    'http://localhost:5000/api/gemini/predict',
    json=predict_data
)
result = predict_response.json()

if result['success']:
    gemini = result['gemini_result']
    print("Analysis:", gemini['analysis_markdown'])
    print("Risk:", gemini['risk_level'])
    print("Status:", gemini['outcome_summary']['status'])
    print("Confidence:", gemini['outcome_summary']['confidence'])
    print("\nRecommendations:")
    for rec in gemini['recommendations']:
        print(f"  - {rec}")
```

### Example 3: Comparing Normal vs Gemini Modes

```javascript
// Frontend JavaScript example

// Test with Normal AI
const normalResult = await fetch('/api/analyze_curriculum', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        current_grades: grades,
        model_filename: 'model.joblib'
    })
}).then(r => r.json());

console.log("Normal AI:", {
    probability: normalResult.prediction_result.prob_pass,
    confidence: normalResult.prediction_result.confidence
});

// Test with Gemini
const geminiResult = await fetch('/api/gemini/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        course_grades: grades,
        analysis_goal: 'ทำนายและแนะนำ'
    })
}).then(r => r.json());

console.log("Gemini:", {
    status: geminiResult.gemini_result.outcome_summary.status,
    confidence: geminiResult.gemini_result.outcome_summary.confidence,
    recommendations: geminiResult.gemini_result.recommendations
});
```

---

## ⚠️ ข้อควรระวัง (Cautions)

### 1. API Rate Limits
- Gemini API มีข้อจำกัดจำนวน requests ต่อนาที
- หากใช้บ่อยเกินไป อาจถูก rate limit
- แนะนำ: ใช้ cache สำหรับ requests ที่ซ้ำๆ

### 2. Cost
- Gemini API มีค่าใช้จ่าย (ขึ้นอยู่กับ tier)
- `gemini-1.5-flash`: ถูกที่สุด แต่มี rate limit
- `gemini-1.5-pro`: แพงกว่า แต่ให้ผลลัพธ์ดีกว่า

### 3. Response Time
- Gemini analysis ใช้เวลา ~2-5 วินาที
- Normal AI ใช้เวลา ~0.1-0.5 วินาที
- แนะนำ: ใช้ Gemini เฉพาะเมื่อต้องการ insights เชิงลึก

### 4. Privacy
- ข้อมูลที่ส่งให้ Gemini จะผ่าน Google servers
- อย่าส่งข้อมูลส่วนตัวที่ sensitive
- แนะนำ: Anonymize ข้อมูลก่อนส่ง

---

## 🐛 Troubleshooting

### ปัญหา: Gemini ไม่พร้อมใช้งาน

**อาการ:**
- Badge แสดงสีแดง "ปิดอยู่"
- ไม่สามารถเลือก Gemini Mode ได้

**วิธีแก้:**
1. ตรวจสอบว่าตั้งค่า `GEMINI_API_KEY` แล้วหรือไม่
```bash
echo $GEMINI_API_KEY
```

2. ตรวจสอบ log ของแอพ
```bash
tail -f app_startup.log | grep -i gemini
```

3. ทดสอบ API Key
```python
import google.generativeai as genai
import os

# ใช้ model name จาก environment หรือ default
model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel(model_name)
response = model.generate_content("Test")
print(response.text)
```

### ปัญหา: Gemini ส่ง Error

**อาการ:**
- API returns 503 หรือ 500
- Error message: "Gemini API is not configured"

**วิธีแก้:**
1. ตรวจสอบว่า `google-generativeai` ติดตั้งแล้ว
```bash
pip install google-generativeai
```

2. ตรวจสอบ API Key ถูกต้อง
3. ตรวจสอบ network connectivity
4. ลอง restart แอพ

### ปัญหา: Response ช้า

**อาการ:**
- Gemini analysis ใช้เวลานาน (>10 วินาที)

**วิธีแก้:**
1. ลดขนาด payload (ส่งข้อมูลน้อยลง)
2. ใช้ `gemini-1.5-flash` แทน `gemini-1.5-pro`
3. เพิ่ม timeout setting
4. ใช้ caching สำหรับ requests ที่ซ้ำ

---

## 📚 เอกสารเพิ่มเติม

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Gemini API Quickstart](https://ai.google.dev/tutorials/python_quickstart)
- [Gemini Pricing](https://ai.google.dev/pricing)

---

## 🎓 สรุป

ระบบมีการผสานรวม Gemini AI อย่างสมบูรณ์ใน 3 จุดหลัก:

1. ✅ **การเทรน**: Gemini วิเคราะห์ไฟล์เทรนและให้ insights
2. ✅ **การทำนาย**: Gemini วิเคราะห์เกรดและให้คำแนะนำ
3. ✅ **การทดสอบ**: ผู้ใช้เลือกได้ว่าจะใช้ AI แบบไหน

ระบบพร้อมใช้งานทันที เพียงแค่ตั้งค่า `GEMINI_API_KEY` ✨
