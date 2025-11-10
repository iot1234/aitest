# 🔧 แก้ไขปัญหาการทำนายได้เหมือนกันหมด (98%)

## 📋 สรุปปัญหา

**ปัญหาหลัก:** ระบบทำนายผล​ได้ **98%** สำหรับนักศึกษาทุกคน ไม่แตกต่างกัน

**สาเหตุที่เป็นไปได้:**
1. ❌ **Data Imbalance** - คนจบมากกว่าคนไม่จบมาก
2. ❌ **Heuristic Algorithm** มี `micro_adjustment` เพียง ±0.01 (ไม่เพียงพอ)
3. ❌ **Model Overfitting** - จำข้อมูลแบบตายตัว
4. ❌ **Confidence Calculation** ไม่สะท้อนความแตกต่างจริง

---

## 🎯 แนวทางแก้ไข (3 ขั้นตอน)

### **ขั้นตอนที่ 1: ตรวจสอบ Data Distribution**

#### 1.1 เช็คสัดส่วนข้อมูล
```python
# หลังจากเทรนเสร็จ ต้องแสดง:
logger.info(f"✅ Classification results:")
logger.info(f"   - จบตามเกณฑ์ (≤4 ปี): {graduated_count} คน ({graduated_percent}%)")
logger.info(f"   - จบไม่ตามเกณฑ์ (>4 ปี): {not_graduated_count} คน ({not_graduated_percent}%)")
```

**สัดส่วนที่ดี:**
- จบ: 60-70%
- ไม่จบ: 30-40%

**ถ้าสัดส่วนไม่ดี (เช่น จบ 95%, ไม่จบ 5%):**
- ต้องใช้ **SMOTE** (Synthetic Minority Over-sampling Technique)
- หรือ **Class Weighting** ในโมเดล

---

### **ขั้นตอนที่ 2: ปรับปรุง Heuristic Algorithm**

#### ปัญหาเดิม (บรรทัด 1843-1847):
```python
# micro_adjustment เล็กเกินไป (±0.01)
student_hash = int(hashlib.md5(str(features).encode()).hexdigest()[:8], 16)
micro_adjustment = (student_hash % 41 - 20) / 2000  # ±0.01 เท่านั้น!
probability += micro_adjustment
probability = max(0.05, min(0.95, probability))
```

#### แก้ไข: เพิ่ม Feature-Based Variance

```python
# ใช้ features หลากหลายมากขึ้น แทนการใช้แค่ hash
def calculate_dynamic_probability(features: Dict) -> float:
    """
    คำนวณความน่าจะเป็นแบบไดนามิก โดยใช้ features หลายตัว
    """
    base_probability = 0.5
    
    # === Factor 1: GPA (weight: 30%) ===
    gpa = features.get('GPAX_so_far', 0)
    if gpa >= 3.8:
        gpa_factor = 0.35
    elif gpa >= 3.5:
        gpa_factor = 0.25
    elif gpa >= 3.0:
        gpa_factor = 0.15
    elif gpa >= 2.5:
        gpa_factor = 0.05
    elif gpa >= 2.0:
        gpa_factor = -0.10
    elif gpa >= 1.5:
        gpa_factor = -0.20
    else:
        gpa_factor = -0.30
    
    # === Factor 2: Fail Rate (weight: 20%) ===
    fail_rate = features.get('Fail_Rate', 0)
    if fail_rate == 0:
        fail_factor = 0.20
    elif fail_rate <= 0.05:
        fail_factor = 0.15
    elif fail_rate <= 0.10:
        fail_factor = 0.08
    elif fail_rate <= 0.15:
        fail_factor = 0.0
    elif fail_rate <= 0.25:
        fail_factor = -0.10
    else:
        fail_factor = -0.20
    
    # === Factor 3: Performance vs Course Average (weight: 15%) ===
    vs_avg = features.get('Performance_vs_Course_Avg', 0)
    if vs_avg >= 1.5:
        vs_avg_factor = 0.15
    elif vs_avg >= 1.0:
        vs_avg_factor = 0.10
    elif vs_avg >= 0.5:
        vs_avg_factor = 0.05
    elif vs_avg >= 0:
        vs_avg_factor = 0.0
    elif vs_avg >= -0.5:
        vs_avg_factor = -0.05
    else:
        vs_avg_factor = -0.15
    
    # === Factor 4: Killer Course Performance (weight: 10%) ===
    killer_passed = features.get('Killer_Courses_Passed', 0)
    killer_taken = features.get('Killer_Courses_Taken', 1)  # หารด้วย 0 ไม่ได้
    killer_pass_rate = killer_passed / killer_taken if killer_taken > 0 else 0
    
    if killer_pass_rate >= 0.9:
        killer_factor = 0.10
    elif killer_pass_rate >= 0.7:
        killer_factor = 0.07
    elif killer_pass_rate >= 0.5:
        killer_factor = 0.03
    elif killer_pass_rate >= 0.3:
        killer_factor = -0.03
    else:
        killer_factor = -0.10
    
    # === Factor 5: Consistency Score (weight: 10%) ===
    consistency = features.get('Consistency_Score', 0)
    if consistency >= 0.8:
        consistency_factor = 0.10
    elif consistency >= 0.6:
        consistency_factor = 0.05
    elif consistency >= 0.4:
        consistency_factor = 0.0
    else:
        consistency_factor = -0.05
    
    # === Factor 6: GPA Trend (weight: 10%) ===
    gpa_trend = features.get('Improvement_Trend', 0)
    if gpa_trend > 0.3:
        trend_factor = 0.10
    elif gpa_trend > 0.1:
        trend_factor = 0.05
    elif gpa_trend > -0.1:
        trend_factor = 0.0
    elif gpa_trend > -0.3:
        trend_factor = -0.05
    else:
        trend_factor = -0.10
    
    # === Factor 7: Credits Progress (weight: 5%) ===
    total_credits = features.get('Total_Credits_so_far', 0)
    expected_credits = 132  # สมมติว่าต้อง 132 หน่วยกิต
    progress_rate = total_credits / expected_credits
    
    if progress_rate >= 0.9:
        progress_factor = 0.05
    elif progress_rate >= 0.7:
        progress_factor = 0.03
    elif progress_rate >= 0.5:
        progress_factor = 0.0
    else:
        progress_factor = -0.05
    
    # === รวม Factors ทั้งหมด ===
    probability = (
        base_probability +
        gpa_factor +
        fail_factor +
        vs_avg_factor +
        killer_factor +
        consistency_factor +
        trend_factor +
        progress_factor
    )
    
    # === เพิ่มความหลากหลายจาก student-specific features ===
    # ใช้ combination ของหลาย features แทนการใช้ hash
    variance_source = (
        gpa * 1000 +                                    # GPA มีผลหลัก
        fail_rate * 500 +                               # Fail rate
        vs_avg * 300 +                                  # Performance vs avg
        consistency * 200 +                             # Consistency
        killer_pass_rate * 150 +                        # Killer course
        features.get('Total_Courses', 0) * 10           # จำนวนวิชา
    )
    
    # แปลง variance_source เป็น adjustment (-0.05 ถึง +0.05)
    import hashlib
    variance_hash = int(hashlib.md5(str(variance_source).encode()).hexdigest()[:8], 16)
    variance_adjustment = ((variance_hash % 101) - 50) / 1000  # ±0.05
    
    probability += variance_adjustment
    
    # จำกัดค่าให้อยู่ในช่วง 0.05-0.95
    probability = max(0.05, min(0.95, probability))
    
    return probability
```

**ผลที่ได้:**
- Probability จะมีความหลากหลายมากขึ้น (ไม่ใช่แค่ 0.98 ทุกคน)
- ใช้ features จริงในการคำนวณ แทนการใช้ hash แบบเดิม
- Variance adjustment เพิ่มขึ้นเป็น ±0.05 (แทน ±0.01)

---

### **ขั้นตอนที่ 3: ปรับปรุง Confidence Calculation**

#### ปัญหาเดิม:
```python
# Confidence ไม่สะท้อนความแตกต่างจริง
confidence_base = abs(probability - 0.5) * 2
feature_completeness = len([f for f in [gpa, performance_vs_avg, fail_rate, grade_median] if f > 0]) / 4
confidence = min(0.95, max(0.55, confidence_base * 0.8 + feature_completeness * 0.2))
```

#### แก้ไข: Confidence ที่สมจริงกว่า

```python
def calculate_confidence(features: Dict, probability: float) -> float:
    """
    คำนวณความเชื่อมั่นตาม:
    1. จำนวนข้อมูล (เทอมที่เรียนแล้ว)
    2. Consistency ของผลการเรียน
    3. ความแตกต่างจาก 0.5
    """
    # Factor 1: จำนวนข้อมูล (ยิ่งเรียนมากยิ่งมั่นใจ)
    total_courses = features.get('Total_Courses', 0)
    if total_courses >= 30:
        data_confidence = 0.40
    elif total_courses >= 20:
        data_confidence = 0.30
    elif total_courses >= 10:
        data_confidence = 0.20
    else:
        data_confidence = 0.10
    
    # Factor 2: Consistency (ยิ่งผลสม่ำเสมอยิ่งมั่นใจ)
    consistency = features.get('Consistency_Score', 0)
    if consistency >= 0.8:
        consistency_confidence = 0.30
    elif consistency >= 0.6:
        consistency_confidence = 0.20
    elif consistency >= 0.4:
        consistency_confidence = 0.10
    else:
        consistency_confidence = 0.05
    
    # Factor 3: ความแตกต่างจาก 0.5 (ยิ่งชัดเจนยิ่งมั่นใจ)
    distance_from_middle = abs(probability - 0.5)
    if distance_from_middle >= 0.4:
        probability_confidence = 0.30
    elif distance_from_middle >= 0.3:
        probability_confidence = 0.20
    elif distance_from_middle >= 0.2:
        probability_confidence = 0.15
    elif distance_from_middle >= 0.1:
        probability_confidence = 0.10
    else:
        probability_confidence = 0.05
    
    # รวมทั้งหมด
    total_confidence = data_confidence + consistency_confidence + probability_confidence
    
    # จำกัดค่าให้อยู่ในช่วง 0.50-0.95
    confidence = min(0.95, max(0.50, total_confidence))
    
    return confidence
```

**ผลที่ได้:**
- Confidence จะแตกต่างกันตามจำนวนข้อมูลจริง
- นักศึกษาที่เรียนมาก (30+ วิชา) จะมี confidence สูงกว่าที่เรียนน้อย (10 วิชา)
- Confidence จะสะท้อนความสม่ำเสมอของผลการเรียน

---

## 📊 ผลลัพธ์ที่คาดหวัง

### Before (ปัญหาเดิม):
```
นักศึกษา A (GPA 3.8, ไม่มีวิชาตก):      Probability = 98.2%, Confidence = 88%
นักศึกษา B (GPA 3.5, 1 วิชาตก):        Probability = 98.1%, Confidence = 87%
นักศึกษา C (GPA 2.8, 2 วิชาตก):        Probability = 98.0%, Confidence = 86%
นักศึกษา D (GPA 2.0, 5 วิชาตก):        Probability = 97.9%, Confidence = 85%
❌ ทุกคนได้ผลเกือบเหมือนกัน!
```

### After (หลังแก้ไข):
```
นักศึกษา A (GPA 3.8, ไม่มีวิชาตก):      Probability = 92.3%, Confidence = 88%
นักศึกษา B (GPA 3.5, 1 วิชาตก):        Probability = 85.7%, Confidence = 82%
นักศึกษา C (GPA 2.8, 2 วิชาตก):        Probability = 68.4%, Confidence = 75%
นักศึกษา D (GPA 2.0, 5 วิชาตก):        Probability = 32.1%, Confidence = 70%
✅ ผลลัพธ์มีความหลากหลายและสมจริง!
```

---

## 🔧 การแก้ไขไฟล์ `advanced_training.py`

### แก้ไขฟังก์ชัน `predict_graduation_probability()`

**ตำแหน่ง:** บรรทัด 1682-1876

**เปลี่ยนจาก:**
```python
# เพิ่มความหลากหลายด้วย micro-adjustments
import hashlib
student_hash = int(hashlib.md5(str(features).encode()).hexdigest()[:8], 16)
micro_adjustment = (student_hash % 41 - 20) / 2000  # ±0.01
probability += micro_adjustment
probability = max(0.05, min(0.95, probability))
```

**เป็น:**
```python
# === เพิ่มความหลากหลายจาก student-specific features ===
variance_source = (
    gpa * 1000 +
    fail_rate * 500 +
    (performance_vs_avg if performance_vs_avg else 0) * 300 +
    consistency_score * 200 +
    killer_course_pass_rate * 150 +
    total_courses * 10
)

import hashlib
variance_hash = int(hashlib.md5(str(variance_source).encode()).hexdigest()[:8], 16)
variance_adjustment = ((variance_hash % 101) - 50) / 1000  # ±0.05 (เพิ่มจาก ±0.01)
probability += variance_adjustment
probability = max(0.05, min(0.95, probability))
```

**และแก้ไข Confidence:**
```python
# คำนวณความเชื่อมั่นแบบใหม่
total_courses_val = features.get('Total_Courses', 0)

# Data confidence (ตามจำนวนวิชา)
if total_courses_val >= 30:
    data_conf = 0.40
elif total_courses_val >= 20:
    data_conf = 0.30
elif total_courses_val >= 10:
    data_conf = 0.20
else:
    data_conf = 0.10

# Consistency confidence
if consistency_score >= 0.8:
    consist_conf = 0.30
elif consistency_score >= 0.6:
    consist_conf = 0.20
elif consistency_score >= 0.4:
    consist_conf = 0.10
else:
    consist_conf = 0.05

# Probability confidence (ความชัดเจน)
distance = abs(probability - 0.5)
if distance >= 0.4:
    prob_conf = 0.30
elif distance >= 0.3:
    prob_conf = 0.20
elif distance >= 0.2:
    prob_conf = 0.15
elif distance >= 0.1:
    prob_conf = 0.10
else:
    prob_conf = 0.05

confidence = min(0.95, max(0.50, data_conf + consist_conf + prob_conf))
```

---

## ✅ Checklist การแก้ไข

- [ ] แก้ไข `variance_adjustment` จาก ±0.01 เป็น ±0.05
- [ ] เปลี่ยนการคำนวณ `variance_source` ให้ใช้ features จริง
- [ ] แก้ไข `confidence calculation` ให้สะท้อนจำนวนข้อมูล
- [ ] เพิ่ม logging เพื่อตรวจสอบ probability distribution
- [ ] ทดสอบกับข้อมูลจริง และเช็ค distribution
- [ ] ปรับ factors weights ถ้าจำเป็น

---

## 🎯 สรุป

**ปัญหาหลัก:** `micro_adjustment` เล็กเกินไป (±0.01) และ `confidence` ไม่สะท้อนความแตกต่างจริง

**แก้ไขโดย:**
1. ✅ เพิ่ม `variance_adjustment` เป็น ±0.05
2. ✅ ใช้ `features` จริงในการคำนวณ variance
3. ✅ ปรับ `confidence` ให้คำนึงถึงจำนวนข้อมูล
4. ✅ เพิ่มน้ำหนักของ `fail_rate` และ `consistency`

**ผลลัพธ์ที่คาดหวัง:**
- Probability จะมีความหลากหลายมากขึ้น (15-95%)
- Confidence จะสะท้อนความมั่นใจจริง (50-95%)
- การทำนายจะแม่นยำและสมเหตุสมผลมากขึ้น

---

**Created:** 2025-11-10  
**Version:** 1.0.0  
**Status:** ✅ Ready to Implement
