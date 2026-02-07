# advanced_training.py - COMPLETE ENHANCED VERSION WITH AUTOMATIC GRADUATION CALCULATION
# ✅ Fixed: Course code normalization (old 3-part → new 4-part)
# ✅ Fixed: Name-based course matching fallback
# ✅ Fixed: GROUPNAME1 deduplication
# ✅ Fixed: พ.ศ./ค.ศ. year mismatch handling
# ✅ Fixed: Graduation label for students still studying
# ✅ Fixed: Temporal snapshot target leakage
# ✅ Fixed: W_count tracking
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# --- Course Code Normalization Utilities ---
_COURSE_CODE_MAPPING = None
_COURSE_NAME_KEYWORDS = None

def _get_course_code_mapping():
    """Lazy-load course code mapping from config"""
    global _COURSE_CODE_MAPPING
    if _COURSE_CODE_MAPPING is None:
        try:
            from config import Config
            _COURSE_CODE_MAPPING = getattr(Config, 'COURSE_CODE_MAPPING', {})
        except Exception:
            _COURSE_CODE_MAPPING = {}
    return _COURSE_CODE_MAPPING

def _get_course_name_keywords():
    """Lazy-load course name keywords from config"""
    global _COURSE_NAME_KEYWORDS
    if _COURSE_NAME_KEYWORDS is None:
        try:
            from config import Config
            _COURSE_NAME_KEYWORDS = getattr(Config, 'COURSE_NAME_KEYWORDS', {})
        except Exception:
            _COURSE_NAME_KEYWORDS = {}
    return _COURSE_NAME_KEYWORDS

def normalize_course_code(code: str, course_name: str = None) -> str:
    """
    Normalize course code: แปลงรหัสเก่า (3 ส่วน) → ใหม่ (4 ส่วน)
    ถ้ารหัสไม่ match ให้ลอง match จากชื่อวิชา
    
    Args:
        code: รหัสวิชา เช่น '02-011-109' หรือ '02-005-011-109'
        course_name: ชื่อวิชา (ไทย/อังกฤษ) สำหรับ fallback matching
    Returns:
        รหัสวิชาที่ normalize แล้ว
    """
    if not code or not isinstance(code, str):
        return code
    
    code = code.strip()
    
    # 1. ลอง exact match จาก mapping
    mapping = _get_course_code_mapping()
    if code in mapping:
        return mapping[code]
    
    # 2. ถ้าเป็นรหัส 4 ส่วนอยู่แล้ว → คืนค่าเดิม
    if re.match(r'^\d{2}-\d{3}-\d{3}-\d{3}$', code):
        return code
    
    # 3. ลอง pattern matching: XX-YYY-ZZZ → XX-000-YYY-ZZZ (เติม 000)
    m = re.match(r'^(\d{2})-(\d{3})-(\d{3})$', code)
    if m:
        prefix, mid, suffix = m.groups()
        # ลองเติม 000, 005 etc. แล้วดูว่า match กับ mapping value ไหม
        for padded in [f'{prefix}-000-{mid}-{suffix}', f'{prefix}-005-{mid}-{suffix}']:
            # Check if this padded code exists in mapping values or COURSES_DATA
            if padded in mapping.values():
                return padded
        # ไม่ match → ลอง default padding
        return f'{prefix}-000-{mid}-{suffix}'
    
    # 4. Fallback: match จากชื่อวิชา
    if course_name and isinstance(course_name, str):
        name_keywords = _get_course_name_keywords()
        course_name_clean = course_name.strip()
        # ลอง exact match ก่อน จากนั้น substring match
        for keyword, mapped_code in name_keywords.items():
            if keyword in course_name_clean:
                return mapped_code
    
    return code


def deduplicate_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """
    ลบข้อมูลซ้ำที่เกิดจาก GROUPNAME1 (วิชาเดียวกันอยู่หลายกลุ่มวิชา)
    เก็บเฉพาะแถวที่ไม่ซ้ำตาม (นักศึกษา, รหัสวิชา, ปีการศึกษา, เทอม, เกรด)
    """
    original_len = len(df)
    
    # หาคอลัมน์ที่เกี่ยวข้อง (ก่อน lowercasing)
    student_cols = [c for c in df.columns if any(k in c.lower().replace(' ', '_').replace('-', '_') 
                    for k in ['dummy_studentno', 'student_id', 'studentno', 'รหัสนักศึกษา'])]
    course_cols = [c for c in df.columns if any(k in c.lower().replace(' ', '_').replace('-', '_') 
                    for k in ['course_code', 'course', 'รหัสวิชา'])]
    year_cols = [c for c in df.columns if any(k in c.lower().replace(' ', '_').replace('-', '_') 
                    for k in ['ปีการศึกษา', 'academic_year', 'year'])]
    term_cols = [c for c in df.columns if any(k in c.lower().replace(' ', '_').replace('-', '_') 
                    for k in ['เทอม', 'semester', 'term'])]
    grade_cols = [c for c in df.columns if c.lower().strip() in ['grade', 'เกรด']]
    
    dedup_cols = []
    if student_cols: dedup_cols.append(student_cols[0])
    if course_cols: dedup_cols.append(course_cols[0])
    if year_cols: dedup_cols.append(year_cols[0])
    if term_cols: dedup_cols.append(term_cols[0])
    if grade_cols: dedup_cols.append(grade_cols[0])
    
    if len(dedup_cols) >= 3:  # ต้องมีอย่างน้อย student+course+grade
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"🧹 Dedup: ลบข้อมูลซ้ำ {removed} แถว (จาก {original_len} → {len(df)})")
    
    return df

class AdvancedFeatureEngineer:
    """
    Advanced Context-Aware Feature Engineering System
    ✅ รองรับ Transcript Format (1 นักศึกษา = หลายแถว)
    ✅ คำนวณการจบอัตโนมัติ (≤4 ปี = จบตามเกณฑ์)
    ✅ สร้าง Dynamic Snapshots ตามช่วงเวลาการเรียน
    """
    
    def __init__(self, grade_mapping: Dict[str, float]):
        """Initialize with grade mapping configuration"""
        self.grade_mapping = grade_mapping
        self.course_profiles = {}
        self.student_profiles = {}
        self.global_statistics = {}
        
    def create_dynamic_snapshot_features(self, student_data: pd.DataFrame, course_profiles: Dict[str, Dict], 
                                         snapshot_point: str = 'all') -> Dict[str, float]:
        """
        สร้าง Dynamic Snapshot Features ที่สามารถทำงานได้กับข้อมูลไม่ว่ากี่เทอมก็ตาม
        เพิ่มความสามารถเชิงบริบท (Context-Aware) โดยใช้ Course DNA Profiles
        
        Args:
            student_data: ข้อมูลของนักศึกษาคนหนึ่ง
            course_profiles: DNA profiles ของทุกวิชา
            snapshot_point: จุดเวลาที่ต้องการสร้าง snapshot ('all', 'semester_X', etc.)
        
        Returns:
            Dict ของ features ที่มีจำนวนคงที่เสมอ
        """
        # หาคอลัมน์ที่จำเป็น
        grade_col = self._find_column(student_data, ['grade', 'เกรด'])
        grade_point_col = self._find_column(student_data, ['grade_point', 'คะแนนเกรด', 'gpa_point'])
        credit_col = self._find_column(student_data, ['credit', 'หน่วยกิต'])
        course_col = self._find_column(student_data, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        semester_col = self._find_column(student_data, ['semester', 'เทอม', 'term'])
        academic_year_col = self._find_column(student_data, ['academic_year', 'ปีการศึกษา', 'year'])
        
        # เตรียมข้อมูลพื้นฐาน
        grades = []
        credits = []
        courses = []
        grade_letters = []
        semesters = []
        academic_years = []
        
        for _, row in student_data.iterrows():
            if pd.notna(row[grade_col]):
                # แปลงเกรด
                grade_point_val = row[grade_point_col] if grade_point_col and grade_point_col in row else None
                grade_val = self._convert_grade_to_numeric(row[grade_col], grade_point_val)
                
                if grade_val is not None:
                    grades.append(grade_val)
                    grade_letters.append(str(row[grade_col]).upper())
                    courses.append(str(row[course_col]) if course_col else 'UNKNOWN')
                    
                    # หน่วยกิต
                    try:
                        credit = float(row[credit_col]) if credit_col and pd.notna(row[credit_col]) else 3
                        credits.append(credit)
                    except:
                        credits.append(3)
                    
                    # เทอมและปีการศึกษา
                    if semester_col and pd.notna(row[semester_col]):
                        semesters.append(int(row[semester_col]))
                    else:
                        semesters.append(1)
                    
                    if academic_year_col and pd.notna(row[academic_year_col]):
                        academic_years.append(int(row[academic_year_col]))
                    else:
                        academic_years.append(2023)
        
        if not grades:
            return self._get_empty_dynamic_snapshot_features()
        
        # === 1. คุณลักษณะภาพรวม (Overall Features) ===
        total_credits = sum(credits)
        weighted_gpa = np.average(grades, weights=credits) if credits else np.mean(grades)
        
        # === 2. คุณลักษณะการกระจาย (Distribution Features) ===
        grade_counts = {
            'A_count': sum(1 for g in grades if g >= 3.5),
            'B_count': sum(1 for g in grades if 2.5 <= g < 3.5),
            'C_count': sum(1 for g in grades if 1.5 <= g < 2.5),
            'D_count': sum(1 for g in grades if 0.5 <= g < 1.5),
            'F_count': sum(1 for g in grades if g == 0),
            'W_count': sum(1 for gl in grade_letters if gl == 'W')
        }
        
        # === 3. คุณลักษณะแนวโน้ม (Trend Features) ===
        trend_features = self._calculate_dynamic_trend_features(grades, credits, semesters, academic_years)
        
        # === 4. คุณลักษณะเชิงบริบท (Context-Aware Features) ===
        context_features = self._calculate_dynamic_context_aware_features(courses, grades, course_profiles)
        
        # === 5. คุณลักษณะความเสี่ยง (Risk Features) ===
        risk_features = self._calculate_dynamic_risk_features(grades, grade_letters, courses, course_profiles)
        
        # === 6. คุณลักษณะเชิงเปรียบเทียบ (Comparative Features) ===
        comparative_features = self._calculate_dynamic_comparative_features(courses, grades, course_profiles)
        
        # รวม features ทั้งหมด
        snapshot_features = {
            # === Overall Performance ===
            'Total_Credits_so_far': total_credits,
            'GPAX_so_far': weighted_gpa,
            'GPA_simple': np.mean(grades),
            'Total_Courses': len(grades),
            'Grade_Std': np.std(grades) if len(grades) > 1 else 0,
            'Grade_Range': max(grades) - min(grades) if grades else 0,
            
            # === Grade Distribution ===
            'A_Count': grade_counts['A_count'],
            'B_Count': grade_counts['B_count'],
            'C_Count': grade_counts['C_count'],
            'D_Count': grade_counts['D_count'],
            'F_Count': grade_counts['F_count'],
            'W_Count': grade_counts['W_count'],
            
            'A_Rate': grade_counts['A_count'] / len(grades),
            'B_Rate': grade_counts['B_count'] / len(grades),
            'C_Rate': grade_counts['C_count'] / len(grades),
            'Low_Grade_Rate': (grade_counts['D_count'] + grade_counts['F_count']) / len(grades),
            'Pass_Rate': sum(1 for g in grades if g > 0) / len(grades),
            
            # === Academic Progress ===
            'Semesters_Studied': len(set(zip(academic_years, semesters))),
            'Academic_Years_Span': max(academic_years) - min(academic_years) + 1 if academic_years else 1,
            'Avg_Credits_Per_Semester': total_credits / len(set(zip(academic_years, semesters))) if academic_years else total_credits,
            
            # === Performance Consistency ===
            'Consistency_Score': 1 / (1 + np.std(grades)) if len(grades) > 1 else 1,
            'Improvement_Trend': trend_features.get('improvement_trend', 0),
            'Recent_Performance': trend_features.get('recent_performance', weighted_gpa),
            
            # === Context-Aware Features ===
            **context_features,
            
            # === Risk Indicators ===
            **risk_features,
            
            # === Comparative Analysis ===
            **comparative_features
        }
        
        return snapshot_features
    
    def _calculate_dynamic_trend_features(self, grades: List[float], credits: List[float], 
                                         semesters: List[int], academic_years: List[int]) -> Dict[str, float]:
        """คำนวณคุณลักษณะแนวโน้มแบบไดนามิก"""
        if len(grades) < 2:
            return {
                'improvement_trend': 0,
                'recent_performance': np.mean(grades) if grades else 0,
                'gpa_volatility': 0,
                'last_semester_gpa': np.mean(grades) if grades else 0,
                'semester_performance_trend': 0
            }
        
        # จัดกลุ่มตามเทอม
        semester_data = {}
        for i, (year, sem) in enumerate(zip(academic_years, semesters)):
            key = f"{year}_{sem}"
            if key not in semester_data:
                semester_data[key] = {'grades': [], 'credits': []}
            semester_data[key]['grades'].append(grades[i])
            semester_data[key]['credits'].append(credits[i])
        
        # คำนวณ GPA แต่ละเทอม
        semester_gpas = []
        sorted_semesters = sorted(semester_data.keys())
        
        for sem_key in sorted_semesters:
            sem_grades = semester_data[sem_key]['grades']
            sem_credits = semester_data[sem_key]['credits']
            sem_gpa = np.average(sem_grades, weights=sem_credits) if sem_credits else np.mean(sem_grades)
            semester_gpas.append(sem_gpa)
        
        # คำนวณแนวโน้ม
        if len(semester_gpas) >= 2:
            # Linear trend
            x = np.arange(len(semester_gpas))
            trend_slope = np.polyfit(x, semester_gpas, 1)[0] if len(semester_gpas) > 1 else 0
            
            # Recent vs Early performance
            recent_avg = np.mean(semester_gpas[-2:]) if len(semester_gpas) >= 2 else semester_gpas[-1]
            early_avg = np.mean(semester_gpas[:2]) if len(semester_gpas) >= 2 else semester_gpas[0]
            improvement = recent_avg - early_avg
            
            return {
                'improvement_trend': trend_slope,
                'recent_performance': recent_avg,
                'gpa_volatility': np.std(semester_gpas),
                'last_semester_gpa': semester_gpas[-1],
                'semester_performance_trend': improvement
            }
        else:
            return {
                'improvement_trend': 0,
                'recent_performance': semester_gpas[0] if semester_gpas else 0,
                'gpa_volatility': 0,
                'last_semester_gpa': semester_gpas[0] if semester_gpas else 0,
                'semester_performance_trend': 0
            }
    
    def _calculate_dynamic_context_aware_features(self, courses: List[str], grades: List[float], 
                                                 course_profiles: Dict[str, Dict]) -> Dict[str, float]:
        """คำนวณคุณลักษณะเชิงบริบทแบบไดนามิก โดยใช้ Course DNA Profiles"""
        
        context_features = {
            'Avg_Course_Difficulty': 0,
            'Performance_vs_Course_Avg': 0,
            'Killer_Courses_Taken': 0,
            'Killer_Courses_Passed': 0,
            'Easy_Courses_Taken': 0,
            'GPA_Booster_Courses': 0,
            'Above_Course_Avg_Count': 0,
            'Below_Course_Avg_Count': 0,
            'Excellence_in_Hard_Courses': 0,
            'Struggle_in_Easy_Courses': 0,
            'Competitive_Courses_Performance': 0,
            'Grade_Inflation_Benefit': 0
        }
        
        if not course_profiles or not courses:
            return context_features
        
        # วิเคราะห์แต่ละวิชาที่เรียน
        course_difficulties = []
        performance_vs_avg = []
        killer_taken = 0
        killer_passed = 0
        easy_taken = 0
        gpa_booster_taken = 0
        above_avg = 0
        below_avg = 0
        excellence_hard = 0
        struggle_easy = 0
        competitive_performance = []
        inflation_benefit = 0
        
        for course, grade in zip(courses, grades):
            if course in course_profiles:
                profile = course_profiles[course]
                
                # เก็บความยากของวิชา
                course_difficulties.append(profile['difficulty_score'])
                
                # เปรียบเทียบกับค่าเฉลี่ยของวิชา
                grade_diff = grade - profile['avg_grade']
                performance_vs_avg.append(grade_diff)
                
                if grade_diff > 0:
                    above_avg += 1
                else:
                    below_avg += 1
                
                # วิเคราะห์ประเภทวิชา
                if profile['is_killer_course']:
                    killer_taken += 1
                    if grade > 0:  # ผ่าน
                        killer_passed += 1
                    if grade >= profile['excellence_threshold']:
                        excellence_hard += 1
                
                if profile['is_easy_course']:
                    easy_taken += 1
                    if grade < profile['struggle_threshold']:
                        struggle_easy += 1
                
                if profile['is_gpa_booster']:
                    gpa_booster_taken += 1
                
                # Competitive performance
                competitive_performance.append(grade * profile['competitive_index'])
                
                # Grade inflation benefit
                if profile['grade_inflation'] and grade >= 3.0:
                    inflation_benefit += 1
        
        # คำนวณค่าเฉลี่ย
        total_courses = len(courses)
        if total_courses > 0:
            context_features.update({
                'Avg_Course_Difficulty': np.mean(course_difficulties) if course_difficulties else 0,
                'Performance_vs_Course_Avg': np.mean(performance_vs_avg) if performance_vs_avg else 0,
                'Killer_Courses_Taken': killer_taken,
                'Killer_Courses_Passed': killer_passed,
                'Killer_Course_Pass_Rate': killer_passed / killer_taken if killer_taken > 0 else 0,
                'Easy_Courses_Taken': easy_taken,
                'GPA_Booster_Courses': gpa_booster_taken,
                'Above_Course_Avg_Count': above_avg,
                'Below_Course_Avg_Count': below_avg,
                'Above_Avg_Rate': above_avg / total_courses,
                'Excellence_in_Hard_Courses': excellence_hard,
                'Struggle_in_Easy_Courses': struggle_easy,
                'Competitive_Courses_Performance': np.mean(competitive_performance) if competitive_performance else 0,
                'Grade_Inflation_Benefit': inflation_benefit
            })
        
        return context_features
    
    def _calculate_dynamic_risk_features(self, grades: List[float], grade_letters: List[str], 
                                        courses: List[str], course_profiles: Dict[str, Dict]) -> Dict[str, float]:
        """คำนวณคุณลักษณะความเสี่ยงแบบไดนามิก"""
        
        risk_features = {
            'Consecutive_Low_Grades': 0,
            'Recent_Decline_Signal': 0,
            'High_Risk_Course_Failures': 0,
            'Withdrawal_Pattern': 0,
            'Academic_Probation_Risk': 0,
            'Course_Load_Stress': 0,
            'Foundation_Course_Weakness': 0
        }
        
        if not grades:
            return risk_features
        
        # ตรวจสอบเกรดต่ำติดต่อกัน
        consecutive_low = 0
        max_consecutive_low = 0
        for grade in grades:
            if grade < 2.0:  # ต่ำกว่า C
                consecutive_low += 1
                max_consecutive_low = max(max_consecutive_low, consecutive_low)
            else:
                consecutive_low = 0
        
        # ตรวจสอบการลดลงล่าสุด
        recent_decline = 0
        if len(grades) >= 4:
            recent_avg = np.mean(grades[-2:])
            earlier_avg = np.mean(grades[-4:-2])
            if recent_avg < earlier_avg - 0.5:
                recent_decline = 1
        
        # ตรวจสอบการตกในวิชาเสี่ยงสูง
        high_risk_failures = 0
        foundation_weakness = 0
        
        for course, grade, grade_letter in zip(courses, grades, grade_letters):
            if course in course_profiles:
                profile = course_profiles[course]
                
                # วิชาเสี่ยงสูง
                if profile['risk_level'] == 'high_risk' and grade == 0:
                    high_risk_failures += 1
                
                # วิชาพื้นฐาน (สมมติว่าเป็นวิชาที่มี course code ขึ้นต้นด้วย 0)
                if course.startswith('0') and grade < 2.0:
                    foundation_weakness += 1
        
        # ตรวจสอบ pattern การถอน
        w_count = sum(1 for gl in grade_letters if gl == 'W')
        withdrawal_pattern = 1 if w_count >= 2 else 0
        
        # ความเสี่ยง Academic Probation (GPA < 2.0)
        current_gpa = np.mean(grades)
        probation_risk = 1 if current_gpa < 2.0 else 0
        
        risk_features.update({
            'Consecutive_Low_Grades': max_consecutive_low,
            'Recent_Decline_Signal': recent_decline,
            'High_Risk_Course_Failures': high_risk_failures,
            'Withdrawal_Pattern': withdrawal_pattern,
            'Academic_Probation_Risk': probation_risk,
            'Foundation_Course_Weakness': foundation_weakness,
            'Overall_Risk_Score': (max_consecutive_low * 0.2 + recent_decline * 0.3 + 
                                 high_risk_failures * 0.2 + withdrawal_pattern * 0.15 + 
                                 probation_risk * 0.15)
        })
        
        return risk_features
    
    def _calculate_dynamic_comparative_features(self, courses: List[str], grades: List[float], 
                                              course_profiles: Dict[str, Dict]) -> Dict[str, float]:
        """คำนวณคุณลักษณะเชิงเปรียบเทียบแบบไดนามิก"""
        
        comparative_features = {
            'Relative_Performance_Score': 0,
            'Percentile_Performance': 0,
            'Difficulty_Adjusted_GPA': 0,
            'Competitive_Advantage': 0,
            'Course_Selection_Strategy': 0
        }
        
        if not course_profiles or not courses:
            return comparative_features
        
        # คำนวณ performance เทียบกับ percentile ของแต่ละวิชา
        percentile_scores = []
        difficulty_weights = []
        competitive_scores = []
        
        for course, grade in zip(courses, grades):
            if course in course_profiles:
                profile = course_profiles[course]
                
                # หา percentile ของเกรดนี้ในวิชานี้
                if grade >= profile['percentile_90']:
                    percentile_score = 0.95
                elif grade >= profile['percentile_75']:
                    percentile_score = 0.80
                elif grade >= profile['percentile_50']:
                    percentile_score = 0.60
                elif grade >= profile['percentile_25']:
                    percentile_score = 0.30
                else:
                    percentile_score = 0.10
                
                percentile_scores.append(percentile_score)
                difficulty_weights.append(profile['difficulty_score'])
                
                # Competitive score
                competitive_score = grade * (1 + profile['competitive_index'])
                competitive_scores.append(competitive_score)
        
        if percentile_scores:
            # Relative Performance Score
            relative_performance = np.mean(percentile_scores)
            
            # Difficulty-Adjusted GPA
            if difficulty_weights and len(difficulty_weights) == len(grades):
                difficulty_adjusted_gpa = np.average(grades, weights=difficulty_weights)
            else:
                difficulty_adjusted_gpa = np.mean(grades)
            
            # Competitive Advantage
            competitive_advantage = np.mean(competitive_scores) if competitive_scores else 0
            
            # Course Selection Strategy (เลือกวิชาง่ายหรือยาก)
            avg_difficulty = np.mean(difficulty_weights) if difficulty_weights else 0.5
            if avg_difficulty > 0.7:
                strategy_score = 1  # เลือกวิชายาก = กล้าเสี่ยง
            elif avg_difficulty < 0.3:
                strategy_score = -1  # เลือกวิชาง่าย = เล่นปลอดภัย
            else:
                strategy_score = 0  # สมดุล
            
            comparative_features.update({
                'Relative_Performance_Score': relative_performance,
                'Percentile_Performance': relative_performance * 100,
                'Difficulty_Adjusted_GPA': difficulty_adjusted_gpa,
                'Competitive_Advantage': competitive_advantage,
                'Course_Selection_Strategy': strategy_score
            })
        
        return comparative_features
    
    def _get_empty_dynamic_snapshot_features(self) -> Dict[str, float]:
        """สร้าง dynamic snapshot features เปล่าสำหรับกรณีที่ไม่มีข้อมูล"""
        return {
            # Overall Performance
            'Total_Credits_so_far': 0,
            'GPAX_so_far': 0,
            'GPA_simple': 0,
            'Total_Courses': 0,
            'Grade_Std': 0,
            'Grade_Range': 0,
            
            # Grade Distribution
            'A_Count': 0, 'B_Count': 0, 'C_Count': 0, 'D_Count': 0, 'F_Count': 0, 'W_Count': 0,
            'A_Rate': 0, 'B_Rate': 0, 'C_Rate': 0, 'Low_Grade_Rate': 0, 'Pass_Rate': 0,
            
            # Academic Progress
            'Semesters_Studied': 0,
            'Academic_Years_Span': 0,
            'Avg_Credits_Per_Semester': 0,
            
            # Performance Consistency
            'Consistency_Score': 0,
            'Improvement_Trend': 0,
            'Recent_Performance': 0,
            
            # Context-Aware Features
            'Avg_Course_Difficulty': 0,
            'Performance_vs_Course_Avg': 0,
            'Killer_Courses_Taken': 0,
            'Killer_Courses_Passed': 0,
            'Killer_Course_Pass_Rate': 0,
            'Easy_Courses_Taken': 0,
            'GPA_Booster_Courses': 0,
            'Above_Course_Avg_Count': 0,
            'Below_Course_Avg_Count': 0,
            'Above_Avg_Rate': 0,
            'Excellence_in_Hard_Courses': 0,
            'Struggle_in_Easy_Courses': 0,
            'Competitive_Courses_Performance': 0,
            'Grade_Inflation_Benefit': 0,
            
            # Risk Features
            'Consecutive_Low_Grades': 0,
            'Recent_Decline_Signal': 0,
            'High_Risk_Course_Failures': 0,
            'Withdrawal_Pattern': 0,
            'Academic_Probation_Risk': 0,
            'Foundation_Course_Weakness': 0,
            'Overall_Risk_Score': 0,
            
            # Comparative Features
            'Relative_Performance_Score': 0,
            'Percentile_Performance': 0,
            'Difficulty_Adjusted_GPA': 0,
            'Competitive_Advantage': 0,
            'Course_Selection_Strategy': 0
        }

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method: เตรียมข้อมูลสำหรับการเทรนแบบ Advanced Context-Aware
        รองรับ Transcript Format ที่ 1 นักศึกษา = หลายแถว
        ✅ ปรับปรุง: auto-detect columns, dedup, normalize, GROUPNAME1 features
        """
        logger.info("\U0001f680 Starting Advanced Context-Aware Feature Engineering...")
        logger.info(f"\U0001f4ca Input data shape: {df.shape}")
        
        try:
            # Step 0: Deduplicate ข้อมูลซ้ำจาก GROUPNAME1 ก่อน clean
            df = deduplicate_transcript(df)
            
            # Step 0.5: Auto-detect columns ก่อน clean (ชื่อคอลัมน์ยังเป็นต้นฉบับ)
            self._detected_columns = self._auto_detect_columns(df)
            
            # Step 1: ตรวจสอบและทำความสะอาดข้อมูล
            df = self._clean_data(df)
            
            # Step 1.5: Normalize course codes (เก่า → ใหม่)
            df = self._normalize_course_codes(df)
            
            # Step 2: สร้าง Course DNA Profiles จากข้อมูลทั้งหมด
            logger.info("\U0001f9ec Creating Course DNA profiles...")
            self.course_profiles = self._create_course_dna_profiles(df)
            logger.info(f"\u2705 Created DNA profiles for {len(self.course_profiles)} courses")
            
            # Step 2.5: สร้าง Course Group Profiles (GROUPNAME1)
            self.course_group_profiles = self._create_course_group_profiles(df)
            
            # Step 3: แปลงข้อมูล Transcript เป็น Student Records พร้อมคำนวณการจบอัตโนมัติ
            logger.info("\U0001f465 Transforming transcript data to student records...")
            student_records = self._transform_transcript_to_students(df)
            logger.info(f"\u2705 Processed {len(student_records)} unique students")
            
            # Step 4: สร้าง Dynamic Snapshots สำหรับแต่ละนักศึกษา
            logger.info("\U0001f4f8 Creating dynamic temporal snapshots...")
            all_snapshots = []
            
            for student_id, student_record in student_records.items():
                snapshots = self._create_temporal_snapshots(student_id, student_record)
                all_snapshots.extend(snapshots)
            
            logger.info(f"\u2705 Created {len(all_snapshots)} training snapshots")
            
            if not all_snapshots:
                raise ValueError("No snapshots created! Check your data format.")
            
            # Step 5: Generate Advanced Features
            logger.info("\U0001f527 Generating advanced contextual features...")
            X = pd.DataFrame(all_snapshots)
            X = self._generate_advanced_features(X)
            
            # Step 6: Extract target variable
            if 'graduated' not in X.columns:
                raise ValueError("No 'graduated' column found in features!")
                
            y = X['graduated'].astype(int)
            
            # Log class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"\U0001f4ca Target distribution: {dict(zip(unique_classes, class_counts))}")
            
            # Remove non-feature columns
            X = X.drop(columns=['graduated', 'student_id', 'snapshot_id'], errors='ignore')
            
            # Step 7: Feature selection and normalization
            X = self._select_and_normalize_features(X)
            
            logger.info(f"\u2705 Feature engineering completed!")
            logger.info(f"\U0001f4ca Final shape: X={X.shape}, y={y.shape}")
            logger.info(f"\U0001f4ca Features created: {list(X.columns)[:20]}...")
            
            return X, y
            
        except Exception as e:
            logger.error(f"\u274c Error in feature engineering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_course_group_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        สร้าง profile สำหรับแต่ละกลุ่มวิชา (GROUPNAME1)
        ใช้เป็น feature เพิ่มเติมสำหรับวิเคราะห์ว่า นศ. ผ่านวิชากลุ่มใดได้ดี/แย่
        """
        group_col = self._find_column(df, ['groupname1', 'groupname', 'กลุ่มวิชา', 'course_group', 'group'])
        grade_col = self._find_column(df, ['grade', 'เกรด'])
        grade_point_col = self._find_column(df, ['grade_point', 'คะแนนเกรด', 'gpa_point'])
        
        if not group_col or not grade_col:
            return {}
        
        profiles = {}
        for group_name in df[group_col].dropna().unique():
            group_data = df[df[group_col] == group_name]
            grades = []
            for _, row in group_data.iterrows():
                gp_val = row.get(grade_point_col) if grade_point_col else None
                g = self._convert_grade_to_numeric(row[grade_col], gp_val)
                if g is not None:
                    grades.append(g)
            
            if len(grades) >= 3:
                profiles[str(group_name)] = {
                    'avg_grade': np.mean(grades),
                    'fail_rate': sum(1 for g in grades if g == 0) / len(grades),
                    'pass_rate': sum(1 for g in grades if g > 0) / len(grades),
                    'sample_size': len(grades),
                    'is_core': 'บังคับ' in str(group_name) or 'แกน' in str(group_name),
                    'is_elective': 'เลือก' in str(group_name),
                }
        
        if profiles:
            logger.info(f"\U0001f4da Created course group profiles for {len(profiles)} groups")
        return profiles
    
    def _normalize_course_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize course codes: แปลงรหัสเก่า → ใหม่
        ใช้ทั้ง code mapping และ name-based matching
        """
        course_col = self._find_column(df, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        if not course_col:
            return df
        
        # หา course name column สำหรับ fallback
        name_col_th = self._find_column(df, ['course_title_th', 'ชื่อวิชา', 'วิชา'])
        name_col_en = self._find_column(df, ['course_title_en', 'course_name'])
        name_col = name_col_th or name_col_en
        
        original_codes = df[course_col].nunique()
        
        # Normalize แต่ละแถว
        def _normalize_row(row):
            code = str(row[course_col]).strip() if pd.notna(row[course_col]) else ''
            name = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else None
            return normalize_course_code(code, name)
        
        df[course_col] = df.apply(_normalize_row, axis=1)
        
        new_codes = df[course_col].nunique()
        if new_codes != original_codes:
            logger.info(f"🔄 Course code normalization: {original_codes} → {new_codes} unique codes")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ทำความสะอาดข้อมูล"""
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle special characters in column names
        df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
        
        return df
    
    def _create_course_dna_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        สร้าง DNA Profile ของแต่ละวิชาจากข้อมูลทั้งหมด
        วิเคราะห์ความยาก-ง่าย, อัตราการตก, การกระจายของเกรด
        เพิ่มความสามารถเชิงบริบท (Context-Aware) สำหรับการเปรียบเทียบ
        """
        course_profiles = {}
        
        # หาคอลัมน์ที่เกี่ยวข้อง
        course_col = self._find_column(df, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        grade_col = self._find_column(df, ['grade', 'เกรด'])
        grade_point_col = self._find_column(df, ['grade_point', 'คะแนนเกรด', 'gpa_point'])
        credit_col = self._find_column(df, ['credit', 'หน่วยกิต'])
        
        if not course_col or not grade_col:
            logger.warning("Cannot find course or grade columns for DNA profiling")
            return {}
        
        # วิเคราะห์แต่ละวิชา
        unique_courses = df[course_col].dropna().unique()
        logger.info(f"🧬 Analyzing {len(unique_courses)} unique courses for DNA profiling...")
        
        for course in unique_courses:
            course_data = df[df[course_col] == course]
            
            if len(course_data) < 3:  # ลดเกณฑ์ขั้นต่ำเพื่อให้ครอบคลุมมากขึ้น
                continue
            
            # เก็บเกรดทั้งหมดของวิชานี้
            grades = []
            grade_letters = []
            credits = []
            
            for _, row in course_data.iterrows():
                if pd.notna(row[grade_col]):
                    # ส่ง grade_point ด้วยถ้ามี
                    grade_point_val = row[grade_point_col] if grade_point_col and grade_point_col in row else None
                    grade_val = self._convert_grade_to_numeric(row[grade_col], grade_point_val)
                    if grade_val is not None:
                        grades.append(grade_val)
                        grade_letters.append(str(row[grade_col]).upper())
                        
                        # เก็บหน่วยกิต
                        if credit_col and credit_col in row.index:
                            try:
                                credit = float(row[credit_col])
                                credits.append(credit)
                            except:
                                credits.append(3)  # default
                        else:
                            credits.append(3)  # default
            
            if len(grades) < 3:
                continue
            
            # คำนวณ DNA ของวิชาแบบ Context-Aware
            profile = {
                'course_id': str(course),
                'sample_size': len(grades),
                'avg_credit': np.mean(credits) if credits else 3,
                
                # === Core Statistics ===
                'avg_grade': np.mean(grades),
                'median_grade': np.median(grades),
                'std_grade': np.std(grades) if len(grades) > 1 else 0,
                'min_grade': np.min(grades),
                'max_grade': np.max(grades),
                
                # === Performance Distribution ===
                'fail_rate': sum(1 for g in grades if g == 0) / len(grades),
                'withdraw_rate': sum(1 for g in grade_letters if g == 'W') / len(grade_letters),
                'a_rate': sum(1 for g in grades if g >= 3.5) / len(grades),
                'b_plus_rate': sum(1 for g in grades if 3.0 <= g < 3.5) / len(grades),
                'b_rate': sum(1 for g in grades if 2.5 <= g < 3.0) / len(grades),
                'c_plus_rate': sum(1 for g in grades if 2.0 <= g < 2.5) / len(grades),
                'c_rate': sum(1 for g in grades if 1.5 <= g < 2.0) / len(grades),
                'low_grade_rate': sum(1 for g in grades if 0 < g < 2.0) / len(grades),
                'pass_rate': sum(1 for g in grades if g > 0) / len(grades),
                
                # === Difficulty Indicators ===
                'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                'is_killer_course': sum(1 for g in grades if g == 0) / len(grades) > 0.25,
                'is_easy_course': np.mean(grades) > 3.0 and sum(1 for g in grades if g >= 3.5) / len(grades) > 0.4,
                'is_gpa_booster': np.mean(grades) > 3.2 and np.std(grades) < 0.6,
                'is_inconsistent': np.std(grades) > 1.2,
                
                # === Percentiles for Comparison ===
                'percentile_10': np.percentile(grades, 10),
                'percentile_25': np.percentile(grades, 25),
                'percentile_50': np.percentile(grades, 50),
                'percentile_75': np.percentile(grades, 75),
                'percentile_90': np.percentile(grades, 90),
                
                # === Advanced Context Features ===
                'excellence_threshold': np.percentile(grades, 80),  # เกรดที่ถือว่าดีในวิชานี้
                'struggle_threshold': np.percentile(grades, 20),   # เกรดที่ถือว่าแย่ในวิชานี้
                'competitive_index': self._calculate_competitive_index(grades),
                'grade_inflation': 1 if np.mean(grades) > 3.0 and sum(1 for g in grades if g >= 3.5) / len(grades) > 0.5 else 0,
                
                # === Classification ===
                'course_type': self._classify_course_type(grades, grade_letters),
                'difficulty_level': self._classify_difficulty_level(grades, grade_letters),
                'risk_level': self._calculate_risk_level(grades, grade_letters),
                
                # === Contextual Insights ===
                'typical_grade': self._find_typical_grade(grades),
                'grade_distribution': self._create_grade_distribution(grades),
                'success_rate': sum(1 for g in grades if g >= 2.0) / len(grades),  # C หรือดีกว่า
                
                # === Comparative Metrics ===
                'relative_difficulty': 0,  # จะคำนวณหลังจากวิเคราะห์ทุกวิชาแล้ว
                'percentile_rank': 0       # จะคำนวณหลังจากวิเคราะห์ทุกวิชาแล้ว
            }
            
            course_profiles[str(course)] = profile
        
        # คำนวณ Relative Difficulty และ Percentile Rank
        if course_profiles:
            self._calculate_relative_metrics(course_profiles)
        
        logger.info(f"✅ Created DNA profiles for {len(course_profiles)} courses")
        
        # สรุปสถิติ Course DNA
        self._summarize_course_dna_stats(course_profiles)
        
        return course_profiles
    
    def _calculate_competitive_index(self, grades: List[float]) -> float:
        """คำนวณดัชนีความแข่งขัน (0-1) ยิ่งสูงยิ่งแข่งขันหนัก"""
        if not grades or len(grades) < 3:
            return 0.5
        
        # ดูการกระจายของเกรด ถ้ากระจายมาก = แข่งขันหนัก
        std_grade = np.std(grades)
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        high_grade_rate = sum(1 for g in grades if g >= 3.5) / len(grades)
        
        # คำนวณดัชนี
        competitive_score = (
            std_grade * 0.4 +           # ความแปรปรวนสูง = แข่งขันหนัก
            fail_rate * 0.3 +           # คนตกเยอะ = แข่งขันหนัก
            (1 - high_grade_rate) * 0.3 # คนได้เกรดสูงน้อย = แข่งขันหนัก
        )
        
        return min(1.0, max(0.0, competitive_score))
    
    def _classify_difficulty_level(self, grades: List[float], grade_letters: List[str]) -> str:
        """จำแนกระดับความยากของวิชา"""
        if not grades:
            return 'unknown'
        
        avg_grade = np.mean(grades)
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        a_rate = sum(1 for g in grades if g >= 3.5) / len(grades)
        
        if fail_rate > 0.3:
            return 'very_hard'
        elif fail_rate > 0.15 and avg_grade < 2.5:
            return 'hard'
        elif avg_grade > 3.2 and a_rate > 0.4:
            return 'easy'
        elif avg_grade > 2.8 and fail_rate < 0.05:
            return 'moderate_easy'
        else:
            return 'moderate'
    
    def _calculate_risk_level(self, grades: List[float], grade_letters: List[str]) -> str:
        """คำนวณระดับความเสี่ยงของวิชา"""
        if not grades:
            return 'unknown'
        
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        withdraw_rate = sum(1 for g in grade_letters if g == 'W') / len(grade_letters)
        low_grade_rate = sum(1 for g in grades if 0 < g < 2.0) / len(grades)
        
        total_risk = fail_rate + withdraw_rate + low_grade_rate * 0.5
        
        if total_risk > 0.4:
            return 'high_risk'
        elif total_risk > 0.2:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def _find_typical_grade(self, grades: List[float]) -> float:
        """หาเกรดที่พบบ่อยที่สุด (mode) หรือ median"""
        if not grades:
            return 0
        
        # ปัดเกรดเป็นทศนิยม 1 ตำแหน่ง แล้วหา mode
        rounded_grades = [round(g * 2) / 2 for g in grades]  # ปัดเป็น 0.5
        
        from collections import Counter
        grade_counts = Counter(rounded_grades)
        most_common = grade_counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        else:
            return np.median(grades)
    
    def _create_grade_distribution(self, grades: List[float]) -> Dict[str, float]:
        """สร้างการกระจายของเกรดแบบละเอียด"""
        if not grades:
            return {}
        
        total = len(grades)
        return {
            'A_rate': sum(1 for g in grades if g >= 3.5) / total,
            'B_rate': sum(1 for g in grades if 2.5 <= g < 3.5) / total,
            'C_rate': sum(1 for g in grades if 1.5 <= g < 2.5) / total,
            'D_rate': sum(1 for g in grades if 0.5 <= g < 1.5) / total,
            'F_rate': sum(1 for g in grades if g == 0) / total
        }
    
    def _calculate_relative_metrics(self, course_profiles: Dict[str, Dict]):
        """คำนวณ metrics เชิงเปรียบเทียบระหว่างวิชา"""
        if not course_profiles:
            return
        
        # เก็บค่าเฉลี่ยของทุกวิชา
        all_avg_grades = [profile['avg_grade'] for profile in course_profiles.values()]
        all_difficulty_scores = [profile['difficulty_score'] for profile in course_profiles.values()]
        
        # คำนวณ percentile rank สำหรับแต่ละวิชา
        for course_id, profile in course_profiles.items():
            # Relative difficulty (เทียบกับวิชาอื่น)
            easier_courses = sum(1 for avg in all_avg_grades if avg > profile['avg_grade'])
            profile['relative_difficulty'] = easier_courses / len(all_avg_grades)
            
            # Percentile rank ของความยาก
            harder_courses = sum(1 for diff in all_difficulty_scores if diff > profile['difficulty_score'])
            profile['percentile_rank'] = harder_courses / len(all_difficulty_scores)
    
    def _summarize_course_dna_stats(self, course_profiles: Dict[str, Dict]):
        """สรุปสถิติ Course DNA"""
        if not course_profiles:
            return
        
        total_courses = len(course_profiles)
        killer_courses = sum(1 for p in course_profiles.values() if p['is_killer_course'])
        easy_courses = sum(1 for p in course_profiles.values() if p['is_easy_course'])
        gpa_boosters = sum(1 for p in course_profiles.values() if p['is_gpa_booster'])
        
        avg_difficulty = np.mean([p['difficulty_score'] for p in course_profiles.values()])
        avg_fail_rate = np.mean([p['fail_rate'] for p in course_profiles.values()])
        
        logger.info(f"📊 Course DNA Summary:")
        logger.info(f"   - Total courses analyzed: {total_courses}")
        logger.info(f"   - Killer courses (>25% fail): {killer_courses} ({killer_courses/total_courses*100:.1f}%)")
        logger.info(f"   - Easy courses: {easy_courses} ({easy_courses/total_courses*100:.1f}%)")
        logger.info(f"   - GPA boosters: {gpa_boosters} ({gpa_boosters/total_courses*100:.1f}%)")
        logger.info(f"   - Average difficulty score: {avg_difficulty:.3f}")
        logger.info(f"   - Average fail rate: {avg_fail_rate*100:.1f}%")
    
    def _transform_transcript_to_students(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        แปลงข้อมูล Transcript (หลายแถวต่อนักศึกษา) เป็น Student Records
        พร้อมคำนวณสถานะการจบอัตโนมัติตามจำนวนปีที่เรียน
        
        ✅ ปรับปรุง: กรองนศ.ที่ยังเรียนไม่ครบ, ใช้เกณฑ์ GPA+หน่วยกิต+ปี ร่วม
        """
        student_records = {}
        
        # หาคอลัมน์ Student ID - รองรับหลาย format
        student_col = self._find_column(df, ['dummy_studentno', 'dummy studentno', 'studentno', 'student_id', 'student', 'รหัสนักศึกษา', 'id', 'name', 'ชื่อ'])
        
        if not student_col:
            # ถ้าไม่มี student ID ให้สร้างจากชื่อหรือ index
            logger.warning("Cannot find Student ID column, creating dummy IDs")
            df['dummy_student_id'] = [f"student_{i}" for i in range(len(df))]
            student_col = 'dummy_student_id'
        
        # หาคอลัมน์สำคัญสำหรับคำนวณ graduation
        grade_col = self._find_column(df, ['grade', 'เกรด'])
        credit_col = self._find_column(df, ['credit', 'หน่วยกิต'])
        grade_point_col = self._find_column(df, ['grade_point', 'คะแนนเกรด', 'gpa_point'])
        entry_year_col = self._find_column(df, ['ปีที่เข้า', 'entry_year', 'admission_year'])
        academic_year_col = self._find_column(df, ['ปีการศึกษา', 'academic_year', 'year'])
        
        # Group by student
        df[student_col] = df[student_col].fillna('unknown')
        df[student_col] = df[student_col].astype(str)
        unique_students = df[student_col].unique()
        
        logger.info(f"📊 Processing {len(unique_students)} students...")
        
        graduation_stats = {'graduated': 0, 'not_graduated': 0, 'excluded': 0}
        
        # --- กำหนดว่า "ปีปัจจุบัน" คืออะไร เพื่อกรอง นศ. ที่ยังเรียนอยู่ ---
        current_calendar_year = datetime.now().year  # ค.ศ. ปัจจุบัน (2026)
        # หา max academic year ในข้อมูลเพื่อตรวจว่าเป็น พ.ศ. หรือ ค.ศ.
        max_acad_year = 0
        if academic_year_col:
            try:
                max_acad_year = int(df[academic_year_col].dropna().max())
            except:
                pass
        
        # ถ้า max > 2500 แสดงว่าเป็น พ.ศ.
        is_be_system = max_acad_year > 2500
        current_acad_year_be = current_calendar_year + 543 if is_be_system else current_calendar_year
        
        for i, student_id in enumerate(unique_students):
            try:
                student_data = df[df[student_col] == student_id].copy()
                
                if student_data.empty:
                    continue
                
                # คำนวณจำนวนปีที่เรียนจากข้อมูล (แก้ไข พ.ศ./ค.ศ.)
                years_studied = self._calculate_years_studied(student_data)
                
                # --- ตรวจว่า นศ. มีข้อมูลเพียงพอหรือยัง ---
                # กรอง นศ. ที่เข้ามาน้อยกว่า 4 ปี → ยังตัดสินไม่ได้ว่าจบหรือไม่
                entry_year = None
                if entry_year_col:
                    try:
                        entry_year = int(student_data[entry_year_col].dropna().iloc[0])
                    except:
                        pass
                
                # คำนวณว่า นศ.คนนี้ มีเวลา >= 4 ปี ที่จะจบหรือยัง
                min_years_to_graduate = 4
                student_has_enough_time = True
                
                if entry_year:
                    # แปลง entry_year เป็น พ.ศ. ถ้าจำเป็น
                    entry_be = entry_year + 543 if entry_year < 2500 else entry_year
                    potential_years = (current_acad_year_be - entry_be + 1)
                    
                    if potential_years < min_years_to_graduate:
                        # นศ. เข้ามาน้อยกว่า 4 ปี → ยังตัดสินไม่ได้
                        student_has_enough_time = False
                
                if not student_has_enough_time:
                    graduation_stats['excluded'] += 1
                    # ยังคง include ใน training แต่ใช้ GPA+trajectory-based label
                    gpa = self._calculate_student_gpa(student_data, grade_col, credit_col, grade_point_col)
                    # คำนวณ pass rate (ไม่ F ไม่ W)
                    pass_grades = 0
                    total_graded = 0
                    for _, row in student_data.iterrows():
                        g = str(row.get(grade_col, '')).strip().upper()
                        if g in ('A', 'B+', 'B', 'C+', 'C', 'D+', 'D'):
                            pass_grades += 1
                            total_graded += 1
                        elif g in ('F',):
                            total_graded += 1
                    pass_rate = pass_grades / total_graded if total_graded > 0 else 0.5
                    # นศ.ที่ยังเรียนอยู่: GPA >= 2.0 AND pass_rate > 60% → คาดว่าจบ
                    graduated_status = 1 if (gpa >= 2.0 and pass_rate > 0.6) else 0
                else:
                    # --- คำนวณ graduation ด้วยเกณฑ์ผสม (ปี + GPA + หน่วยกิต) ---
                    gpa = self._calculate_student_gpa(student_data, grade_col, credit_col, grade_point_col)
                    total_credits = 0
                    passed_credits = 0
                    if credit_col:
                        try:
                            total_credits = student_data[credit_col].astype(float).sum()
                            # นับเฉพาะหน่วยกิตที่ผ่าน (ไม่รวม F, W)
                            for _, row in student_data.iterrows():
                                g = str(row.get(grade_col, '')).strip().upper()
                                if g in ('A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'S'):
                                    try:
                                        passed_credits += float(row[credit_col])
                                    except:
                                        passed_credits += 3
                        except:
                            total_credits = len(student_data) * 3
                            passed_credits = total_credits * 0.7
                    
                    # เกณฑ์ผสม: เรียน ≤ 5 ปี AND GPA >= 2.0 AND หน่วยกิตที่ผ่าน >= 120
                    # (หลักสูตรวิศวกรรมต้อง ~136 หน่วยกิต แต่ใช้ 120 เป็น threshold)
                    min_credits = 120
                    graduated_status = 1 if (years_studied <= 5 and gpa >= 2.0 and passed_credits >= min_credits) else 0
                    
                    # กรณีที่เรียนนาน (6+ ปี) แต่ GPA ดี = อาจจะแค่ช้า
                    if years_studied > 5 and years_studied <= 7 and gpa >= 2.5 and passed_credits >= min_credits:
                        graduated_status = 1  # จบช้าแต่จบได้
                
                # นับสถิติ
                if graduated_status == 1:
                    graduation_stats['graduated'] += 1
                else:
                    graduation_stats['not_graduated'] += 1
                
                # Log progress ทุก 50 students
                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i+1}/{len(unique_students)} students...")
                
                # เรียงตามเวลา
                student_data = self._sort_by_time(student_data)
                
                student_records[str(student_id)] = {
                    'data': student_data,
                    'graduated': graduated_status,
                    'years_studied': years_studied,
                    'gpa': gpa,
                    'has_enough_time': student_has_enough_time
                }
                
            except Exception as e:
                logger.warning(f"Error processing student {student_id}: {e}")
                continue
        
        # สรุปผล
        logger.info(f"✅ Classification results:")
        logger.info(f"   - จบตามเกณฑ์: {graduation_stats['graduated']} คน")
        logger.info(f"   - ไม่จบตามเกณฑ์: {graduation_stats['not_graduated']} คน")
        logger.info(f"   - นศ.ยังเรียนอยู่ (ใช้ GPA-based label): {graduation_stats['excluded']} คน")
        
        return student_records
    
    def _calculate_student_gpa(self, student_data: pd.DataFrame, grade_col, credit_col, grade_point_col) -> float:
        """คำนวณ GPA ของนักศึกษาจากข้อมูล"""
        if not grade_col:
            return 2.0  # default
        
        grades = []
        credits = []
        
        for _, row in student_data.iterrows():
            if pd.notna(row.get(grade_col)):
                gp_val = row.get(grade_point_col) if grade_point_col else None
                grade_val = self._convert_grade_to_numeric(row[grade_col], gp_val)
                if grade_val is not None:
                    grades.append(grade_val)
                    try:
                        c = float(row[credit_col]) if credit_col and pd.notna(row.get(credit_col)) else 3
                        credits.append(c)
                    except:
                        credits.append(3)
        
        if not grades:
            return 2.0
        
        total_points = sum(g * c for g, c in zip(grades, credits))
        total_credits = sum(credits)
        return total_points / total_credits if total_credits > 0 else 0
    
    def _calculate_years_studied(self, student_data: pd.DataFrame) -> int:
        """
        คำนวณจำนวนปีที่เรียนจากข้อมูล transcript
        ✅ แก้ไข: ปีที่เข้า (ค.ศ.) กับ ปีการศึกษา (พ.ศ.) แปลงให้ตรงกัน
        """
        entry_year_col = self._find_column(student_data, ['ปีที่เข้า', 'entry_year', 'admission_year'])
        academic_year_col = self._find_column(student_data, ['ปีการศึกษา', 'academic_year', 'year'])
        
        # Method 1: ใช้ "ปีที่เข้า" + "ปีการศึกษา" (แม่นยำที่สุด)
        if entry_year_col and academic_year_col:
            try:
                entry_years = student_data[entry_year_col].dropna().unique()
                academic_years = student_data[academic_year_col].dropna().unique()
                
                if len(entry_years) > 0 and len(academic_years) > 0:
                    entry_year = int(entry_years[0])
                    
                    year_values = []
                    for y in academic_years:
                        year_int = self._convert_year_to_int(y)
                        if year_int:
                            year_values.append(year_int)
                    
                    if year_values:
                        last_academic_year = max(year_values)
                        
                        # --- แปลงให้เป็นระบบเดียวกัน ---
                        # ตรวจว่า entry_year เป็น ค.ศ. หรือ พ.ศ.
                        entry_is_ce = entry_year < 2500  # <2500 = ค.ศ.
                        acad_is_be = last_academic_year > 2500  # >2500 = พ.ศ.
                        
                        if entry_is_ce and acad_is_be:
                            # ปีที่เข้าเป็น ค.ศ., ปีการศึกษาเป็น พ.ศ. → แปลง entry เป็น พ.ศ.
                            entry_year_be = entry_year + 543
                            years_studied = last_academic_year - entry_year_be + 1
                        elif not entry_is_ce and not acad_is_be:
                            # ปีที่เข้าเป็น พ.ศ., ปีการศึกษาเป็น ค.ศ. → แปลง entry เป็น ค.ศ.
                            entry_year_ce = entry_year - 543
                            years_studied = last_academic_year - entry_year_ce + 1
                        else:
                            # เป็นระบบเดียวกันอยู่แล้ว
                            years_studied = last_academic_year - entry_year + 1
                        
                        return max(1, min(10, years_studied))
                        
            except Exception as e:
                logger.debug(f"Error in Method 1: {e}")
        
        # Method 2: ใช้ปีการศึกษาเพียงอย่างเดียว (range)
        if academic_year_col:
            try:
                years = student_data[academic_year_col].dropna().unique()
                if len(years) > 0:
                    year_values = []
                    for y in years:
                        year_int = self._convert_year_to_int(y)
                        if year_int:
                            year_values.append(year_int)
                    
                    if year_values:
                        return max(1, min(10, max(year_values) - min(year_values) + 1))
            except Exception as e:
                logger.debug(f"Error in Method 2: {e}")
        
        # Method 3: นับจำนวนเทอมที่แตกต่างกัน (รวม year+term)
        term_col = self._find_column(student_data, ['term', 'semester', 'ภาคเรียน', 'เทอม'])
        if term_col and academic_year_col:
            try:
                unique_terms = student_data.drop_duplicates(subset=[academic_year_col, term_col])
                # นับเฉพาะเทอม 1 และ 2 (ไม่นับ summer)
                main_terms = len(unique_terms[unique_terms[term_col].isin([1, 2, '1', '2'])])
                if main_terms > 0:
                    return max(1, min(10, (main_terms + 1) // 2))
            except:
                pass
        
        # Method 4: นับจากจำนวนวิชา (fallback)
        total_courses = len(student_data)
        courses_per_year = 14
        return max(1, min(8, (total_courses + courses_per_year - 1) // courses_per_year))
    
    def _convert_year_to_int(self, year_value) -> Optional[int]:
        """แปลงปีเป็นตัวเลข รองรับ พ.ศ./ค.ศ."""
        if pd.isna(year_value):
            return None
        
        try:
            year_str = str(year_value).strip()
            
            # ลบอักขระพิเศษ
            year_str = year_str.replace('/', '').replace('-', '').replace(' ', '')
            
            # แปลงเป็นตัวเลข
            year_int = int(float(year_str))
            
            # ตรวจสอบว่าเป็นปีที่สมเหตุสมผล
            if 1900 <= year_int <= 2100:  # ค.ศ.
                return year_int
            elif 2400 <= year_int <= 2700:  # พ.ศ.
                return year_int
            else:
                return None
                
        except (ValueError, TypeError):
            return None
    
    def _sort_by_time(self, student_data: pd.DataFrame) -> pd.DataFrame:
        """เรียงข้อมูลตามเวลา"""
        # Try to sort by year and term
        year_col = self._find_column(student_data, ['year', 'ปี', 'ปีการศึกษา'])
        term_col = self._find_column(student_data, ['term', 'semester', 'เทอม'])
        
        if year_col and term_col:
            try:
                return student_data.sort_values([year_col, term_col])
            except:
                pass
        elif year_col:
            try:
                return student_data.sort_values(year_col)
            except:
                pass
        
        return student_data
    
    def _create_temporal_snapshots(self, student_id: str, student_record: Dict) -> List[Dict]:
        """
        สร้าง Dynamic Snapshots สำหรับนักศึกษาแต่ละคน
        จำลองสถานการณ์ในแต่ละช่วงเวลา (เทอม) ของการเรียน
        """
        snapshots = []
        student_data = student_record['data']
        graduated = student_record['graduated']
        
        # หาคอลัมน์ที่ต้องใช้
        course_col = self._find_column(student_data, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        grade_col = self._find_column(student_data, ['grade', 'เกรด'])
        credit_col = self._find_column(student_data, ['credit', 'หน่วยกิต'])
        year_col = self._find_column(student_data, ['year', 'ปี', 'ปีการศึกษา'])
        term_col = self._find_column(student_data, ['term', 'semester', 'เทอม'])
        
        if not course_col or not grade_col:
            return snapshots
        
        # กำหนด breakpoints สำหรับการสร้าง snapshots
        if year_col and term_col:
            # Group by year-term
            student_data['time_key'] = student_data[year_col].astype(str) + '_' + student_data[term_col].astype(str)
            time_groups = student_data.groupby('time_key')
            
            accumulated_data = pd.DataFrame()
            for time_key, group_data in time_groups:
                accumulated_data = pd.concat([accumulated_data, group_data])
                snapshot = self._create_snapshot_features(
                    student_id=student_id,
                    snapshot_id=f"{student_id}_{time_key}",
                    courses_data=accumulated_data,
                    course_col=course_col,
                    grade_col=grade_col,
                    credit_col=credit_col,
                    graduated=graduated
                )
                if snapshot:
                    snapshots.append(snapshot)
        else:
            # สร้าง snapshots ทุกๆ กลุ่มของวิชา (simulate terms)
            courses_per_term = 6
            total_courses = len(student_data)
            
            for i in range(courses_per_term, total_courses + 1, courses_per_term):
                current_data = student_data.iloc[:i]
                snapshot = self._create_snapshot_features(
                    student_id=student_id,
                    snapshot_id=f"{student_id}_snapshot_{i//courses_per_term}",
                    courses_data=current_data,
                    course_col=course_col,
                    grade_col=grade_col,
                    credit_col=credit_col,
                    graduated=graduated
                )
                if snapshot:
                    snapshots.append(snapshot)
            
            # เพิ่ม final snapshot ด้วยข้อมูลทั้งหมด
            if total_courses % courses_per_term != 0 or len(snapshots) == 0:
                final_snapshot = self._create_snapshot_features(
                    student_id=student_id,
                    snapshot_id=f"{student_id}_final",
                    courses_data=student_data,
                    course_col=course_col,
                    grade_col=grade_col,
                    credit_col=credit_col,
                    graduated=graduated
                )
                if final_snapshot:
                    snapshots.append(final_snapshot)
        
        return snapshots
    
    def _create_snapshot_features(self, student_id: str, snapshot_id: str, 
                                 courses_data: pd.DataFrame, course_col: str, 
                                 grade_col: str, credit_col: str, 
                                 graduated: int) -> Dict:
        """
        สร้าง Standardized Feature Set สำหรับ snapshot หนึ่งๆ
        นี่คือหัวใจของระบบ - สร้างชุด features มาตรฐานที่โมเดลต้องการ
        """
        grades = []
        credits = []
        grade_letters = []  # ✅ เพิ่ม: เก็บ grade letter สำหรับนับ W
        course_grades_detail = {}
        
        # หา grade_point column
        grade_point_col = self._find_column(courses_data, ['grade_point', 'คะแนนเกรด', 'gpa_point'])
        
        # Context-aware features
        contextual_features = {
            'vs_avg_scores': [],
            'passed_killer': 0,
            'struggled_easy': 0,
            'better_than_avg': 0,
            'worse_than_avg': 0
        }
        
        # ประมวลผลแต่ละวิชา
        for _, row in courses_data.iterrows():
            if pd.notna(row[course_col]) and pd.notna(row[grade_col]):
                course_id = str(row[course_col])
                
                # ส่ง grade_point ด้วยถ้ามี
                grade_point_val = row[grade_point_col] if grade_point_col and grade_point_col in row else None
                grade_val = self._convert_grade_to_numeric(row[grade_col], grade_point_val)
                
                if grade_val is None:
                    continue
                
                grades.append(grade_val)
                grade_letters.append(str(row[grade_col]).strip().upper())  # ✅ เก็บ grade letter
                if credit_col and credit_col in row.index:
                    try:
                        credit = float(row[credit_col])
                        credits.append(credit)
                    except:
                        credits.append(3)  # default
                else:
                    credits.append(3)
                
                course_grades_detail[course_id] = grade_val
                
                # เปรียบเทียบกับ Course DNA
                if course_id in self.course_profiles:
                    profile = self.course_profiles[course_id]
                    
                    # Performance vs average
                    vs_avg = grade_val - profile['avg_grade']
                    contextual_features['vs_avg_scores'].append(vs_avg)
                    
                    if vs_avg > 0:
                        contextual_features['better_than_avg'] += 1
                    else:
                        contextual_features['worse_than_avg'] += 1
                    
                    # Performance in different course types
                    if profile['is_killer_course'] and grade_val > 0:
                        contextual_features['passed_killer'] += 1
                    
                    if profile['is_easy_course'] and grade_val < 2.0:
                        contextual_features['struggled_easy'] += 1
        
        if not grades:
            return None
        
        # คำนวณ GPA แบบถ่วงน้ำหนักด้วยหน่วยกิต
        if credits and len(credits) == len(grades):
            total_points = sum(g * c for g, c in zip(grades, credits))
            total_credits = sum(credits)
            gpa = total_points / total_credits if total_credits > 0 else 0
        else:
            gpa = np.mean(grades)
        
        # คำนวณ features ล่าสุด (Recent features)
        recent_window = min(6, len(grades))  # ดู 6 วิชาล่าสุด
        recent_grades = grades[-recent_window:] if len(grades) > recent_window else grades
        
        # === Retake Analysis (วิชาที่เรียนซ้ำ) ===
        retake_count = 0
        retake_improved = 0
        seen_courses = {}
        for course_id, gv in course_grades_detail.items():
            if course_id in seen_courses:
                retake_count += 1
                if gv > seen_courses[course_id]:
                    retake_improved += 1
            seen_courses[course_id] = gv
        
        # === Course Group Analysis (GROUPNAME1) ===
        group_col = self._find_column(courses_data, ['groupname1', 'groupname', 'กลุ่มวิชา', 'course_group', 'group'])
        core_gpa = 0
        core_count = 0
        elective_gpa = 0
        elective_count = 0
        group_diversity = 0
        
        if group_col and hasattr(self, 'course_group_profiles'):
            groups_seen = set()
            for _, row in courses_data.iterrows():
                if pd.notna(row.get(group_col)):
                    grp = str(row[group_col])
                    groups_seen.add(grp)
                    gp_val = row.get(self._find_column(courses_data, ['grade_point', 'คะแนนเกรด']) or '__none__')
                    g = self._convert_grade_to_numeric(row[grade_col], gp_val if pd.notna(gp_val) else None)
                    if g is not None:
                        if 'บังคับ' in grp or 'แกน' in grp or 'พื้นฐาน' in grp:
                            core_gpa += g
                            core_count += 1
                        elif 'เลือก' in grp:
                            elective_gpa += g
                            elective_count += 1
            group_diversity = len(groups_seen)
        
        core_gpa_avg = core_gpa / core_count if core_count > 0 else 0
        elective_gpa_avg = elective_gpa / elective_count if elective_count > 0 else 0
        
        # === Credit Accumulation Rate ===
        total_credits_val = sum(credits) if credits else len(grades) * 3
        # ประมาณปีที่เรียนจากจำนวน snapshots (semesters)
        n_semesters = max(1, len(set(zip(
            [str(row.get(self._find_column(courses_data, ['ปีการศึกษา', 'academic_year', 'year']) or '__none__', ''))
             for _, row in courses_data.iterrows()],
            [str(row.get(self._find_column(courses_data, ['เทอม', 'term', 'semester']) or '__none__', ''))
             for _, row in courses_data.iterrows()]
        ))))
        credits_per_semester = total_credits_val / n_semesters
        
        # สร้าง STANDARDIZED FEATURE SET
        features = {
            'student_id': student_id,
            'snapshot_id': snapshot_id,
            
            # === Overall Features ===
            'GPAX_so_far': gpa,
            'Total_Credits_so_far': total_credits_val,
            'Total_Courses_so_far': len(grades),
            'Total_F_Count_so_far': sum(1 for g in grades if g == 0),
            'Total_W_Count_so_far': sum(1 for gl in grade_letters if gl == 'W'),
            
            # === Trend & Recent Features ===
            'GPA_last_window': np.mean(recent_grades) if recent_grades else 0,
            'GPA_trend': self._calculate_gpa_trend(grades),
            'Credits_last_window': sum(credits[-recent_window:]) if credits else recent_window * 3,
            'Improvement_potential': self._calculate_improvement_potential(grades),
            
            # === Insightful Features ===
            'Core_Courses_Below_C_recent': sum(1 for g in recent_grades if g < 2.0),
            'Failed_Core_Course_Count': sum(1 for g in grades if g == 0),
            'High_Grade_Rate': sum(1 for g in grades if g >= 3.5) / len(grades) if grades else 0,
            'Low_Grade_Rate': sum(1 for g in grades if 0 < g < 2.0) / len(grades) if grades else 0,
            
            # === Statistical Features ===
            'Grade_Mean': np.mean(grades),
            'Grade_Std': np.std(grades) if len(grades) > 1 else 0,
            'Grade_Min': np.min(grades),
            'Grade_Max': np.max(grades),
            'Grade_Median': np.median(grades),
            
            # === Context-Aware Features (Course DNA) ===
            'Avg_vs_Course_Avg': np.mean(contextual_features['vs_avg_scores']) if contextual_features['vs_avg_scores'] else 0,
            'Std_vs_Course_Avg': np.std(contextual_features['vs_avg_scores']) if len(contextual_features['vs_avg_scores']) > 1 else 0,
            'Passed_Killer_Courses': contextual_features['passed_killer'],
            'Struggled_Easy_Courses': contextual_features['struggled_easy'],
            'Better_Than_Avg_Count': contextual_features['better_than_avg'],
            'Worse_Than_Avg_Count': contextual_features['worse_than_avg'],
            
            # === Risk Indicators ===
            'At_Risk_Flag': 1 if gpa < 2.0 else 0,
            'High_Performer_Flag': 1 if gpa >= 3.25 else 0,
            'Consistency_Score': 1 / (1 + np.std(grades)) if len(grades) > 1 else 1,
            
            # === Performance Rates ===
            'Pass_Rate': sum(1 for g in grades if g > 0) / len(grades) if grades else 0,
            'Fail_Rate': sum(1 for g in grades if g == 0) / len(grades) if grades else 0,
            
            # === NEW: Retake Features ===
            'Retake_Count': retake_count,
            'Retake_Improved_Rate': retake_improved / retake_count if retake_count > 0 else 0,
            
            # === NEW: Course Group Features ===
            'Core_GPA': core_gpa_avg,
            'Elective_GPA': elective_gpa_avg,
            'Core_vs_Elective_Gap': core_gpa_avg - elective_gpa_avg if core_count > 0 and elective_count > 0 else 0,
            'Group_Diversity': group_diversity,
            
            # === NEW: Credit Trajectory ===
            'Credits_Per_Semester': credits_per_semester,
            'Credit_Completion_Ratio': total_credits_val / 136.0,  # 136 = typical engineering requirement
            
            # === NEW: Grade Distribution Shape ===
            'Grade_Skewness': float(pd.Series(grades).skew()) if len(grades) > 2 else 0,
            'Grade_Kurtosis': float(pd.Series(grades).kurtosis()) if len(grades) > 3 else 0,
            
            # === Target Variable ===
            'graduated': graduated
        }
        
        return features
    
    def _generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้าง Advanced Features เพิ่มเติม (Feature Engineering)
        """
        # Interaction features
        if 'GPAX_so_far' in df.columns and 'Total_F_Count_so_far' in df.columns:
            df['GPA_Fail_Interaction'] = df['GPAX_so_far'] * (1 + df['Total_F_Count_so_far'])
            
            if 'Fail_Rate' in df.columns:
                df['Risk_Score'] = (4 - df['GPAX_so_far']) * df['Fail_Rate']
            else:
                df['Risk_Score'] = (4 - df['GPAX_so_far']) * 0.1
        
        # Performance consistency
        if 'Passed_Killer_Courses' in df.columns and 'Struggled_Easy_Courses' in df.columns:
            df['Performance_Consistency'] = (
                df['Passed_Killer_Courses'] - df['Struggled_Easy_Courses'] * 2
            )
        
        # Academic strength
        if 'GPAX_so_far' in df.columns and 'Grade_Std' in df.columns:
            df['Academic_Strength'] = df['GPAX_so_far'] / (1 + df['Grade_Std'])
        
        # Polynomial features for key metrics
        if 'GPAX_so_far' in df.columns:
            df['GPAX_Squared'] = df['GPAX_so_far'] ** 2
            df['GPAX_Log'] = np.log1p(df['GPAX_so_far'])
        
        # Ratio features
        if 'Better_Than_Avg_Count' in df.columns and 'Worse_Than_Avg_Count' in df.columns:
            df['Performance_Ratio'] = (
                df['Better_Than_Avg_Count'] / 
                (df['Worse_Than_Avg_Count'] + 1)  # +1 to avoid division by zero
            )
        
        # Progress indicators
        if 'Total_Credits_so_far' in df.columns:
            expected_credits_per_year = 36  # ประมาณ 36 หน่วยกิตต่อปี
            df['Progress_Rate'] = df['Total_Credits_so_far'] / (expected_credits_per_year * 4)
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if col not in ['student_id', 'snapshot_id']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _select_and_normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        เลือกและปรับปรุง features สุดท้าย
        """
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X = X.drop(columns=to_drop, errors='ignore')
        
        # Remove features with very low variance
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        
        try:
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        except:
            logger.warning("Could not apply variance threshold")
        
        return X
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """หาชื่อคอลัมน์จากรายการที่เป็นไปได้ - รองรับหลาย format + content-based detection"""
        if df is None or df.empty:
            return None
            
        # Normalize column names: lowercase, remove spaces/underscores
        def normalize(s):
            return str(s).lower().replace(' ', '').replace('_', '').replace('-', '')
        
        df_columns_normalized = [normalize(col) for col in df.columns]
        
        for name in possible_names:
            name_normalized = normalize(name)
            
            # Exact normalized match first
            if name_normalized in df_columns_normalized:
                idx = df_columns_normalized.index(name_normalized)
                return df.columns[idx]
            
            # Partial match
            for col, col_normalized in zip(df.columns, df_columns_normalized):
                if name_normalized in col_normalized or col_normalized in name_normalized:
                    return col
        
        return None
    
    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        ตรวจจับคอลัมน์อัตโนมัติจากเนื้อหาข้อมูล (content-based detection)
        รองรับ CSV ที่ชื่อคอลัมน์ไม่ตรงมาตรฐาน
        """
        detected = {
            'student_id': None, 'course_code': None, 'grade': None,
            'credit': None, 'grade_point': None, 'entry_year': None,
            'academic_year': None, 'term': None, 'course_name_th': None,
            'course_name_en': None, 'course_group': None
        }
        
        # 1. ลอง name-based ก่อน
        name_map = {
            'student_id': ['dummy_studentno', 'dummy studentno', 'studentno', 'student_id', 'student', 'รหัสนักศึกษา', 'id'],
            'course_code': ['course_code', 'course', 'subject', 'รหัสวิชา'],
            'grade': ['grade', 'เกรด'],
            'credit': ['credit', 'หน่วยกิต'],
            'grade_point': ['grade_point', 'คะแนนเกรด', 'gpa_point'],
            'entry_year': ['ปีที่เข้า', 'entry_year', 'admission_year'],
            'academic_year': ['ปีการศึกษา', 'academic_year', 'year'],
            'term': ['เทอม', 'semester', 'term'],
            'course_name_th': ['course_title_th', 'ชื่อวิชา', 'วิชา'],
            'course_name_en': ['course_title_en', 'course_name'],
            'course_group': ['groupname1', 'groupname', 'กลุ่มวิชา', 'course_group', 'group']
        }
        
        for key, names in name_map.items():
            col = self._find_column(df, names)
            if col:
                detected[key] = col
        
        # 2. Content-based detection สำหรับคอลัมน์ที่ยังหาไม่เจอ
        used_cols = set(v for v in detected.values() if v)
        
        for col in df.columns:
            if col in used_cols:
                continue
            
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            sample_vals = sample.head(100)
            
            # Detect student ID: many unique text values with pattern like A110018-3
            if not detected['student_id']:
                str_vals = sample_vals.astype(str)
                if sample.nunique() > len(df) * 0.01 and str_vals.str.match(r'^[A-Za-z]\d').any():
                    detected['student_id'] = col
                    used_cols.add(col)
                    continue
            
            # Detect course code: patterns like XX-YYY-ZZZ or XX-YYY-ZZZ-NNN
            if not detected['course_code']:
                str_vals = sample_vals.astype(str)
                if str_vals.str.match(r'^\d{2}-\d{3}-\d{3}').mean() > 0.5:
                    detected['course_code'] = col
                    used_cols.add(col)
                    continue
            
            # Detect grade: A, B+, B, C+, C, D+, D, F, W, I, S, U
            if not detected['grade']:
                str_vals = sample_vals.astype(str).str.strip().str.upper()
                grade_set = {'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F', 'W', 'I', 'S', 'U'}
                if str_vals.isin(grade_set).mean() > 0.7:
                    detected['grade'] = col
                    used_cols.add(col)
                    continue
            
            # Detect credit: small integers 1-6
            if not detected['credit']:
                try:
                    num_vals = pd.to_numeric(sample_vals, errors='coerce').dropna()
                    if len(num_vals) > 0 and num_vals.between(1, 6).mean() > 0.9 and num_vals.nunique() <= 6:
                        detected['credit'] = col
                        used_cols.add(col)
                        continue
                except:
                    pass
            
            # Detect grade_point: float 0.0-4.0
            if not detected['grade_point']:
                try:
                    num_vals = pd.to_numeric(sample_vals, errors='coerce').dropna()
                    if len(num_vals) > 0 and num_vals.between(0, 4).mean() > 0.9 and num_vals.nunique() > 3:
                        detected['grade_point'] = col
                        used_cols.add(col)
                        continue
                except:
                    pass
            
            # Detect year columns: integers 2000-2099 (CE) or 2500-2599 (BE)
            if not detected['entry_year'] or not detected['academic_year']:
                try:
                    num_vals = pd.to_numeric(sample_vals, errors='coerce').dropna()
                    if len(num_vals) > 0:
                        is_year = (num_vals.between(2000, 2099) | num_vals.between(2500, 2599)).mean() > 0.9
                        if is_year:
                            # entry_year มีค่า unique น้อย (ปีเดียว per student), academic_year มากกว่า
                            if not detected['entry_year'] and num_vals.nunique() <= 15:
                                detected['entry_year'] = col
                                used_cols.add(col)
                                continue
                            elif not detected['academic_year']:
                                detected['academic_year'] = col
                                used_cols.add(col)
                                continue
                except:
                    pass
            
            # Detect term/semester: integers 1-3
            if not detected['term']:
                try:
                    num_vals = pd.to_numeric(sample_vals, errors='coerce').dropna()
                    if len(num_vals) > 0 and num_vals.between(1, 3).mean() > 0.95 and num_vals.nunique() <= 3:
                        detected['term'] = col
                        used_cols.add(col)
                        continue
                except:
                    pass
        
        logger.info(f"\U0001f50d Auto-detected columns: {', '.join(f'{k}={v}' for k, v in detected.items() if v)}")
        return detected
    
    def _convert_grade_to_numeric(self, grade, grade_point=None) -> Optional[float]:
        """
        แปลงเกรดเป็นตัวเลข
        ให้ความสำคัญกับ GRADE_POINT จากข้อมูลก่อน แล้วค่อยใช้การแปลงเกรดตัวอักษร
        """
        # Method 1: ใช้ GRADE_POINT จากข้อมูลโดยตรง (ความสำคัญสูงสุด)
        if grade_point is not None and not pd.isna(grade_point):
            try:
                numeric_point = float(grade_point)
                # ตรวจสอบว่าอยู่ในช่วงที่สมเหตุสมผล (0-4)
                if 0 <= numeric_point <= 4:
                    return numeric_point
            except (ValueError, TypeError):
                pass
        
        # Method 2: ถ้าไม่มี GRADE_POINT หรือไม่ถูกต้อง ให้ลองแปลงจากเกรด
        if pd.isna(grade):
            return None
        
        # ลองแปลงเป็นตัวเลขโดยตรง
        try:
            numeric = float(grade)
            if 0 <= numeric <= 4:
                return numeric
        except (ValueError, TypeError):
            pass
        
        # แปลงจากตัวอักษร (fallback)
        grade_str = str(grade).strip().upper()
        return self.grade_mapping.get(grade_str)
    
    def _calculate_difficulty_score(self, grades: List[float], grade_letters: List[str]) -> float:
        """คำนวณความยากของวิชา (0-1)"""
        if not grades:
            return 0.5
        
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        low_grade_rate = sum(1 for g in grades if 0 < g < 2.0) / len(grades)
        avg_grade = np.mean(grades)
        std_grade = np.std(grades) if len(grades) > 1 else 0
        
        # Weighted difficulty score
        difficulty = (
            fail_rate * 0.4 +
            low_grade_rate * 0.25 +
            max(0, (2.5 - avg_grade) / 2.5) * 0.2 +
            min(std_grade / 2.0, 1.0) * 0.15
        )
        
        return min(1.0, max(0.0, difficulty))
    
    def _classify_course_type(self, grades: List[float], grade_letters: List[str]) -> str:
        """จำแนกประเภทของวิชา"""
        if not grades:
            return 'unknown'
        
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        avg_grade = np.mean(grades)
        std_grade = np.std(grades) if len(grades) > 1 else 0
        
        if fail_rate > 0.3:
            return 'killer'
        elif avg_grade > 3.0 and std_grade < 0.5:
            return 'easy'
        elif fail_rate < 0.1 and avg_grade > 2.5:
            return 'normal'
        elif std_grade > 1.0:
            return 'inconsistent'
        else:
            return 'challenging'
    
    def _calculate_improvement_potential(self, grades: List[float]) -> float:
        """คำนวณศักยภาพในการพัฒนา"""
        if len(grades) < 2:
            return 0.5
        
        # เปรียบเทียบครึ่งแรกกับครึ่งหลัง
        mid = len(grades) // 2
        first_half = grades[:mid]
        second_half = grades[mid:]
        
        if first_half and second_half:
            improvement = np.mean(second_half) - np.mean(first_half)
            return min(1.0, max(0.0, (improvement + 2) / 4))
        
        return 0.5
    
    def _calculate_gpa_trend(self, grades: List[float]) -> float:
        """คำนวณแนวโน้ม GPA"""
        if len(grades) < 2:
            return 0
        
        # Simple linear regression
        x = np.arange(len(grades))
        if len(grades) > 1:
            try:
                slope, _ = np.polyfit(x, grades, 1)
                return np.clip(slope, -1, 1)
            except:
                return 0
        
        return 0


# Keep the existing train_ensemble_model function - enhanced with CV and calibration
def train_ensemble_model(X, y):
    """
    Train ensemble model with advanced techniques
    Enhanced: StratifiedKFold CV, calibration, better hyperparameters
    """
    logger.info("\U0001f680 Starting Advanced Ensemble Model Training...")
    logger.info(f"\U0001f4ca Input shape: X={X.shape}, y={y.shape}")
    
    try:
        from collections import Counter
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"\U0001f4ca Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Ensure minimum samples per class
        min_class_count = min(class_counts) if len(class_counts) > 0 else 0
        
        if len(unique_classes) < 2:
            logger.warning("\u26a0\ufe0f Only one class found! Adding synthetic minority class...")
            minority_class = 1 if unique_classes[0] == 0 else 0
            for _ in range(max(3, int(len(X) * 0.1))):
                X = pd.concat([X, X.iloc[[0]]], ignore_index=True)
                y = pd.concat([y, pd.Series([minority_class])], ignore_index=True)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"\U0001f4ca After synthetic: {dict(zip(unique_classes, class_counts))}")
        
        if min_class_count < 3:
            logger.warning(f"\u26a0\ufe0f Insufficient samples in minority class: {min_class_count}")
            minority_class = unique_classes[np.argmin(class_counts)]
            needed = 3 - min_class_count
            minority_indices = np.where(y == minority_class)[0]
            if len(minority_indices) > 0:
                for _ in range(needed):
                    idx = np.random.choice(minority_indices)
                    X = pd.concat([X, X.iloc[[idx]]], ignore_index=True)
                    y = pd.concat([y, pd.Series([minority_class])], ignore_index=True)
        
        # --- Cross-Validation setup ---
        n_splits = min(5, min(Counter(y).values()))
        n_splits = max(2, n_splits)
        
        # Adaptive test size
        test_size = min(0.2, max(0.1, 10 / len(X)))
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"\U0001f4ca Train/Test split: {len(X_train)}/{len(X_test)}")
        
        # Apply SMOTE if possible
        try:
            min_samples = min(Counter(y_train).values())
            if min_samples >= 2:
                k_neighbors = min(5, min_samples - 1)
                k_neighbors = max(1, k_neighbors)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                logger.info(f"\u2705 Applied SMOTE. New distribution: {Counter(y_train_resampled)}")
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
        except Exception as e:
            logger.warning(f"\u26a0\ufe0f SMOTE not applied: {e}")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
        
        # --- Train models with better hyperparameters ---
        models = {}
        cv_scores = {}
        
        # Random Forest - tuned
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=1,
                class_weight='balanced',
                oob_score=True
            )
            rf.fit(X_train_resampled, y_train_resampled)
            models['rf'] = rf
            
            # Cross-validation score
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                rf_cv = cross_val_score(rf, X_train_resampled, y_train_resampled, cv=cv, scoring='f1')
                cv_scores['rf'] = rf_cv.mean()
                logger.info(f"\u2705 Random Forest trained (CV F1: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}, OOB: {rf.oob_score_:.3f})")
            except:
                logger.info("\u2705 Random Forest trained successfully")
            
            # Log feature importance
            if hasattr(rf, 'feature_importances_'):
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                top_features = importances.nlargest(10)
                logger.info(f"\U0001f3af Top 10 important features:")
                for feat, imp in top_features.items():
                    logger.info(f"   - {feat}: {imp:.4f}")
                    
        except Exception as e:
            logger.error(f"\u274c Random Forest training failed: {e}")
        
        # Gradient Boosting - tuned
        try:
            gb = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            gb.fit(X_train_resampled, y_train_resampled)
            models['gb'] = gb
            
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                gb_cv = cross_val_score(gb, X_train_resampled, y_train_resampled, cv=cv, scoring='f1')
                cv_scores['gb'] = gb_cv.mean()
                logger.info(f"\u2705 Gradient Boosting trained (CV F1: {gb_cv.mean():.3f} +/- {gb_cv.std():.3f})")
            except:
                logger.info("\u2705 Gradient Boosting trained successfully")
        except Exception as e:
            logger.error(f"\u274c Gradient Boosting training failed: {e}")
        
        # Logistic Regression - tuned
        try:
            lr = LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear',
                C=0.5
            )
            lr.fit(X_train_scaled, y_train_resampled)
            models['lr'] = lr
            
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                lr_cv = cross_val_score(lr, X_train_scaled, y_train_resampled, cv=cv, scoring='f1')
                cv_scores['lr'] = lr_cv.mean()
                logger.info(f"\u2705 Logistic Regression trained (CV F1: {lr_cv.mean():.3f} +/- {lr_cv.std():.3f})")
            except:
                logger.info("\u2705 Logistic Regression trained successfully")
        except Exception as e:
            logger.error(f"\u274c Logistic Regression training failed: {e}")
        
        # --- Weighted Ensemble based on CV scores ---
        model_weights = {}
        if cv_scores:
            total_cv = sum(cv_scores.values())
            if total_cv > 0:
                model_weights = {k: v / total_cv for k, v in cv_scores.items()}
            else:
                model_weights = {k: 1.0 / len(cv_scores) for k in cv_scores}
            logger.info(f"\U0001f3af Model weights (from CV): {', '.join(f'{k}={v:.3f}' for k, v in model_weights.items())}")
        
        # Evaluate ensemble
        if len(X_test) > 0 and models:
            predictions_proba = []
            weights_list = []
            for name, model in models.items():
                if name == 'lr':
                    pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    pred_proba = model.predict_proba(X_test)[:, 1]
                predictions_proba.append(pred_proba)
                weights_list.append(model_weights.get(name, 1.0 / len(models)))
            
            # Weighted average ensemble
            weights_arr = np.array(weights_list)
            weights_arr = weights_arr / weights_arr.sum()
            ensemble_proba = np.average(predictions_proba, axis=0, weights=weights_arr)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, ensemble_pred)
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        else:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            logger.warning("\u26a0\ufe0f No test set available, metrics set to 0 (unknown)")
        
        logger.info(f"\U0001f4ca Model Performance:")
        logger.info(f"   - Accuracy: {accuracy:.3f}")
        logger.info(f"   - Precision: {precision:.3f}")
        logger.info(f"   - Recall: {recall:.3f}")
        logger.info(f"   - F1-Score: {f1:.3f}")
        if cv_scores:
            logger.info(f"   - Avg CV F1: {np.mean(list(cv_scores.values())):.3f}")
        
        return {
            'models': models,
            'scaler': scaler,
            'model_weights': model_weights,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'training_samples': len(X_train_resampled),
            'validation_samples': len(X_test),
            'feature_names': list(X.columns)
        }
        
    except Exception as e:
        logger.error(f"\u274c Error in ensemble training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# Backward compatibility
CurriculumAnalyzer = type('CurriculumAnalyzer', (), {})
CourseRetakeSimulator = type('CourseRetakeSimulator', (), {})
CourseNameNormalizer = type('CourseNameNormalizer', (), {})

class ContextAwarePredictor:
    """คลาสสำหรับการทำนายแบบ Context-Aware พร้อม Explainability"""
    
    def __init__(self, feature_engineer: AdvancedFeatureEngineer, models: Dict = None, 
                 scaler: Any = None, feature_names: List[str] = None, model_weights: Dict = None):
        self.feature_engineer = feature_engineer
        self.models = models or {}
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.model_weights = model_weights or {}  # weighted ensemble
        
        # Initialize explainer
        try:
            from explainable_ai import ExplainablePredictor
            self.explainer = ExplainablePredictor(
                models=self.models,
                feature_names=self.feature_names,
                course_profiles=feature_engineer.course_profiles if hasattr(feature_engineer, 'course_profiles') else {},
                courses_data=[]
            )
        except Exception as e:
            logger.warning(f"Could not initialize ExplainablePredictor: {e}")
            self.explainer = None
    
    def predict_graduation_probability(self, student_data: pd.DataFrame, explain: bool = True) -> Dict[str, Any]:
        """
        ทำนายความน่าจะเป็นการจบการศึกษาสำหรับนักศึกษาใหม่
        ใช้ระบบ Context-Aware พร้อม Explainability
        
        Args:
            student_data: ข้อมูลนักศึกษา
            explain: ถ้า True จะคืนคำอธิบายละเอียด
        
        Returns:
            Dict ที่มีผลการทำนายและคำอธิบาย (ถ้า explain=True)
        """
        if not hasattr(self.feature_engineer, 'course_profiles') or not self.feature_engineer.course_profiles:
            logger.warning("⚠️ Course profiles not available. Please train the model first.")
            return {'probability': 0.5, 'confidence': 0.0, 'features_used': 0, 'courses_analyzed': 0}
        
        # สร้าง features ใช้วิธีเดียวกับการเทรน
        # หาคอลัมน์ที่ต้องใช้
        course_col = self.feature_engineer._find_column(student_data, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        grade_col = self.feature_engineer._find_column(student_data, ['grade', 'เกรด'])
        credit_col = self.feature_engineer._find_column(student_data, ['credit', 'หน่วยกิต'])
        
        if not course_col or not grade_col:
            logger.warning("⚠️ Cannot find required columns (course_code, grade)")
            return {'probability': 0.5, 'confidence': 0.0, 'features_used': 0, 'courses_analyzed': 0}
        
        # สร้าง features ใช้ method เดียวกับ training
        features = self.feature_engineer._create_snapshot_features(
            student_id='prediction',
            snapshot_id='prediction_snapshot',
            courses_data=student_data,
            course_col=course_col,
            grade_col=grade_col,
            credit_col=credit_col,
            graduated=0  # dummy value, not used for prediction
        )
        
        if not features or features.get('Total_Courses_so_far', 0) == 0:
            return {'probability': 0.5, 'confidence': 0.0, 'features_used': 0, 'courses_analyzed': 0}
        
        # แปลงเป็น DataFrame
        X = pd.DataFrame([features])
        
        # เพิ่ม advanced features (เหมือนตอนเทรน)
        X = self.feature_engineer._generate_advanced_features(X)
        
        # ลบคอลัมน์ที่ไม่ใช่ feature
        X = X.drop(columns=['graduated', 'student_id', 'snapshot_id'], errors='ignore')
        
        # ===== สำคัญ: จับคู่ features ให้ตรงกับตอนเทรน =====
        if self.feature_names and len(self.feature_names) > 0:
            # เติม missing columns ด้วย 0
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            # เลือกเฉพาะ columns ที่โมเดลต้องการ
            X = X[self.feature_names]
        else:
            # fallback: ทำความสะอาดข้อมูล
            X = X.select_dtypes(include=[np.number])
        
        X = X.fillna(0)
        
        # ===================================================================
        # ใช้โมเดล AI จริง (Ensemble Prediction) - ไม่มี Heuristic Fallback
        # ===================================================================
        
        if not self.models or len(self.models) == 0:
            # ไม่มีโมเดล - แจ้งเตือนและ raise exception
            logger.error("❌ ไม่พบโมเดล AI! กรุณาเทรนโมเดลก่อนใช้งาน")
            raise ValueError(
                "โมเดล AI ยังไม่ถูกเทรน กรุณาเทรนโมเดลก่อนใช้งาน\n"
                "ไปที่หน้า 'จัดการโมเดล' > 'เทรนโมเดล' > อัปโหลดไฟล์ CSV"
            )
        
        # ทำนายด้วย Weighted Ensemble
        predictions = []
        model_confidences = {}
        weights = []
        
        try:
            # 1. Random Forest
            if 'rf' in self.models:
                rf_pred = self.models['rf'].predict_proba(X)[0][1]
                predictions.append(rf_pred)
                model_confidences['rf'] = rf_pred
                weights.append(self.model_weights.get('rf', 1.0))
            
            # 2. Gradient Boosting
            if 'gb' in self.models:
                gb_pred = self.models['gb'].predict_proba(X)[0][1]
                predictions.append(gb_pred)
                model_confidences['gb'] = gb_pred
                weights.append(self.model_weights.get('gb', 1.0))
            
            # 3. Logistic Regression (ต้องใช้ scaler)
            if 'lr' in self.models and self.scaler:
                X_scaled = self.scaler.transform(X)
                lr_pred = self.models['lr'].predict_proba(X_scaled)[0][1]
                predictions.append(lr_pred)
                model_confidences['lr'] = lr_pred
                weights.append(self.model_weights.get('lr', 1.0))
            
            # คำนวณ Weighted Ensemble
            if len(predictions) == 0:
                raise ValueError("ไม่สามารถทำนายได้ด้วยโมเดลใดๆ")
            
            weights_arr = np.array(weights)
            if weights_arr.sum() > 0:
                weights_arr = weights_arr / weights_arr.sum()
                probability = float(np.average(predictions, weights=weights_arr))
            else:
                probability = float(np.mean(predictions))
            logger.info(f"Weighted Ensemble Prediction: {probability:.3f}")
            
            # คำนวณความมั่นใจ (variance ของการทำนาย)
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                # ยิ่ง std น้อย = โมเดลเห็นพ้อง = มั่นใจมาก
                confidence = max(0.5, min(0.95, 1.0 - prediction_std))
            else:
                # มีโมเดลเดียว - ใช้ความชัดเจนของ probability
                distance = abs(probability - 0.5)
                confidence = max(0.5, min(0.95, 0.5 + distance))
            
            # ดึง Feature Importance (จาก Random Forest)
            feature_importance = {}
            if 'rf' in self.models:
                importances = self.models['rf'].feature_importances_
                feature_names = X.columns.tolist()
                importance_dict = dict(zip(feature_names, importances))
                # เรียงจากมากไปน้อย (แสดงแค่ 10 อันดับแรก)
                sorted_importance = dict(sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10])
                feature_importance = sorted_importance
            
            result = {
                'probability': probability,
                'confidence': confidence,
                'features_used': len(X.columns),
                'courses_analyzed': features.get('Total_Courses_so_far', features.get('Total_Courses', 0)),
                'prediction_method': 'AI_MODEL',  # บอกว่าใช้โมเดล AI
                'models_used': list(self.models.keys()),  # ['rf', 'gb', 'lr']
                'model_confidence': model_confidences,  # {'rf': 0.85, 'gb': 0.82, 'lr': 0.80}
                'feature_importance': feature_importance,  # Top 10 features
                'factors': {}  # สำหรับ backward compatibility
            }
            
            # เพิ่มคำอธิบายถ้าต้องการ
            if explain and self.explainer:
                try:
                    explanation = self.explainer.explain_prediction(features, result)
                    result['explanation'] = explanation
                except Exception as e:
                    logger.warning(f"Could not generate explanation: {e}")
                    result['explanation'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise ValueError(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        
        # === ลบ Heuristic Fallback ทั้งหมดแล้ว ===
        # ตอนนี้ระบบใช้โมเดล AI จริงเท่านั้น (ไม่มีเงื่อนไข if-else แบบเดิมอีกต่อไป)

