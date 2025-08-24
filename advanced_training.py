# advanced_training.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering for Context-Aware Model Training
    สร้าง features ที่ลึกซึ้งและมีความหมายเชิงบริบท
    """
    
    def __init__(self, grade_mapping: Dict[str, float]):
        self.grade_mapping = grade_mapping
        self.course_profiles = {}
        self.student_snapshots = []
        
    def create_course_dna(self, historical_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Phase 1: สร้าง DNA ของแต่ละวิชา (Course Profiling)
        วิเคราะห์ความยาก-ง่าย และลักษณะเฉพาะของแต่ละวิชา
        """
        logger.info("🧬 Creating Course DNA profiles...")
        
        course_dna = {}
        
        # ดึงชื่อคอลัมน์ที่เป็นวิชา (ไม่ใช่ metadata)
        exclude_cols = ['ชื่อ', 'ปีที่เข้า', 'ปีที่จบ', 'graduated', 'gpa', 'student_id']
        course_columns = [col for col in historical_data.columns 
                         if col not in exclude_cols and 
                         not any(kw in col.lower() for kw in ['gpa', 'เกรด', 'สำเร็จ', 'จบ'])]
        
        for course_col in course_columns:
            grades = []
            for _, row in historical_data.iterrows():
                grade_val = row.get(course_col)
                if pd.notna(grade_val) and str(grade_val).strip():
                    # แปลงเกรดเป็นตัวเลข
                    numeric_grade = self._convert_grade_to_numeric(grade_val)
                    if numeric_grade is not None:
                        grades.append(numeric_grade)
            
            if grades:
                # คำนวณ DNA ของวิชา
                course_dna[course_col] = {
                    'avg_grade': np.mean(grades),
                    'std_grade': np.std(grades) if len(grades) > 1 else 0,
                    'fail_rate': sum(1 for g in grades if g == 0) / len(grades),
                    'a_rate': sum(1 for g in grades if g >= 3.5) / len(grades),
                    'w_rate': sum(1 for g in grades if g == 0) / len(grades),  # W = 0
                    'difficulty_score': self._calculate_difficulty_score(grades),
                    'variance': np.var(grades) if len(grades) > 1 else 0,
                    'median_grade': np.median(grades),
                    'percentile_25': np.percentile(grades, 25) if len(grades) > 3 else min(grades),
                    'percentile_75': np.percentile(grades, 75) if len(grades) > 3 else max(grades),
                    'sample_size': len(grades)
                }
        
        self.course_profiles = course_dna
        logger.info(f"✅ Created DNA profiles for {len(course_dna)} courses")
        return course_dna
    
    def create_temporal_snapshots(self, student_data: pd.DataFrame, 
                                 course_dna: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Phase 2: สร้าง Temporal Snapshots 
        จำลองการเดินทางของนักศึกษาแต่ละคนผ่านเวลา
        """
        logger.info("📸 Creating temporal snapshots for each student...")
        
        snapshots = []
        
        # สำหรับนักศึกษาแต่ละคน
        for idx, row in student_data.iterrows():
            student_name = row.get('ชื่อ', f'Student_{idx}')
            graduated = row.get('graduated', 0)
            
            # เก็บเกรดของวิชาทั้งหมด
            course_grades = {}
            for col in course_dna.keys():
                if col in row.index:
                    grade_val = row[col]
                    if pd.notna(grade_val) and str(grade_val).strip():
                        numeric_grade = self._convert_grade_to_numeric(grade_val)
                        if numeric_grade is not None:
                            course_grades[col] = numeric_grade
            
            # สร้าง snapshots ตามจำนวนวิชาที่เรียน (simulate terms)
            if course_grades:
                # แบ่งวิชาเป็นเทอม (สมมติ 5-6 วิชาต่อเทอม)
                courses_list = list(course_grades.items())
                courses_per_term = 5
                num_terms = max(1, (len(courses_list) + courses_per_term - 1) // courses_per_term)
                
                cumulative_grades = []
                
                for term in range(1, min(num_terms + 1, 9)):  # Max 8 terms
                    # วิชาที่เรียนจนถึงเทอมนี้
                    end_idx = min(term * courses_per_term, len(courses_list))
                    term_courses = courses_list[:end_idx]
                    
                    # สร้าง features สำหรับ snapshot นี้
                    snapshot_features = self._create_snapshot_features(
                        student_name=student_name,
                        term_number=term,
                        courses_taken=dict(term_courses),
                        course_dna=course_dna,
                        graduated=graduated,
                        cumulative_grades=cumulative_grades
                    )
                    
                    snapshots.append(snapshot_features)
                    
                    # เก็บเกรดสะสม
                    term_new_courses = courses_list[(term-1)*courses_per_term:end_idx]
                    cumulative_grades.extend([grade for _, grade in term_new_courses])
        
        snapshot_df = pd.DataFrame(snapshots)
        logger.info(f"✅ Created {len(snapshots)} temporal snapshots from {len(student_data)} students")
        return snapshot_df
    
    def _create_snapshot_features(self, student_name: str, term_number: int,
                                 courses_taken: Dict[str, float],
                                 course_dna: Dict[str, Dict[str, float]],
                                 graduated: int,
                                 cumulative_grades: List[float]) -> Dict[str, Any]:
        """
        Phase 3: สร้าง Hyper-Contextual Features
        คุณลักษณะอัจฉริยะที่มีบริบทสูง
        """
        
        features = {
            'student_name': student_name,
            'term_number': term_number,
            'graduated': graduated  # Target variable
        }
        
        if not courses_taken:
            return features
        
        grades = list(courses_taken.values())
        
        # === A. Cumulative Features (ภาพรวมสะสม) ===
        all_grades = cumulative_grades + grades
        features['gpax_so_far'] = np.mean(all_grades) if all_grades else 0
        features['total_courses_taken'] = len(all_grades)
        features['min_grade_overall'] = min(all_grades) if all_grades else 0
        features['max_grade_overall'] = max(all_grades) if all_grades else 0
        features['std_grade_overall'] = np.std(all_grades) if len(all_grades) > 1 else 0
        
        # Relative Performance เทียบกับ Course Average
        relative_performances = []
        for course, grade in courses_taken.items():
            if course in course_dna:
                relative_perf = grade - course_dna[course]['avg_grade']
                relative_performances.append(relative_perf)
        
        features['total_relative_performance'] = sum(relative_performances) if relative_performances else 0
        features['avg_relative_performance'] = np.mean(relative_performances) if relative_performances else 0
        
        # === B. Trend Features (แนวโน้ม) ===
        # GPA ในเทอมปัจจุบัน
        current_term_grades = grades[-len(courses_taken):]
        features['gpa_current_term'] = np.mean(current_term_grades) if current_term_grades else 0
        
        # GPA Trend (เทียบกับเทอมก่อน)
        if cumulative_grades:
            prev_gpa = np.mean(cumulative_grades)
            features['gpa_trend'] = features['gpa_current_term'] - prev_gpa
            features['gpa_momentum'] = features['gpa_trend'] / (prev_gpa + 0.01)  # Percentage change
        else:
            features['gpa_trend'] = 0
            features['gpa_momentum'] = 0
        
        # Performance Consistency
        if len(all_grades) > 2:
            features['grade_consistency'] = 1 / (np.std(all_grades) + 0.01)
        else:
            features['grade_consistency'] = 1
        
        # === C. Interaction Features (ปฏิสัมพันธ์ซับซ้อน) ===
        
        # จำนวนวิชาหินที่ทำได้ไม่ดี
        hard_course_struggles = 0
        easy_course_failures = 0
        
        for course, grade in courses_taken.items():
            if course in course_dna:
                course_profile = course_dna[course]
                
                # วิชาหิน (fail_rate > 30%)
                if course_profile['fail_rate'] > 0.3:
                    if grade < course_profile['avg_grade']:
                        hard_course_struggles += 1
                
                # วิชาง่าย (a_rate > 40%) แต่ทำได้ไม่ดี
                if course_profile['a_rate'] > 0.4:
                    if grade < 2.0:  # ได้ต่ำกว่า C
                        easy_course_failures += 1
        
        features['hard_course_struggle_count'] = hard_course_struggles
        features['easy_course_underperformance_count'] = easy_course_failures
        
        # Workload Impact
        features['workload_credits'] = len(courses_taken)
        if features['workload_credits'] > 0:
            features['workload_performance_ratio'] = features['gpa_current_term'] / features['workload_credits']
        else:
            features['workload_performance_ratio'] = 0
        
        # === D. Risk Indicators ===
        features['fail_count'] = sum(1 for g in all_grades if g == 0)
        features['fail_rate'] = features['fail_count'] / len(all_grades) if all_grades else 0
        features['at_risk'] = 1 if features['gpax_so_far'] < 2.0 else 0
        features['critical_risk'] = 1 if features['gpax_so_far'] < 1.5 else 0
        
        # Course Difficulty Weighted GPA
        weighted_grades = []
        for course, grade in courses_taken.items():
            if course in course_dna:
                difficulty = course_dna[course]['difficulty_score']
                weighted_grades.append(grade * difficulty)
        
        features['difficulty_weighted_gpa'] = np.mean(weighted_grades) if weighted_grades else 0
        
        # === E. Pattern Recognition Features ===
        # Improvement Detection
        if len(all_grades) >= 3:
            first_half = all_grades[:len(all_grades)//2]
            second_half = all_grades[len(all_grades)//2:]
            features['improvement_indicator'] = np.mean(second_half) - np.mean(first_half)
        else:
            features['improvement_indicator'] = 0
        
        # Learning Stability
        if term_number > 1:
            features['terms_studied'] = term_number
            features['avg_performance_per_term'] = features['gpax_so_far'] / term_number
        else:
            features['terms_studied'] = 1
            features['avg_performance_per_term'] = features['gpax_so_far']
        
        return features
    
    def _convert_grade_to_numeric(self, grade) -> Optional[float]:
        """แปลงเกรดเป็นตัวเลข"""
        if pd.isna(grade):
            return None
        
        # ถ้าเป็นตัวเลขอยู่แล้ว
        try:
            numeric = float(grade)
            if 0 <= numeric <= 4:
                return numeric
        except (ValueError, TypeError):
            pass
        
        # แปลงจากตัวอักษร
        grade_str = str(grade).strip().upper()
        return self.grade_mapping.get(grade_str, None)
    
    def _calculate_difficulty_score(self, grades: List[float]) -> float:
        """คำนวณคะแนนความยากของวิชา"""
        if not grades:
            return 0.5
        
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        avg_grade = np.mean(grades)
        
        # Difficulty = high fail rate + low average
        difficulty = (fail_rate * 0.5) + ((4 - avg_grade) / 4 * 0.5)
        return min(1.0, max(0.0, difficulty))
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method: เตรียมข้อมูลสำหรับการเทรน
        """
        logger.info("🚀 Starting advanced feature engineering...")
        
        # Step 1: Create Course DNA
        course_dna = self.create_course_dna(df)
        
        # Step 2: Create Temporal Snapshots
        snapshot_df = self.create_temporal_snapshots(df, course_dna)
        
        # Step 3: แยก features และ target
        if 'graduated' in snapshot_df.columns:
            target_col = 'graduated'
            feature_cols = [col for col in snapshot_df.columns 
                          if col not in ['graduated', 'student_name']]
            
            X = snapshot_df[feature_cols].fillna(0)
            y = snapshot_df[target_col]
            
            logger.info(f"✅ Prepared {len(X)} training samples with {len(feature_cols)} features")
            logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")
            
            return X, y
        else:
            logger.error("❌ No 'graduated' column found in snapshot data")
            return pd.DataFrame(), pd.Series()


class ModelEvaluator:
    """
    ประเมินและเปรียบเทียบโมเดล
    """
    
    @staticmethod
    def evaluate_model_performance(y_true, y_pred, y_prob=None):
        """คำนวณ metrics ทั้งหมด"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix,
            classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = 0.5
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification Report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        return metrics
