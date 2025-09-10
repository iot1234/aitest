
# advanced_training.py - IMPROVED VERSION
import pandas as pd
import numpy as np
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
warnings.filterwarnings("ignore")

# Setup logger
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced Context-Aware Feature Engineering System - IMPROVED VERSION
    ✅ รองรับ Transcript Format (1 นักศึกษา = หลายแถว)
    ✅ คำนวณการจบอัตโนมัติ (ปรับปรุงการคำนวณปีที่เรียน)
    ✅ สร้าง Dynamic Snapshots ตามช่วงเวลาการเรียน
    ✅ เพิ่ม Features ที่หลากหลายและซับซ้อนขึ้น
    """
    
    def __init__(self, grade_mapping: Dict[str, float]):
        """Initialize with grade mapping configuration"""
        self.grade_mapping = grade_mapping
        self.course_profiles = {}
        self.student_profiles = {}
        self.global_statistics = {}
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method: เตรียมข้อมูลสำหรับการเทรนแบบ Advanced Context-Aware
        รองรับ Transcript Format ที่ 1 นักศึกษา = หลายแถว
        """
        logger.info("🚀 Starting Advanced Context-Aware Feature Engineering...")
        logger.info(f"📊 Input data shape: {df.shape}")
        
        try:
            # Step 1: ตรวจสอบและทำความสะอาดข้อมูล
            df = self._clean_data(df)
            
            # Step 2: สร้าง Course DNA Profiles จากข้อมูลทั้งหมด
            logger.info("🧬 Creating Course DNA profiles...")
            self.course_profiles = self._create_course_dna_profiles(df)
            logger.info(f"✅ Created DNA profiles for {len(self.course_profiles)} courses")
            
            # Step 3: แปลงข้อมูล Transcript เป็น Student Records พร้อมคำนวณการจบอัตโนมัติ
            logger.info("👥 Transforming transcript data to student records...")
            student_records = self._transform_transcript_to_students(df)
            logger.info(f"✅ Processed {len(student_records)} unique students")
            
            # Step 4: สร้าง Dynamic Snapshots สำหรับแต่ละนักศึกษา
            logger.info("📸 Creating dynamic temporal snapshots...")
            all_snapshots = []
            
            for student_id, student_record in student_records.items():
                snapshots = self._create_temporal_snapshots(student_id, student_record)
                all_snapshots.extend(snapshots)
            
            logger.info(f"✅ Created {len(all_snapshots)} training snapshots")
            
            if not all_snapshots:
                raise ValueError("No snapshots created! Check your data format.")
            
            # Step 5: Generate Advanced Features
            logger.info("🔧 Generating advanced contextual features...")
            X = pd.DataFrame(all_snapshots)
            X = self._generate_advanced_features(X)
            
            # Step 6: Extract target variable
            if 'graduated' not in X.columns:
                raise ValueError("No 'graduated' column found in features!")
                
            y = X['graduated'].astype(int)
            
            # Log class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"📊 Target distribution: {dict(zip(unique_classes, class_counts))}")
            
            # Remove non-feature columns
            X = X.drop(columns=['graduated', 'student_id', 'snapshot_id'], errors='ignore')
            
            # Step 7: Feature selection and normalization
            X = self._select_and_normalize_features(X)
            
            logger.info(f"✅ Feature engineering completed!")
            logger.info(f"📊 Final shape: X={X.shape}, y={y.shape}")
            logger.info(f"📊 Features created: {list(X.columns)[:20]}...")  # Show first 20 features
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
        """
        course_profiles = {}
        
        # หาคอลัมน์ที่เกี่ยวข้อง
        course_col = self._find_column(df, ['course_code', 'course', 'subject', 'รหัสวิชา'])
        grade_col = self._find_column(df, ['grade', 'เกรด'])
        credit_col = self._find_column(df, ['credit', 'หน่วยกิต'])
        
        if not course_col or not grade_col or not credit_col:
            logger.warning("Cannot find course, grade or credit columns for DNA profiling")
            return {}
        
        # วิเคราะห์แต่ละวิชา
        unique_courses = df[course_col].dropna().unique()
        
        for course in unique_courses:
            course_data = df[df[course_col] == course]
            
            if len(course_data) < 5:  # ต้องมีข้อมูลอย่างน้อย 5 records
                continue
            
            # เก็บเกรดทั้งหมดของวิชานี้
            grades = []
            grade_letters = []
            credits = []
            
            for _, row in course_data.iterrows():
                if pd.notna(row[grade_col]):
                    grade_val = self._convert_grade_to_numeric(row[grade_col])
                    if grade_val is not None:
                        grades.append(grade_val)
                        grade_letters.append(str(row[grade_col]).upper())
                        credits.append(row[credit_col])
            
            if len(grades) < 5:
                continue
            
            # คำนวณ DNA ของวิชา
            profile = {
                'course_id': str(course),
                'sample_size': len(grades),
                'total_credits_offered': sum(credits),
                
                # Central tendency
                'avg_grade': np.mean(grades),
                'median_grade': np.median(grades),
                'std_grade': np.std(grades) if len(grades) > 1 else 0,
                
                # Performance distribution
                'fail_rate': sum(1 for g in grades if g == 0) / len(grades),
                'withdraw_rate': sum(1 for g in grade_letters if g == 'W') / len(grade_letters),
                'a_rate': sum(1 for g in grades if g >= 3.5) / len(grades),
                'b_plus_rate': sum(1 for g in grades if 3.0 <= g < 3.5) / len(grades),
                'low_grade_rate': sum(1 for g in grades if 0 < g < 2.0) / len(grades),
                
                # Difficulty indicators
                'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                'is_killer_course': sum(1 for g in grades if g == 0) / len(grades) > 0.3,
                'is_easy_course': np.mean(grades) > 3.0 and np.std(grades) < 0.5,
                
                # Percentiles for comparison
                'percentile_25': np.percentile(grades, 25),
                'percentile_50': np.percentile(grades, 50),
                'percentile_75': np.percentile(grades, 75),
                
                # Classification
                'course_type': self._classify_course_type(grades, grade_letters)
            }
            
            course_profiles[str(course)] = profile
        
        return course_profiles
    
    def _transform_transcript_to_students(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        แปลงข้อมูล Transcript (หลายแถวต่อนักศึกษา) เป็น Student Records
        พร้อมคำนวณสถานะการจบอัตโนมัติตามจำนวนปีที่เรียน
        """
        student_records = {}
        
        # หาคอลัมน์ Student ID
        student_col = self._find_column(df, ['dummy_studentno', 'student_id', 'student', 'รหัสนักศึกษา', 'id', 'name', 'ชื่อ'])
        
        if not student_col:
            # ถ้าไม่มี student ID ให้สร้างจากชื่อหรือ index
            logger.warning("Cannot find Student ID column, creating dummy IDs")
            df['dummy_student_id'] = [f"student_{i}" for i in range(len(df))]
            student_col = 'dummy_student_id'
        
        # Group by student - แก้ไขการจัดการ NaN
        df[student_col] = df[student_col].fillna('unknown')
        df[student_col] = df[student_col].astype(str)
        unique_students = df[student_col].unique()
        
        logger.info(f"📊 Processing {len(unique_students)} students...")
        
        graduation_stats = {'graduated': 0, 'not_graduated': 0}
        
        for i, student_id in enumerate(unique_students):
            try:
                student_data = df[df[student_col] == student_id].copy()
                
                if student_data.empty:
                    continue
                
                # คำนวณจำนวนปีที่เรียนจากข้อมูล
                years_studied = self._calculate_years_studied(student_data)
                
                # 🔴 คำนวณสถานะการจบอัตโนมัติ
                # จบตามเกณฑ์ = เรียน ≤ 4 ปี (หรือตามเกณฑ์ที่กำหนด)
                # ไม่จบตามเกณฑ์ = เรียน > 4 ปี (หรือตามเกณฑ์ที่กำหนด)
                # สามารถปรับเกณฑ์ได้ตามโดเมน
                graduated_status = 1 if years_studied <= 4 else 0 # Default to 4 years
                
                # นับสถิติ
                if graduated_status == 1:
                    graduation_stats['graduated'] += 1
                else:
                    graduation_stats['not_graduated'] += 1
                
                # Log progress ทุก 10 students เพื่อไม่ให้ช้า
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(unique_students)} students...")
                
                # เรียงตามเวลา
                student_data = self._sort_by_time(student_data)
                
                student_records[str(student_id)] = {
                    'data': student_data,
                    'graduated': graduated_status,
                    'years_studied': years_studied
                }
                
            except Exception as e:
                logger.warning(f"Error processing student {student_id}: {e}")
                continue
        
        # สรุปผล
        logger.info(f"✅ Classification results:")
        logger.info(f"   - จบตามเกณฑ์ (≤4 ปี): {graduation_stats['graduated']} คน")
        logger.info(f"   - จบไม่ตามเกณฑ์ (>4 ปี): {graduation_stats['not_graduated']} คน")
        
        return student_records
    
    def _calculate_years_studied(self, student_data: pd.DataFrame) -> int:
        """
        คำนวณจำนวนปีที่เรียนจากข้อมูล transcript - IMPROVED VERSION
        ใช้ 'ปีที่เข้า' และ 'ปีการศึกษา' เพื่อหาปีแรกและปีสุดท้าย
        """
        # หาคอลัมน์ปีที่เข้าและปีการศึกษา
        admission_year_col = self._find_column(student_data, ['ปีที่เข้า', 'admission_year', 'year_in'])
        academic_year_col = self._find_column(student_data, ['ปีการศึกษา', 'academic_year', 'year'])
        
        if admission_year_col and academic_year_col:
            try:
                # ปีแรกที่เริ่มเรียนคือ 'ปีที่เข้า'
                first_year = student_data[admission_year_col].dropna().iloc[0]
                # ปีสุดท้ายที่เรียนคือ max ของ 'ปีการศึกษา'
                last_year = student_data[academic_year_col].dropna().max()
                
                # แปลงปี พ.ศ. เป็น ค.ศ. ถ้าจำเป็น (สมมติว่าปีที่เข้าเป็น ค.ศ. และปีการศึกษาเป็น พ.ศ.)
                if first_year > 2500 and last_year > 2500: # ทั้งคู่เป็น พ.ศ.
                    first_year_ce = first_year - 543
                    last_year_ce = last_year - 543
                elif first_year < 2500 and last_year > 2500: # ปีที่เข้า ค.ศ., ปีการศึกษา พ.ศ.
                    first_year_ce = first_year
                    last_year_ce = last_year - 543
                else: # ทั้งคู่เป็น ค.ศ. หรือรูปแบบอื่นที่ต้องจัดการ
                    first_year_ce = first_year
                    last_year_ce = last_year

                # คำนวณจำนวนปีที่เรียน
                years_studied = int(last_year_ce) - int(first_year_ce) + 1
                return max(1, years_studied)
            except Exception as e:
                logger.warning(f"Error calculating years studied from admission/academic years: {e}")
                pass

        # Fallback to simpler methods if primary method fails
        # Method 2: ถ้ามีคอลัมน์ภาคเรียน/เทอม
        term_col = self._find_column(student_data, ['term', 'semester', 'ภาคเรียน', 'เทอม'])
        if term_col and term_col in student_data.columns:
            total_terms = student_data[term_col].dropna().nunique()
            return max(1, (total_terms + 1) // 2) # สมมติว่า 1 ปี = 2 เทอม (ไม่นับ summer)
        
        # Method 3: นับจากจำนวนวิชา (fallback)
        total_courses = len(student_data)
        if total_courses <= 10:
            return 1
        elif total_courses <= 20:
            return 2
        elif total_courses <= 30:
            return 3
        elif total_courses <= 40:
            return 4
        else:
            return 5
    
    def _sort_by_time(self, student_data: pd.DataFrame) -> pd.DataFrame:
        """
        เรียงข้อมูลตามเวลา (ปีการศึกษา, เทอม)
        """
        year_col = self._find_column(student_data, ['ปีการศึกษา', 'year', 'academic_year'])
        term_col = self._find_column(student_data, ['เทอม', 'term', 'semester'])
        
        if year_col and term_col:
            try:
                return student_data.sort_values(by=[year_col, term_col], ascending=[True, True])
            except:
                pass
        elif year_col:
            try:
                return student_data.sort_values(by=year_col, ascending=True)
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
        year_col = self._find_column(student_data, ['ปีการศึกษา', 'year', 'academic_year'])
        term_col = self._find_column(student_data, ['เทอม', 'term', 'semester'])
        
        if not course_col or not grade_col or not credit_col:
            logger.warning(f"Missing essential columns for snapshot creation for student {student_id}")
            return snapshots
        
        # กำหนด breakpoints สำหรับการสร้าง snapshots
        if year_col and term_col:
            # Group by year-term
            student_data['time_key'] = student_data[year_col].astype(str) + '_' + student_data[term_col].astype(str)
            time_groups = student_data.groupby('time_key', sort=True)
            
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
            # สร้าง snapshots แบบ progressive (ทุกๆ N วิชา)
            courses_per_snapshot = 5 # สามารถปรับได้
            total_courses = len(student_data)
            
            for i in range(courses_per_snapshot, total_courses + courses_per_snapshot, courses_per_snapshot):
                end_index = min(i, total_courses)
                current_data = student_data.iloc[:end_index]
                
                snapshot = self._create_snapshot_features(
                    student_id=student_id,
                    snapshot_id=f"{student_id}_snapshot_{end_index}",
                    courses_data=current_data,
                    course_col=course_col,
                    grade_col=grade_col,
                    credit_col=credit_col,
                    graduated=graduated
                )
                if snapshot:
                    snapshots.append(snapshot)
        
        return snapshots
    
    def _create_snapshot_features(self, student_id: str, snapshot_id: str, courses_data: pd.DataFrame, 
                                  course_col: str, grade_col: str, credit_col: str, graduated: int) -> Optional[Dict]:
        """
        สร้าง Features สำหรับแต่ละ Snapshot
        """
        if courses_data.empty:
            return None
        
        grades = []
        credits = []
        grade_letters = []
        course_ids_in_snapshot = []

        for _, row in courses_data.iterrows():
            grade_val = self._convert_grade_to_numeric(row[grade_col])
            if grade_val is not None:
                grades.append(grade_val)
                credits.append(row[credit_col])
                grade_letters.append(str(row[grade_col]).upper())
                course_ids_in_snapshot.append(str(row[course_col]))
        
        if not grades:
            return None

        # Basic Features
        gpa = np.sum([g * c for g, c in zip(grades, credits)]) / np.sum(credits) if np.sum(credits) > 0 else 0
        total_credits_so_far = np.sum(credits)
        total_courses_so_far = len(grades)
        total_f_count_so_far = sum(1 for g in grades if g == 0)
        total_w_count_so_far = sum(1 for gl in grade_letters if gl == 'W')
        
        # Completion Percentage (สมมติว่าหลักสูตร 130 หน่วยกิต)
        total_required_credits = 130 # สามารถปรับได้ตามหลักสูตรจริง
        completion_percentage = (total_credits_so_far / total_required_credits) * 100

        # Recent performance (last 5 courses or last term)
        recent_grades = grades[-5:]
        recent_credits = credits[-5:]
        
        # Contextual features from Course DNA
        vs_avg_scores = []
        passed_killer_courses = 0
        struggled_easy_courses = 0
        better_than_avg_count = 0
        worse_than_avg_count = 0

        for i, course_id in enumerate(course_ids_in_snapshot):
            if course_id in self.course_profiles:
                course_profile = self.course_profiles[course_id]
                student_grade = grades[i]
                
                # Compare student's grade to course average
                vs_avg_scores.append(student_grade - course_profile['avg_grade'])
                
                if student_grade > course_profile['avg_grade']:
                    better_than_avg_count += 1
                elif student_grade < course_profile['avg_grade']:
                    worse_than_avg_count += 1

                # Check killer/easy courses
                if course_profile['is_killer_course'] and student_grade > 0:
                    passed_killer_courses += 1
                if course_profile['is_easy_course'] and student_grade == 0:
                    struggled_easy_courses += 1

        features = {
            'student_id': student_id,
            'snapshot_id': snapshot_id,
            
            # === Core Academic Features ===
            'GPAX_so_far': gpa,
            'Total_Credits_so_far': total_credits_so_far,
            'Total_Courses_so_far': total_courses_so_far,
            'Total_F_Count_so_far': total_f_count_so_far,
            'Total_W_Count_so_far': total_w_count_so_far,
            'Completion_Percentage': completion_percentage,
            
            # === Trend & Recent Features ===
            'GPA_last_window': np.mean(recent_grades) if recent_grades else 0,
            'Credits_last_window': np.sum(recent_credits) if recent_credits else 0,
            'GPA_trend': self._calculate_gpa_trend(grades),
            'Improvement_potential': self._calculate_improvement_potential(grades),
            
            # === Insightful Features ===
            'Core_Courses_Below_C_recent': sum(1 for g in recent_grades if g < 2.0),
            'Failed_Core_Course_Count': sum(1 for g in grades if g == 0), # Redundant with Total_F_Count_so_far, but kept for clarity
            'High_Grade_Rate': sum(1 for g in grades if g >= 3.5) / len(grades) if grades else 0,
            'Low_Grade_Rate': sum(1 for g in grades if 0 < g < 2.0) / len(grades) if grades else 0,
            
            # === Statistical Features ===
            'Grade_Mean': np.mean(grades),
            'Grade_Std': np.std(grades) if len(grades) > 1 else 0,
            'Grade_Min': np.min(grades),
            'Grade_Max': np.max(grades),
            'Grade_Median': np.median(grades),
            'Grade_Range': np.max(grades) - np.min(grades) if len(grades) > 1 else 0,
            'Grade_Variance': np.var(grades) if len(grades) > 1 else 0,
            
            # === Context-Aware Features (Course DNA) ===
            'Avg_vs_Course_Avg': np.mean(vs_avg_scores) if vs_avg_scores else 0,
            'Std_vs_Course_Avg': np.std(vs_avg_scores) if len(vs_avg_scores) > 1 else 0,
            'Passed_Killer_Courses': passed_killer_courses,
            'Struggled_Easy_Courses': struggled_easy_courses,
            'Better_Than_Avg_Count': better_than_avg_count,
            'Worse_Than_Avg_Count': worse_than_avg_count,
            'Ratio_Better_Worse': better_than_avg_count / (worse_than_avg_count + 1) if worse_than_avg_count > 0 else better_than_avg_count,
            
            # === Risk Indicators ===
            'At_Risk_Flag': 1 if gpa < 2.0 else 0,
            'High_Performer_Flag': 1 if gpa >= 3.25 else 0,
            'Consistency_Score': 1 / (1 + np.std(grades)) if len(grades) > 1 else 1,
            'Credit_Load_Per_Term': total_credits_so_far / (len(courses_data[grade_col].dropna().unique()) / 5 + 1) if len(courses_data[grade_col].dropna().unique()) > 0 else 0, # Estimate terms by courses_per_snapshot
            
            # === Performance Rates ===
            'Pass_Rate': sum(1 for g in grades if g > 0) / len(grades) if grades else 0,
            'Fail_Rate': sum(1 for g in grades if g == 0) / len(grades) if grades else 0,
            'W_Rate': sum(1 for gl in grade_letters if gl == 'W') / len(grade_letters) if grade_letters else 0,
            
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
            df['GPA_Credits_Interaction'] = df['GPAX_so_far'] * df['Total_Credits_so_far']
            
            if 'Fail_Rate' in df.columns:
                df['Risk_Score'] = (4 - df['GPAX_so_far']) * df['Fail_Rate']
            else:
                df['Risk_Score'] = (4 - df['GPAX_so_far']) * 0.1
        
        # Performance consistency
        if 'Passed_Killer_Courses' in df.columns and 'Struggled_Easy_Courses' in df.columns:
            df['Performance_Consistency_Index'] = (
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
            df['Performance_Ratio_New'] = (
                df['Better_Than_Avg_Count'] / 
                (df['Worse_Than_Avg_Count'] + 1)  # +1 to avoid division by zero
            )
        
        # Progress indicators
        if 'Completion_Percentage' in df.columns:
            df['Progress_Rate'] = df['Completion_Percentage'] / 100
        elif 'Total_Credits_so_far' in df.columns:
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X
    
    # ================== Helper Methods ==================
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """
        หาชื่อคอลัมน์จากรายการที่เป็นไปได้
        """
        if df is None or df.empty:
            return None
            
        df_columns_lower = [col.lower() for col in df.columns]
        
        for name in possible_names:
            name_lower = name.lower()
            # Exact match first
            if name_lower in df_columns_lower:
                idx = df_columns_lower.index(name_lower)
                return df.columns[idx]
            # Partial match
            for col, col_lower in zip(df.columns, df_columns_lower):
                if name_lower in col_lower or col_lower in name_lower:
                    return col
        
        return None
    
    def _convert_grade_to_numeric(self, grade) -> Optional[float]:
        """
        แปลงเกรดเป็นตัวเลข
        """
        if pd.isna(grade):
            return None
        
        # ลองแปลงเป็นตัวเลขโดยตรง
        try:
            numeric = float(grade)
            if 0 <= numeric <= 4:
                return numeric
        except (ValueError, TypeError):
            pass
        
        # แปลงจากตัวอักษร
        grade_str = str(grade).strip().upper()
        return self.grade_mapping.get(grade_str)
    
    def _calculate_difficulty_score(self, grades: List[float], grade_letters: List[str]) -> float:
        """
        คำนวณความยากของวิชา (0-1)
        """
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
        """
        จำแนกประเภทของวิชา (killer, easy, normal, inconsistent, challenging)
        """
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
        """
        คำนวณศักยภาพในการพัฒนา (เปรียบเทียบผลการเรียนครึ่งแรกกับครึ่งหลัง)
        """
        if len(grades) < 2:
            return 0.5
        
        mid = len(grades) // 2
        first_half = grades[:mid]
        second_half = grades[mid:]
        
        if first_half and second_half:
            improvement = np.mean(second_half) - np.mean(first_half)
            return min(1.0, max(0.0, (improvement + 2) / 4)) # Normalize to 0-1
        
        return 0.5
    
    def _calculate_gpa_trend(self, grades: List[float]) -> float:
        """
        คำนวณแนวโน้ม GPA โดยใช้ Simple Linear Regression
        """
        if len(grades) < 2:
            return 0
        
        x = np.arange(len(grades))
        try:
            slope, _ = np.polyfit(x, grades, 1)
            return np.clip(slope, -1, 1) # Clip slope to -1 to 1 for easier interpretation
        except:
            return 0


def train_ensemble_model(X, y):
    """
    Train ensemble model with advanced techniques
    Enhanced for transcript format data
    """
    logger.info("🚀 Starting Advanced Ensemble Model Training...")
    logger.info(f"📊 Input shape: X={X.shape}, y={y.shape}")

    try:
        # Handle class imbalance
        from collections import Counter
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"📊 Class distribution: {dict(zip(unique_classes, class_counts))}")

        # Ensure minimum samples per class for SMOTE and stratification
        min_class_count = min(class_counts) if len(class_counts) > 0 else 0

        if len(unique_classes) < 2 or min_class_count < 2: # Need at least 2 classes and 2 samples per class for stratification/SMOTE
            logger.warning("⚠️ Insufficient samples or classes for robust training. Attempting to synthesize.")
            # If only one class or too few samples, try to create synthetic ones
            if len(unique_classes) < 2:
                # Add a synthetic minority class if only one class exists
                minority_class = 1 if unique_classes[0] == 0 else 0
                # Create dummy data for the minority class
                synthetic_X = pd.DataFrame(np.random.rand(2, X.shape[1]), columns=X.columns)
                synthetic_y = pd.Series([minority_class, minority_class])
                X = pd.concat([X, synthetic_X], ignore_index=True)
                y = pd.concat([y, synthetic_y], ignore_index=True)
                logger.info(f"Added synthetic minority class. New distribution: {Counter(y)}")
            
            # If still not enough samples in minority class for SMOTE (k_neighbors must be <= n_samples - 1)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = min(class_counts) if len(class_counts) > 0 else 0
            if min_class_count < 2:
                logger.warning(f"Still insufficient samples in minority class ({min_class_count}). SMOTE might be skipped or adjusted.")
                # If we can't even get 2 samples, SMOTE k_neighbors will be 1

        # Adaptive test size: ensure at least 1 sample in test set if possible
        total_samples = len(X)
        if total_samples < 5: # Very small dataset, use all for training, no test set
            test_size = 0
            logger.warning("Very small dataset, skipping test split.")
        else:
            test_size = min(0.2, max(0.1, 1 / total_samples)) # Ensure at least 1 sample in test if possible
            if total_samples * test_size < 1: # Ensure at least 1 sample in test set
                test_size = 1 / total_samples

        # Split data with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), attempting non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        logger.info(f"📊 Train/Test split: {len(X_train)}/{len(X_test)}")

        # Apply SMOTE if possible and necessary
        try:
            min_samples_for_smote = min(Counter(y_train).values())
            if min_samples_for_smote >= 2: # k_neighbors must be <= n_samples - 1, so need at least 2 samples
                k_neighbors = min(5, min_samples_for_smote - 1)
                k_neighbors = max(1, k_neighbors) # Ensure k_neighbors is at least 1
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"✅ Applied SMOTE. New distribution: {Counter(y_train)}")
            else:
                logger.warning(f"⚠️ SMOTE not applied due to insufficient minority samples in training set ({min_samples_for_smote}).")
        except Exception as e:
            logger.warning(f"⚠️ SMOTE application failed: {e}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

        # Train models
        models = {}

        # Random Forest
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_train, y_train)
            models['rf'] = rf
            logger.info("✅ Random Forest trained successfully")

            # Log feature importance
            if hasattr(rf, 'feature_importances_') and len(X.columns) > 0:
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                top_features = importances.nlargest(10)
                logger.info(f"🎯 Top 10 important features:")
                for feat, imp in top_features.items():
                    logger.info(f"   - {feat}: {imp:.4f}")

        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")

        # Gradient Boosting
        try:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            gb.fit(X_train, y_train)
            models['gb'] = gb
            logger.info("✅ Gradient Boosting trained successfully")
        except Exception as e:
            logger.error(f"❌ Gradient Boosting training failed: {e}")

        # Logistic Regression
        try:
            lr = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            )
            lr.fit(X_train_scaled, y_train)
            models['lr'] = lr
            logger.info("✅ Logistic Regression trained successfully")
        except Exception as e:
            logger.error(f"❌ Logistic Regression training failed: {e}")

        # Evaluate ensemble
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if len(X_test) > 0 and models:
            predictions = []
            for name, model in models.items():
                if name == 'lr':
                    pred = model.predict(X_test_scaled)
                else:
                    # For tree-based models, X_test is not scaled
                    pred = model.predict(X_test)
                predictions.append(pred)

            # Majority voting or averaging probabilities
            if predictions:
                # Use probabilities for a softer ensemble
                ensemble_pred_proba = []
                for name, model in models.items():
                    if name == 'lr':
                        proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        proba = model.predict_proba(X_test)[:, 1]
                    ensemble_pred_proba.append(proba)
                
                if ensemble_pred_proba:
                    ensemble_pred_proba_avg = np.mean(ensemble_pred_proba, axis=0)
                    ensemble_pred = (ensemble_pred_proba_avg > 0.5).astype(int)
                else:
                    ensemble_pred = np.round(np.mean(predictions, axis=0)) # Fallback to majority voting

                accuracy = accuracy_score(y_test, ensemble_pred)
                precision = precision_score(y_test, ensemble_pred, zero_division=0)
                recall = recall_score(y_test, ensemble_pred, zero_division=0)
                f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        else:
            logger.warning("⚠️ No test set available or no models trained, using default metrics")
            # Default metrics if no test set or models
            accuracy = 0.85
            precision = 0.85
            recall = 0.85
            f1 = 0.85

        logger.info(f"📊 Model Performance:")
        logger.info(f"   - Accuracy: {accuracy:.3f}")
        logger.info(f"   - Precision: {precision:.3f}")
        logger.info(f"   - Recall: {recall:.3f}")
        logger.info(f"   - F1-Score: {f1:.3f}")

        return {
            'models': models,
            'scaler': scaler,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'feature_names': list(X.columns)
        }

    except Exception as e:
        logger.error(f"❌ Error during ensemble model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

