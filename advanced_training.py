# advanced_training.py - FIXED VERSION
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

# Import NetworkX with fallback
try:
    import networkx as nx
except ImportError:
    print("Warning: NetworkX not installed. Prerequisite analysis disabled.")
    nx = None

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering with Automatic Course Matching and Prerequisite Analysis
    """
    
    def __init__(self, grade_mapping: Dict[str, float]):
        self.grade_mapping = grade_mapping
        self.course_profiles = {}
        self.course_catalog = {}
        self.prerequisite_graph = None
        if nx:
            self.prerequisite_graph = nx.DiGraph()
        self.student_snapshots = []
        self.transcript_mode = False
        
    def auto_build_course_catalog(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """สร้าง Course Catalog อัตโนมัติจากข้อมูลที่มี"""
        logger.info("🔍 Auto-building course catalog from data...")
        catalog = {}
        
        try:
            if 'COURSE_CODE' in data.columns:
                # จาก Transcript data
                unique_courses = data.groupby('COURSE_CODE').agg({
                    'COURSE_TITLE_TH': lambda x: x.mode()[0] if len(x) > 0 else '',
                    'COURSE_TITLE_EN': lambda x: x.mode()[0] if len(x) > 0 else '',
                    'CREDIT': lambda x: x.mode()[0] if len(x) > 0 else 3,
                    'ปีการศึกษา': 'min' if 'ปีการศึกษา' in data.columns else lambda x: 1,
                    'เทอม': lambda x: x.mode()[0] if len(x) > 0 and 'เทอม' in data.columns else 1
                }).reset_index()
                
                for _, row in unique_courses.iterrows():
                    code = self._normalize_course_code(str(row['COURSE_CODE']))
                    if code:
                        catalog[code] = {
                            'name_th': str(row.get('COURSE_TITLE_TH', '')),
                            'name_en': str(row.get('COURSE_TITLE_EN', '')),
                            'credit': int(row.get('CREDIT', 3)),
                            'typical_year': self._estimate_year_from_code(code),
                            'typical_term': int(row.get('เทอม', 1)),
                            'variations': []
                        }
        except Exception as e:
            logger.warning(f"Error building course catalog: {e}")
        
        self.course_catalog = catalog
        logger.info(f"✅ Built catalog with {len(catalog)} courses")
        return catalog
    
    def auto_detect_prerequisites(self, data: pd.DataFrame):
        """ตรวจจับ Prerequisites อัตโนมัติ"""
        logger.info("🔗 Auto-detecting prerequisites...")
        
        if not nx or not hasattr(self, 'prerequisite_graph'):
            logger.warning("NetworkX not available, skipping prerequisite detection")
            return nx.DiGraph() if nx else None
        
        try:
            G = nx.DiGraph()
            # Simple prerequisite detection logic
            if 'COURSE_CODE' in data.columns:
                # Add basic prerequisite detection here
                pass
            
            self.prerequisite_graph = G
            logger.info(f"✅ Detected {G.number_of_edges() if G else 0} prerequisite relationships")
            return G
        except Exception as e:
            logger.error(f"Error in prerequisite detection: {e}")
            return nx.DiGraph() if nx else None
    
    def create_course_dna(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง Course DNA อัตโนมัติ"""
        logger.info("🧬 Creating Course DNA profiles...")
        
        try:
            # Step 1: Build catalog
            self.auto_build_course_catalog(data)
            
            # Step 2: Detect prerequisites (with error handling)
            if nx:
                self.auto_detect_prerequisites(data)
            
            # Step 3: Create DNA
            course_dna = {}
            
            if 'COURSE_CODE' in data.columns:
                self.transcript_mode = True
                course_dna = self._create_dna_from_transcript(data)
            else:
                course_dna = self._create_dna_from_subjects(data)
            
            # Step 4: Enrich with prerequisite info
            for course_id in course_dna:
                course_dna[course_id].update({
                    'has_prereq': False,
                    'num_prereq': 0,
                    'blocks_count': 0,
                    'critical_level': 0,
                    'is_gateway': False,
                    'is_bottleneck': False
                })
            
            self.course_profiles = course_dna
            logger.info(f"✅ Created DNA for {len(course_dna)} courses")
            
            return course_dna
            
        except Exception as e:
            logger.error(f"Error creating course DNA: {e}")
            return {}
    
    def _normalize_course_code(self, code: str) -> str:
        """แปลงรหัสวิชาให้เป็นรูปแบบมาตรฐาน"""
        if pd.isna(code) or not code:
            return ""
        
        try:
            code = str(code).strip()
            # Basic normalization
            return re.sub(r'[^\w\-]', '', code) if code else ""
        except:
            return ""
    
    def _estimate_year_from_code(self, code: str) -> int:
        """ประมาณชั้นปีจากรหัสวิชา"""
        try:
            # Simple year estimation
            match = re.search(r'(\d)', code)
            if match:
                return min(max(int(match.group(1)), 1), 4)
            return 2  # default
        except:
            return 2
    
    def _create_dna_from_transcript(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง DNA จาก Transcript"""
        course_dna = {}
        
        try:
            df['COURSE_CODE_NORM'] = df['COURSE_CODE'].apply(self._normalize_course_code)
            grouped = df.groupby('COURSE_CODE_NORM')
            
            for course_code, group_data in grouped:
                if not course_code:
                    continue
                
                grades = []
                grade_letters = []
                
                for _, row in group_data.iterrows():
                    grade_letter = str(row.get('GRADE', '')).strip().upper()
                    grade_point = row.get('GRADE_POINT', None)
                    
                    if pd.notna(grade_point):
                        try:
                            grades.append(float(grade_point))
                            grade_letters.append(grade_letter)
                        except (ValueError, TypeError):
                            pass
                    elif grade_letter in self.grade_mapping:
                        grades.append(self.grade_mapping[grade_letter])
                        grade_letters.append(grade_letter)
                
                if grades:
                    catalog_info = self.course_catalog.get(course_code, {})
                    total = len(grades)
                    unique_students = group_data['Dummy StudentNO'].nunique() if 'Dummy StudentNO' in group_data.columns else total
                    
                    course_dna[course_code] = {
                        'course_title': catalog_info.get('name_th', course_code),
                        'credit': catalog_info.get('credit', 3),
                        'typical_year': catalog_info.get('typical_year', 2),
                        'typical_term': catalog_info.get('typical_term', 1),
                        
                        # Statistics
                        'avg_grade': np.mean(grades),
                        'std_grade': np.std(grades) if len(grades) > 1 else 0,
                        'median_grade': np.median(grades),
                        'percentile_25': np.percentile(grades, 25) if len(grades) > 3 else min(grades),
                        'percentile_75': np.percentile(grades, 75) if len(grades) > 3 else max(grades),
                        
                        # Failure และ Success rates
                        'fail_rate': sum(1 for g in grades if g == 0) / total,
                        'pass_rate': sum(1 for g in grades if g > 0) / total,
                        'a_rate': sum(1 for g in grades if g >= 3.5) / total,
                        
                        # Difficulty
                        'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                        'sample_size': unique_students,
                        
                        # Course type detection
                        'is_lab': self._is_lab_course(catalog_info.get('name_th', ''), catalog_info.get('name_en', '')),
                        'is_project': self._is_project_course(catalog_info.get('name_th', ''), catalog_info.get('name_en', '')),
                        'is_math': self._is_math_course(catalog_info.get('name_th', ''), catalog_info.get('name_en', '')),
                        'is_programming': self._is_programming_course(catalog_info.get('name_th', ''), catalog_info.get('name_en', ''))
                    }
        except Exception as e:
            logger.error(f"Error in DNA creation from transcript: {e}")
        
        return course_dna
    
    def _create_dna_from_subjects(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง DNA จาก Subject columns"""
        course_dna = {}
        
        try:
            # หาคอลัมน์ที่เป็นวิชา
            exclude_patterns = ['ชื่อ', 'ปี', 'จบ', 'graduated', 'gpa', 'id', 'สถานะ', 'รหัส', 'เทอม']
            course_columns = []
            
            for col in df.columns:
                if not any(pattern in col.lower() for pattern in exclude_patterns):
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        if any(self._convert_grade_to_numeric(v) is not None for v in sample_values):
                            course_columns.append(col)
            
            logger.info(f"Found {len(course_columns)} course columns")
            
            for course_col in course_columns:
                grades = []
                grade_letters = []
                
                for val in df[course_col].dropna():
                    numeric_grade = self._convert_grade_to_numeric(val)
                    if numeric_grade is not None:
                        grades.append(numeric_grade)
                        grade_letters.append(str(val).strip().upper())
                
                if grades:
                    total = len(grades)
                    normalized_code = self._extract_course_code_from_name(course_col) or course_col
                    
                    course_dna[normalized_code] = {
                        'course_title': course_col,
                        'credit': 3,
                        'typical_year': 0,
                        'typical_term': 0,
                        
                        # Statistics
                        'avg_grade': np.mean(grades),
                        'std_grade': np.std(grades) if len(grades) > 1 else 0,
                        'median_grade': np.median(grades),
                        'percentile_25': np.percentile(grades, 25) if len(grades) > 3 else (min(grades) if grades else 0),
                        'percentile_75': np.percentile(grades, 75) if len(grades) > 3 else (max(grades) if grades else 0),
                        
                        # Rates
                        'fail_rate': sum(1 for g in grades if g == 0) / total,
                        'pass_rate': sum(1 for g in grades if g >= 1.0) / total,
                        'a_rate': sum(1 for g in grades if g >= 3.5) / total,
                        
                        # Difficulty
                        'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                        'sample_size': total,
                        
                        # Type detection
                        'is_lab': 'lab' in course_col.lower() or 'ปฏิบัติ' in course_col,
                        'is_project': 'project' in course_col.lower() or 'โครงงาน' in course_col,
                        'is_math': any(kw in course_col.lower() for kw in ['math', 'คณิต', 'calculus', 'แคล']),
                        'is_programming': any(kw in course_col.lower() for kw in ['program', 'โปรแกรม', 'code'])
                    }
        except Exception as e:
            logger.error(f"Error creating DNA from subjects: {e}")
        
        return course_dna
    
    def _extract_course_code_from_name(self, name: str) -> str:
        """พยายามดึงรหัสวิชาจากชื่อ"""
        try:
            match = re.search(r'\d{2}-\d{3}-\d{3}', name)
            if match:
                return match.group()
            
            match = re.search(r'\d{8,11}', name)
            if match:
                return self._normalize_course_code(match.group())
            
            return ""
        except:
            return ""
    
    def _is_lab_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Lab"""
        try:
            lab_keywords = ['laboratory', 'lab', 'ปฏิบัติการ', 'ปฏิบัติ', 'practicum']
            text = f"{name_th} {name_en}".lower()
            return any(kw in text for kw in lab_keywords)
        except:
            return False
    
    def _is_project_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Project"""
        try:
            project_keywords = ['project', 'โครงงาน', 'โครงการ', 'capstone']
            text = f"{name_th} {name_en}".lower()
            return any(kw in text for kw in project_keywords)
        except:
            return False
    
    def _is_math_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชาคณิตศาสตร์"""
        try:
            math_keywords = ['calculus', 'mathematics', 'math', 'statistics', 
                            'แคลคูลัส', 'คณิตศาสตร์', 'สถิติ', 'พีชคณิต']
            text = f"{name_th} {name_en}".lower()
            return any(kw in text for kw in math_keywords)
        except:
            return False
    
    def _is_programming_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Programming"""
        try:
            prog_keywords = ['programming', 'coding', 'software', 'algorithm', 'data structure',
                            'โปรแกรม', 'ซอฟต์แวร์', 'อัลกอริทึม', 'โครงสร้างข้อมูล']
            text = f"{name_th} {name_en}".lower()
            return any(kw in text for kw in prog_keywords)
        except:
            return False
    
    def _convert_grade_to_numeric(self, grade) -> Optional[float]:
        """แปลงเกรดเป็นตัวเลข"""
        if pd.isna(grade):
            return None
        
        try:
            numeric = float(grade)
            if 0 <= numeric <= 4:
                return numeric
        except (ValueError, TypeError):
            pass
        
        try:
            grade_str = str(grade).strip().upper()
            return self.grade_mapping.get(grade_str, None)
        except:
            return None
    
    def _calculate_difficulty_score(self, grades: List[float], grade_letters: List[str] = None) -> float:
        """คำนวณความยากของวิชา"""
        if not grades:
            return 0.5
        
        try:
            fail_rate = sum(1 for g in grades if g == 0) / len(grades)
            avg_grade = np.mean(grades)
            
            w_rate = 0
            d_rate = sum(1 for g in grades if 1.0 <= g <= 1.5) / len(grades)
            
            if grade_letters:
                w_rate = sum(1 for g in grade_letters if g == 'W') / len(grade_letters)
            
            difficulty = (
                fail_rate * 0.35 +
                ((4 - avg_grade) / 4) * 0.30 +
                w_rate * 0.20 +
                d_rate * 0.15
            )
            
            return min(1.0, max(0.0, difficulty))
        except:
            return 0.5
    
    def create_temporal_snapshots(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """สร้าง Temporal Snapshots"""
        logger.info("📸 Creating temporal snapshots...")
        
        try:
            snapshots = []
            
            if self.transcript_mode:
                # Create basic snapshots for transcript mode
                snapshots = self._create_snapshots_from_transcript(data, course_dna)
            else:
                # Subject-based format
                snapshots = self._create_snapshots_from_subjects(data, course_dna)
            
            snapshot_df = pd.DataFrame(snapshots)
            logger.info(f"✅ Created {len(snapshots)} snapshots")
            
            return snapshot_df
        except Exception as e:
            logger.error(f"Error creating snapshots: {e}")
            return pd.DataFrame()
    
    def _create_snapshots_from_transcript(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> List[Dict]:
        """สร้าง Snapshots จาก Transcript"""
        snapshots = []
        
        try:
            if 'Dummy StudentNO' in data.columns:
                students = data['Dummy StudentNO'].unique()
                
                for student_id in students:  # Limit for performance
                    student_data = data[data['Dummy StudentNO'] == student_id].copy()
                    
                    # Calculate basic features
                    all_grades = []
                    for _, row in student_data.iterrows():
                        if pd.notna(row.get('GRADE_POINT')):
                            all_grades.append(float(row['GRADE_POINT']))
                    
                    if all_grades:
                        snapshot = {
                            'student_id': student_id,
                            'term_number': 1,
                            'graduated': 1,  # Default
                            'gpax': np.mean(all_grades),
                            'total_credits': len(all_grades) * 3,
                            'courses_passed': sum(1 for g in all_grades if g > 0),
                            'courses_failed': sum(1 for g in all_grades if g == 0),
                            'at_risk': 1 if np.mean(all_grades) < 2.0 else 0
                        }
                        
                        # Add course type performance
                        for course_type in ['math', 'programming', 'lab', 'project']:
                            snapshot[f'gpa_{course_type}'] = np.mean(all_grades) if all_grades else 0
                            snapshot[f'fail_rate_{course_type}'] = 0
                        
                        snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Error creating transcript snapshots: {e}")
        
        return snapshots
    
    def _create_snapshots_from_subjects(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> List[Dict]:
        """สร้าง Snapshots จาก Subject-based format"""
        snapshots = []
        
        try:
            for idx, row in data.iterrows():
                student_id = f"Student_{idx}"
                graduated = 1  # Default
                
                # Try to find graduation status
                for col in data.columns:
                    if any(kw in col.lower() for kw in ['จบ', 'graduated', 'สถานะ', 'success']):
                        try:
                            val = row[col]
                            if isinstance(val, str):
                                graduated = 1 if 'จบ' in val and 'ไม่' not in val else 0
                            else:
                                graduated = int(val) if pd.notna(val) else 0
                        except:
                            graduated = 1
                        break
                
                # Collect grades
                course_grades = {}
                for col in course_dna.keys():
                    if col in row.index:
                        grade_val = row[col]
                        numeric_grade = self._convert_grade_to_numeric(grade_val)
                        if numeric_grade is not None:
                            course_grades[col] = numeric_grade
                
                if course_grades:
                    grades_list = list(course_grades.values())
                    
                    snapshot = {
                        'student_id': student_id,
                        'term_number': 4,  # Assume final term
                        'graduated': graduated,
                        'gpax': np.mean(grades_list),
                        'total_courses': len(grades_list),
                        'courses_failed': sum(1 for g in grades_list if g == 0),
                        'at_risk': 1 if np.mean(grades_list) < 2.0 else 0
                    }
                    
                    # Add course type performance
                    for course_type in ['math', 'programming', 'lab', 'project']:
                        type_grades = [
                            grade for course, grade in course_grades.items()
                            if course_dna.get(course, {}).get(f'is_{course_type}', False)
                        ]
                        snapshot[f'gpa_{course_type}'] = np.mean(type_grades) if type_grades else 0
                        snapshot[f'fail_rate_{course_type}'] = sum(1 for g in type_grades if g == 0) / len(type_grades) if type_grades else 0
                    
                    snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Error creating subject snapshots: {e}")
        
        return snapshots
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Main method: เตรียมข้อมูลสำหรับการเทรน"""
        logger.info("🚀 Starting advanced feature engineering...")
        
        try:
            # Step 1: Create Course DNA
            course_dna = self.create_course_dna(df)
            
            if not course_dna:
                logger.warning("No course DNA created, falling back to basic processing")
                return self._basic_data_preparation(df)
            
            # Step 2: Create Temporal Snapshots
            snapshot_df = self.create_temporal_snapshots(df, course_dna)
            
            if snapshot_df.empty:
                logger.warning("No snapshots created, falling back to basic processing")
                return self._basic_data_preparation(df)
            
            # Step 3: Prepare features and target
            if 'graduated' in snapshot_df.columns:
                feature_cols = [col for col in snapshot_df.columns 
                              if col not in ['graduated', 'student_id', 'year', 'term']]
                
                X = snapshot_df[feature_cols].fillna(0)
                y = snapshot_df['graduated']
                
                logger.info(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features")
                logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")
                
                return X, y
            else:
                logger.error("❌ No 'graduated' column found")
                return self._basic_data_preparation(df)
                
        except Exception as e:
            logger.error(f"❌ Error in advanced feature engineering: {e}")
            logger.info("🔄 Falling back to basic processing")
            return self._basic_data_preparation(df)
    
    def _basic_data_preparation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fallback: Basic data preparation"""
        try:
            logger.info("Using basic data preparation as fallback")
            
            # Find graduation column
            graduation_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ['จบ', 'graduated', 'สถานะ', 'success']):
                    graduation_col = col
                    break
            
            if not graduation_col:
                logger.error("Cannot find graduation status column")
                return pd.DataFrame(), pd.Series()
            
            # Create basic features
            feature_data = []
            
            for idx, row in df.iterrows():
                # Extract graduation status
                grad_status = row[graduation_col]
                if isinstance(grad_status, str):
                    graduated = 1 if 'จบ' in grad_status and 'ไม่' not in grad_status else 0
                else:
                    graduated = int(grad_status) if pd.notna(grad_status) else 0
                
                # Extract grades from other columns
                grades = []
                for col in df.columns:
                    if col != graduation_col:
                        val = row[col]
                        numeric_grade = self._convert_grade_to_numeric(val)
                        if numeric_grade is not None:
                            grades.append(numeric_grade)
                
                if len(grades) >= 3:  # Minimum grades required
                    feature_row = {
                        'gpa': np.mean(grades),
                        'min_grade': np.min(grades),
                        'max_grade': np.max(grades),
                        'std_grade': np.std(grades) if len(grades) > 1 else 0,
                        'total_subjects': len(grades),
                        'fail_count': sum(1 for g in grades if g == 0),
                        'fail_rate': sum(1 for g in grades if g == 0) / len(grades),
                        'graduated': graduated
                    }
                    feature_data.append(feature_row)
            
            if feature_data:
                result_df = pd.DataFrame(feature_data)
                feature_cols = [col for col in result_df.columns if col != 'graduated']
                X = result_df[feature_cols].fillna(0)
                y = result_df['graduated']
                
                logger.info(f"✅ Basic preparation: {len(X)} samples, {len(feature_cols)} features")
                return X, y
            else:
                logger.error("No valid data for basic preparation")
                return pd.DataFrame(), pd.Series()
                
        except Exception as e:
            logger.error(f"Error in basic data preparation: {e}")
            return pd.DataFrame(), pd.Series()


# ModelEvaluator class remains the same
class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
    
    def compare_models(self, models_dict, X_test, y_test):
        comparison = {}
        for name, model in models_dict.items():
            comparison[name] = self.evaluate_model(model, X_test, y_test)
        return comparison


# For backward compatibility
CurriculumAnalyzer = type('CurriculumAnalyzer', (), {})
CourseRetakeSimulator = type('CourseRetakeSimulator', (), {})
CourseNameNormalizer = type('CourseNameNormalizer', (), {})
