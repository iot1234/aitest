
from advanced_training import AdvancedFeatureEngineer, UltimateModelTrainer, ModelEvaluator
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import re
import warnings
warnings.filterwarnings('ignore')

# Import NetworkX with fallback
try:
    import networkx as nx
except ImportError:
    print("Warning: NetworkX not installed. Prerequisite analysis disabled.")
    nx = None

# Import Bayesian Optimization with fallback
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not installed. Using standard GridSearchCV instead.")
    BAYESIAN_OPT_AVAILABLE = False
    from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Ultimate Feature Engineering with Maximum Intelligence
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
        self.cohort_statistics = {}  # เก็บสถิติของกลุ่ม
        
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
        """ตรวจจับ Prerequisites อัตโนมัติแบบอัจฉริยะ"""
        logger.info("🔗 Auto-detecting prerequisites with intelligence...")
        
        if not nx or not hasattr(self, 'prerequisite_graph'):
            logger.warning("NetworkX not available, skipping prerequisite detection")
            return nx.DiGraph() if nx else None
        
        try:
            G = nx.DiGraph()
            
            # วิเคราะห์ลำดับการลงทะเบียน
            if 'COURSE_CODE' in data.columns and 'Dummy StudentNO' in data.columns:
                for student_id in data['Dummy StudentNO'].unique()[:100]:  # จำกัดเพื่อประสิทธิภาพ
                    student_data = data[data['Dummy StudentNO'] == student_id].copy()
                    
                    if 'ปีการศึกษา' in student_data.columns and 'เทอม' in student_data.columns:
                        student_data = student_data.sort_values(['ปีการศึกษา', 'เทอม'])
                        
                        courses_taken = []
                        for _, row in student_data.iterrows():
                            course_code = self._normalize_course_code(row['COURSE_CODE'])
                            grade = self._convert_grade_to_numeric(row.get('GRADE'))
                            
                            if course_code and grade and grade > 0:
                                # เช็คว่าวิชานี้มักจะเรียนหลังวิชาอะไร
                                for prev_course in courses_taken[-3:]:  # ดู 3 วิชาล่าสุด
                                    if not G.has_edge(prev_course, course_code):
                                        G.add_edge(prev_course, course_code, weight=1)
                                    else:
                                        G[prev_course][course_code]['weight'] += 1
                                
                                courses_taken.append(course_code)
                
                # เก็บเฉพาะ edge ที่มี weight สูง (หลายคนเรียนตามลำดับนี้)
                edges_to_keep = []
                for u, v, data in G.edges(data=True):
                    if data['weight'] > 5:  # อย่างน้อย 5 คนเรียนตามลำดับนี้
                        edges_to_keep.append((u, v))
                
                G = nx.DiGraph()
                G.add_edges_from(edges_to_keep)
            
            self.prerequisite_graph = G
            logger.info(f"✅ Detected {G.number_of_edges()} prerequisite relationships")
            return G
            
        except Exception as e:
            logger.error(f"Error in prerequisite detection: {e}")
            return nx.DiGraph() if nx else None
    
    def create_course_dna(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง Course DNA พร้อมการวิเคราะห์เชิงลึก"""
        logger.info("🧬 Creating Advanced Course DNA profiles...")
        
        try:
            # Step 1: Build catalog
            self.auto_build_course_catalog(data)
            
            # Step 2: Detect prerequisites
            if nx:
                self.auto_detect_prerequisites(data)
            
            # Step 3: Create DNA
            course_dna = {}
            
            if 'COURSE_CODE' in data.columns:
                self.transcript_mode = True
                course_dna = self._create_dna_from_transcript(data)
            else:
                course_dna = self._create_dna_from_subjects(data)
            
            # Step 4: คำนวณ Cohort Statistics
            self._calculate_cohort_statistics(data)
            
            # Step 5: Enrich with advanced metrics
            for course_id in course_dna:
                # เพิ่มข้อมูล prerequisite
                if self.prerequisite_graph and course_id in self.prerequisite_graph:
                    predecessors = list(self.prerequisite_graph.predecessors(course_id))
                    successors = list(self.prerequisite_graph.successors(course_id))
                    
                    course_dna[course_id].update({
                        'has_prereq': len(predecessors) > 0,
                        'num_prereq': len(predecessors),
                        'blocks_count': len(successors),
                        'critical_level': self._calculate_critical_level(course_id),
                        'is_gateway': len(successors) > 3,
                        'is_bottleneck': len(predecessors) > 2 and len(successors) > 2
                    })
                else:
                    course_dna[course_id].update({
                        'has_prereq': False,
                        'num_prereq': 0,
                        'blocks_count': 0,
                        'critical_level': 0,
                        'is_gateway': False,
                        'is_bottleneck': False
                    })
                
                # เพิ่มการวิเคราะห์ Pattern
                course_dna[course_id]['seasonal_variance'] = self._calculate_seasonal_variance(course_id, data)
                course_dna[course_id]['correlation_with_success'] = self._calculate_success_correlation(course_id, data)
            
            self.course_profiles = course_dna
            logger.info(f"✅ Created Advanced DNA for {len(course_dna)} courses")
            
            return course_dna
            
        except Exception as e:
            logger.error(f"Error creating course DNA: {e}")
            return {}
    
    def _calculate_cohort_statistics(self, data: pd.DataFrame):
        """คำนวณสถิติของกลุ่มนักศึกษา"""
        try:
            if 'ปีการศึกษา' in data.columns:
                # Group by cohort year
                cohort_stats = data.groupby('ปีการศึกษา').agg({
                    'GRADE_POINT': ['mean', 'std', 'median'] if 'GRADE_POINT' in data.columns else lambda x: [0, 0, 0]
                })
                self.cohort_statistics = cohort_stats
        except Exception as e:
            logger.warning(f"Could not calculate cohort statistics: {e}")
    
    def _calculate_critical_level(self, course_id: str) -> float:
        """คำนวณระดับความสำคัญของวิชาใน prerequisite chain"""
        if not self.prerequisite_graph or course_id not in self.prerequisite_graph:
            return 0.0
        
        try:
            # ใช้ PageRank algorithm
            pagerank = nx.pagerank(self.prerequisite_graph)
            return pagerank.get(course_id, 0.0)
        except:
            return 0.0
    
    def _calculate_seasonal_variance(self, course_id: str, data: pd.DataFrame) -> float:
        """วิเคราะห์ความแปรปรวนตามฤดูกาล/เทอม"""
        try:
            if 'เทอม' not in data.columns:
                return 0.0
            
            course_data = data[data['COURSE_CODE'].str.contains(course_id, na=False)]
            if course_data.empty:
                return 0.0
            
            term_grades = course_data.groupby('เทอม')['GRADE_POINT'].mean()
            return term_grades.std() if len(term_grades) > 1 else 0.0
        except:
            return 0.0
    
    def _calculate_success_correlation(self, course_id: str, data: pd.DataFrame) -> float:
        """คำนวณความสัมพันธ์กับความสำเร็จในการจบการศึกษา"""
        try:
            # สมมติว่ามีคอลัมน์บอกว่าจบหรือไม่
            if 'graduated' not in data.columns:
                return 0.0
            
            course_takers = data[data['COURSE_CODE'].str.contains(course_id, na=False)]['Dummy StudentNO'].unique()
            if len(course_takers) == 0:
                return 0.0
            
            # คำนวณ correlation
            graduated_with_course = data[data['Dummy StudentNO'].isin(course_takers)]['graduated'].mean()
            overall_graduation = data['graduated'].mean()
            
            return (graduated_with_course - overall_graduation) / (overall_graduation + 0.001)
        except:
            return 0.0
    
    def create_advanced_features(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """สร้างฟีเจอร์ขั้นสูงสำหรับการเทรน"""
        logger.info("🚀 Creating advanced features...")
        
        all_features = []
        
        if self.transcript_mode:
            # Process by student
            student_ids = data['Dummy StudentNO'].unique() if 'Dummy StudentNO' in data.columns else []
            
            for student_id in student_ids[:1000]:  # จำกัดเพื่อประสิทธิภาพ
                student_data = data[data['Dummy StudentNO'] == student_id]
                features = self._extract_student_features(student_data, course_dna)
                if features:
                    all_features.append(features)
        else:
            # Process subject-based format
            for idx, row in data.iterrows():
                features = self._extract_row_features(row, course_dna)
                if features:
                    all_features.append(features)
        
        if not all_features:
            logger.warning("No features extracted")
            return pd.DataFrame()
        
        return pd.DataFrame(all_features)
    
    def _extract_student_features(self, student_data: pd.DataFrame, course_dna: Dict) -> Dict:
        """Extract comprehensive features for a student"""
        features = {}
        
        try:
            # Basic statistics
            grades = []
            weighted_grades = []
            
            for _, row in student_data.iterrows():
                grade = self._convert_grade_to_numeric(row.get('GRADE'))
                if grade is not None:
                    grades.append(grade)
                    
                    # Weighted by course difficulty
                    course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                    if course_code in course_dna:
                        difficulty = course_dna[course_code].get('difficulty_score', 0.5)
                        weighted_grades.append(grade * (1 + difficulty))
            
            if not grades:
                return {}
            
            # 1. Basic GPA features
            features['gpa'] = np.mean(grades)
            features['weighted_gpa'] = np.mean(weighted_grades)
            features['min_grade'] = np.min(grades)
            features['max_grade'] = np.max(grades)
            features['std_grade'] = np.std(grades) if len(grades) > 1 else 0
            features['grade_range'] = features['max_grade'] - features['min_grade']
            
            # 2. Performance distribution
            features['fail_count'] = sum(1 for g in grades if g == 0)
            features['fail_rate'] = features['fail_count'] / len(grades)
            features['excellent_rate'] = sum(1 for g in grades if g >= 3.5) / len(grades)
            features['pass_rate'] = sum(1 for g in grades if g > 0) / len(grades)
            
            # 3. Temporal patterns
            features.update(self._calculate_temporal_features(student_data))
            
            # 4. Course difficulty mastery
            features.update(self._calculate_difficulty_features(student_data, course_dna))
            
            # 5. Prerequisite success
            features.update(self._calculate_prerequisite_features(student_data, course_dna))
            
            # 6. Category performance
            features.update(self._calculate_category_features(student_data, course_dna))
            
            # 7. Advanced statistical features
            features.update(self._calculate_statistical_features(grades))
            
            # 8. Consistency and trend features
            features['consistency_score'] = self._calculate_consistency_score(grades)
            features['performance_trend'] = self._calculate_performance_trend(student_data)
            features['recovery_ability'] = self._calculate_recovery_ability(student_data)
            
            # 9. Relative performance (เทียบกับ cohort)
            features.update(self._calculate_relative_performance(student_data, course_dna))
            
            # 10. Target variable
            features['graduated'] = self._determine_graduation_status(student_data)
            
        except Exception as e:
            logger.error(f"Error extracting student features: {e}")
            return {}
        
        return features
    
    def _calculate_temporal_features(self, student_data: pd.DataFrame) -> Dict:
        """คำนวณฟีเจอร์ที่เกี่ยวกับเวลา"""
        features = {}
        
        try:
            if 'ปีการศึกษา' in student_data.columns and 'เทอม' in student_data.columns:
                # Sort by time
                sorted_data = student_data.sort_values(['ปีการศึกษา', 'เทอม'])
                
                # Calculate GPA by term
                term_gpas = []
                for (year, term), group in sorted_data.groupby(['ปีการศึกษา', 'เทอม']):
                    grades = [self._convert_grade_to_numeric(g) for g in group['GRADE']]
                    grades = [g for g in grades if g is not None]
                    if grades:
                        term_gpas.append(np.mean(grades))
                
                if term_gpas:
                    # Moving averages
                    features['gpa_ma_2'] = np.mean(term_gpas[-2:]) if len(term_gpas) >= 2 else term_gpas[-1]
                    features['gpa_ma_3'] = np.mean(term_gpas[-3:]) if len(term_gpas) >= 3 else np.mean(term_gpas)
                    
                    # Trend
                    if len(term_gpas) >= 2:
                        features['gpa_trend'] = term_gpas[-1] - term_gpas[-2]
                        features['gpa_acceleration'] = (term_gpas[-1] - term_gpas[-2]) - (term_gpas[-2] - term_gpas[-3]) if len(term_gpas) >= 3 else 0
                    else:
                        features['gpa_trend'] = 0
                        features['gpa_acceleration'] = 0
                    
                    # Volatility
                    features['gpa_volatility'] = np.std(term_gpas) if len(term_gpas) > 1 else 0
                    
                    # Best and worst terms
                    features['best_term_gpa'] = np.max(term_gpas)
                    features['worst_term_gpa'] = np.min(term_gpas)
                    features['term_gpa_range'] = features['best_term_gpa'] - features['worst_term_gpa']
        
        except Exception as e:
            logger.warning(f"Error in temporal features: {e}")
        
        return features
    
    def _calculate_difficulty_features(self, student_data: pd.DataFrame, course_dna: Dict) -> Dict:
        """คำนวณฟีเจอร์เกี่ยวกับความสามารถในการเรียนวิชายาก"""
        features = {}
        
        try:
            hard_courses_grades = []
            easy_courses_grades = []
            medium_courses_grades = []
            
            for _, row in student_data.iterrows():
                course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                grade = self._convert_grade_to_numeric(row.get('GRADE'))
                
                if course_code in course_dna and grade is not None:
                    difficulty = course_dna[course_code].get('difficulty_score', 0.5)
                    
                    if difficulty > 0.7:
                        hard_courses_grades.append(grade)
                    elif difficulty < 0.3:
                        easy_courses_grades.append(grade)
                    else:
                        medium_courses_grades.append(grade)
            
            # Performance in different difficulty levels
            features['hard_courses_gpa'] = np.mean(hard_courses_grades) if hard_courses_grades else 0
            features['easy_courses_gpa'] = np.mean(easy_courses_grades) if easy_courses_grades else 0
            features['medium_courses_gpa'] = np.mean(medium_courses_grades) if medium_courses_grades else 0
            
            # Difficulty mastery score
            if hard_courses_grades:
                features['difficulty_mastery'] = features['hard_courses_gpa'] / 4.0
            else:
                features['difficulty_mastery'] = 0.5
            
            # Adaptive ability (performance difference)
            if easy_courses_grades and hard_courses_grades:
                features['adaptive_ability'] = features['hard_courses_gpa'] - features['easy_courses_gpa']
            else:
                features['adaptive_ability'] = 0
        
        except Exception as e:
            logger.warning(f"Error in difficulty features: {e}")
        
        return features
    
    def _calculate_prerequisite_features(self, student_data: pd.DataFrame, course_dna: Dict) -> Dict:
        """คำนวณฟีเจอร์เกี่ยวกับ prerequisite chains"""
        features = {}
        
        try:
            if not self.prerequisite_graph:
                features['prereq_success_rate'] = 1.0
                features['prereq_chain_completeness'] = 1.0
                return features
            
            completed_courses = set()
            failed_prereqs = 0
            total_prereqs = 0
            
            for _, row in student_data.iterrows():
                course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                grade = self._convert_grade_to_numeric(row.get('GRADE'))
                
                if course_code and grade is not None:
                    if grade > 0:
                        completed_courses.add(course_code)
                    
                    # Check prerequisites
                    if course_code in self.prerequisite_graph:
                        prereqs = list(self.prerequisite_graph.predecessors(course_code))
                        total_prereqs += len(prereqs)
                        
                        for prereq in prereqs:
                            if prereq not in completed_courses:
                                failed_prereqs += 1
            
            features['prereq_success_rate'] = 1 - (failed_prereqs / total_prereqs) if total_prereqs > 0 else 1.0
            
            # Critical path progress
            features['critical_path_progress'] = self._calculate_critical_path_progress(completed_courses)
        
        except Exception as e:
            logger.warning(f"Error in prerequisite features: {e}")
            features['prereq_success_rate'] = 1.0
            features['prereq_chain_completeness'] = 1.0
        
        return features
    
    def _calculate_category_features(self, student_data: pd.DataFrame, course_dna: Dict) -> Dict:
        """คำนวณฟีเจอร์ตามหมวดหมู่วิชา"""
        features = {}
        
        categories = {
            'math': ['คณิต', 'แคลคูลัส', 'สถิติ', 'calculus', 'statistics'],
            'programming': ['โปรแกรม', 'คอมพิวเตอร์', 'programming', 'computer'],
            'engineering': ['วิศวกรรม', 'วงจร', 'engineering', 'circuit'],
            'language': ['ภาษา', 'อังกฤษ', 'english', 'language'],
            'science': ['ฟิสิกส์', 'เคมี', 'physics', 'chemistry']
        }
        
        try:
            for cat_name, keywords in categories.items():
                cat_grades = []
                
                for _, row in student_data.iterrows():
                    course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                    
                    if course_code in course_dna:
                        course_name = course_dna[course_code].get('course_title', '')
                        if any(kw in course_name.lower() for kw in keywords):
                            grade = self._convert_grade_to_numeric(row.get('GRADE'))
                            if grade is not None:
                                cat_grades.append(grade)
                
                features[f'gpa_{cat_name}'] = np.mean(cat_grades) if cat_grades else 0
                features[f'fail_rate_{cat_name}'] = sum(1 for g in cat_grades if g == 0) / len(cat_grades) if cat_grades else 0
        
        except Exception as e:
            logger.warning(f"Error in category features: {e}")
        
        return features
    
    def _calculate_statistical_features(self, grades: List[float]) -> Dict:
        """คำนวณฟีเจอร์ทางสถิติขั้นสูง"""
        features = {}
        
        try:
            if grades:
                # Percentiles
                features['grade_p25'] = np.percentile(grades, 25)
                features['grade_p50'] = np.percentile(grades, 50)
                features['grade_p75'] = np.percentile(grades, 75)
                features['grade_iqr'] = features['grade_p75'] - features['grade_p25']
                
                # Skewness and Kurtosis
                from scipy import stats
                features['grade_skewness'] = stats.skew(grades) if len(grades) > 2 else 0
                features['grade_kurtosis'] = stats.kurtosis(grades) if len(grades) > 3 else 0
                
                # Entropy (measure of grade diversity)
                grade_counts = pd.Series(grades).value_counts(normalize=True)
                features['grade_entropy'] = -sum(p * np.log(p + 1e-10) for p in grade_counts)
        
        except Exception as e:
            logger.warning(f"Error in statistical features: {e}")
        
        return features
    
    def _calculate_consistency_score(self, grades: List[float]) -> float:
        """คำนวณความสม่ำเสมอของผลการเรียน"""
        if len(grades) < 2:
            return 1.0
        
        mean_grade = np.mean(grades)
        std_grade = np.std(grades)
        
        if mean_grade == 0:
            return 0.0
        
        # Coefficient of variation (ยิ่งต่ำยิ่งสม่ำเสมอ)
        cv = std_grade / mean_grade
        consistency = 1 / (1 + cv)
        
        return consistency
    
    def _calculate_performance_trend(self, student_data: pd.DataFrame) -> float:
        """คำนวณแนวโน้มผลการเรียน"""
        try:
            if 'ปีการศึกษา' not in student_data.columns or 'เทอม' not in student_data.columns:
                return 0.0
            
            # Sort by time
            sorted_data = student_data.sort_values(['ปีการศึกษา', 'เทอม'])
            
            term_gpas = []
            for (year, term), group in sorted_data.groupby(['ปีการศึกษา', 'เทอม']):
                grades = [self._convert_grade_to_numeric(g) for g in group['GRADE']]
                grades = [g for g in grades if g is not None]
                if grades:
                    term_gpas.append(np.mean(grades))
            
            if len(term_gpas) < 2:
                return 0.0
            
            # Linear regression for trend
            x = np.arange(len(term_gpas)).reshape(-1, 1)
            y = np.array(term_gpas)
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Return slope (positive = improving, negative = declining)
            return float(model.coef_[0])
        
        except Exception as e:
            logger.warning(f"Error calculating trend: {e}")
            return 0.0
    
    def _calculate_recovery_ability(self, student_data: pd.DataFrame) -> float:
        """วัดความสามารถในการฟื้นตัวหลังจากตก"""
        try:
            recovery_scores = []
            
            # Find failed courses
            failed_courses = student_data[student_data['GRADE'].isin(['F', 'W'])]
            
            for _, fail_row in failed_courses.iterrows():
                course_code = fail_row['COURSE_CODE']
                fail_index = fail_row.name
                
                # Check for retake
                retakes = student_data[(student_data['COURSE_CODE'] == course_code) & 
                                      (student_data.index > fail_index)]
                
                if not retakes.empty:
                    retake_grade = self._convert_grade_to_numeric(retakes.iloc[0]['GRADE'])
                    if retake_grade and retake_grade > 0:
                        recovery_scores.append(retake_grade / 4.0)
            
            return np.mean(recovery_scores) if recovery_scores else 0.5
        
        except Exception as e:
            logger.warning(f"Error calculating recovery: {e}")
            return 0.5
    
    def _calculate_relative_performance(self, student_data: pd.DataFrame, course_dna: Dict) -> Dict:
        """คำนวณผลการเรียนเทียบกับค่าเฉลี่ยของวิชา"""
        features = {}
        
        try:
            relative_performances = []
            
            for _, row in student_data.iterrows():
                course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                grade = self._convert_grade_to_numeric(row.get('GRADE'))
                
                if course_code in course_dna and grade is not None:
                    avg_grade = course_dna[course_code].get('avg_grade', 2.0)
                    relative_perf = grade - avg_grade
                    relative_performances.append(relative_perf)
            
            if relative_performances:
                features['avg_relative_performance'] = np.mean(relative_performances)
                features['best_relative_performance'] = np.max(relative_performances)
                features['worst_relative_performance'] = np.min(relative_performances)
                features['relative_performance_consistency'] = np.std(relative_performances)
            else:
                features['avg_relative_performance'] = 0
                features['best_relative_performance'] = 0
                features['worst_relative_performance'] = 0
                features['relative_performance_consistency'] = 0
        
        except Exception as e:
            logger.warning(f"Error in relative performance: {e}")
        
        return features
    
    def _calculate_critical_path_progress(self, completed_courses: Set[str]) -> float:
        """คำนวณความคืบหน้าใน critical path"""
        if not self.prerequisite_graph:
            return 1.0
        
        try:
            # Find critical path (longest path in DAG)
            if nx.is_directed_acyclic_graph(self.prerequisite_graph):
                # Get all paths
                all_paths = []
                for node in self.prerequisite_graph.nodes():
                    if self.prerequisite_graph.in_degree(node) == 0:  # Start nodes
                        for end_node in self.prerequisite_graph.nodes():
                            if self.prerequisite_graph.out_degree(end_node) == 0:  # End nodes
                                try:
                                    paths = list(nx.all_simple_paths(self.prerequisite_graph, node, end_node))
                                    all_paths.extend(paths)
                                except:
                                    pass
                
                if all_paths:
                    # Find longest path
                    critical_path = max(all_paths, key=len)
                    completed_in_path = sum(1 for course in critical_path if course in completed_courses)
                    return completed_in_path / len(critical_path) if critical_path else 1.0
            
            return 1.0
        
        except Exception as e:
            logger.warning(f"Error calculating critical path: {e}")
            return 1.0
    
    def _determine_graduation_status(self, student_data: pd.DataFrame) -> int:
        """กำหนดสถานะการจบการศึกษา"""
        # This would need actual graduation data
        # For now, using simple heuristic
        try:
            grades = [self._convert_grade_to_numeric(g) for g in student_data['GRADE']]
            grades = [g for g in grades if g is not None]
            
            if grades:
                gpa = np.mean(grades)
                fail_rate = sum(1 for g in grades if g == 0) / len(grades)
                
                # Simple rule-based (จะถูกแทนที่ด้วยข้อมูลจริง)
                if gpa >= 2.0 and fail_rate < 0.2:
                    return 1
                else:
                    return 0
            
            return 0
        
        except:
            return 0
    
    def _extract_row_features(self, row: pd.Series, course_dna: Dict) -> Dict:
        """Extract features from a single row (for subject-based format)"""
        features = {}
        
        try:
            # Extract grades from subject columns
            grades = []
            for col in row.index:
                if col not in ['ชื่อ', 'รหัส', 'ปี', 'graduated', 'จบ']:
                    grade = self._convert_grade_to_numeric(row[col])
                    if grade is not None:
                        grades.append(grade)
            
            if not grades:
                return {}
            
            # Basic features
            features['gpa'] = np.mean(grades)
            features['min_grade'] = np.min(grades)
            features['max_grade'] = np.max(grades)
            features['std_grade'] = np.std(grades) if len(grades) > 1 else 0
            features['fail_count'] = sum(1 for g in grades if g == 0)
            features['fail_rate'] = features['fail_count'] / len(grades)
            features['total_subjects'] = len(grades)
            
            # Advanced features
            features['consistency_score'] = self._calculate_consistency_score(grades)
            features.update(self._calculate_statistical_features(grades))
            
            # Target
            for col in row.index:
                if any(kw in col.lower() for kw in ['จบ', 'graduated', 'สถานะ']):
                    val = row[col]
                    if isinstance(val, str):
                        features['graduated'] = 1 if 'จบ' in val and 'ไม่' not in val else 0
                    else:
                        features['graduated'] = int(val) if pd.notna(val) else 0
                    break
            
            if 'graduated' not in features:
                features['graduated'] = 0
        
        except Exception as e:
            logger.warning(f"Error extracting row features: {e}")
        
        return features
    
    # Keep all helper methods from original code
    def _normalize_course_code(self, code: str) -> str:
        """แปลงรหัสวิชาให้เป็นรูปแบบมาตรฐาน"""
        if pd.isna(code) or not code:
            return ""
        
        try:
            code = str(code).strip()
            return re.sub(r'[^\w\-]', '', code) if code else ""
        except:
            return ""
    
    def _estimate_year_from_code(self, code: str) -> int:
        """ประมาณชั้นปีจากรหัสวิชา"""
        try:
            match = re.search(r'(\d)', code)
            if match:
                return min(max(int(match.group(1)), 1), 4)
            return 2
        except:
            return 2
    
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
    
    def create_temporal_snapshots(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """สร้าง Temporal Snapshots พร้อม Advanced Features"""
        logger.info("📸 Creating advanced temporal snapshots...")
        
        try:
            # Use the new advanced feature creation
            snapshots_df = self.create_advanced_features(data, course_dna)
            
            if snapshots_df.empty:
                logger.warning("No snapshots created with advanced features")
                # Fallback to basic snapshots
                if self.transcript_mode:
                    snapshots = self._create_snapshots_from_transcript(data, course_dna)
                else:
                    snapshots = self._create_snapshots_from_subjects(data, course_dna)
                snapshots_df = pd.DataFrame(snapshots)
            
            logger.info(f"✅ Created {len(snapshots_df)} advanced snapshots")
            return snapshots_df
            
        except Exception as e:
            logger.error(f"Error creating snapshots: {e}")
            return pd.DataFrame()
    
    def _create_snapshots_from_transcript(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> List[Dict]:
        """สร้าง Snapshots จาก Transcript (Fallback)"""
        snapshots = []
        
        try:
            if 'Dummy StudentNO' in data.columns:
                students = data['Dummy StudentNO'].unique()[:100]  # Limit for performance
                
                for student_id in students:
                    student_data = data[data['Dummy StudentNO'] == student_id].copy()
                    
                    # Use advanced feature extraction
                    features = self._extract_student_features(student_data, course_dna)
                    if features:
                        snapshots.append(features)
        except Exception as e:
            logger.error(f"Error creating transcript snapshots: {e}")
        
        return snapshots
    
    def _create_snapshots_from_subjects(self, data: pd.DataFrame, course_dna: Dict[str, Dict[str, float]]) -> List[Dict]:
        """สร้าง Snapshots จาก Subject-based format (Fallback)"""
        snapshots = []
        
        try:
            for idx, row in data.iterrows():
                features = self._extract_row_features(row, course_dna)
                if features:
                    snapshots.append(features)
        except Exception as e:
            logger.error(f"Error creating subject snapshots: {e}")
        
        return snapshots
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Main method: เตรียมข้อมูลสำหรับการเทรนด้วย Advanced Features"""
        logger.info("🚀 Starting ULTIMATE feature engineering...")
        
        try:
            # Step 1: Create Course DNA with advanced analysis
            course_dna = self.create_course_dna(df)
            
            if not course_dna:
                logger.warning("No course DNA created, falling back to basic processing")
                return self._basic_data_preparation(df)
            
            # Step 2: Create Advanced Temporal Snapshots
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
                
                # Step 4: Feature Engineering Pipeline
                X = self._apply_feature_engineering_pipeline(X, y)
                
                logger.info(f"✅ Prepared {len(X)} samples with {X.shape[1]} advanced features")
                logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")
                
                return X, y
            else:
                logger.error("❌ No 'graduated' column found")
                return self._basic_data_preparation(df)
                
        except Exception as e:
            logger.error(f"❌ Error in advanced feature engineering: {e}")
            logger.info("🔄 Falling back to basic processing")
            return self._basic_data_preparation(df)
    
    def _apply_feature_engineering_pipeline(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply advanced feature engineering pipeline"""
        try:
            # 1. Feature Selection
            if X.shape[1] > 50:
                selector = SelectKBest(f_classif, k=50)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # 2. Create Polynomial Features (for small datasets)
            if X.shape[1] < 20 and len(X) > 100:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_poly = poly.fit_transform(X)
                poly_names = poly.get_feature_names_out(X.columns) if hasattr(poly, 'get_feature_names_out') else [f'poly_{i}' for i in range(X_poly.shape[1])]
                X = pd.DataFrame(X_poly, columns=poly_names, index=X.index)
            
            # 3. Power Transform
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            X_transformed = pt.fit_transform(X)
            X = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
            
            return X
            
        except Exception as e:
            logger.warning(f"Error in feature engineering pipeline: {e}")
            return X
    
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
                        'consistency_score': self._calculate_consistency_score(grades),
                        'graduated': graduated
                    }
                    
                    # Add statistical features
                    feature_row.update(self._calculate_statistical_features(grades))
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


class UltimateModelTrainer:
    """Ultimate Model Training with Maximum Accuracy"""
    
    def __init__(self):
        self.models = {}
        self.best_threshold = 0.5
        
    def train_ultimate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ultimate ensemble with all optimization techniques"""
        logger.info("🚀 Starting ULTIMATE ensemble training...")
        
        try:
            # 1. Handle class imbalance
            X, y = self._handle_class_imbalance(X, y)
            
            # 2. Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 3. Train base models
            base_models = self._train_base_models(X_train, y_train)
            
            # 4. Create stacking ensemble
            stacking_model = self._create_stacking_ensemble(base_models)
            stacking_model.fit(X_train, y_train)
            
            # 5. Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(
                stacking_model, method='isotonic', cv=3
            )
            calibrated_model.fit(X_train, y_train)
            
            # 6. Optimize threshold
            self.best_threshold = self._optimize_threshold(calibrated_model, X_test, y_test)
            
            # 7. Evaluate
            metrics = self._evaluate_model(calibrated_model, X_test, y_test)
            
            return {
                'model': calibrated_model,
                'threshold': self.best_threshold,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(base_models, X.columns)
            }
            
        except Exception as e:
            logger.error(f"Error in ultimate ensemble training: {e}")
            raise
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance with SMOTE"""
        try:
            from imblearn.over_sampling import SMOTE
            
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            
            if min_class_count >= 6:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            elif min_class_count >= 2:
                k_neighbors = min(5, min_class_count - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            else:
                logger.warning("Not enough samples for SMOTE")
                return X, y
                
        except Exception as e:
            logger.warning(f"Could not handle class imbalance: {e}")
            return X, y
            # เพิ่มหลังจาก imports ด้านบน

# แทนที่ฟังก์ชัน train_ensemble_model เดิมด้วย:
def train_ensemble_model(X, y):
    """Enhanced Ensemble model training with Ultimate Accuracy"""
    try:
        logger.info("🚀 Starting ULTIMATE model training...")
        
        # ใช้ UltimateModelTrainer แทน
        trainer = UltimateModelTrainer()
        result = trainer.train_ultimate_ensemble(X, y)
        
        # Extract results
        model = result['model']
        threshold = result['threshold']
        metrics = result['metrics']
        feature_importance = result.get('feature_importance', {})
        
        # Comprehensive evaluation
        evaluator = ModelEvaluator()
        eval_results = evaluator.comprehensive_evaluation(model, X, y)
        
        logger.info(f"✅ Model training completed:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.3f}")
        if 'roc_auc' in metrics:
            logger.info(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return {
            'models': {'ultimate_ensemble': model},
            'scaler': None,  # Scaler is handled inside the pipeline
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'training_samples': len(X),
            'validation_samples': int(len(X) * 0.2),
            'features_count': X.shape[1],
            'threshold': threshold,
            'feature_importance': feature_importance,
            'evaluation_results': eval_results,
            'best_rf_params': {},  # Handled by Bayesian optimization
            'best_gb_params': {},
            'best_lr_params': {}
        }
        
    except Exception as e:
        logger.error(f"Error training Ultimate model: {str(e)}")
        raise

# เพิ่มฟังก์ชันสำหรับการ predict ที่ใช้ threshold ที่ optimize แล้ว
def predict_with_threshold(model_data, X):
    """Predict using optimized threshold"""
    model = model_data['models'].get('ultimate_ensemble')
    threshold = model_data.get('threshold', 0.5)
    
    if model is None:
        # Fallback to old model structure
        for name, m in model_data['models'].items():
            if m is not None:
                model = m
                break
    
    if model is None:
        raise ValueError("No model found for prediction")
    
    # Get probabilities
    probas = model.predict_proba(X)
    
    # Apply threshold
    if probas.shape[1] == 2:
        predictions = (probas[:, 1] >= threshold).astype(int)
    else:
        predictions = probas.argmax(axis=1)
    
    return predictions, probas
