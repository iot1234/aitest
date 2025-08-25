# advanced_training.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import networkx as nx
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering with Automatic Course Matching and Prerequisite Analysis
    """
    
    def __init__(self, grade_mapping: Dict[str, float]):
        self.grade_mapping = grade_mapping
        self.course_profiles = {}
        self.course_catalog = {}  # จะสร้างจากข้อมูลที่มี
        self.prerequisite_graph = nx.DiGraph()
        self.student_snapshots = []
        self.transcript_mode = False
        
    def auto_build_course_catalog(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        สร้าง Course Catalog อัตโนมัติจากข้อมูลที่มี
        ไม่ต้อง hardcode รายวิชา
        """
        logger.info("🔍 Auto-building course catalog from data...")
        catalog = {}
        
        if 'COURSE_CODE' in data.columns:
            # จาก Transcript data
            unique_courses = data.groupby('COURSE_CODE').agg({
                'COURSE_TITLE_TH': lambda x: x.mode()[0] if len(x) > 0 else '',
                'COURSE_TITLE_EN': lambda x: x.mode()[0] if len(x) > 0 else '',
                'CREDIT': lambda x: x.mode()[0] if len(x) > 0 else 3,
                'ปีการศึกษา': 'min',
                'เทอม': lambda x: x.mode()[0] if len(x) > 0 else 1
            }).reset_index()
            
            for _, row in unique_courses.iterrows():
                code = self._normalize_course_code(row['COURSE_CODE'])
                catalog[code] = {
                    'name_th': row['COURSE_TITLE_TH'],
                    'name_en': row['COURSE_TITLE_EN'],
                    'credit': row['CREDIT'],
                    'typical_year': self._estimate_year_from_code(code),
                    'typical_term': row['เทอม'],
                    'variations': self._collect_name_variations(data, row['COURSE_CODE'])
                }
        
        self.course_catalog = catalog
        logger.info(f"✅ Built catalog with {len(catalog)} courses")
        return catalog
    
    def auto_detect_prerequisites(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        ตรวจจับ Prerequisites อัตโนมัติจากรูปแบบการลงทะเบียน
        """
        logger.info("🔗 Auto-detecting prerequisites from enrollment patterns...")
        G = nx.DiGraph()
        
        if 'COURSE_CODE' not in data.columns:
            return G
        
        # วิเคราะห์จากลำดับการเรียน
        students = data['Dummy StudentNO'].unique() if 'Dummy StudentNO' in data.columns else []
        prerequisite_patterns = {}
        
        for student in students:
            student_data = data[data['Dummy StudentNO'] == student].copy()
            student_data = student_data.sort_values(['ปีการศึกษา', 'เทอม'])
            
            # Track ลำดับการเรียน
            courses_taken = []
            for _, row in student_data.iterrows():
                course = self._normalize_course_code(row['COURSE_CODE'])
                grade = row.get('GRADE', '')
                
                if grade not in ['F', 'W', 'I']:  # ผ่าน
                    # เช็คว่าวิชานี้มักเรียนหลังวิชาอะไร
                    for prev_course in courses_taken:
                        key = (prev_course, course)
                        if key not in prerequisite_patterns:
                            prerequisite_patterns[key] = 0
                        prerequisite_patterns[key] += 1
                    
                    courses_taken.append(course)
        
        # กำหนด threshold (ถ้า > 70% เรียนตามลำดับนี้ = น่าจะเป็น prerequisite)
        threshold = len(students) * 0.7
        
        for (prereq, course), count in prerequisite_patterns.items():
            if count >= threshold:
                G.add_edge(prereq, course, weight=count/len(students))
                logger.info(f"   Found: {prereq} → {course} (confidence: {count/len(students):.2%})")
        
        # เพิ่มการตรวจจับจากรหัสวิชา (วิชาที่มีเลขต่อท้าย)
        for course in self.course_catalog:
            # ตรวจจับ pattern เช่น XXX-101 -> XXX-102
            match = re.match(r'(\d+-\d+-\d+-)(\d+)', course)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                
                # หาวิชาที่น่าจะเป็น prerequisite
                if number > 101:  # ไม่ใช่วิชาแรก
                    for prev_num in [number-1, number-100, number-99]:
                        potential_prereq = f"{prefix}{prev_num:03d}"
                        if potential_prereq in self.course_catalog:
                            G.add_edge(potential_prereq, course, weight=0.5)
        
        self.prerequisite_graph = G
        logger.info(f"✅ Detected {G.number_of_edges()} prerequisite relationships")
        return G
    
    def create_course_dna(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        สร้าง Course DNA อัตโนมัติ
        """
        logger.info("🧬 Creating Course DNA profiles...")
        
        # Step 1: Build catalog
        self.auto_build_course_catalog(data)
        
        # Step 2: Detect prerequisites
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
            # คำนวณ critical level
            blocks_count = len(list(self.prerequisite_graph.successors(course_id)))
            prereq_count = len(list(self.prerequisite_graph.predecessors(course_id)))
            
            course_dna[course_id].update({
                'has_prereq': prereq_count > 0,
                'num_prereq': prereq_count,
                'blocks_count': blocks_count,
                'critical_level': blocks_count,
                'is_gateway': blocks_count >= 3,
                'is_bottleneck': blocks_count >= 5
            })
        
        self.course_profiles = course_dna
        logger.info(f"✅ Created DNA for {len(course_dna)} courses")
        
        # แสดงวิชาที่สำคัญ
        critical_courses = [(c, d['critical_level']) for c, d in course_dna.items() 
                          if d.get('critical_level', 0) >= 3]
        if critical_courses:
            critical_courses.sort(key=lambda x: x[1], reverse=True)
            logger.info("🎯 Critical courses detected:")
            for course, level in critical_courses[:5]:
                logger.info(f"   - {course}: blocks {level} courses")
        
        return course_dna
    
    def _normalize_course_code(self, code: str) -> str:
        """แปลงรหัสวิชาให้เป็นรูปแบบมาตรฐาน"""
        if pd.isna(code):
            return ""
        
        code = str(code).strip()
        
        # ลบช่องว่างและอักขระพิเศษ
        code_clean = re.sub(r'[\s\-_]', '', code)
        
        # ใส่ขีดกลางในตำแหน่งที่ถูกต้อง
        if len(code_clean) >= 8:
            if len(code_clean) == 8:  # XXXXXXXX
                code = f"{code_clean[:2]}-{code_clean[2:5]}-{code_clean[5:8]}"
            elif len(code_clean) >= 11:  # XXXXXXXXXXX
                code = f"{code_clean[:2]}-{code_clean[2:5]}-{code_clean[5:8]}-{code_clean[8:11]}"
        
        return code
    
    def _estimate_year_from_code(self, code: str) -> int:
        """ประมาณชั้นปีจากรหัสวิชา"""
        # Pattern: XX-XXX-YXX โดย Y คือชั้นปี
        match = re.match(r'\d+-\d+-(\d)\d+', code)
        if match:
            year = int(match.group(1))
            return min(max(year, 1), 4)  # จำกัด 1-4
        return 2  # default
    
    def _collect_name_variations(self, data: pd.DataFrame, course_code: str) -> List[str]:
        """รวบรวมชื่อวิชาที่แตกต่างกันแต่มีรหัสเดียวกัน"""
        variations = set()
        
        course_data = data[data['COURSE_CODE'] == course_code]
        
        if 'COURSE_TITLE_TH' in course_data.columns:
            variations.update(course_data['COURSE_TITLE_TH'].dropna().unique())
        if 'COURSE_TITLE_EN' in course_data.columns:
            variations.update(course_data['COURSE_TITLE_EN'].dropna().unique())
        
        return list(variations)
    
    def _create_dna_from_transcript(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง DNA จาก Transcript"""
        course_dna = {}
        
        # Normalize codes
        df['COURSE_CODE_NORM'] = df['COURSE_CODE'].apply(self._normalize_course_code)
        grouped = df.groupby('COURSE_CODE_NORM')
        
        for course_code, group_data in grouped:
            if not course_code:
                continue
            
            grades = []
            grade_letters = []
            retake_students = set()
            
            # วิเคราะห์การลงซ้ำ
            student_attempts = group_data.groupby('Dummy StudentNO').size()
            for student, count in student_attempts.items():
                if count > 1:
                    retake_students.add(student)
            
            for _, row in group_data.iterrows():
                grade_letter = str(row.get('GRADE', '')).strip().upper()
                grade_point = row.get('GRADE_POINT', None)
                
                if pd.notna(grade_point):
                    grades.append(float(grade_point))
                    grade_letters.append(grade_letter)
                elif grade_letter in self.grade_mapping:
                    grades.append(self.grade_mapping[grade_letter])
                    grade_letters.append(grade_letter)
            
            if grades:
                catalog_info = self.course_catalog.get(course_code, {})
                grade_counts = pd.Series(grade_letters).value_counts()
                total = len(grades)
                unique_students = group_data['Dummy StudentNO'].nunique()
                
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
                    
                    # Failure และ Retake
                    'fail_rate': sum(1 for g in grade_letters if g in ['F', 'W', 'I']) / total,
                    'f_rate': grade_counts.get('F', 0) / total,
                    'w_rate': grade_counts.get('W', 0) / total,
                    'd_rate': sum(grade_counts.get(g, 0) for g in ['D', 'D+']) / total,
                    'retake_rate': len(retake_students) / unique_students if unique_students > 0 else 0,
                    
                    # Success
                    'a_rate': grade_counts.get('A', 0) / total,
                    'pass_rate': sum(1 for g in grades if g >= 1.0) / total,
                    
                    # Difficulty
                    'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                    'sample_size': unique_students,
                    
                    # Course type (auto-detect)
                    'is_lab': self._is_lab_course(catalog_info.get('name_th', ''), 
                                                 catalog_info.get('name_en', '')),
                    'is_project': self._is_project_course(catalog_info.get('name_th', ''), 
                                                         catalog_info.get('name_en', '')),
                    'is_math': self._is_math_course(catalog_info.get('name_th', ''), 
                                                   catalog_info.get('name_en', '')),
                    'is_programming': self._is_programming_course(catalog_info.get('name_th', ''), 
                                                                 catalog_info.get('name_en', ''))
                }
        
        return course_dna
    
    def _create_dna_from_subjects(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """สร้าง DNA จาก Subject columns"""
        course_dna = {}
        
        # หาคอลัมน์ที่เป็นวิชา
        exclude_patterns = ['ชื่อ', 'ปี', 'จบ', 'graduated', 'gpa', 'id', 'สถานะ', 'รหัส', 'เทอม']
        course_columns = []
        
        for col in df.columns:
            if not any(pattern in col.lower() for pattern in exclude_patterns):
                # ตรวจสอบว่ามีข้อมูลเกรด
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # ถ้าเป็นเกรด
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
                grade_counts = pd.Series(grade_letters).value_counts() if grade_letters else pd.Series()
                total = len(grades)
                
                # พยายามจับคู่กับรหัสวิชา
                normalized_code = self._extract_course_code_from_name(course_col)
                if not normalized_code:
                    normalized_code = course_col
                
                course_dna[normalized_code] = {
                    'course_title': course_col,
                    'credit': 3,  # default
                    'typical_year': 0,
                    'typical_term': 0,
                    
                    # Statistics
                    'avg_grade': np.mean(grades),
                    'std_grade': np.std(grades) if len(grades) > 1 else 0,
                    'median_grade': np.median(grades),
                    'percentile_25': np.percentile(grades, 25) if len(grades) > 3 else min(grades),
                    'percentile_75': np.percentile(grades, 75) if len(grades) > 3 else max(grades),
                    
                    # Failure
                    'fail_rate': sum(1 for g in grades if g == 0) / total,
                    'f_rate': grade_counts.get('F', 0) / total if grade_letters else 0,
                    'w_rate': grade_counts.get('W', 0) / total if grade_letters else 0,
                    'd_rate': sum(1 for g in grades if 1.0 <= g <= 1.5) / total,
                    'retake_rate': 0,
                    
                    # Success
                    'a_rate': sum(1 for g in grades if g >= 3.5) / total,
                    'pass_rate': sum(1 for g in grades if g >= 1.0) / total,
                    
                    # Difficulty
                    'difficulty_score': self._calculate_difficulty_score(grades, grade_letters),
                    'sample_size': total,
                    
                    # Type detection
                    'is_lab': 'lab' in course_col.lower() or 'ปฏิบัติ' in course_col,
                    'is_project': 'project' in course_col.lower() or 'โครงงาน' in course_col,
                    'is_math': any(kw in course_col.lower() for kw in ['math', 'คณิต', 'calculus', 'แคล']),
                    'is_programming': any(kw in course_col.lower() for kw in ['program', 'โปรแกรม', 'code'])
                }
        
        return course_dna
    
    def _extract_course_code_from_name(self, name: str) -> str:
        """พยายามดึงรหัสวิชาจากชื่อ"""
        # Pattern: XX-XXX-XXX
        match = re.search(r'\d{2}-\d{3}-\d{3}', name)
        if match:
            return match.group()
        
        # Pattern: XXXXXXXX
        match = re.search(r'\d{8,11}', name)
        if match:
            code = match.group()
            return self._normalize_course_code(code)
        
        return ""
    
    def _is_lab_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Lab"""
        lab_keywords = ['laboratory', 'lab', 'ปฏิบัติการ', 'ปฏิบัติ', 'practicum']
        text = f"{name_th} {name_en}".lower()
        return any(kw in text for kw in lab_keywords)
    
    def _is_project_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Project"""
        project_keywords = ['project', 'โครงงาน', 'โครงการ', 'capstone']
        text = f"{name_th} {name_en}".lower()
        return any(kw in text for kw in project_keywords)
    
    def _is_math_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชาคณิตศาสตร์"""
        math_keywords = ['calculus', 'mathematics', 'math', 'statistics', 
                        'แคลคูลัส', 'คณิตศาสตร์', 'สถิติ', 'พีชคณิต']
        text = f"{name_th} {name_en}".lower()
        return any(kw in text for kw in math_keywords)
    
    def _is_programming_course(self, name_th: str, name_en: str) -> bool:
        """ตรวจสอบว่าเป็นวิชา Programming"""
        prog_keywords = ['programming', 'coding', 'software', 'algorithm', 'data structure',
                        'โปรแกรม', 'ซอฟต์แวร์', 'อัลกอริทึม', 'โครงสร้างข้อมูล']
        text = f"{name_th} {name_en}".lower()
        return any(kw in text for kw in prog_keywords)
    
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
    
    def _calculate_difficulty_score(self, grades: List[float], 
                                   grade_letters: List[str] = None) -> float:
        """คำนวณความยากของวิชา"""
        if not grades:
            return 0.5
        
        fail_rate = sum(1 for g in grades if g == 0) / len(grades)
        avg_grade = np.mean(grades)
        
        w_rate = 0
        d_rate = sum(1 for g in grades if 1.0 <= g <= 1.5) / len(grades)
        
        if grade_letters:
            w_rate = sum(1 for g in grade_letters if g == 'W') / len(grade_letters)
        
        # Weighted difficulty
        difficulty = (
            fail_rate * 0.35 +           # น้ำหนักการตก
            ((4 - avg_grade) / 4) * 0.30 + # เกรดเฉลี่ยต่ำ
            w_rate * 0.20 +              # การถอน
            d_rate * 0.15                # ได้ D
        )
        
        return min(1.0, max(0.0, difficulty))
    
    def create_temporal_snapshots(self, data: pd.DataFrame, 
                                 course_dna: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """สร้าง Temporal Snapshots"""
        logger.info("📸 Creating temporal snapshots...")
        
        snapshots = []
        
        if self.transcript_mode:
            data['COURSE_CODE_NORM'] = data['COURSE_CODE'].apply(self._normalize_course_code)
            students = data['Dummy StudentNO'].unique()
            
            for student_id in students:
                student_data = data[data['Dummy StudentNO'] == student_id].copy()
                student_data = student_data.sort_values(['ปีการศึกษา', 'เทอม'])
                
                # ตรวจสอบการจบ
                graduated = self._check_graduation_status(student_data)
                
                # Track progress
                failed_courses = []
                passed_courses = set()
                retake_count = {}
                cumulative_grades = []
                
                terms = student_data[['ปีการศึกษา', 'เทอม']].drop_duplicates()
                
                for idx, (year, term) in enumerate(terms.values):
                    # ข้อมูลจนถึงเทอมนี้
                    data_until = student_data[
                        (student_data['ปีการศึกษา'] < year) | 
                        ((student_data['ปีการศึกษา'] == year) & (student_data['เทอม'] <= term))
                    ]
                    
                    # เทอมปัจจุบัน
                    current_term = student_data[
                        (student_data['ปีการศึกษา'] == year) & (student_data['เทอม'] == term)
                    ]
                    
                    # Update status
                    for _, row in current_term.iterrows():
                        course = row['COURSE_CODE_NORM']
                        grade = row['GRADE']
                        
                        if grade in ['F', 'W', 'I']:
                            if course not in failed_courses:
                                failed_courses.append(course)
                            retake_count[course] = retake_count.get(course, 0) + 1
                        else:
                            if course in failed_courses:
                                failed_courses.remove(course)
                            passed_courses.add(course)
                            
                            if pd.notna(row['GRADE_POINT']):
                                cumulative_grades.append(row['GRADE_POINT'])
                    
                    # Create snapshot
                    snapshot = self._create_snapshot_features(
                        student_id=student_id,
                        term_number=idx + 1,
                        year=year,
                        term=term,
                        data_until=data_until,
                        current_term=current_term,
                        course_dna=course_dna,
                        failed_courses=failed_courses,
                        passed_courses=passed_courses,
                        retake_count=retake_count,
                        cumulative_grades=cumulative_grades,
                        graduated=graduated
                    )
                    
                    snapshots.append(snapshot)
        
        else:
            # Subject-based format
            snapshots = self._create_snapshots_from_subjects(data, course_dna)
        
        snapshot_df = pd.DataFrame(snapshots)
        logger.info(f"✅ Created {len(snapshots)} snapshots")
        
        return snapshot_df
    
    def _check_graduation_status(self, student_data: pd.DataFrame) -> int:
        """ตรวจสอบสถานะการจบ"""
        # Check critical failures
        critical_patterns = ['project', 'โครงงาน', 'coop', 'สหกิจ']
        
        for _, row in student_data.iterrows():
            # Check if it's a critical course
            is_critical = any(
                pattern in str(row.get('COURSE_TITLE_TH', '')).lower() or
                pattern in str(row.get('COURSE_TITLE_EN', '')).lower()
                for pattern in critical_patterns
            )
            
            # Also check by course code pattern
            if re.search(r'40[1-4]|30[2-4]', str(row.get('COURSE_CODE', ''))):
                is_critical = True
            
            if is_critical and row.get('GRADE') in ['F', 'W']:
                return 0  # ไม่จบ
        
        return 1  # จบ
    
    def _create_snapshot_features(self, **kwargs) -> Dict[str, Any]:
        """สร้าง Features สำหรับ Snapshot"""
        features = {
            'student_id': kwargs['student_id'],
            'term_number': kwargs['term_number'],
            'year': kwargs['year'],
            'term': kwargs['term'],
            'graduated': kwargs['graduated']
        }
        
        data_until = kwargs['data_until']
        current_term = kwargs['current_term']
        course_dna = kwargs['course_dna']
        failed_courses = kwargs['failed_courses']
        passed_courses = kwargs['passed_courses']
        
        # Academic performance
        all_grades = data_until['GRADE_POINT'].dropna().values
        if len(all_grades) > 0:
            features['gpax'] = np.mean(all_grades)
            features['total_credits'] = data_until['CREDIT'].sum()
            features['courses_passed'] = len(passed_courses)
            features['courses_failed'] = len(failed_courses)
        else:
            features.update({'gpax': 0, 'total_credits': 0, 
                           'courses_passed': 0, 'courses_failed': 0})
        
        # Current term performance
        term_grades = current_term['GRADE_POINT'].dropna().values
        if len(term_grades) > 0:
            features['gpa_term'] = np.mean(term_grades)
            features['credits_term'] = current_term['CREDIT'].sum()
        else:
            features['gpa_term'] = 0
            features['credits_term'] = 0
        
        # Prerequisite impact
        blocked_courses = set()
        for failed in failed_courses:
            # Find courses blocked by this failure
            if failed in self.prerequisite_graph:
                blocked_courses.update(self.prerequisite_graph.successors(failed))
        
        features['blocked_courses'] = len(blocked_courses)
        features['critical_blocks'] = sum(
            1 for c in blocked_courses 
            if course_dna.get(c, {}).get('critical_level', 0) >= 3
        )
        
        # Retake analysis
        features['total_retakes'] = sum(kwargs['retake_count'].values())
        features['unique_retakes'] = len(kwargs['retake_count'])
        
        # Risk indicators
        features['at_risk'] = 1 if features.get('gpax', 0) < 2.0 else 0
        features['high_risk'] = 1 if len(failed_courses) >= 3 or features.get('gpax', 0) < 1.5 else 0
        
        # Course type performance
        for course_type in ['math', 'programming', 'lab', 'project']:
            type_grades = []
            for _, row in data_until.iterrows():
                course_code = self._normalize_course_code(row.get('COURSE_CODE', ''))
                if course_code in course_dna:
                    if course_dna[course_code].get(f'is_{course_type}', False):
                        if pd.notna(row['GRADE_POINT']):
                            type_grades.append(row['GRADE_POINT'])
            
            features[f'gpa_{course_type}'] = np.mean(type_grades) if type_grades else 0
            features[f'fail_rate_{course_type}'] = (
                sum(1 for g in type_grades if g == 0) / len(type_grades) 
                if type_grades else 0
            )
        
        return features
    
    def _create_snapshots_from_subjects(self, data: pd.DataFrame, 
                                       course_dna: Dict[str, Dict[str, float]]) -> List[Dict]:
        """สร้าง Snapshots จาก Subject-based format"""
        snapshots = []
        
        for idx, row in data.iterrows():
            student_id = f"Student_{idx}"
            graduated = row.get('graduated', row.get('จบการศึกษา', row.get('สถานะ', 0)))
            
            # Convert to binary
            if isinstance(graduated, str):
                graduated = 1 if 'จบ' in graduated and 'ไม่' not in graduated else 0
            
            # Collect grades
            course_grades = {}
            for col in course_dna.keys():
                if col in row.index:
                    grade_val = row[col]
                    numeric_grade = self._convert_grade_to_numeric(grade_val)
                    if numeric_grade is not None:
                        course_grades[col] = numeric_grade
            
            if course_grades:
                # Simulate progression
                courses_list = list(course_grades.items())
                courses_per_term = 5
                num_terms = max(1, (len(courses_list) + courses_per_term - 1) // courses_per_term)
                
                cumulative_grades = []
                
                for term in range(1, min(num_terms + 1, 9)):
                    end_idx = min(term * courses_per_term, len(courses_list))
                    courses_so_far = dict(courses_list[:end_idx])
                    
                    snapshot = {
                        'student_id': student_id,
                        'term_number': term,
                        'graduated': graduated,
                        'gpax': np.mean(list(courses_so_far.values())),
                        'total_courses': len(courses_so_far),
                        'courses_failed': sum(1 for g in courses_so_far.values() if g == 0),
                        'at_risk': 1 if np.mean(list(courses_so_far.values())) < 2.0 else 0
                    }
                    
                    # Add course type performance
                    for course_type in ['math', 'programming', 'lab', 'project']:
                        type_grades = [
                            grade for course, grade in courses_so_far.items()
                            if course_dna.get(course, {}).get(f'is_{course_type}', False)
                        ]
                        snapshot[f'gpa_{course_type}'] = np.mean(type_grades) if type_grades else 0
                    
                    snapshots.append(snapshot)
        
        return snapshots
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method: เตรียมข้อมูลสำหรับการเทรน"""
        logger.info("🚀 Starting advanced feature engineering...")
        
        # Step 1: Create Course DNA
        course_dna = self.create_course_dna(df)
        
        # Step 2: Create Temporal Snapshots
        snapshot_df = self.create_temporal_snapshots(df, course_dna)
        
        # Step 3: Prepare features and target
        if 'graduated' in snapshot_df.columns:
            feature_cols = [col for col in snapshot_df.columns 
                          if col not in ['graduated', 'student_id', 'year', 'term']]
            
            X = snapshot_df[feature_cols].fillna(0)
            y = snapshot_df['graduated']
            
            logger.info(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features")
            logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")
            
            # Show important features
            if len(X) > 0:
                feature_stats = X.describe().T[['mean', 'std']].sort_values('std', ascending=False)
                logger.info("\n🎯 Top 10 features by variability:")
                for feat in feature_stats.head(10).index:
                    logger.info(f"   - {feat}: mean={feature_stats.loc[feat, 'mean']:.3f}, "
                              f"std={feature_stats.loc[feat, 'std']:.3f}")
            
            return X, y
        else:
            logger.error("❌ No 'graduated' column found")
            return pd.DataFrame(), pd.Series()


# For backward compatibility
CurriculumAnalyzer = type('CurriculumAnalyzer', (), {})
CourseRetakeSimulator = type('CourseRetakeSimulator', (), {})
CourseNameNormalizer = type('CourseNameNormalizer', (), {})

# Usage example
if __name__ == "__main__":
    # Test with your data
    df = pd.read_csv('tt22.csv')
    
    engineer = AdvancedFeatureEngineer(
        grade_mapping={
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'I': 0.0,
            'S': 0.0, 'U': 0.0
        }
    )
    
    X, y = engineer.prepare_training_data(df)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Features: {list(X.columns)[:20]}")
    print(f"Target distribution:\n{y.value_counts()}")
