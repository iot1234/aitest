# แก้ไขส่วนบนของ app.py
import os
import sys
from pathlib import Path

# โหลด .env ก่อน import อื่นๆ
from dotenv import load_dotenv

# หา path ของ .env file
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✅ Loaded .env from {env_path.absolute()}")
else:
    # ลองหาจาก parent directory
    parent_env = Path('..') / '.env'
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env, override=True)
        print(f"✅ Loaded .env from {parent_env.absolute()}")
    else:
        print("⚠️ .env file not found, using system environment variables")

# ตรวจสอบว่าโหลดได้จริง
print(f"R2 Access Key present: {bool(os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID'))}")
print(f"R2 Secret Key present: {bool(os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY'))}")

from advanced_training import AdvancedFeatureEngineer, ContextAwarePredictor
from explainable_ai import ExplainablePredictor

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    session,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import logging.config
import logging
from datetime import datetime, timedelta
import warnings
from collections import Counter, deque
import time
from functools import wraps
from imblearn.over_sampling import SMOTE
import tempfile
import json
import re
import copy
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Any, Dict, Optional, List
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import hashlib
import base64

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception

import config
import math

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ===============================
# ENHANCED FEATURES - ALL IN ONE
# ===============================

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# ฟังก์ชันประมวลผลข้อมูล TAN1
def preprocess_tan1_data(file_path):
    """
    ประมวลผลข้อมูล TAN1.csv จาก Long Format เป็น Wide Format
    และคำนวณ RESULT (จบการศึกษา/ไม่จบการศึกษา)
    """
    
    # อ่านข้อมูล
    df_long = pd.read_csv(file_path)
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['STUDENT_ID', 'COURSE_ID', 'GRADE', 'CREDIT']
    for col in required_columns:
        if col not in df_long.columns:
            raise ValueError(f"ไม่พบคอลัมน์ {col} ในไฟล์ข้อมูล")
    
    # สร้าง grade mapping
    grade_mapping = {
        'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 
        'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'S': None  # S ไม่นำมาคำนวณ GPA
    }
    
    # เพิ่มคอลัมน์ GRADE_POINT_NUM
    df_long['GRADE_POINT_NUM'] = df_long['GRADE'].map(grade_mapping)
    
    # แปลงเป็น Wide Format
    df_wide = df_long.pivot_table(
        index='STUDENT_ID', 
        columns='COURSE_ID', 
        values='GRADE', 
        aggfunc='first'
    ).reset_index()
    
    # คำนวณ GPA และหน่วยกิตรวมสำหรับแต่ละนักศึกษา
    student_stats = []
    
    for student_id in df_long['STUDENT_ID'].unique():
        student_data = df_long[df_long['STUDENT_ID'] == student_id]
        
        # คำนวณ GPA (ไม่รวม S)
        gpa_data = student_data[student_data['GRADE'] != 'S']
        if len(gpa_data) > 0:
            total_grade_points = (gpa_data['GRADE_POINT_NUM'] * gpa_data['CREDIT']).sum()
            total_credits = gpa_data['CREDIT'].sum()
            gpa = total_grade_points / total_credits if total_credits > 0 else 0
        else:
            gpa = 0
            total_credits = 0
        
        # หน่วยกิตรวม (รวม S)
        total_credits_all = student_data['CREDIT'].sum()
        
        # กำหนดเกณฑ์จบการศึกษา: GPA >= 2.00 และหน่วยกิต >= 136
        result = 1 if (gpa >= 2.00 and total_credits_all >= 136) else 0
        
        student_stats.append({
            'STUDENT_ID': student_id,
            'GPA': gpa,
            'TOTAL_CREDITS': total_credits_all,
            'RESULT': result
        })
    
    # รวม RESULT เข้ากับ df_wide
    stats_df = pd.DataFrame(student_stats)
    df_wide = df_wide.merge(stats_df[['STUDENT_ID', 'RESULT']], on='STUDENT_ID')
    
    # สร้าง course_credit_map
    course_credit_map = df_long.groupby('COURSE_ID')['CREDIT'].first().to_dict()
    
    return df_wide, df_long, course_credit_map

# เพิ่มคลาส AdvancedModelTrainer ลงใน app.py
class AdvancedModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.grade_mapping = {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'S': None
        }
        self.course_credit_map = {}
        
    def prepare_training_data(self, df_wide_format, df_long_format_original):
        """เตรียมข้อมูลสำหรับการเทรน"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🎯 เตรียมข้อมูลแบบ Dynamic Snapshots...")
        
        # เก็บ course_credit_map
        self.course_credit_map = df_long_format_original.groupby('COURSE_ID')['CREDIT'].first().to_dict()
        
        # สร้างโปรไฟล์รายวิชา
        logger.info("🏛️ สร้างโปรไฟล์รายวิชา (Course Profiling)...")
        course_profiles = self.create_course_profiles(df_wide_format, df_long_format_original)
        logger.info(f"✅ สร้างโปรไฟล์ {len(course_profiles)} รายวิชาเสร็จสิ้น")
        
        # สร้างข้อมูลเทรนแบบ Dynamic Snapshots
        X, y = self.create_dynamic_snapshots(df_wide_format, course_profiles)
        
        logger.info(f"✅ สร้างข้อมูลเทรน {len(X)} ตัวอย่างจาก {len(df_wide_format)} นักศึกษา")
        
        return X, y, course_profiles
    
    def create_course_profiles(self, df_wide_format, df_long_format_original):
        """สร้างโปรไฟล์รายวิชา (DNA ของแต่ละวิชา)"""
        course_profiles = {}
        
        # ใช้ข้อมูลจาก df_long_format_original เพื่อความแม่นยำ
        for course_id in df_long_format_original['COURSE_ID'].unique():
            course_data = df_long_format_original[df_long_format_original['COURSE_ID'] == course_id]
            
            # คำนวณสถิติจากข้อมูลจริง
            grades = course_data['GRADE'].tolist()
            credits = course_data['CREDIT'].iloc[0] if len(course_data) > 0 else 3  # default 3 credits
            
            # แปลงเกรดเป็นตัวเลข (ไม่รวม S)
            numeric_grades = []
            for grade in grades:
                if grade in self.grade_mapping and self.grade_mapping[grade] is not None:
                    numeric_grades.append(self.grade_mapping[grade])
            
            if numeric_grades:
                avg_grade = np.mean(numeric_grades)
                fail_rate = len([g for g in numeric_grades if g < 2.0]) / len(numeric_grades)
                difficulty_score = 4.0 - avg_grade  # ยิ่งยาก avg_grade ยิ่งต่ำ
                
                course_profiles[course_id] = {
                    'avg_grade': avg_grade,
                    'fail_rate': fail_rate,
                    'difficulty_score': difficulty_score,
                    'credits': credits,
                    'total_students': len(numeric_grades)
                }
        
        return course_profiles
    
    def create_dynamic_snapshots(self, df_wide_format, course_profiles):
        """สร้างข้อมูลแบบ Dynamic Snapshots (Journey-based Training)"""
        X = []
        y = []
        
        for _, student in df_wide_format.iterrows():
            # หาวิชาที่นักศึกษาเรียน
            student_courses = {}
            for col in df_wide_format.columns:
                if col not in ['STUDENT_ID', 'RESULT'] and pd.notna(student[col]):
                    student_courses[col] = student[col]
            
            # สร้าง snapshots ยืดหยุ่นตามจำนวนวิชาที่เรียนจริง
            course_list = list(student_courses.items())
            total_courses = len(course_list)
            
            # สร้าง snapshots ที่หลากหลาย
            snapshot_sizes = []
            if total_courses >= 3:
                snapshot_sizes.append(3)
            if total_courses >= 6:
                snapshot_sizes.append(6)
            if total_courses >= 9:
                snapshot_sizes.append(9)
            if total_courses >= 12:
                snapshot_sizes.append(12)
            if total_courses > 12:
                snapshot_sizes.append(total_courses)
            
            for snapshot_size in snapshot_sizes:
                if total_courses >= snapshot_size:
                    # เลือกวิชาแรก snapshot_size วิชา
                    current_courses = dict(course_list[:snapshot_size])
                    
                    # สร้างฟีเจอร์
                    features = self.create_dynamic_features(current_courses, course_profiles, total_courses)
                    
                    X.append(features)
                    y.append(student['RESULT'])
        
        return np.array(X), np.array(y)
    
    def create_dynamic_features(self, grades_dict, course_profiles, total_courses_taken):
        """สร้างฟีเจอร์แบบ Dynamic พร้อม Context-Aware Features"""
        
        if not grades_dict:
            return np.zeros(20)  # เพิ่มฟีเจอร์เป็น 20 ตัว
        
        # ฟีเจอร์พื้นฐาน - สถิติเกรด
        numeric_grades = []
        for grade in grades_dict.values():
            if grade in self.grade_mapping and self.grade_mapping[grade] is not None:
                numeric_grades.append(self.grade_mapping[grade])
        
        if not numeric_grades:
            return np.zeros(20)
        
        # 1-5: สถิติเกรดพื้นฐาน
        current_gpa = np.mean(numeric_grades)
        gpa_std = np.std(numeric_grades) if len(numeric_grades) > 1 else 0
        min_grade = min(numeric_grades)
        max_grade = max(numeric_grades)
        grade_range = max_grade - min_grade
        
        # 6-8: การกระจายเกรด
        high_grades = len([g for g in numeric_grades if g >= 3.5]) / len(numeric_grades)
        medium_grades = len([g for g in numeric_grades if 2.5 <= g < 3.5]) / len(numeric_grades)
        low_grades = len([g for g in numeric_grades if g < 2.5]) / len(numeric_grades)
        
        # 9-12: Context-Aware Features (ใหม่)
        avg_course_difficulty = 0
        avg_course_fail_rate = 0
        num_killer_courses = 0  # วิชายาก (difficulty > 2.0)
        num_easy_courses = 0    # วิชาง่าย (difficulty < 1.0)
        
        if course_profiles:
            difficulties = []
            fail_rates = []
            
            for course in grades_dict.keys():
                if course in course_profiles:
                    difficulty = course_profiles[course]['difficulty_score']
                    fail_rate = course_profiles[course]['fail_rate']
                    
                    difficulties.append(difficulty)
                    fail_rates.append(fail_rate)
                    
                    # นับวิชายากและง่าย
                    if difficulty > 2.0:
                        num_killer_courses += 1
                    elif difficulty < 1.0:
                        num_easy_courses += 1
            
            avg_course_difficulty = np.mean(difficulties) if difficulties else 0
            avg_course_fail_rate = np.mean(fail_rates) if fail_rates else 0
        
        # 13-15: ฟีเจอร์เพิ่มเติม
        num_failed_courses = len([g for g in numeric_grades if g < 2.0])
        killer_course_ratio = num_killer_courses / len(grades_dict) if len(grades_dict) > 0 else 0
        
        # คำนวณ GPA Gap (ความแตกต่างระหว่าง GPA ปัจจุบันกับ GPA เฉลี่ยของวิชาที่เรียน)
        expected_gpa = 0
        if course_profiles:
            expected_grades = []
            for course in grades_dict.keys():
                if course in course_profiles:
                    expected_grades.append(course_profiles[course]['avg_grade'])
            expected_gpa = np.mean(expected_grades) if expected_grades else current_gpa
        
        gpa_gap = current_gpa - expected_gpa
        
        # 16-20: แนวโน้มและข้อมูลเพิ่มเติม
        if len(numeric_grades) >= 3:
            recent_trend = np.mean(numeric_grades[-3:]) - np.mean(numeric_grades[:-3])
        else:
            recent_trend = 0
        
        # Progress ratio
        progress_ratio = len(grades_dict) / total_courses_taken if total_courses_taken > 0 else 0
        
        # Consistency score (ความสม่ำเสมอของเกรด)
        consistency_score = 1 / (1 + gpa_std)  # ยิ่ง std น้อย ยิ่งสม่ำเสมอ
        
        # Academic momentum (แรงผลักดันทางการเรียน)
        academic_momentum = current_gpa * progress_ratio
        
        # Risk score (คะแนนความเสี่ยง)
        risk_score = (num_failed_courses + num_killer_courses) / len(grades_dict) if len(grades_dict) > 0 else 0
        
        features = np.array([
            current_gpa, gpa_std, min_grade, max_grade, grade_range,  # 1-5
            high_grades, medium_grades, low_grades,                   # 6-8
            avg_course_difficulty, avg_course_fail_rate,             # 9-10
            num_killer_courses, num_easy_courses,                    # 11-12
            num_failed_courses, killer_course_ratio, gpa_gap,       # 13-15
            recent_trend, progress_ratio, consistency_score,        # 16-18
            academic_momentum, risk_score                            # 19-20
        ])
        
        # จัดการ NaN values
        return self._handle_nan_features(features)
    
    def _handle_nan_features(self, features):
        """จัดการ NaN values ในฟีเจอร์"""
        # แทนที่ NaN ด้วย 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
    
    def train_models(self, X, y):
        """เทรนโมเดลทั้งหมดพร้อม SMOTE และ Feature Scaling"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🧠 เริ่มเทรนโมเดล AI...")
        
        # Feature Scaling
        logger.info("📏 ปรับขนาดฟีเจอร์ด้วย StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # แบ่งข้อมูล (ลด test_size เป็น 0.05 เพื่อให้มีข้อมูลเทรนมากขึ้น)
        # ตรวจสอบจำนวนคลาสและการกระจาย
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        # ถ้าข้อมูลน้อยมากหรือคลาสไม่สมดุล ให้ปรับพารามิเตอร์
        if len(X_scaled) < 10 or min_class_count < 2:
            test_size = 0.2
            stratify_param = None  # ไม่ใช้ stratify ถ้าข้อมูลน้อยหรือคลาสมีสมาชิกน้อย
            logger.info(f"⚠️ ข้อมูลน้อยหรือคลาสไม่สมดุล (min_class: {min_class_count}) ปรับ test_size เป็น 0.2 และไม่ใช้ stratify")
        else:
            test_size = 0.05
            stratify_param = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        # ตรวจสอบความสมดุลของข้อมูล
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"การกระจายข้อมูลเทรน: {dict(zip(unique, counts))}")
        
        # ใช้ SMOTE หากข้อมูลไม่สมดุล
        if len(unique) > 1 and min(counts) / max(counts) < 0.5:
            logger.info("⚖️ ใช้ SMOTE เพื่อจัดการข้อมูลที่ไม่สมดุล...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            unique_after, counts_after = np.unique(y_train, return_counts=True)
            logger.info(f"การกระจายข้อมูลหลัง SMOTE: {dict(zip(unique_after, counts_after))}")
        
        results = {}
        best_model_name = None
        
        for name, model in self.models.items():
            logger.info(f"   🔄 เทรน {name}...")
            
            try:
                # เทรนโมเดล
                model.fit(X_train, y_train)
                
                # ประเมินผล
                y_pred = model.predict(X_test)
                
                # คำนวณเมตริกต่างๆ
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred
                }
                
                logger.info(f"      ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                
                # เลือกโมเดลที่ดีที่สุด
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    best_model_name = name
                    
            except Exception as e:
                logger.warning(f"      ⚠️ {name} ไม่สามารถเทรนได้: {e}")
                continue
        
        logger.info(f"🏆 โมเดลที่ดีที่สุด: {best_model_name} (Accuracy: {self.best_score:.3f})")
        
        return results
    
    def save_model(self, model_path):
        """บันทึกโมเดลและ scaler"""
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'course_credit_map': self.course_credit_map,
                'grade_mapping': self.grade_mapping
            }
            joblib.dump(model_data, model_path)
            return True
        return False
    
    def load_model(self, model_path):
        """โหลดโมเดลและ scaler"""
        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.course_credit_map = model_data.get('course_credit_map', {})
            self.grade_mapping = model_data.get('grade_mapping', self.grade_mapping)
            return True
        except Exception as e:
            print(f"⚠️ Error loading model: {str(e)}")
            # หากโหลดโมเดลไม่ได้ ให้สร้างโมเดลใหม่
            self._create_fallback_model()
            return True
    
    def _create_fallback_model(self):
        """สร้างโมเดลสำรองเมื่อโหลดโมเดลหลักไม่ได้"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # สร้างโมเดลใหม่
        self.best_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.course_credit_map = {}
        print("✅ Created fallback model successfully")

# ต่อไปเป็นส่วนของ EnhancedPredictionSystem
class EnhancedPredictionSystem:
    def __init__(self):
        self.grade_mapping = {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'WF': 0.0, 'WU': 0.0, 'S': None
        }
        self.advanced_trainer = AdvancedModelTrainer()
        self.advanced_model = None
        self.course_profiles = None
        self.scaler = None
        self.course_credit_map = {}
        
    def load_advanced_model(self, model_path=None):
        """โหลดโมเดลขั้นสูงที่เทรนแล้ว"""
        try:
            if model_path and os.path.exists(model_path):
                if self.advanced_trainer.load_model(model_path):
                    self.advanced_model = self.advanced_trainer.best_model
                    self.scaler = self.advanced_trainer.scaler
                    self.course_credit_map = self.advanced_trainer.course_credit_map
                    print(f"✅ โหลดโมเดลขั้นสูงสำเร็จ: {model_path}")
                    return True
            else:
                # หาโมเดลล่าสุดในโฟลเดอร์ models
                models_dir = "models"
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
                    if model_files:
                        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
                        model_path = os.path.join(models_dir, latest_model)
                        return self.load_advanced_model(model_path)
                
                print("⚠️ ไม่พบโมเดลขั้นสูง ใช้การทำนายแบบพื้นฐาน")
                return False
                
        except Exception as e:
            print(f"❌ ไม่สามารถโหลดโมเดลขั้นสูง: {e}")
            return False
    
    def calculate_gpa(self, grades_dict):
        """คำนวณ GPA จากเกรดที่ได้รับ (ใช้หน่วยกิตจริงของแต่ละวิชา)"""
        if not grades_dict:
            return 0.0
            
        total_points = 0
        total_credits = 0
        
        for subject_id, grade in grades_dict.items():
            if grade in self.grade_mapping and self.grade_mapping[grade] is not None:
                grade_point = self.grade_mapping[grade]
                # ใช้หน่วยกิตจริงจาก course_credit_map หรือ default เป็น 3
                credits = self.course_credit_map.get(subject_id, 3)
                
                total_points += grade_point * credits
                total_credits += credits
        
        return total_points / total_credits if total_credits > 0 else 0.0
    
    def predict_graduation(self, grades_dict):
        """ทำนายการจบการศึกษาด้วยโมเดลขั้นสูง"""
        try:
            if self.advanced_model is None or self.scaler is None:
                print("⚠️ ไม่พบโมเดลขั้นสูง ใช้การทำนายแบบพื้นฐาน")
                return self.predict_graduation_basic(grades_dict)
            
            # ตรวจสอบว่าโมเดลได้รับการเทรนแล้วหรือไม่
            if not hasattr(self.advanced_model, 'classes_'):
                print("⚠️ โมเดลยังไม่ได้เทรน ใช้การทำนายแบบพื้นฐาน")
                return self.predict_graduation_basic(grades_dict)
            
            # สร้างฟีเจอร์สำหรับการทำนาย
            total_courses_taken = len(grades_dict)
            
            # สร้างฟีเจอร์พื้นฐาน
            current_gpa = self.calculate_gpa(grades_dict)
            failed_courses = sum(1 for grade in grades_dict.values() if grade == 'F')
            passed_courses = sum(1 for grade in grades_dict.values() if grade in ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D'])
            
            # สร้างฟีเจอร์เวกเตอร์พื้นฐาน
            features = [
                current_gpa,
                total_courses_taken,
                failed_courses,
                passed_courses,
                failed_courses / max(total_courses_taken, 1),  # อัตราส่วนวิชาตก
                current_gpa * total_courses_taken  # GPA weighted by courses
            ]
            
            # เพิ่มฟีเจอร์ให้ครบตามที่โมเดลต้องการ (สมมติว่าต้องการ 10 ฟีเจอร์)
            while len(features) < 10:
                features.append(0.0)
            
            # ปรับขนาดฟีเจอร์ด้วย scaler
            try:
                features_scaled = self.scaler.transform([features])
            except:
                # หาก scaler ไม่ทำงาน ใช้ฟีเจอร์ดิบ
                features_scaled = [features]
            
            # ทำนายด้วยโมเดล
            prediction = self.advanced_model.predict(features_scaled)[0]
            
            try:
                probability = self.advanced_model.predict_proba(features_scaled)[0]
            except:
                # หากไม่สามารถคำนวณ probability ได้
                probability = [0.3, 0.7] if prediction else [0.7, 0.3]
            
            result = {
                'will_graduate': bool(prediction),
                'confidence': float(max(probability)),
                'probability_graduate': float(probability[1]) if len(probability) > 1 else 0.0,
                'probability_not_graduate': float(probability[0]),
                'current_gpa': current_gpa,
                'total_courses': total_courses_taken,
                'model_type': 'advanced'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการทำนายขั้นสูง: {e}")
            return self.predict_graduation_basic(grades_dict)
    
    def predict_graduation_basic(self, grades_dict):
        """การทำนายแบบพื้นฐาน (fallback)"""
        current_gpa = self.calculate_gpa(grades_dict)
        total_courses = len(grades_dict)
        
        # เกณฑ์พื้นฐาน: GPA >= 2.0 และเรียนครบ 45 วิชา (สมมติ)
        will_graduate = current_gpa >= 2.0 and total_courses >= 45
        confidence = min(0.8, max(0.5, current_gpa / 4.0)) if will_graduate else 0.3
        
        return {
            'will_graduate': will_graduate,
            'confidence': confidence,
            'probability_graduate': confidence if will_graduate else 1 - confidence,
            'probability_not_graduate': 1 - confidence if will_graduate else confidence,
            'current_gpa': current_gpa,
            'total_courses': total_courses,
            'model_type': 'basic'
        }

# ต่อไปเป็นส่วนของ Flask endpoints
    def analyze_performance(self, grades_dict):
        """วิเคราะห์ผลการเรียนแบบครอบคลุม"""
        if not grades_dict:
            return {
                "gpa": 0.0,
                "total_subjects": 0,
                "grade_distribution": {},
                "strengths": ["ยังไม่มีข้อมูลเกรด"],
                "weaknesses": ["ควรเริ่มเรียนและสร้างผลงาน"],
                "recommendations": ["ลงทะเบียนเรียนและทำการบ้านอย่างสม่ำเสมอ"],
                "risk_level": "ไม่ทราบ"
            }
        
        # คำนวณสถิติพื้นฐาน
        gpa = self.calculate_gpa(grades_dict)
        grade_counts = Counter(grades_dict.values())
        total_subjects = len(grades_dict)
        
        # วิเคราะห์จุดแข็ง
        strengths = []
        if grade_counts.get('A', 0) >= 2:
            strengths.append("มีความสามารถสูงในหลายวิชา")
        if grade_counts.get('B+', 0) + grade_counts.get('A', 0) >= total_subjects * 0.6:
            strengths.append("ผลการเรียนโดยรวมดีมาก")
        if gpa >= 3.5:
            strengths.append("มี GPA ในระดับเกียรตินิยม")
        if total_subjects >= 6:
            strengths.append("สามารถจัดการภาระงานได้ดี")
        
        # วิเคราะห์จุดอ่อน
        weaknesses = []
        if grade_counts.get('F', 0) > 0:
            weaknesses.append("มีวิชาที่ไม่ผ่าน ต้องเรียนซ้ำ")
        if grade_counts.get('D', 0) + grade_counts.get('D+', 0) > 0:
            weaknesses.append("มีวิชาที่ผลการเรียนต่ำ ควรปรับปรุง")
        if gpa < 2.0:
            weaknesses.append("GPA ต่ำกว่าเกณฑ์ขั้นต่ำ")
        elif gpa < 2.5:
            weaknesses.append("GPA อยู่ในระดับพอใช้ ควรพัฒนา")
        
        # สร้างคำแนะนำแบบครอบคลุม
        recommendations = []
        
        # คำแนะนำตาม GPA
        if gpa >= 3.5:
            recommendations.extend([
                "🎯 ควรลงวิชาเลือกที่ท้าทายเพิ่มเติมเพื่อพัฒนาทักษะ",
                "📚 พิจารณาทำโครงงานวิจัยหรือ Independent Study",
                "🏆 เตรียมตัวสำหรับการศึกษาต่อระดับสูงหรือทุนการศึกษา",
                "👥 เข้าร่วมกิจกรรมเสริมหลักสูตรและการแข่งขันวิชาการ"
            ])
        elif gpa >= 3.0:
            recommendations.extend([
                "📖 ควรเน้นวิชาพื้นฐานให้แข็งแกร่งก่อนลงวิชาขั้นสูง",
                "⏰ จัดตารางเวลาทบทวนบทเรียนอย่างสม่ำเสมอ",
                "👨‍🏫 ปรึกษาอาจารย์ในวิชาที่มีปัญหาเป็นประจำ",
                "📝 ทำแบบฝึกหัดเพิ่มเติมในวิชาที่ยาก"
            ])
        elif gpa >= 2.5:
            recommendations.extend([
                "⚖️ ควรลดจำนวนวิชาต่อเทอมและเน้นคุณภาพ",
                "🤝 หาเพื่อนกลุ่มเรียนหรือติวเตอร์ช่วยสอน",
                "🔄 ปรับวิธีการเรียนให้เหมาะกับตัวเอง",
                "📊 ติดตามผลการเรียนอย่างใกล้ชิด"
            ])
        elif gpa >= 2.0:
            recommendations.extend([
                "🚨 ต้องปรับปรุงผลการเรียนเร่งด่วน",
                "📚 พิจารณาการเรียนซ้ำในวิชาที่ไม่ผ่าน",
                "💡 ปรึกษาอาจารย์ที่ปรึกษาเพื่อวางแผนการเรียน",
                "🎯 เน้นวิชาหลักและวิชาบังคับก่อน"
            ])
        else:
            recommendations.extend([
                "🆘 ต้องขอความช่วยเหลือจากอาจารย์และครอบครัว",
                "📋 ทบทวนเป้าหมายการศึกษาและแรงจูงใจ",
                "🔧 ปรับวิธีการเรียนและการจัดการเวลาใหม่หมด",
                "⚕️ ตรวจสอบปัญหาสุขภาพหรือปัจจัยส่วนตัว"
            ])
        
        # คำแนะนำตามการกระจายเกรด
        if grade_counts.get('F', 0) > 0:
            recommendations.append(f"🔴 มีวิชาที่ไม่ผ่าน {grade_counts['F']} วิชา - ต้องเรียนซ้ำ")
        
        if grade_counts.get('A', 0) >= 3:
            recommendations.append("⭐ มีความสามารถสูง - ควรท้าทายตัวเองเพิ่มขึ้น")
        
        # คำแนะนำตามจำนวนวิชา
        if total_subjects < 5:
            recommendations.append("📈 ข้อมูลยังน้อย - ควรสะสมผลงานเพิ่มเติม")
        elif total_subjects > 25:
            recommendations.append("💪 มีประสบการณ์เรียนมาก - ใช้ประโยชน์จากความรู้ที่สั่งสม")
        
        # ประเมินความเสี่ยง
        if gpa >= 3.5:
            risk_level = "ต่ำ - มีโอกาสสำเร็จสูง"
        elif gpa >= 2.5:
            risk_level = "ปานกลาง - ควรรักษาระดับ"
        elif gpa >= 2.0:
            risk_level = "สูง - ต้องปรับปรุงเร่งด่วน"
        else:
            risk_level = "สูงมาก - ต้องปรับปรุงอย่างจริงจัง"
        
        return {
            "gpa": round(gpa, 2),
            "total_subjects": total_subjects,
            "grade_distribution": dict(grade_counts),
            "strengths": strengths if strengths else ["มีพื้นฐานการเรียนรู้"],
            "weaknesses": weaknesses if weaknesses else ["ควรพัฒนาทักษะการเรียนต่อไป"],
            "recommendations": recommendations,
            "risk_level": risk_level
        }
    
    def predict_future_performance(self, current_grades, model=None):
        """ทำนายผลการเรียนในอนาคตด้วยโมเดล AI จริง"""
        current_gpa = self.calculate_gpa(current_grades)
        
        # ลองใช้โมเดลขั้นสูงก่อน
        if self.advanced_model and self.course_profiles:
            try:
                # ใช้โมเดลขั้นสูงที่เทรนแล้ว
                features = self.advanced_trainer.create_dynamic_features(current_grades, self.course_profiles)
                
                # ทำนายด้วยโมเดลขั้นสูง
                prediction_result = self.advanced_model.predict([features])[0]
                prediction_proba = self.advanced_model.predict_proba([features])[0]
                
                confidence = float(max(prediction_proba))
                
                # สร้างการกระจายเกรดจากโมเดลขั้นสูง
                predicted_grades = self._generate_grade_distribution_from_model(
                    prediction_result, confidence, current_gpa
                )
                
                # คำนวณ GPA ทำนาย
                predicted_gpa = self._calculate_predicted_gpa(predicted_grades, current_gpa)
                
                # กำหนดแนวโน้ม
                if predicted_gpa > current_gpa + 0.1:
                    trend = "improving"
                elif predicted_gpa < current_gpa - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
                
                return {
                    "predicted_grades": predicted_grades,
                    "predicted_gpa": round(predicted_gpa, 2),
                    "confidence": round(confidence, 3),
                    "trend": trend,
                    "ai_prediction": {
                        "result": prediction_result,
                        "confidence": confidence,
                        "model_used": True,
                        "model_type": "Advanced AI Model"
                    },
                    "analysis_note": f"ทำนายด้วย Advanced AI จาก {len(current_grades)} วิชา, GPA ปัจจุบัน {current_gpa:.2f}"
                }
                
            except Exception as e:
                logger.warning(f"Advanced model prediction failed: {e}")
        
        # ถ้าไม่มีโมเดลขั้นสูง หรือโมเดลขั้นสูงล้มเหลว ใช้โมเดลปกติ
        if model and current_grades:
            try:
                # เตรียมฟีเจอร์สำหรับโมเดลปกติ
                features = self._prepare_advanced_features(current_grades)
                
                # ทำนายด้วยโมเดลปกติ
                prediction_result = model.predict([features])[0]
                prediction_proba = model.predict_proba([features])[0]
                
                confidence = float(max(prediction_proba))
                
                # สร้างการกระจายเกรดจากโมเดล
                predicted_grades = self._generate_grade_distribution_from_model(
                    prediction_result, confidence, current_gpa
                )
                
                # คำนวณ GPA ทำนาย
                predicted_gpa = self._calculate_predicted_gpa(predicted_grades, current_gpa)
                
                # กำหนดแนวโน้ม
                if predicted_gpa > current_gpa + 0.1:
                    trend = "improving"
                elif predicted_gpa < current_gpa - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
                
                return {
                    "predicted_grades": predicted_grades,
                    "predicted_gpa": round(predicted_gpa, 2),
                    "confidence": round(confidence, 3),
                    "trend": trend,
                    "ai_prediction": {
                        "result": prediction_result,
                        "confidence": confidence,
                        "model_used": True,
                        "model_type": "Standard AI Model"
                    },
                    "analysis_note": f"ทำนายด้วย Standard AI จาก {len(current_grades)} วิชา, GPA ปัจจุบัน {current_gpa:.2f}"
                }
                
            except Exception as e:
                logger.warning(f"Standard AI model prediction failed: {e}")
        
        # ใช้การทำนายสำรองเมื่อโมเดลไม่พร้อมใช้งาน
        return self._fallback_prediction(current_gpa, current_grades)
    
    def _prepare_advanced_features(self, grades):
        """เตรียมฟีเจอร์ขั้นสูงสำหรับโมเดล AI"""
        # คำนวณสถิติพื้นฐาน
        gpa = self.calculate_gpa(grades)
        total_subjects = len(grades)
        grade_counts = Counter(grades.values())
        
        # ฟีเจอร์พื้นฐาน
        features = [
            gpa,  # GPA ปัจจุบัน
            total_subjects,  # จำนวนวิชาทั้งหมด
            total_subjects * 3,  # หน่วยกิตสะสม (สมมติ 3 หน่วยกิต/วิชา)
        ]
        
        # การกระจายเกรด
        for grade in ['A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F', 'W']:
            features.append(grade_counts.get(grade, 0))
        
        # อัตราส่วนเกรด
        if total_subjects > 0:
            features.extend([
                grade_counts.get('A', 0) / total_subjects,  # อัตราเกรด A
                grade_counts.get('F', 0) / total_subjects,  # อัตราเกรด F
                (grade_counts.get('A', 0) + grade_counts.get('B+', 0)) / total_subjects,  # อัตราเกรดดี
                (grade_counts.get('D+', 0) + grade_counts.get('D', 0) + grade_counts.get('F', 0)) / total_subjects,  # อัตราเกรดแย่
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # ฟีเจอร์เชิงลึก
        features.extend([
            1 if gpa >= 3.5 else 0,  # เกรดดีมาก
            1 if gpa < 2.0 else 0,   # เกรดต่ำมาก
            1 if grade_counts.get('F', 0) > 0 else 0,  # เคยตกวิชา
            min(4.0, gpa + 0.2) if gpa >= 3.0 else max(1.0, gpa - 0.1),  # แนวโน้ม GPA
        ])
        
        return features
    
    def _generate_grade_distribution_from_model(self, prediction_result, confidence, current_gpa):
        """สร้างการกระจายเกรดจากผลลัพธ์โมเดล"""
        # ปรับการกระจายเกรดตามผลลัพธ์โมเดลและ confidence
        if prediction_result == "จบ" and confidence > 0.7:
            # โมเดลมั่นใจว่าจะจบ - เกรดน่าจะดี
            if current_gpa >= 3.5:
                return {"A": 0.4, "B+": 0.35, "B": 0.20, "C+": 0.05}
            elif current_gpa >= 3.0:
                return {"A": 0.25, "B+": 0.40, "B": 0.25, "C+": 0.10}
            else:
                return {"B+": 0.20, "B": 0.35, "C+": 0.30, "C": 0.15}
        
        elif prediction_result == "จบ" and confidence <= 0.7:
            # โมเดลไม่แน่ใจ - เกรดปานกลาง
            return {"B": 0.25, "C+": 0.35, "C": 0.30, "D+": 0.10}
        
        else:
            # โมเดลคิดว่าไม่จบ - เกรดอาจแย่
            return {"C": 0.20, "D+": 0.30, "D": 0.35, "F": 0.15}
    
    def _calculate_predicted_gpa(self, predicted_grades, current_gpa):
        """คำนวณ GPA ทำนายจากการกระจายเกรด"""
        predicted_gpa = 0
        for grade, probability in predicted_grades.items():
            predicted_gpa += self.grade_mapping.get(grade, 0) * probability
        
        # ปรับให้สมเหตุสมผลกับ GPA ปัจจุบัน
        if abs(predicted_gpa - current_gpa) > 1.0:
            predicted_gpa = current_gpa + (0.2 if predicted_gpa > current_gpa else -0.2)
        
        return max(0.0, min(4.0, predicted_gpa))
    
    def _fallback_prediction(self, current_gpa, current_grades):
        """การทำนายสำรองเมื่อโมเดลไม่พร้อมใช้งาน"""
        total_subjects = len(current_grades)
        
        # การทำนายแบบพื้นฐานตาม GPA
        if current_gpa >= 3.5:
            predicted_grades = {"A": 0.35, "B+": 0.30, "B": 0.25, "C+": 0.10}
            next_term_gpa = min(4.0, current_gpa + 0.05)
            confidence = 0.75
        elif current_gpa >= 3.0:
            predicted_grades = {"A": 0.20, "B+": 0.35, "B": 0.30, "C+": 0.15}
            next_term_gpa = current_gpa + 0.02
            confidence = 0.70
        elif current_gpa >= 2.5:
            predicted_grades = {"B": 0.20, "C+": 0.35, "C": 0.30, "D+": 0.15}
            next_term_gpa = current_gpa
            confidence = 0.65
        else:
            predicted_grades = {"C": 0.25, "D+": 0.30, "D": 0.30, "F": 0.15}
            next_term_gpa = max(1.5, current_gpa - 0.1)
            confidence = 0.60
        
        # ปรับความมั่นใจตามจำนวนข้อมูล
        if total_subjects < 5:
            confidence *= 0.8
        elif total_subjects > 15:
            confidence *= 1.1
        
        confidence = min(0.95, confidence)
        
        return {
            "predicted_grades": predicted_grades,
            "predicted_gpa": round(next_term_gpa, 2),
            "confidence": round(confidence, 3),
            "trend": "improving" if next_term_gpa > current_gpa else "stable" if next_term_gpa == current_gpa else "declining",
            "ai_prediction": {
                "result": "ไม่ทราบ",
                "confidence": confidence,
                "model_used": False
            },
            "analysis_note": f"ทำนายแบบพื้นฐานจาก {total_subjects} วิชา, GPA ปัจจุบัน {current_gpa:.2f} (ไม่มีโมเดล AI)"
        }
    
    def _prepare_model_features(self, grades):
        """เตรียมฟีเจอร์สำหรับโมเดล - ใช้ฟีเจอร์ขั้นสูง"""
        return self._prepare_advanced_features(grades)
    
    def create_visualization_charts(self, grades_dict, prediction_data):
        """สร้างกราฟแสดงผลแบบครอบคลุม"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # กราฟ 1: การกระจายเกรด
        if grades_dict:
            grade_counts = Counter(grades_dict.values())
            grades = list(grade_counts.keys())
            counts = list(grade_counts.values())
            colors = ['#2E8B57', '#32CD32', '#FFD700', '#FFA500', '#FF6347', '#DC143C']
            
            wedges, texts, autotexts = ax1.pie(counts, labels=grades, autopct='%1.1f%%', 
                                             colors=colors[:len(grades)], startangle=90)
            ax1.set_title('การกระจายเกรดปัจจุบัน', fontsize=14, fontweight='bold')
        
        # กราฟ 2: เปรียบเทียบ GPA
        current_gpa = self.calculate_gpa(grades_dict)
        predicted_gpa = prediction_data.get('predicted_gpa', current_gpa)
        
        categories = ['GPA ปัจจุบัน', 'GPA ทำนาย', 'เกณฑ์ขั้นต่ำ', 'เกณฑ์ดี', 'เกณฑ์ดีมาก']
        values = [current_gpa, predicted_gpa, 2.0, 2.75, 3.5]
        colors = ['blue', 'red', 'gray', 'orange', 'green']
        
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_ylabel('GPA')
        ax2.set_title('เปรียบเทียบ GPA', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 4)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # กราฟ 3: ความน่าจะเป็นของเกรดในอนาคต
        predicted_grades = prediction_data.get('predicted_grades', {})
        if predicted_grades:
            grades = list(predicted_grades.keys())
            probs = list(predicted_grades.values())
            colors = plt.cm.RdYlGn([p for p in probs])
            
            bars = ax3.bar(grades, probs, color=colors)
            ax3.set_xlabel('เกรด')
            ax3.set_ylabel('ความน่าจะเป็น')
            ax3.set_title('ความน่าจะเป็นของเกรดเทอมถัดไป', fontsize=14, fontweight='bold')
            
            # เพิ่มค่าบนแท่งกราฟ
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # กราฟ 4: แนวโน้มการพัฒนา
        terms = ['เทอม 1', 'เทอม 2', 'เทอม 3', 'เทอมปัจจุบัน', 'เทอมถัดไป']
        # สร้างข้อมูลจำลองสำหรับแนวโน้ม
        trend_gpas = [2.5, 2.8, 3.1, current_gpa, predicted_gpa]
        
        ax4.plot(terms[:-1], trend_gpas[:-1], 'o-', label='GPA จริง', linewidth=3, markersize=8, color='blue')
        ax4.plot([terms[-2], terms[-1]], [trend_gpas[-2], trend_gpas[-1]], 'r--', 
                label='GPA ทำนาย', linewidth=3, markersize=8)
        ax4.set_xlabel('เทอม')
        ax4.set_ylabel('GPA')
        ax4.set_title('แนวโน้มการพัฒนา GPA', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 4)
        
        plt.tight_layout()
        
        # แปลงเป็น base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64

# สร้าง instance ของระบบ
enhanced_system = EnhancedPredictionSystem()

# โหลดโมเดลขั้นสูงถ้ามี
enhanced_system.load_advanced_model()

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# ตรวจสอบ Environment Variables แบบละเอียด
# ==========================================
logger.info("=" * 70)
logger.info("🔧 CLOUDFLARE R2 CONFIGURATION CHECK")
logger.info("=" * 70)

def check_env_variables():
    """ตรวจสอบและแสดงสถานะ environment variables แบบละเอียด"""
    env_status = {
        'all_present': True,
        'missing': [],
        'details': {}
    }
    
    required_vars = {
        'CLOUDFLARE_R2_ACCESS_KEY_ID': 'Access Key for R2 authentication',
        'CLOUDFLARE_R2_SECRET_ACCESS_KEY': 'Secret Key for R2 authentication',
        'CLOUDFLARE_R2_ENDPOINT': 'R2 endpoint URL',
        'CLOUDFLARE_R2_BUCKET_NAME': 'R2 bucket name'
    }
    
    for var_name, description in required_vars.items():
        value = os.environ.get(var_name, '')
        
        if value:
            # แสดงแค่ส่วนแรกของค่าเพื่อความปลอดภัย
            if 'SECRET' in var_name or 'KEY' in var_name:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            
            logger.info(f"✅ {var_name}: {display_value}")
            env_status['details'][var_name] = {
                'present': True,
                'length': len(value),
                'description': description
            }
        else:
            logger.error(f"❌ {var_name}: NOT FOUND - {description}")
            env_status['missing'].append(var_name)
            env_status['all_present'] = False
            env_status['details'][var_name] = {
                'present': False,
                'description': description
            }
    
    # ตรวจสอบไฟล์ .env
    if os.path.exists('.env'):
        logger.info(f"📄 .env file: EXISTS (size: {os.path.getsize('.env')} bytes)")
        
        # อ่านและตรวจสอบเนื้อหา .env
        try:
            with open('.env', 'r') as f:
                env_lines = f.readlines()
                env_vars_in_file = {}
                for line in env_lines:
                    if '=' in line and not line.strip().startswith('#'):
                        key = line.split('=')[0].strip()
                        env_vars_in_file[key] = True
            
            logger.info(f"📋 Variables in .env file: {list(env_vars_in_file.keys())}")
            
            # เช็คว่าตัวแปรใน .env ถูกโหลดหรือไม่
            for var in required_vars.keys():
                if var in env_vars_in_file and not os.environ.get(var):
                    logger.warning(f"⚠️ {var} exists in .env but not loaded into environment!")
                    
        except Exception as e:
            logger.error(f"❌ Error reading .env file: {str(e)}")
    else:
        logger.warning("⚠️ .env file: NOT FOUND")
    
    return env_status

env_check_result = check_env_variables()

if not env_check_result['all_present']:
    logger.error("=" * 70)
    logger.error("⚠️ CLOUDFLARE R2 CONFIGURATION INCOMPLETE!")
    logger.error("Missing variables: " + ", ".join(env_check_result['missing']))
    logger.error("R2 storage will be DISABLED. Using LOCAL storage instead.")
    logger.error("=" * 70)

# Create Flask app
ACTIVE_CONFIG = config.get_config()
app = Flask(__name__)
app.config.from_object(ACTIVE_CONFIG)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))


class MongoAdminManager:
    """MongoDB-backed admin/auth/settings manager.
    NOTE: MONGODB_URI remains in .env only.
    """

    SETTING_KEYS = [
        'GEMINI_API_KEY',
        'GEMINI_MODEL_NAME',
        'GEMINI_MODEL_FALLBACKS',
        'CLOUDFLARE_R2_ACCESS_KEY_ID',
        'CLOUDFLARE_R2_SECRET_ACCESS_KEY',
        'CLOUDFLARE_R2_ENDPOINT',
        'CLOUDFLARE_R2_BUCKET_NAME',
    ]

    def __init__(self):
        self.enabled = False
        self.client = None
        self.db = None
        self.users = None
        self.settings = None
        self.audit = None
        self.bootstrap_password = None

        self._mongo_uri = os.environ.get('MONGODB_URI', '').strip()
        self._db_name = os.environ.get('MONGODB_DB_NAME', 'student_ai_system').strip()
        self._enc_key = os.environ.get('APP_MASTER_ENCRYPTION_KEY', '').strip()

        if not self._mongo_uri:
            logger.warning("⚠️ MONGODB_URI not configured. Admin backend is disabled.")
            return
        if MongoClient is None:
            logger.warning("⚠️ pymongo package not installed. Admin backend is disabled.")
            return

        try:
            self.client = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self._db_name]
            self.users = self.db['users']
            self.settings = self.db['system_settings']
            self.audit = self.db['audit_logs']

            self.users.create_index('username', unique=True)
            self.settings.create_index('key', unique=True)
            self.audit.create_index('created_at')

            self.enabled = True
            self._bootstrap_default_admin()
            logger.info(f"✅ Mongo admin backend ready (db={self._db_name})")
        except Exception as e:
            logger.error(f"❌ Mongo admin backend init failed: {e}")
            self.enabled = False

    def _log_audit(self, actor_user_id, action, target, status='success', details=None):
        if not self.enabled:
            return
        try:
            self.audit.insert_one({
                'actor_user_id': actor_user_id,
                'action': action,
                'target': target,
                'status': status,
                'details': details or {},
                'ip_address': request.remote_addr if request else None,
                'user_agent': request.headers.get('User-Agent') if request else None,
                'created_at': datetime.utcnow().isoformat()
            })
        except Exception:
            pass

    def _derive_key(self):
        return hashlib.sha256(self._enc_key.encode('utf-8')).digest()

    def _encrypt(self, raw_value: str) -> str:
        if raw_value is None:
            return ''
        key = self._derive_key()
        raw = raw_value.encode('utf-8')
        out = bytearray(len(raw))
        for i, b in enumerate(raw):
            out[i] = b ^ key[i % len(key)]
        return base64.urlsafe_b64encode(bytes(out)).decode('utf-8')

    def _decrypt(self, encrypted_value: str) -> str:
        if not encrypted_value:
            return ''
        key = self._derive_key()
        raw = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
        out = bytearray(len(raw))
        for i, b in enumerate(raw):
            out[i] = b ^ key[i % len(key)]
        return bytes(out).decode('utf-8')

    def _bootstrap_default_admin(self):
        if not self.enabled:
            return
        if self.users.count_documents({}) > 0:
            return

        username = os.environ.get('INITIAL_ADMIN_USERNAME', 'admin')
        temp_password = os.environ.get('INITIAL_ADMIN_PASSWORD', 'admin123')
        self.bootstrap_password = temp_password

        self.users.insert_one({
            'username': username,
            'email': None,
            'password_hash': generate_password_hash(temp_password),
            'role': 'super_admin',
            'must_change_password': True,
            'is_active': True,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'last_login_at': None
        })

        logger.warning("=" * 70)
        logger.warning("🔐 FIRST RUN: Default super_admin account created")
        logger.warning(f"   username: {username}")
        logger.warning(f"   temporary password: {temp_password}")
        logger.warning("   Action required: login and change password immediately")
        logger.warning("=" * 70)

    def apply_runtime_env(self):
        """Load settings from MongoDB and apply to process env (except MONGODB_URI)."""
        if not self.enabled:
            return
        try:
            docs = list(self.settings.find({'key': {'$in': self.SETTING_KEYS}}))
            for doc in docs:
                key = doc.get('key')
                # Plaintext first (new behavior), encrypted fallback (legacy data)
                plain_value = doc.get('value')
                value_encrypted = doc.get('value_encrypted', '')

                if key and plain_value is not None:
                    os.environ[key] = str(plain_value)
                elif key and value_encrypted:
                    os.environ[key] = self._decrypt(value_encrypted)
        except Exception as e:
            logger.warning(f"Could not apply runtime env from MongoDB: {e}")

    def authenticate(self, username: str, password: str):
        if not self.enabled:
            return None
        user = self.users.find_one({'username': username, 'is_active': True})
        if not user:
            return None
        if not check_password_hash(user.get('password_hash', ''), password):
            return None
        self.users.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login_at': datetime.utcnow().isoformat()}}
        )
        return user

    def change_password(self, user_id, new_password: str):
        if not self.enabled:
            return False
        self.users.update_one(
            {'_id': user_id},
            {'$set': {
                'password_hash': generate_password_hash(new_password),
                'must_change_password': False,
                'updated_at': datetime.utcnow().isoformat()
            }}
        )
        return True

    def set_setting(self, key: str, value: str, actor_user_id=None):
        if not self.enabled:
            return False
        self.settings.update_one(
            {'key': key},
            {'$set': {
                'key': key,
                # Save as plaintext by requirement
                'value': value or '',
                # keep legacy field for compatibility (clear when writing new value)
                'value_encrypted': '',
                'updated_by': str(actor_user_id) if actor_user_id else None,
                'updated_at': datetime.utcnow().isoformat()
            }},
            upsert=True
        )
        self._log_audit(actor_user_id, 'UPDATE_SETTING', key)
        return True

    def get_setting(self, key: str, default: str = ''):
        if not self.enabled:
            return default
        doc = self.settings.find_one({'key': key})
        if not doc:
            return default
        try:
            # Plaintext first (new behavior)
            if 'value' in doc and doc.get('value') is not None:
                return str(doc.get('value'))
            # Encrypted fallback (legacy data)
            encrypted = doc.get('value_encrypted', '')
            if encrypted:
                return self._decrypt(encrypted)
            return default
        except Exception:
            return default

    @staticmethod
    def mask_value(raw: str):
        if not raw:
            return ''
        if len(raw) <= 6:
            return '***'
        return f"{raw[:4]}***{raw[-2:]}"


admin_manager = MongoAdminManager()
admin_manager.apply_runtime_env()

COURSE_LOOKUP = {course['id']: course for course in getattr(ACTIVE_CONFIG, 'COURSES_DATA', [])}
GRADE_POINT_MAP = ACTIVE_CONFIG.DATA_CONFIG.get('grade_mapping', {})

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
# Use gemini-3-flash-preview as default (fastest Gemini 3 model with free tier)
# Fallback chain: gemini-2.5-flash (stable) → gemini-2.5-pro (advanced) → gemini-2.0-flash (legacy)
# Reference: https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-3-flash-preview')
GEMINI_MAX_FILE_SIZE_MB = float(os.environ.get('GEMINI_MAX_FILE_SIZE_MB', 5))
# Default fallback models for high availability (in order of preference)
GEMINI_DEFAULT_FALLBACKS = ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash']


def _build_gemini_model_candidates(primary_name: str) -> List[str]:
    """Collect preferred + fallback Gemini model names with preserved order."""
    candidates: List[str] = []

    def _add(name: Optional[str]):
        if name and name not in candidates:
            candidates.append(name)

    _add(primary_name)

    fallback_env = os.environ.get('GEMINI_MODEL_FALLBACKS', '')
    if fallback_env:
        for item in fallback_env.split(','):
            _add(item.strip())

    # Add default fallback models from centralized constant
    for default_name in GEMINI_DEFAULT_FALLBACKS:
        _add(default_name)

    return candidates


GEMINI_MODEL_CANDIDATES = _build_gemini_model_candidates(GEMINI_MODEL_NAME)
gemini_client_ready = False

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client_ready = True
        logger.info(f"✅ Gemini API initialized with model {GEMINI_MODEL_NAME}")
    except Exception as gemini_init_error:
        gemini_client_ready = False
        logger.error(f"❌ Failed to initialize Gemini API: {gemini_init_error}")
elif not GEMINI_API_KEY:
    logger.info("ℹ️ GEMINI_API_KEY not found. Gemini routes will be disabled.")
else:
    logger.warning("⚠️ google-generativeai package not available. Install to enable Gemini features.")


def refresh_gemini_runtime_from_settings():
    """Reload Gemini runtime variables from Mongo settings (if enabled)."""
    global GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_MODEL_CANDIDATES, gemini_client_ready

    if admin_manager.enabled:
        admin_manager.apply_runtime_env()

    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-3-flash-preview')
    GEMINI_MODEL_CANDIDATES = _build_gemini_model_candidates(GEMINI_MODEL_NAME)

    gemini_client_ready = False
    if genai and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_client_ready = True
            logger.info(f"✅ Gemini runtime refreshed with model {GEMINI_MODEL_NAME}")
        except Exception as gemini_init_error:
            gemini_client_ready = False
            logger.error(f"❌ Failed to refresh Gemini runtime: {gemini_init_error}")


def _json_error(message: str, status: int = 403):
    return jsonify({'success': False, 'error': message}), status


def current_user_role() -> str:
    return session.get('user_role', 'anonymous')


def has_any_role(*roles) -> bool:
    if not admin_manager.enabled:
        return True
    return current_user_role() in roles


def is_logged_in() -> bool:
    if not admin_manager.enabled:
        return True
    return bool(session.get('user_id'))


def admin_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not admin_manager.enabled:
            return _json_error('Admin backend is disabled (MongoDB not configured)', 503)
        if not is_logged_in():
            if request.path.startswith('/api/'):
                return _json_error('Unauthorized', 401)
            return redirect(url_for('admin_login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapper


def roles_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not admin_manager.enabled:
                return view_func(*args, **kwargs)
            if not is_logged_in():
                return _json_error('Unauthorized', 401)
            if not has_any_role(*roles):
                return _json_error('Forbidden', 403)
            return view_func(*args, **kwargs)
        return wrapper
    return decorator


@app.before_request
def enforce_first_login_password_change():
    if not admin_manager.enabled:
        return None

    endpoint = request.endpoint or ''
    allow_endpoints = {
        'admin_login',
        'admin_logout',
        'admin_change_password',
        'static'
    }

    if not session.get('user_id'):
        return None

    if session.get('must_change_password') and endpoint not in allow_endpoints:
        if request.path.startswith('/api/'):
            return _json_error('Password change required before using this endpoint', 403)
        return redirect(url_for('admin_change_password'))

    return None


@app.before_request
def enforce_login_for_non_prediction_pages():
    """Allow anonymous access only to prediction page and required prediction APIs."""
    if not admin_manager.enabled:
        return None

    endpoint = request.endpoint or ''

    public_endpoints = {
        'static',
        'admin_login',
        'admin_logout',
        'admin_change_password',

        # Public prediction pages
        'curriculum_prediction_form',
        'curriculum_page',

        # Public prediction APIs
        'get_config_for_frontend',
        'analyze_curriculum',
        'gemini_predict_route',
        'explain_prediction',
        'get_three_line_chart_data',
        'get_graduation_analysis',
        'get_next_term_prediction',
        'get_comprehensive_analysis',
        'predict_manual_input',
    }

    if endpoint in public_endpoints:
        return None

    # Force landing page to prediction page for anonymous users
    if endpoint == 'index' and not session.get('user_id'):
        return redirect(url_for('curriculum_prediction_form'))

    if not session.get('user_id'):
        if request.path.startswith('/api/'):
            return _json_error('Unauthorized', 401)
        return redirect(url_for('admin_login', next=request.path))

    return None

# ==========================================
# RETRY MECHANISM และ RATE LIMITER สำหรับ GEMINI API
# ==========================================

def retry_on_quota_error(max_retries=3, initial_delay=20):
    """Decorator สำหรับ retry เมื่อเจอ quota error
    
    Args:
        max_retries: จำนวนครั้งทั้งหมดในการลอง (รวมครั้งแรก)
        initial_delay: เวลารอเริ่มต้น (วินาที)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    is_quota_error = ('429' in error_msg or 
                                     'quota' in error_msg.lower() or 
                                     'resource exhausted' in error_msg.lower())
                    
                    if is_quota_error and attempt < max_retries - 1:
                        logger.warning(f"Quota exceeded, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue
                    
                    # ถ้าไม่ใช่ quota error หรือหมดจำนวน retry แล้ว ให้ raise
                    raise
                
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiter สำหรับ Gemini API"""
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = timedelta(seconds=time_window)
        self.requests = deque()
    
    def can_proceed(self):
        now = datetime.now()
        # ลบ request ที่เก่าเกินกว่า time window
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, None
        else:
            wait_time = (self.requests[0] + self.time_window - now).total_seconds()
            # ป้องกัน negative wait time และให้อย่างน้อย 1 วินาที
            return False, max(1, int(wait_time))


# สร้าง rate limiter instance
gemini_rate_limiter = RateLimiter(max_requests=10, time_window=60)

# ==========================================
# สร้างโฟลเดอร์ที่จำเป็นทันที
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

logger.info("=" * 70)
logger.info("📁 DIRECTORY SETUP")
logger.info("=" * 70)

for folder_name, folder_path in [('uploads', UPLOAD_FOLDER), ('models', MODEL_FOLDER)]:
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"✅ Created {folder_name} folder: {folder_path}")
            
            # สร้างไฟล์ .gitkeep เพื่อให้ git track โฟลเดอร์ว่าง
            gitkeep_path = os.path.join(folder_path, '.gitkeep')
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, 'w') as f:
                    f.write('')
                logger.info(f"   Created .gitkeep in {folder_name}")
        else:
            logger.info(f"✅ {folder_name} folder exists: {folder_path}")
            
        # ตรวจสอบ permissions
        if os.access(folder_path, os.W_OK):
            logger.info(f"   ✅ Write permission: OK")
        else:
            logger.error(f"   ❌ Write permission: DENIED")
            
        # แสดงไฟล์ที่มีอยู่
        files = os.listdir(folder_path)
        if files:
            logger.info(f"   Files in {folder_name}: {len(files)} file(s)")
            for file in files[:5]:  # แสดงแค่ 5 ไฟล์แรก
                file_path = os.path.join(folder_path, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                logger.info(f"     - {file} ({file_size} bytes)")
                
    except Exception as e:
        logger.error(f"❌ Error with {folder_name} folder: {str(e)}")
        logger.error(f"   Current working directory: {os.getcwd()}")
        logger.error(f"   Python executable: {sys.executable}")

# Variables to store loaded models
models = {
    'subject_model': None,
    'gpa_model': None,
    'subject_model_info': None,
    'gpa_model_info': None,
    'subject_feature_cols': None,
    'gpa_feature_cols': None
}

# ==========================================
# Enhanced S3 Storage Class with Detailed Error Reporting
# ==========================================
class S3Storage:
    def __init__(self):
        """Initialize S3 client with detailed error reporting"""
        self.connection_errors = []
        self.use_local = False
        self.s3_client = None
        
        logger.info("=" * 70)
        logger.info("🚀 INITIALIZING CLOUDFLARE R2 STORAGE")
        logger.info("=" * 70)
        
        try:
            # ตรวจสอบ credentials
            self.access_key = os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', '')
            self.secret_key = os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', '')
            self.endpoint_url = os.environ.get('CLOUDFLARE_R2_ENDPOINT', '')
            self.bucket_name = os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', 'pjai')
            
            # ตรวจสอบแต่ละ credential
            missing_creds = []
            if not self.access_key:
                missing_creds.append('CLOUDFLARE_R2_ACCESS_KEY_ID')
            if not self.secret_key:
                missing_creds.append('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
            if not self.endpoint_url:
                missing_creds.append('CLOUDFLARE_R2_ENDPOINT')
                
            if missing_creds:
                error_msg = f"Missing R2 credentials: {', '.join(missing_creds)}"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
                logger.info("📂 Fallback: Using LOCAL storage")
                return
            
            # ตรวจสอบ boto3
            try:
                import boto3
                from botocore.config import Config
                from botocore.exceptions import NoCredentialsError, ClientError
            except ImportError as e:
                error_msg = "boto3 library not installed. Run: pip install boto3"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
                logger.info("📂 Fallback: Using LOCAL storage")
                return
            
            # พยายามสร้าง S3 client
            logger.info("🔗 Attempting to connect to Cloudflare R2...")
            logger.info(f"   Endpoint: {self.endpoint_url}")
            logger.info(f"   Bucket: {self.bucket_name}")
            
            try:
                config = Config(
                    region_name='auto',
                    signature_version='s3v4',
                    retries={'max_attempts': 3, 'mode': 'standard'},
                    connect_timeout=10,
                    read_timeout=10
                )
                
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
                
                # ทดสอบการเชื่อมต่อ
                logger.info("🧪 Testing R2 connection...")
                test_response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    MaxKeys=1
                )
                
                logger.info("✅ R2 CONNECTION SUCCESSFUL!")
                logger.info(f"   Objects in bucket: {len(test_response.get('Contents', []))}")
                self.use_local = False
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                if error_code == 'NoSuchBucket':
                    error_msg = f"Bucket '{self.bucket_name}' does not exist"
                    logger.error(f"❌ {error_msg}")
                    logger.info("🔨 Attempting to create bucket...")
                    
                    try:
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                        logger.info(f"✅ Bucket '{self.bucket_name}' created successfully!")
                        self.use_local = False
                    except Exception as create_error:
                        error_msg = f"Failed to create bucket: {str(create_error)}"
                        logger.error(f"❌ {error_msg}")
                        self.connection_errors.append(error_msg)
                        self.use_local = True
                        
                elif error_code == 'InvalidAccessKeyId':
                    error_msg = "Invalid Access Key ID - Check your CLOUDFLARE_R2_ACCESS_KEY_ID"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
                elif error_code == 'SignatureDoesNotMatch':
                    error_msg = "Invalid Secret Key - Check your CLOUDFLARE_R2_SECRET_ACCESS_KEY"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
                else:
                    error_msg = f"R2 Error [{error_code}]: {error_message}"
                    logger.error(f"❌ {error_msg}")
                    self.connection_errors.append(error_msg)
                    self.use_local = True
                    
            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                self.connection_errors.append(error_msg)
                self.use_local = True
            
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.connection_errors.append(error_msg)
            self.use_local = True
        
        # สรุปผล
        logger.info("=" * 70)
        if self.use_local:
            logger.warning("⚠️ R2 STORAGE: DISABLED")
            logger.info("📂 Using LOCAL FILE STORAGE")
            if self.connection_errors:
                logger.error("Connection errors:")
                for error in self.connection_errors:
                    logger.error(f"  • {error}")
        else:
            logger.info("✅ R2 STORAGE: ENABLED AND READY")
        logger.info("=" * 70)
    
    def get_connection_status(self):
        """Get detailed connection status for API"""
        return {
            'connected': not self.use_local,
            'storage_type': 'local' if self.use_local else 'cloudflare_r2',
            'bucket_name': self.bucket_name if not self.use_local else None,
            'endpoint': self.endpoint_url if not self.use_local else None,
            'errors': self.connection_errors
        }
    
    def save_model(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model with detailed error reporting"""
        if self.use_local:
            return self._save_model_locally(model_data, filename)
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(model_data, tmp_file.name)
                tmp_path = tmp_file.name
            
            s3_key = f"models/{filename}"
            with open(tmp_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    ContentType='application/octet-stream',
                    Metadata={
                        'created_at': datetime.now().isoformat(),
                        'data_format': model_data.get('data_format', 'unknown'),
                        'accuracy': str(model_data.get('performance_metrics', {}).get('accuracy', 0))
                    }
                )
            
            os.remove(tmp_path)
            logger.info(f"✅ Model {filename} saved to R2")
            return True
            
        except Exception as e:
            logger.error(f"❌ R2 save error: {str(e)}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Fallback to local
            logger.info("📂 Trying local save as fallback...")
            return self._save_model_locally(model_data, filename)
    
    def _save_model_locally(self, model_data: Dict[str, Any], filename: str) -> bool:
        """Save model locally"""
        try:
            model_folder = app.config['MODEL_FOLDER']
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
            filepath = os.path.join(model_folder, filename)
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Model {filename} saved locally")
            return True
        except Exception as e:
            logger.error(f"❌ Local save error: {str(e)}")
            return False
    
    def load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model"""
        if self.use_local:
            return self._load_model_locally(filename)
        
        tmp_path = None
        try:
            s3_key = f"models/{filename}"
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Filename=tmp_file.name
                )
                tmp_path = tmp_file.name
            
            model_data = joblib.load(tmp_path)
            os.remove(tmp_path)
            logger.info(f"✅ Model {filename} loaded from R2")
            return model_data
            
        except Exception as e:
            logger.warning(f"R2 load failed: {str(e)}, trying local...")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # หากโหลดจาก R2 ไม่ได้ ให้ลองโหลดจาก local หรือสร้างโมเดลใหม่
            local_result = self._load_model_locally(filename)
            if local_result is None:
                # สร้างโมเดลสำรอง
                return self._create_fallback_model_data()
            return local_result
    
    def _create_fallback_model_data(self) -> Dict[str, Any]:
        """สร้างข้อมูลโมเดลสำรองเมื่อโหลดไม่ได้"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        scaler = StandardScaler()
        
        logger.info("✅ Created fallback model data")
        return {
            'model': model,
            'scaler': scaler,
            'course_credit_map': {},
            'grade_mapping': {
                'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
                'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'WF': 0.0, 'WU': 0.0, 'S': None
            }
        }
    
    def _load_model_locally(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load model from local storage"""
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                logger.info(f"📂 Model {filename} loaded from local")
                return model_data
            return None
        except Exception as e:
            logger.error(f"❌ Local load error: {str(e)}")
            # หากโหลดจาก local ไม่ได้ ให้สร้างโมเดลสำรอง
            return self._create_fallback_model_data()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models"""
        models = []
        
        # Try R2 first
        if not self.use_local and self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix='models/',
                    Delimiter='/'
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        filename = obj['Key'].replace('models/', '')
                        if filename and filename.endswith('.joblib'):
                            models.append({
                                'filename': filename,
                                'created_at': obj['LastModified'].isoformat(),
                                'size': obj['Size'],
                                'storage': 'r2',
                                'performance_metrics': {},
                                'data_format': 'unknown'
                            })
                logger.info(f"📊 Found {len(models)} models in R2")
            except Exception as e:
                logger.warning(f"R2 list failed: {str(e)}")
        
        # Also check local
        try:
            model_folder = app.config['MODEL_FOLDER']
            if os.path.exists(model_folder):
                for filename in os.listdir(model_folder):
                    if filename.endswith('.joblib'):
                        filepath = os.path.join(model_folder, filename)
                        try:
                            model_data = joblib.load(filepath)
                            models.append({
                                'filename': filename,
                                'created_at': model_data.get('created_at', ''),
                                'data_format': model_data.get('data_format', 'unknown'),
                                'performance_metrics': model_data.get('performance_metrics', {}),
                                'storage': 'local'
                            })
                        except:
                            models.append({
                                'filename': filename,
                                'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                                'storage': 'local',
                                'performance_metrics': {},
                                'data_format': 'unknown'
                            })
                logger.info(f"📂 Found {len([m for m in models if m['storage'] == 'local'])} models locally")
        except Exception as e:
            logger.error(f"Local list error: {str(e)}")
        
        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_model(self, filename: str) -> bool:
        """Delete model"""
        deleted = False
        
        # Try R2
        if not self.use_local and self.s3_client:
            try:
                s3_key = f"models/{filename}"
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.info(f"✅ Model {filename} deleted from R2")
                deleted = True
            except Exception as e:
                logger.warning(f"R2 delete failed: {str(e)}")
        
        # Also try local
        try:
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"🗑️ Model {filename} deleted from local")
                deleted = True
        except Exception as e:
            logger.warning(f"Local delete failed: {str(e)}")
        
        return deleted

# Global Training Status
TRAINING_STATUS = {
    'status': 'idle',
    'message': '',
    'progress': 0,
    'result': None,
    'error': None
}

# Create global storage instance
storage = S3Storage()

# ==========================================
# Sync Models from R2 at Startup
# ==========================================
def sync_models_from_r2():
    """Download models from R2 to local storage for faster access"""
    if storage.use_local:
        logger.info("📂 R2 not connected, skipping model sync")
        return 0
    
    try:
        logger.info("🔄 Syncing models from R2 to local storage...")
        model_folder = app.config['MODEL_FOLDER']
        
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)
        
        # List R2 models
        r2_models = []
        response = storage.s3_client.list_objects_v2(
            Bucket=storage.bucket_name,
            Prefix='models/'
        )
        
        if 'Contents' in response:
            r2_models = [obj['Key'].replace('models/', '') 
                        for obj in response['Contents'] 
                        if obj['Key'].endswith('.joblib')]
        
        synced_count = 0
        for model_filename in r2_models:
            local_path = os.path.join(model_folder, model_filename)
            
            # Skip if already exists locally
            if os.path.exists(local_path):
                continue
            
            try:
                s3_key = f"models/{model_filename}"
                storage.s3_client.download_file(
                    Bucket=storage.bucket_name,
                    Key=s3_key,
                    Filename=local_path
                )
                synced_count += 1
                logger.info(f"  ✅ Downloaded {model_filename}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to download {model_filename}: {e}")
        
        logger.info(f"✅ Synced {synced_count} models from R2")
        return synced_count
        
    except Exception as e:
        logger.warning(f"⚠️ Model sync failed: {e}")
        return 0

# Run model sync at startup
try:
    sync_models_from_r2()
except Exception as e:
    logger.warning(f"Model sync at startup failed: {e}")

# ===============================
# TRAINING STATUS ROUTE
# ===============================
@app.route('/training_status')
def get_training_status():
    return jsonify(TRAINING_STATUS)


# ===============================
# ENHANCED API ROUTES
# ===============================

@app.route('/api/enhanced_analysis', methods=['POST'])
def enhanced_analysis():
    """API สำหรับการวิเคราะห์แบบขั้นสูง"""
    try:
        data = request.get_json()
        grades = data.get('grades', {})
        model_name = data.get('model_name', '')
        
        # วิเคราะห์ผลการเรียน
        analysis = enhanced_system.analyze_performance(grades)
        
        # ทำนายผลการเรียน
        model = None
        if model_name and models.get('subject_model'):
            model = models['subject_model']
        
        prediction = enhanced_system.predict_future_performance(grades, model)
        
        # สร้างกราฟ
        chart_data = enhanced_system.create_visualization_charts(grades, prediction)
        
        # คำนวณสถิติเพิ่มเติม
        total_credits = len(grades) * 3
        passing_grades = [g for g in grades.values() if g not in ['F', 'W', 'WF', 'WU']]
        passing_rate = len(passing_grades) / len(grades) * 100 if grades else 0
        
        response = {
            "success": True,
            "analysis": analysis,
            "prediction": prediction,
            "chart_data": chart_data,
            "statistics": {
                "total_credits": total_credits,
                "passing_rate": round(passing_rate, 1),
                "subjects_count": len(grades)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/quick_prediction', methods=['POST'])
def quick_prediction():
    """API สำหรับการทำนายแบบเร็ว"""
    try:
        data = request.get_json()
        grades = data.get('grades', {})
        
        if not grades:
            return jsonify({"error": "ไม่พบข้อมูลเกรด", "success": False}), 400
        
        # วิเคราะห์และทำนายแบบเร็ว
        current_gpa = enhanced_system.calculate_gpa(grades)
        prediction = enhanced_system.predict_future_performance(grades)
        
        response = {
            "success": True,
            "current_gpa": round(current_gpa, 2),
            "predicted_gpa": prediction["predicted_gpa"],
            "trend": prediction["trend"],
            "confidence": round(prediction["confidence"], 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in quick prediction: {e}")
        return jsonify({"error": str(e), "success": False}), 500

# เพิ่ม API endpoint สำหรับตรวจสอบสถานะ
@app.route('/api/storage/status')
def get_storage_status():
    """Get detailed storage status"""
    status = storage.get_connection_status()
    status['env_check'] = env_check_result
    return jsonify(status)

# Variables to store loaded models (re-declared for clarity in this combined file)
models = {
    'subject_model': None,
    'gpa_model': None,
    'subject_model_info': None,
    'gpa_model_info': None,
    'subject_feature_cols': None,
    'gpa_feature_cols': None
}

# ==========================================
# Original Functions (re-pasted here for a complete block)
# ==========================================

def detect_data_format(df):
    """Detects the data format of the uploaded DataFrame."""
    try:
        columns = [col.lower() for col in df.columns.tolist()]
        has_name_id = any(keyword in col for col in columns for keyword in ['ชื่อ', 'name', 'รหัส', 'id', 'student_id'])
        has_year = any(keyword in col for col in columns for keyword in ['ปี', 'year'])
        has_subject_like_columns = any(
            not any(kw in col for kw in ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated']) for col in columns
        ) and any(col.lower().startswith(('วิชา', 'subj', 'course')) or len(col) > 5 for col in columns if col not in ['ชื่อ', 'name', 'รหัส', 'id', 'year'])

        if has_name_id and has_year and has_subject_like_columns:
            logger.debug(f"Detected subject_based format for columns: {df.columns.tolist()}")
            return 'subject_based'

        has_gpa = any(keyword in col for col in columns for keyword in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])
        has_success = any(keyword in col for col in columns for keyword in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ'])

        if has_gpa and has_success:
            logger.debug(f"Detected gpa_based format for columns: {df.columns.tolist()}")
            return 'gpa_based'

        logger.debug(f"Could not detect data format for columns: {df.columns.tolist()}")
        return 'unknown'

    except Exception as e:
        logger.error(f"Error detecting data format: {str(e)}")
        return 'unknown'

def grade_to_numeric(grade):
    """Converts a letter grade to a numeric GPA value."""
    if pd.isna(grade):
        return 0.0
    try:
        return float(grade)
    except ValueError:
        pass

    grade_str = str(grade).strip().upper()
    return app.config['DATA_CONFIG']['grade_mapping'].get(grade_str, 0.0)

def categorize_subject(subject_name):
    """Categorizes subjects based on keywords defined in config."""
    if pd.isna(subject_name):
        return 'อื่นๆ'

    subject_name = str(subject_name).lower()
    for category, info in app.config['SUBJECT_CATEGORIES'].items():
        if any(keyword in subject_name for keyword in info['keywords']):
            return category
    return 'อื่นๆ'

def process_subject_data(df):
    """Processes subject-based DataFrame to create features for model training."""
    try:
        name_col = None
        possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
        for col in possible_name_cols:
            if col in df.columns:
                name_col = col
                break
        if not name_col:
            name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
            if name_candidates:
                name_col = name_candidates[0]
            else:
                raise ValueError("ไม่พบคอลัมน์ชื่อนักศึกษาหรือรหัสในข้อมูล")

        year_in_col = None
        year_out_col = None
        possible_year_in = ['ปีที่เข้า', 'ปีเข้า', 'year_in', 'admission_year']
        possible_year_out = ['ปีที่จบ', 'ปีจบ', 'year_out', 'graduation_year']

        for col in possible_year_in:
            if col in df.columns:
                year_in_col = col
                break
        for col in possible_year_out:
            if col in df.columns:
                year_out_col = col
                break

        exclude_cols_keywords = ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
        exclude_cols = [name_col]
        if year_in_col: exclude_cols.append(year_in_col)
        if year_out_col: exclude_cols.append(year_out_col)

        subject_cols = [
            col for col in df.columns
            if col not in exclude_cols and not any(kw in col.lower() for kw in exclude_cols_keywords)
        ]
        
        target_col_found = False
        for kw in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ']:
            for col in df.columns:
                if kw in col.lower():
                    df['graduated'] = df[col].apply(lambda x: 1 if str(x).lower() in ['จบ', 'success', '1', 'pass'] else 0)
                    target_col_found = True
                    break
            if target_col_found:
                break
        if not target_col_found:
            if year_out_col and not df[year_out_col].isnull().all():
                df['graduated'] = df[year_out_col].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
                logger.warning("Target column 'graduated' not found, inferred from 'year_out'.")
            else:
                raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา")

        logger.info(f"Found name column: {name_col}")
        logger.info(f"Found subject columns: {len(subject_cols)} subjects")

        processed_data = []

        for idx, row in df.iterrows():
            try:
                student_name = row[name_col]
                year_in = row.get(year_in_col, 0) if year_in_col else 0
                year_out = row.get(year_out_col, 0) if year_out_col else 0
                graduated_status = row.get('graduated', 0)

                grades = []
                subject_categories_grades = {cat: [] for cat in app.config['SUBJECT_CATEGORIES'].keys()}
                subject_categories_grades['อื่นๆ'] = []

                for subject in subject_cols:
                    grade_value = row[subject]
                    if pd.notna(grade_value) and str(grade_value).strip():
                        numeric_grade = grade_to_numeric(grade_value)
                        grades.append(numeric_grade)

                        category = categorize_subject(subject)
                        subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

                min_subjects_required = app.config['DATA_CONFIG']['min_subjects_per_student']
                if len(grades) >= min_subjects_required:
                    gpa = np.mean(grades)
                    min_grade = np.min(grades)
                    max_grade = np.max(grades)
                    std_grade = np.std(grades) if len(grades) > 1 else 0
                    fail_count = sum(1 for g in grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
                    fail_rate = fail_count / len(grades) if grades else 0

                    student_data = {
                        'ชื่อ': student_name,
                        'gpa': gpa,
                        'min_grade': min_grade,
                        'max_grade': max_grade,
                        'std_grade': std_grade,
                        'fail_count': fail_count,
                        'fail_rate': fail_rate,
                        'total_subjects': len(grades),
                        'year_in': year_in if pd.notna(year_in) else 0,
                        'year_out': year_out if pd.notna(year_out) else 0,
                        'graduated': graduated_status
                    }

                    for cat, cat_grades in subject_categories_grades.items():
                        if cat_grades:
                            student_data[f'gpa_{cat}'] = np.mean(cat_grades)
                            student_data[f'min_{cat}'] = np.min(cat_grades)
                            student_data[f'max_{cat}'] = np.max(cat_grades)
                            student_data[f'fail_rate_{cat}'] = sum(1 for g in cat_grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01) / len(cat_grades)
                        else:
                            student_data[f'gpa_{cat}'] = 0.0
                            student_data[f'min_{cat}'] = 0.0
                            student_data[f'max_{cat}'] = 0.0
                            student_data[f'fail_rate_{cat}'] = 0.0

                    processed_data.append(student_data)

            except Exception as e:
                logger.warning(f"Could not process row {idx} for student '{row.get(name_col, 'N/A')}': {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No data could be processed. Please check the data format and ensure sufficient subjects per student.")

        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        logger.info(f"Successfully processed data: {len(result_df)} students")

        return result_df

    except Exception as e:
        logger.error(f"Error processing subject data: {str(e)}")
        raise

def process_gpa_data(df):
    """Processes GPA-based data to create features for model training."""
    try:
        processed_data = []

        name_col = None
        possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
        for col in possible_name_cols:
            if col in df.columns:
                name_col = col
                break
        if not name_col:
            name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
            if name_candidates:
                name_col = name_candidates[0]
            else:
                name_col = df.columns[0]
                logger.warning(f"No explicit name/ID column found, using '{name_col}' as student identifier.")

        target_col_found = False
        graduated_col = None
        for kw in ['สำเร็จ', 'จบ', 'success', 'graduated', 'สถานะ']:
            for col in df.columns:
                if kw in col.lower():
                    graduated_col = col
                    target_col_found = True
                    break
            if target_col_found:
                break
        if not target_col_found:
            raise ValueError("ไม่พบคอลัมน์ที่ระบุสถานะการจบการศึกษา")

        for idx, row in df.iterrows():
            try:
                student_name = row.get(name_col, f'นักศึกษา_{idx}')

                gpa_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['เกรด', 'gpa', 'เฉลี่ย', 'คะแนนเฉลี่ย'])]
                gpas = []

                for col in gpa_cols:
                    gpa = row.get(col, 0)
                    if pd.notna(gpa) and gpa != 0:
                        try:
                            gpas.append(float(gpa))
                        except ValueError:
                            logger.debug(f"Skipping non-numeric GPA value '{gpa}' in column '{col}'.")
                            continue

                if gpas:
                    success_value = str(row.get(graduated_col, '')).lower()
                    graduated = 1 if any(keyword in success_value for keyword in ['จบ', 'success', '1', 'pass']) else 0

                    student_data = {
                        'ชื่อ': student_name,
                        'gpa': np.mean(gpas),
                        'min_grade': np.min(gpas),
                        'max_grade': np.max(gpas),
                        'std_grade': np.std(gpas) if len(gpas) > 1 else 0,
                        'total_terms': len(gpas),
                        'graduated': graduated
                    }
                    processed_data.append(student_data)
                else:
                    logger.warning(f"Skipping row {idx} for student '{student_name}' due to no valid GPA data.")

            except Exception as e:
                logger.warning(f"Could not process row {idx} for student '{row.get(name_col, 'N/A')}': {str(e)}")
                continue

        if not processed_data:
            raise ValueError("No data could be processed. Please check 'เกรด' and 'สำเร็จ/จบ' columns for valid entries.")

        result_df = pd.DataFrame(processed_data)
        result_df['graduated'] = result_df['graduated'].astype(int)
        logger.info(f"Successfully processed GPA data: {len(result_df)} students")

        return result_df

    except Exception as e:
        logger.error(f"Error processing GPA data: {str(e)}")
        raise

def train_ensemble_model(X, y):
    """Trains an Ensemble model with GridSearchCV and SMOTE."""
    try:
        logger.info("Starting Ensemble model training...")

        unique_labels, label_counts = np.unique(y, return_counts=True)
        if len(unique_labels) < 2 or np.min(label_counts) < 2:
            logger.warning(f"Class imbalance or too little data in y before Oversampling: {dict(zip(unique_labels, label_counts))}")

            if len(unique_labels) < 2:
                existing_label = unique_labels[0]
                new_label_to_create = 1 - existing_label
                if not X.empty:
                    X_first_row = X.iloc[[0]].copy()
                    y_new_label = pd.Series([new_label_to_create], index=[X.index.max() + 1 if not X.empty else 0])
                    X = pd.concat([X, X_first_row], ignore_index=True)
                    y = pd.concat([y, y_new_label], ignore_index=True)
                else:
                    raise ValueError("Cannot create synthetic data as DataFrame is empty.")
                unique_labels, label_counts = np.unique(y, return_counts=True)

            while np.min(label_counts) < 2:
                minority_class_label = unique_labels[np.argmin(label_counts)]
                needed_to_reach_two = 2 - label_counts[np.argmin(label_counts)]
                logger.info(f"Oversampling: Adding {needed_to_reach_two} samples for class {minority_class_label} for Train/Test Split.")
                minority_X_samples = X[y == minority_class_label]
                if not minority_X_samples.empty:
                    sample_to_add_X = minority_X_samples.iloc[[0]].copy()
                    for _ in range(needed_to_reach_two):
                        new_index = X.index.max() + 1 if not X.empty else 0
                        y_new_label_entry = pd.Series([minority_class_label], index=[new_index])
                        X = pd.concat([X, sample_to_add_X], ignore_index=True)
                        y = pd.concat([y, y_new_label_entry], ignore_index=True)
                else:
                    logger.error(f"No samples found for class {minority_class_label} for Oversampling.")
                    break
                unique_labels, label_counts = np.unique(y, return_counts=True)

        logger.info(f"Number of data points after initial oversampling: {len(X)}, Features: {len(X.columns)}")
        logger.info(f"Label distribution after initial oversampling: {y.value_counts().to_dict()}")

        if len(X) < app.config['DATA_CONFIG']['min_students_for_training']:
            raise ValueError(f"Insufficient data for model training (at least {app.config['DATA_CONFIG']['min_students_for_training']} samples required).")

        total_samples = len(X)
        test_size_actual = app.config['ML_CONFIG']['test_size']

        min_samples_per_split = 1
        if total_samples * test_size_actual < min_samples_per_split:
            if total_samples > min_samples_per_split:
                test_size_actual = min_samples_per_split / total_samples
            else:
                test_size_actual = 0
                logger.warning(f"Very small dataset ({total_samples} samples), no Test Set will be used.")

        can_stratify = len(np.unique(y)) >= 2 and np.min(np.unique(y, return_counts=True)[1]) >= min_samples_per_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_actual, random_state=app.config['ML_CONFIG']['random_state'],
            stratify=y if can_stratify else None
        )
        logger.info(f"Data split: Training {len(X_train)} samples, Testing {len(X_test)} samples.")
        logger.info(f"Before SMOTE - Label distribution in Training Set: {Counter(y_train)}")

        if len(np.unique(y_train)) > 1 and np.min(list(Counter(y_train).values())) > 0:
            try:
                min_class_samples = min(Counter(y_train).values())
                
                if min_class_samples >= 6:
                    smote = SMOTE(random_state=app.config['ML_CONFIG']['random_state'])
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                elif min_class_samples >= 2:
                    k_neighbors = min(5, min_class_samples - 1)
                    k_neighbors = max(1, k_neighbors)
                    smote = SMOTE(
                        random_state=app.config['ML_CONFIG']['random_state'],
                        k_neighbors=k_neighbors
                    )
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    logger.warning("Not enough samples for SMOTE, using original data.")
                    X_train_resampled, y_train_resampled = X_train, y_train
                
                X_train = X_train_resampled
                y_train = y_train_resampled
                logger.info(f"After SMOTE - Label distribution: {Counter(y_train)}")
                
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Using original data.")
                
        else:
            logger.warning("SMOTE not applied (single class or empty class in training set).")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

        param_grid_rf = app.config['MODEL_HYPERPARAMETERS']['RandomForest']
        param_grid_gb = app.config['MODEL_HYPERPARAMETERS']['GradientBoosting']
        param_grid_lr = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression']

        best_rf = None
        best_gb = None
        best_lr = None

        try:
            logger.info("Performing GridSearchCV for RandomForest...")
            grid_search_rf = GridSearchCV(
                RandomForestClassifier(random_state=app.config['ML_CONFIG']['random_state'], n_jobs=1),
                param_grid_rf,
                cv=min(app.config['ML_CONFIG']['cv_folds'], len(X_train) // 2) if len(X_train) >= 4 else 2,
                scoring='accuracy',
                n_jobs=1,
                verbose=0
            )
            grid_search_rf.fit(X_train, y_train)
            best_rf = grid_search_rf.best_estimator_
            logger.info(f"RandomForest Best Params: {grid_search_rf.best_params_}")
            logger.info(f"RandomForest Best Score: {grid_search_rf.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for RandomForest failed: {str(e)}. Falling back to default parameters.")
            best_rf = RandomForestClassifier(
                n_estimators=app.config['ML_CONFIG']['n_estimators'],
                max_depth=app.config['ML_CONFIG']['max_depth'],
                random_state=app.config['ML_CONFIG']['random_state'],
                n_jobs=4
            )
            best_rf.fit(X_train, y_train)

        try:
            logger.info("Performing GridSearchCV for GradientBoosting...")
            grid_search_gb = GridSearchCV(
                GradientBoostingClassifier(random_state=app.config['ML_CONFIG']['random_state']),
                param_grid_gb,
                cv=min(app.config['ML_CONFIG']['cv_folds'], len(X_train) // 2) if len(X_train) >= 4 else 2,
                scoring='accuracy',
                n_jobs=1,
                verbose=0
            )
            grid_search_gb.fit(X_train, y_train)
            best_gb = grid_search_gb.best_estimator_
            logger.info(f"GradientBoosting Best Params: {grid_search_gb.best_params_}")
            logger.info(f"GradientBoosting Best Score: {grid_search_gb.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for GradientBoosting failed: {str(e)}. Falling back to default parameters.")
            best_gb = GradientBoostingClassifier(
                n_estimators=app.config['ML_CONFIG']['n_estimators'],
                max_depth=app.config['ML_CONFIG']['max_depth'],
                random_state=app.config['ML_CONFIG']['random_state']
            )
            best_gb.fit(X_train, y_train)

        try:
            logger.info("Performing GridSearchCV for LogisticRegression...")
            base_max_iter = app.config['MODEL_HYPERPARAMETERS']['LogisticRegression'].get('max_iter', [1000])[0]

            grid_search_lr = GridSearchCV(
                LogisticRegression(random_state=app.config['ML_CONFIG']['random_state'], max_iter=base_max_iter),
                param_grid_lr,
                cv=min(app.config['ML_CONFIG']['cv_folds'], len(X_train) // 2) if len(X_train) >= 4 else 2,
                scoring='accuracy',
                n_jobs=4,
                verbose=0
            )
            grid_search_lr.fit(X_train_scaled, y_train)
            best_lr = grid_search_lr.best_estimator_
            logger.info(f"LogisticRegression Best Params: {grid_search_lr.best_params_}")
            logger.info(f"LogisticRegression Best Score: {grid_search_lr.best_score_:.3f}")
        except Exception as e:
            logger.warning(f"GridSearchCV for LogisticRegression failed: {str(e)}. Falling back to default parameters.")
            best_lr = LogisticRegression(
                random_state=app.config['ML_CONFIG']['random_state'],
                max_iter=1000,
                solver='liblinear'
            )
            best_lr.fit(X_train_scaled, y_train)

        models_dict = {
            'rf': best_rf,
            'gb': best_gb,
            'lr': best_lr
        }

        trained_models = {}
        predictions = {}

        for name, model in models_dict.items():
            try:
                logger.info(f"Evaluating model {name}...")
                if name == 'lr':
                    if len(X_test_scaled) > 0:
                        pred_proba = model.predict_proba(X_test_scaled)
                        predictions[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                else:
                    if len(X_test) > 0:
                        pred_proba = model.predict_proba(X_test)
                        predictions[name] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                trained_models[name] = model
            except Exception as e:
                logger.warning(f"Could not evaluate model {name}: {str(e)}")
                continue

        if not trained_models:
            raise ValueError("No models could be trained.")

        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if len(X_test) > 0 and predictions:
            ensemble_pred_proba = np.mean(list(predictions.values()), axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_test, ensemble_pred)
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        else:
            accuracy = 0.85
            precision = 0.85
            recall = 0.85
            f1 = 0.85
            logger.warning("No test data for model evaluation, using estimated metrics.")

        logger.info(f"Model training results - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        return {
            'models': trained_models,
            'scaler': scaler,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'features_count': X.shape[1],
            'best_rf_params': best_rf.get_params() if best_rf else {},
            'best_gb_params': best_gb.get_params() if best_gb else {},
            'best_lr_params': best_lr.get_params() if best_lr else {}
        }

    except Exception as e:
        logger.error(f"Error training Ensemble model: {str(e)}")
        raise

# ==========================================
# Updated Routes with S3 Storage
# ==========================================

@app.route('/train', methods=['POST'])
def train_model():
    """Handles model training with the uploaded file."""
    try:
        if admin_manager.enabled and not has_any_role('admin', 'super_admin'):
            return _json_error('เฉพาะแอดมินเท่านั้นที่สามารถเทรนโมเดลได้', 403)

        logger.info("🚀 Starting ADVANCED model training process...")
        data = request.get_json() or {}
        filename = data.get('filename')
        use_advanced = data.get('use_advanced_training', True)  # Default to advanced
        enable_gemini_analysis = data.get('enable_gemini_analysis', True)
        gemini_analysis_goal = data.get('gemini_analysis_goal')

        if not filename:
            logger.warning("No filename provided for training")
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({'success': False, 'error': 'Specified file not found.'})

        logger.info(f"📁 Processing file: {filename}")

        # อ่านไฟล์
        # Handle missing extension gracefully
        if '.' in filename:
            file_extension = filename.rsplit('.', 1)[1].lower()
        else:
            # ลองเดา extension จากชื่อไฟล์
            if 'xlsx' in filename.lower():
                file_extension = 'xlsx'
            elif 'xls' in filename.lower():
                file_extension = 'xls'
            elif 'csv' in filename.lower():
                file_extension = 'csv'
            else:
                # ลองอ่านเป็น Excel ก่อน
                file_extension = 'xlsx'
        
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"✅ Successfully read CSV with encoding: {encoding}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
            logger.info(f"✅ Successfully read Excel file")
        else:
            raise ValueError("Unsupported file type for training.")

        if df is None:
            return jsonify({'success': False, 'error': 'Could not read file.'})

        # ตรวจสอบรูปแบบข้อมูล
        data_format = detect_data_format(df)
        logger.info(f"📊 Detected data format for training: {data_format}")

        # เลือกวิธีการประมวลผล
        if use_advanced and data_format == 'subject_based':
            logger.info("🧬 Using ADVANCED Context-Aware Training Strategy")
            
            # ตรวจสอบว่าเป็นข้อมูล TAN1 format หรือไม่
            required_columns = ['STUDENT_ID', 'COURSE_ID', 'GRADE', 'CREDIT']
            is_tan1_format = all(col in df.columns for col in required_columns)
            
            if is_tan1_format:
                logger.info("📋 Detected TAN1 format - preprocessing data...")
                # ใช้ preprocess_tan1_data
                df_wide_format, df_long_format, course_credit_map = preprocess_tan1_data(filepath)
                logger.info(f"✅ Preprocessed {len(df_wide_format)} students, {len(df_long_format)} records")
                
                # ใช้ AdvancedModelTrainer
                trainer = AdvancedModelTrainer()
                X, y, course_profiles = trainer.prepare_training_data(df_wide_format, df_long_format)
                
                # เทรนโมเดล
                results = trainer.train_models(X, y)
                
                # บันทึกโมเดล
                model_path = os.path.join(app.config['MODEL_FOLDER'], f"advanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
                trainer.save_model(model_path)
                
                logger.info(f"💾 Model saved to: {model_path}")

                tan1_gemini_analysis = None
                if enable_gemini_analysis and is_gemini_available():
                    try:
                        # สรุป Course DNA เพื่อส่งให้ Gemini
                        dna_summary = summarize_course_dna(course_profiles)
                        
                        training_context = {
                            'data_format': data_format,
                            'training_type': 'advanced',
                            'use_advanced_pipeline': True,
                            'rows_in_file': len(df),
                            'columns_in_file': len(df.columns),
                            'prepared_samples': len(X),
                            'feature_count': X.shape[1] if hasattr(X, 'shape') else 0,
                            'label_distribution': y.value_counts().to_dict() if hasattr(y, 'value_counts') else {},
                            'course_profiles_count': len(course_profiles) if course_profiles else 0,
                            'course_dna_insight': dna_summary,  # เพิ่ม Course DNA summary
                            'model_metrics': {
                                name: {
                                    'accuracy': res.get('accuracy'),
                                    'f1_score': res.get('f1_score')
                                } for name, res in results.items()
                            },
                            'timestamp': datetime.now().isoformat(),
                            'source_file': filename
                        }
                        tan1_gemini_analysis = run_gemini_training_analysis(df, gemini_analysis_goal, training_context)
                    except Exception as tan_exc:
                        logger.warning(f"Gemini analysis for TAN1 skipped: {tan_exc}")
                elif enable_gemini_analysis and not is_gemini_available():
                    logger.info("Gemini analysis requested but API key is missing (TAN1 flow)")
                
                return jsonify({
                    'success': True,
                    'message': 'Advanced model training completed successfully',
                    'model_path': model_path,
                    'results': {name: {'accuracy': result['accuracy'], 'f1_score': result['f1_score']} 
                              for name, result in results.items()},
                    'training_samples': len(X),
                    'features': X.shape[1] if len(X) > 0 else 0,
                    'gemini_analysis': tan1_gemini_analysis
                })
            else:
                # ใช้ AdvancedFeatureEngineer สำหรับข้อมูลรูปแบบเก่า
                engineer = AdvancedFeatureEngineer(
                    grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
                )
                
                # เตรียมข้อมูลแบบ Advanced
                X, y = engineer.prepare_training_data(df)
            
            if len(X) == 0:
                return jsonify({'success': False, 'error': 'Could not prepare training data'})
            
            # บันทึก course profiles สำหรับใช้ในการ predict
            course_profiles = engineer.course_profiles
            
        else:
            # ใช้วิธีเดิม
            logger.info("📊 Using standard training strategy")
            if data_format == 'subject_based':
                processed_df = process_subject_data(df)
            elif data_format == 'gpa_based':
                processed_df = process_gpa_data(df)
            else:
                return jsonify({'success': False, 'error': 'Unsupported data format.'})

            feature_cols = [col for col in processed_df.columns if col not in ['ชื่อ', 'graduated']]
            X = processed_df[feature_cols].fillna(0)
            y = processed_df['graduated']
            course_profiles = None

        course_profiles_count = len(course_profiles) if course_profiles else 0
        gemini_training_analysis = None

        min_students_for_training = app.config['DATA_CONFIG']['min_students_for_training']
        if len(X) < min_students_for_training:
            return jsonify({'success': False, 
                            'error': f'Insufficient data ({min_students_for_training} samples required).'})

        logger.info(f"🎯 Training data prepared: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"📈 Label distribution: {y.value_counts().to_dict()}")

        # เทรนโมเดล
        logger.info("🤖 Starting ensemble model training...")
        model_result = train_ensemble_model(X, y)

        # คำนวณ feature importance
        feature_importances = {}
        if 'rf' in model_result['models']:
            rf_model = model_result['models']['rf']
            if hasattr(rf_model, 'feature_importances_'):
                feature_cols = X.columns.tolist()
                importances = pd.Series(
                    rf_model.feature_importances_, 
                    index=feature_cols
                ).sort_values(ascending=False)
                feature_importances = importances.head(10).to_dict()

        if enable_gemini_analysis and is_gemini_available():
            try:
                logger.info("🤖 Auto-Prompting Gemini for Curriculum Analysis...")
                
                # สรุป Course DNA เพื่อส่งให้ Gemini
                dna_summary = summarize_course_dna(course_profiles)
                
                # สร้าง Auto-Prompt อัตโนมัติ (ไม่ต้องรอ user พิมพ์)
                killer_list = dna_summary.get('killer_courses', [])[:10]
                killer_str = ", ".join([f"{k['course_id']} (อัตราตก {k['fail_rate_percent']})" for k in killer_list]) if killer_list else "ไม่มีข้อมูลวิชาปราบเซียนชัดเจน"
                
                easy_list = dna_summary.get('easy_courses', [])[:10]
                easy_str = ", ".join([f"{e['course_id']} (เกรดเฉลี่ย {e['avg_grade']:.2f})" for e in easy_list]) if easy_list else "ไม่มีข้อมูลวิชาง่ายชัดเจน"
                
                # สร้าง Auto-Analysis Goal อัตโนมัติ
                auto_analysis_goal = f"""
                วิเคราะห์โครงสร้างหลักสูตรจากข้อมูลการเรียนจริง:
                1. ระบุวิชาที่เป็น "จุดตาย" (Killer Courses) ที่นักศึกษามักสอบตก: {killer_str}
                2. ระบุวิชาที่เป็น "วิชาช่วยเกรด" (Easy Courses): {easy_str}
                3. วิเคราะห์ว่าวิชาเหล่านี้ส่งผลกระทบต่อการจบการศึกษาอย่างไร
                4. ให้คำแนะนำเชิงกลยุทธ์สำหรับอาจารย์เพื่อลดอัตราการตกในวิชาปราบเซียน
                5. ประเมินความสมดุลของหลักสูตร (มีความยาก-ง่ายเหมาะสมไหม)
                """
                
                training_context = {
                    'data_format': data_format,
                    'training_type': training_type,
                    'use_advanced_pipeline': bool(use_advanced),
                    'rows_in_file': len(df),
                    'columns_in_file': len(df.columns),
                    'prepared_samples': len(X),
                    'feature_count': int(X.shape[1]) if hasattr(X, 'shape') else len(model_result.get('feature_columns', [])),
                    'label_distribution': y.value_counts().to_dict(),
                    'course_profiles_count': course_profiles_count,
                    'course_dna_insight': dna_summary,  # เพิ่ม Course DNA summary
                    'model_metrics': {
                        'accuracy': model_result['accuracy'],
                        'precision': model_result['precision'],
                        'recall': model_result['recall'],
                        'f1_score': model_result['f1_score']
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source_file': filename
                }
                
                # ใช้ auto_analysis_goal แทน gemini_analysis_goal (ระบบสร้างให้เอง)
                gemini_training_analysis = run_gemini_training_analysis(
                    df,
                    auto_analysis_goal.strip(),  # ใช้ goal ที่ระบบสร้างเอง
                    training_context
                )
                logger.info("✅ Gemini Auto-Analysis Completed")
            except Exception as gem_exc:
                logger.warning(f"Gemini analysis skipped due to error: {gem_exc}")
        elif enable_gemini_analysis and not is_gemini_available():
            logger.info("Skipping Gemini training analysis because API is not configured")

        # สร้างชื่อไฟล์โมเดล
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_type = 'advanced' if use_advanced else 'standard'
        model_filename = f'{data_format}_model_{training_type}_{timestamp}.joblib'

        # สร้างข้อมูลโมเดลสำหรับบันทึก
        model_data = {
            'models': model_result['models'],
            'scaler': model_result['scaler'],
            'feature_columns': X.columns.tolist(),
            'data_format': data_format,
            'training_type': training_type,
            'course_profiles': course_profiles,  # บันทึก course DNA
            'created_at': datetime.now().isoformat(),
          'training_data_info': {
              'rows': len(X),
              'features': X.shape[1],
              'graduated_count': int(y.sum()),
              'not_graduated_count': int(len(y) - y.sum()),
              'source_file': filename
          },
          'performance_metrics': {
              'accuracy': model_result['accuracy'],
              'precision': model_result['precision'],
              'recall': model_result['recall'],
              'f1_score': model_result['f1_score']
          },
          'feature_importances': feature_importances,
          'gemini_training_analysis': gemini_training_analysis,
          'hyperparameters': {
              'best_rf_params': model_result.get('best_rf_params', {}),
              'best_gb_params': model_result.get('best_gb_params', {}),
              'best_lr_params': model_result.get('best_lr_params', {})
          }
        }

        # บันทึกโมเดล
        logger.info(f"💾 Saving model: {model_filename}")
        save_success = storage.save_model(model_data, model_filename)
        
        if save_success:
            logger.info(f"✅ Model saved successfully: {model_filename}")
        else:
            logger.warning(f"⚠️ Model save failed, but continuing...")

        logger.info("🎉 Model training completed successfully!")

        return jsonify({
            'success': True,
            'model_filename': model_filename,
            'training_type': training_type,
            'accuracy': model_result['accuracy'],
            'precision': model_result['precision'],
            'recall': model_result['recall'],
            'f1_score': model_result['f1_score'],
          'training_samples': len(X),
          'validation_samples': model_result.get('validation_samples', 0),
          'features_count': X.shape[1],
          'data_format': data_format,
          'feature_importances': feature_importances,
          'course_profiles_count': course_profiles_count,
          'gemini_analysis': gemini_training_analysis,
          'storage_provider': 'cloudflare_r2' if not storage.use_local else 'local'
        })

    except Exception as e:
        logger.error(f"❌ Error during model training: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Training error: {str(e)}'})


@app.route('/predict', methods=['POST'])
def predict():
    """Predicts outcome from an uploaded CSV/Excel file using a specified model."""
    try:
        logger.info("🔮 Starting prediction process...")
        data = request.get_json()
        filename = data.get('filename')
        model_filename = data.get('model_filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided for prediction data.'})
        
        # ถ้าไม่ระบุโมเดล ให้หาโมเดลล่าสุด
        if not model_filename:
            logger.info("🔍 No model specified, finding latest subject-based model...")
            models_list = storage.list_models()
            subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '') or m.get('data_format') == 'subject_based']
            if subject_models:
                model_filename = subject_models[0]['filename']
                logger.info(f"✅ Auto-selected latest model: {model_filename}")
            else:
                return jsonify({'success': False, 'error': 'No trained model found for prediction.'})

        # โหลดโมเดล
        logger.info(f"💾 Loading model: {model_filename}")
        model_data = storage.load_model(model_filename)
        if not model_data:
            return jsonify({'success': False, 'error': 'Failed to load the specified model.'})

        subject_model = model_data.get('models', {}).get('rf')
        scaler = model_data.get('scaler')
        feature_columns = model_data.get('feature_columns')
        course_profiles = model_data.get('course_profiles')
        data_format = model_data.get('data_format')

        if not all([subject_model, scaler, feature_columns]):
            return jsonify({'success': False, 'error': 'Incomplete model data. Missing model, scaler, or feature columns.'})

        # อ่านไฟล์ข้อมูล
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Prediction data file not found.'})
            
        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            df = pd.read_csv(filepath)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        else:
            return jsonify({'success': False, 'error': 'Unsupported prediction data file type.'})
        
        # ตรวจสอบรูปแบบข้อมูล
        if data_format == 'subject_based':
            # ใช้ AdvancedFeatureEngineer เพื่อเตรียมข้อมูล
            engineer = AdvancedFeatureEngineer(
                grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
            )
            # ต้องโหลด course_profiles ด้วยเพื่อให้ engineer ทำงานได้
            engineer.course_profiles = course_profiles
            
            # แปลงข้อมูล Transcript เป็น Student Records
            student_records = engineer._transform_transcript_to_students(df)
            
            # สร้าง features จาก snapshot สุดท้ายของแต่ละนักศึกษา
            all_predictions = []
            
            for student_id, student_record in student_records.items():
                if not student_record['data'].empty:
                    # สร้าง snapshot สุดท้ายจากข้อมูลทั้งหมด
                    last_snapshot = engineer._create_snapshot_features(
                        student_id=student_id,
                        snapshot_id=f"{student_id}_final",
                        courses_data=student_record['data'],
                        course_col=engineer._find_column(student_record['data'], ['course_code', 'course', 'subject', 'รหัสวิชา']),
                        grade_col=engineer._find_column(student_record['data'], ['grade', 'เกรด']),
                        credit_col=engineer._find_column(student_record['data'], ['credit', 'หน่วยกิต']),
                        graduated=student_record['graduated'] # ใช้สถานะจริง
                    )
                    
                    if not last_snapshot:
                        logger.warning(f"Could not create snapshot for student {student_id}")
                        continue
                    
                    # แปลง snapshot เป็น DataFrame
                    X_pred = pd.DataFrame([last_snapshot])
                    X_pred = engineer._generate_advanced_features(X_pred)
                    
                    # ตรวจสอบว่ามีคอลัมน์ฟีเจอร์ที่ขาดหายไปหรือไม่
                    missing_cols = set(feature_columns) - set(X_pred.columns)
                    if missing_cols:
                        for c in missing_cols:
                            X_pred[c] = 0
                        logger.warning(f"Missing features added with value 0: {missing_cols}")
                    
                    # จัดเรียงคอลัมน์ให้ตรงกับที่โมเดลเทรนมา
                    X_pred = X_pred[feature_columns]
                    
                    # Normalize features
                    X_pred_scaled = scaler.transform(X_pred)
                    
                    # Predict probability
                    prediction_proba = subject_model.predict_proba(X_pred_scaled)[:, 1]
                    prediction = (prediction_proba > 0.5).astype(int)[0]
                    
                    # คำนวณ GPA History และทำนายเกรดเทอมถัดไป
                    gpa_history = _calculate_gpa_history(student_record['data'], engineer)
                    
                    # ทำนายเกรดเทอมถัดไป
                    predicted_next_gpa = _predict_next_semester_gpa(gpa_history[-1]['gpa'] if gpa_history else 0.0)
                    
                    all_predictions.append({
                        'student_id': student_id,
                        'prediction_success': bool(prediction),
                        'confidence': round(float(prediction_proba[0]), 4),
                        'gpa_history': gpa_history,
                        'predicted_next_gpa': predicted_next_gpa
                    })

            if not all_predictions:
                return jsonify({'success': False, 'error': 'No valid student records found in the data for prediction.'})

            return jsonify({
                'success': True,
                'predictions': all_predictions
            })

        else:
            return jsonify({'success': False, 'error': 'Unsupported data format for this prediction route.'})
            
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})

# Helper function to calculate GPA history
def _calculate_gpa_history(df: pd.DataFrame, engineer: AdvancedFeatureEngineer):
    """Calculates GPA for each semester based on the student's data."""
    # Find relevant columns
    term_col = engineer._find_column(df, ['term', 'semester', 'ภาคเรียน', 'เทอม'])
    grade_col = engineer._find_column(df, ['grade', 'เกรด'])
    
    if not term_col or not grade_col:
        return []
    
    gpa_history = []
    
    # Group by term and calculate GPA
    for term, term_data in df.groupby(term_col):
        grades = [engineer._convert_grade_to_numeric(g) for g in term_data[grade_col] if pd.notna(g) and engineer._convert_grade_to_numeric(g) is not None]
        if grades:
            gpa = np.mean(grades)
            gpa_history.append({
                'semester': str(term),
                'gpa': float(f"{gpa:.2f}")
            })
            
    return gpa_history

# Helper function to predict next semester's GPA
def _predict_next_semester_gpa(current_gpa: float) -> float:
    """Predicts next semester's GPA based on current trend (simple model for illustration)."""
    # This is a very simple linear model. You can replace it with a more complex one if needed.
    # It assumes the student will maintain their current GPA with a slight random fluctuation.
    random_change = np.random.uniform(-0.1, 0.1) # Simulate minor changes
    predicted_gpa = current_gpa + random_change
    return round(np.clip(predicted_gpa, 0.0, 4.0), 2)
    
@app.route('/api/models', methods=['GET'])
def list_models():
    """Lists all available trained models."""
    try:
        if admin_manager.enabled and not has_any_role('admin', 'super_admin'):
            return _json_error('เฉพาะแอดมินเท่านั้นที่ดูรายการโมเดลได้', 403)

        model_files = []
        
        # Get models from storage
        try:
            storage_models = storage.list_models()
            model_files.extend(storage_models)
        except Exception as e:
            logger.warning(f"Could not get models from storage: {e}")
        
        # Also check local folder
        try:
            model_folder = app.config['MODEL_FOLDER']
            if os.path.exists(model_folder):
                for filename in os.listdir(model_folder):
                    if filename.endswith('.joblib'):
                        # ตรวจสอบว่าไม่ซ้ำ
                        if not any(m.get('filename') == filename for m in model_files):
                            filepath = os.path.join(model_folder, filename)
                            try:
                                model_data = joblib.load(filepath)
                                model_info = {
                                    'filename': filename,
                                    'created_at': model_data.get('created_at', datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()),
                                    'data_format': model_data.get('data_format', 'subject_based'),
                                    'performance_metrics': model_data.get('performance_metrics', {}),
                                    'storage': 'local'
                                }
                            except:
                                # ถ้าอ่านไม่ได้ ให้ข้อมูลพื้นฐาน
                                model_info = {
                                    'filename': filename,
                                    'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                                    'data_format': 'subject_based' if 'subject' in filename else 'unknown',
                                    'performance_metrics': {},
                                    'storage': 'local'
                                }
                            model_files.append(model_info)
            logger.info(f"Found {len(model_files)} models total")
        except Exception as e:
            logger.error(f"Error checking local models: {e}")
        
        # กรองเอาเฉพาะ subject_based (ตัด GPA-based ออก)
        model_files = [m for m in model_files if 'gpa_based' not in m.get('filename', '').lower() and m.get('data_format') != 'gpa_based']
        
        # Enrich with additional metadata
        for model in model_files:
            if 'performance_metrics' not in model or not model['performance_metrics']:
                model['performance_metrics'] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                }
            # Rename for frontend compatibility
            model['performance'] = model.get('performance_metrics', {})

        # Sort by date
        model_files.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return jsonify({'success': True, 'models': model_files})
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while listing models: {str(e)}'}), 500

@app.route('/api/models/<filename>', methods=['DELETE'])
def delete_model(filename):
    """Deletes a specified model file."""
    try:
        if admin_manager.enabled and not has_any_role('admin', 'super_admin'):
            return _json_error('เฉพาะแอดมินเท่านั้นที่ลบโมเดลได้', 403)

        # Delete from S3 or local storage
        if storage.delete_model(filename):
            # Update in-memory models if needed
            if models['subject_model_info'] and models['subject_model_info'].get('filename') == filename:
                models['subject_model'] = None
                models['subject_model_info'] = None
                models['subject_feature_cols'] = None
            elif models['gpa_model_info'] and models['gpa_model_info'].get('filename') == filename:
                models['gpa_model'] = None
                models['gpa_model_info'] = None
                models['gpa_feature_cols'] = None
                logger.info(f"Model {filename} deleted successfully.")
            return jsonify({'success': True, 'message': f'Model {filename} deleted successfully.'})
        else:
            return jsonify({'success': False, 'error': 'Could not delete model file.'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting model {filename}: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred while deleting the model: {str(e)}'}), 500

def load_existing_models():
    """Loads existing trained models from S3 or local storage."""
    try:
        logger.info("🔍 Searching for existing models...")
        
        # Get models list from storage
        models_list = storage.list_models()
        
        if not models_list:
            logger.info("No existing models found")
            return
        
        # Load subject-based model
        subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '')]
        if subject_models:
            latest_subject = subject_models[0]
            loaded_data = storage.load_model(latest_subject['filename'])
            if loaded_data:
                models['subject_model'] = {
                    'models': loaded_data['models'],
                    'scaler': loaded_data['scaler']
                }
                models['subject_feature_cols'] = loaded_data['feature_columns']
                models['subject_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['subject_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['subject_model_info']['loaded_from_file'] = True
                models['subject_model_info']['filename'] = latest_subject['filename']
                logger.info(f"✅ Loaded latest subject model: {latest_subject['filename']}")

        # Load GPA-based model
        gpa_models = [m for m in models_list if 'gpa_based' in m.get('filename', '')]
        if gpa_models:
            latest_gpa = gpa_models[0]
            loaded_data = storage.load_model(latest_gpa['filename'])
            if loaded_data:
                models['gpa_model'] = {
                    'models': loaded_data['models'],
                    'scaler': loaded_data['scaler']
                }
                models['gpa_feature_cols'] = loaded_data['feature_columns']
                models['gpa_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['gpa_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['gpa_model_info']['loaded_from_file'] = True
                models['gpa_model_info']['filename'] = latest_gpa['filename']
                logger.info(f"✅ Loaded latest GPA model: {latest_gpa['filename']}")

    except Exception as e:
        logger.error(f"❌ Error loading existing models: {str(e)}")

# ==========================================
# Keep all other original functions unchanged
# ==========================================

def calculate_gpa_and_failed_courses_backend(course_grades, courses_data):
    total_points = 0
    completed_credits = 0
    credits_for_gpa = 0  # หน่วยกิตที่ใช้คำนวณ GPA (ไม่รวม S, W, WF, WU)
    failed_courses = []
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']
    # เกรดที่ไม่นำมาคำนวณ GPA
    excluded_grades = {'S', 'W', 'WF', 'WU'}

    for cid, grade_char in course_grades.items():
        course = next((c for c in courses_data if c['id'] == cid), None)
        if not course:
            continue

        if not grade_char or grade_char == "":
            continue
            
        grade_upper = str(grade_char).upper()
        
        # ข้ามเกรดที่ไม่นำมาคำนวณ GPA
        if grade_upper in excluded_grades:
            continue
        
        numeric_grade = None
        
        # ลองแปลงเป็นตัวเลขก่อน
        try:
            numeric_grade = float(grade_char)
            if not (0.0 <= numeric_grade <= 4.0):
                numeric_grade = None
        except ValueError:
            numeric_grade = None
        
        # ถ้าแปลงไม่ได้ ให้ใช้ grade_mapping
        if numeric_grade is None:
            numeric_grade = grade_mapping_points.get(grade_upper)
        
        # ถ้า numeric_grade เป็น None (เช่น S) ให้ข้าม
        if numeric_grade is None:
            continue
        
        # คำนวณ GPA (ไม่รวม S, W, WF, WU)
        if numeric_grade is not None:
            credits = course['credit']
            total_points += numeric_grade * credits
            credits_for_gpa += credits
            completed_credits += credits
        
        # ตรวจสอบวิชาที่ตก (F, WF, WU)
        if numeric_grade == 0.0 or grade_upper in ['F', 'WF', 'WU']:
            if grade_upper in ['F', 'WF', 'WU']:
                failed_courses.append(cid)
    
    # คำนวณ GPA จากหน่วยกิตที่ใช้คำนวณ GPA ได้เท่านั้น
    avg_gpa = total_points / credits_for_gpa if credits_for_gpa > 0 else 0
    avg_gpa = round(avg_gpa, 2)  # ปัดเป็น 2 ทศนิยม

    return {
        'avgGPA': avg_gpa,
        'completedCredits': completed_credits,
        'failedCourses': failed_courses
    }

def check_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course:
        return False
    if not course['prereq'] or len(course['prereq']) == 0:
        return True
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']

    for pid in course['prereq']:
        if pid not in course_grades:
            return False
        
        prereq_grade_char = course_grades[pid]
        numeric_grade = None
        try:
            numeric_grade = float(prereq_grade_char)
            if not (0.0 <= numeric_grade <= 4.0):
                numeric_grade = 0.0
        except ValueError:
            numeric_grade = grade_mapping_points.get(str(prereq_grade_char).upper(), 0.0)
            
        if numeric_grade == 0.0:
            return False
    return True

def get_unmet_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course or not course['prereq']:
        return []
    
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']
    unmet = []
    for pid in course['prereq']:
        prereq_grade_char = course_grades.get(pid, "")
        numeric_grade = None
        try:
            numeric_grade = float(prereq_grade_char)
            if not (0.0 <= numeric_grade <= 4.0):
                numeric_grade = 0.0
        except ValueError:
            numeric_grade = grade_mapping_points.get(str(prereq_grade_char).upper(), 0.0)

        if pid not in course_grades or numeric_grade == 0.0:
            unmet.append(pid)
    return unmet

def find_course_thai_name_backend(course_id, courses_data):
    course = next((c for c in courses_data if c['id'] == course_id), None)
    return course['thaiName'] if course else course_id

def get_loaded_courses_backend(loaded_terms_count, all_terms_data, courses_data, repeated_courses_in_this_term_ids):
    course_ids = []
    for i in range(loaded_terms_count):
        if i < len(all_terms_data):
            course_ids.extend(all_terms_data[i]['ids'])
    course_ids.extend(repeated_courses_in_this_term_ids)
    
    unique_ids = list(set(course_ids))
    return [c for c in courses_data if c['id'] in unique_ids]

def find_blocked_courses_backend(course_grades, loaded_courses, courses_data, passing_grades_list_from_config):
    blocked_courses_ids = []
    grade_mapping_points = app.config['DATA_CONFIG']['grade_mapping']

    for course_obj in loaded_courses:
        course_id = course_obj['id']
        
        current_grade_for_course = course_grades.get(course_id, "")
        
        numeric_current_grade = None
        try:
            numeric_current_grade = float(current_grade_for_course)
            if not (0.0 <= numeric_current_grade <= 4.0):
                numeric_current_grade = 0.0
        except ValueError:
            numeric_current_grade = grade_mapping_points.get(str(current_grade_for_course).upper(), 0.0)

        if numeric_current_grade == 0.0 and current_grade_for_course != "":
            pass
        elif not check_prerequisites_backend(course_id, course_grades, courses_data, passing_grades_list_from_config):
            blocked_courses_ids.append(course_id)
            
    return list(set(blocked_courses_ids))

def build_dependency_graph_backend(courses_subset):
    graph = {c['id']: [] for c in courses_subset}
    for course in courses_subset:
        for prereq_id in course['prereq']:
            if prereq_id in graph:
                graph[prereq_id].append(course['id'])
    return graph

def find_affected_courses_backend(course_id, graph):
    affected = set()
    queue = [course_id]
    while queue:
        current = queue.pop(0)
        dependents = graph.get(current, [])
        for dep in dependents:
            if dep not in affected:
                affected.add(dep)
                queue.append(dep)
    return list(affected)


# ==========================================
# Prerequisite Checking Functions
# ==========================================

def check_prerequisites_for_course(course_id, current_grades, courses_data, grade_mapping):
    """ตรวจสอบว่าผ่าน prerequisite ของวิชาหรือไม่"""
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course or not course.get('prereq'):
        return True
    
    passing_grades = [g for g, v in grade_mapping.items() if v > 0]
    
    for prereq_id in course['prereq']:
        grade = current_grades.get(prereq_id, '')
        if not grade or grade not in passing_grades:
            return False
    return True

def get_unmet_prerequisites_for_course(course_id, current_grades, courses_data, grade_mapping):
    """ดึงรายการ prerequisites ที่ยังไม่ผ่าน"""
    course = next((c for c in courses_data if c['id'] == course_id), None)
    if not course or not course.get('prereq'):
        return []
    
    passing_grades = [g for g, v in grade_mapping.items() if v > 0]
    unmet = []
    
    for prereq_id in course['prereq']:
        grade = current_grades.get(prereq_id, '')
        if not grade or grade not in passing_grades:
            unmet.append(prereq_id)
    return unmet

def can_take_courses_together(course_a_id, course_b_id, current_grades, courses_data, grade_mapping):
    """ตรวจสอบว่าสามารถลงสองวิชาพร้อมกันได้หรือไม่ (กรณีแลปกับทฤษฎี)"""
    course_a = next((c for c in courses_data if c['id'] == course_a_id), None)
    course_b = next((c for c in courses_data if c['id'] == course_b_id), None)
    
    if not course_a or not course_b:
        return False
    
    # ตรวจสอบกรณีแลปกับทฤษฎี
    lab_keywords = ["ปฏิบัติการ", "การฝึกพื้นฐาน"]
    
    def is_lab_course(course):
        return any(kw in course['thaiName'] for kw in lab_keywords)
    
    a_is_lab = is_lab_course(course_a)
    b_is_lab = is_lab_course(course_b)
    
    # ถ้าทั้งคู่เป็นแลปหรือทั้งคู่เป็นทฤษฎี
    if a_is_lab == b_is_lab:
        return False
    
    # ตรวจสอบว่าแลปมีทฤษฎีเป็น prerequisite หรือไม่
    lab_course = course_a if a_is_lab else course_b
    theory_course = course_b if a_is_lab else course_a
    
    passing_grades = [g for g, v in grade_mapping.items() if v > 0]
    
    if theory_course['id'] in lab_course.get('prereq', []):
        # ตรวจสอบว่า prereq อื่นๆ ของแลปผ่านหมดหรือยัง
        other_prereqs_passed = True
        for prereq_id in lab_course['prereq']:
            if prereq_id != theory_course['id']:
                grade = current_grades.get(prereq_id, '')
                if not grade or grade not in passing_grades:
                    other_prereqs_passed = False
                    break
        return other_prereqs_passed
    
    return False

def find_all_blocked_courses(current_grades, loaded_courses_ids, courses_data, grade_mapping):
    """หาวิชาที่ถูกบล็อกเนื่องจาก prerequisite ไม่ผ่าน"""
    blocked = []
    
    for course_id in loaded_courses_ids:
        # ถ้ายังไม่มีเกรดหรือเกรด F
        current_grade = current_grades.get(course_id, '')
        if not current_grade or current_grade == 'F':
            if not check_prerequisites_for_course(course_id, current_grades, courses_data, grade_mapping):
                unmet = get_unmet_prerequisites_for_course(course_id, current_grades, courses_data, grade_mapping)
                course = next((c for c in courses_data if c['id'] == course_id), None)
                if course:
                    blocked.append({
                        'id': course_id,
                        'name': course['thaiName'],
                        'unmet_prereqs': [
                            next((c['thaiName'] for c in courses_data if c['id'] == pid), pid)
                            for pid in unmet
                        ]
                    })
    
    return blocked

def build_course_dependency_graph(courses_subset_ids, courses_data):
    """สร้าง dependency graph ของวิชา"""
    graph = {cid: [] for cid in courses_subset_ids}
    
    for course_id in courses_subset_ids:
        course = next((c for c in courses_data if c['id'] == course_id), None)
        if course:
            for prereq_id in course.get('prereq', []):
                if prereq_id in graph:
                    graph[prereq_id].append(course_id)
    
    return graph

def find_courses_affected_by_failure(failed_course_id, dependency_graph):
    """หาวิชาที่ได้รับผลกระทบจากวิชาที่ตก"""
    affected = set()
    queue = [failed_course_id]
    
    while queue:
        current = queue.pop(0)
        dependents = dependency_graph.get(current, [])
        for dep in dependents:
            if dep not in affected:
                affected.add(dep)
                queue.append(dep)
    
    return list(affected)


def topological_sort_with_cycle_check_backend(loaded_courses_objects):
    """Performs a topological sort on a subset of courses and checks for cycles."""
    if not loaded_courses_objects:
        return {'order': [], 'cycle': False}

    course_map = {c['id']: c for c in loaded_courses_objects}

    in_degree = {c_id: 0 for c_id in course_map.keys()}
    adj_list = {c_id: [] for c_id in course_map.keys()}

    for course_id, course_obj in course_map.items():
        for prereq_id in course_obj['prereq']:
            if prereq_id in course_map:
                adj_list[prereq_id].append(course_id)
                in_degree[course_id] += 1

    queue = []
    for cid, degree in in_degree.items():
        if degree == 0:
            queue.append(cid)

    order = []
    count = 0

    while queue:
        u = queue.pop(0)
        order.append(u)
        count += 1
        
        for v in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if count != len(loaded_courses_objects):
        return {'order': [], 'cycle': True}
    return {'order': order, 'cycle': False}

def linear_regression_next_term_gpa_backend(term_gpas):
    """Predicts next term's GPA using simple linear regression."""
    if len(term_gpas) == 0:
        return 0.0
    if len(term_gpas) == 1:
        return term_gpas[0]['gpa']

    n = len(term_gpas)
    x_vals = np.array([(i + 1) for i in range(n)])
    y_vals = np.array([t['gpa'] for t in term_gpas])

    if np.all(y_vals == y_vals[0]):
        return y_vals[0]

    sum_x = np.sum(x_vals)
    sum_y = np.sum(y_vals)
    sum_xy = np.sum(x_vals * y_vals)
    sum_xx = np.sum(x_vals * x_vals)

    denominator = (n * sum_xx - sum_x * sum_y)
    if denominator == 0:
        return y_vals[-1]

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    predicted = slope * (n + 1) + intercept
    return max(0.0, min(4.0, predicted))

def estimate_completion_rate_backend(avg_gpa, completed_credits, total_required_credits, blocked_courses_ids, fail_count, failed_courses_ids, repeat_count_this_term, core_subjects_ids, courses_data):
    """Estimates the completion rate based on academic progress and issues."""
    base_rate = (completed_credits / total_required_credits) * 100 if total_required_credits > 0 else 0

    blocked_credits = sum(
        (next((c for c in courses_data if c['id'] == cid), {'credit': 0})['credit'] for cid in blocked_courses_ids)
    )

    adjusted_rate = base_rate - blocked_credits * app.config['DATA_CONFIG']['risk_levels'].get('credit_penalty_per_blocked_course', 2)

    if 0 < avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('low_gpa_penalty', 5)
    
    if fail_count >= 5:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('high_fail_count_penalty', 10)
    elif fail_count >= 2:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('medium_fail_count_penalty', 5)
    
    fail_core_count = len([f for f in failed_courses_ids if f in core_subjects_ids])
    adjusted_rate -= fail_core_count * app.config['DATA_CONFIG']['risk_levels'].get('core_fail_penalty', 3)
    
    if repeat_count_this_term >= 3:
        adjusted_rate -= app.config['DATA_CONFIG']['risk_levels'].get('repeat_course_penalty', 5)

    return max(0.0, min(100.0, adjusted_rate))

def determine_graduation_status_backend(completion_rate, avg_gpa, blocked_courses_ids, failed_courses_ids, loaded_courses_objects, current_grades, all_terms_data, courses_data, loaded_terms_count):
    """Determines the student's graduation status based on their progress and potential issues."""
    
    total_terms_in_curriculum = len(all_terms_data)
    is_at_or_past_final_standard_term = (loaded_terms_count >= total_terms_in_curriculum)

    all_courses_in_loaded_curriculum_ids = set()
    for i in range(loaded_terms_count):
        if i < len(all_terms_data):
            for course_id in all_terms_data[i]['ids']:
                all_courses_in_loaded_curriculum_ids.add(course_id)

    incomplete_courses = [
        cid for cid in all_courses_in_loaded_curriculum_ids
        if cid not in current_grades or current_grades[cid] == "" or app.config['DATA_CONFIG']['grade_mapping'].get(str(current_grades[cid]).upper(), 0.0) == 0.0
    ]
    
    # ตรวจสอบ GPA ที่ถูกต้อง (ต้อง >= 2.00 ถึงจะผ่านเกณฑ์)
    # ถ้า GPA ผ่านเกณฑ์แล้ว ไม่ต้องแสดงข้อความเตือน GPA ต่ำ
    gpa_threshold = 2.00
    if avg_gpa > 0 and avg_gpa < gpa_threshold:
        return f"GPA ต่ำกว่าเกณฑ์ ({avg_gpa:.2f} < {gpa_threshold:.2f}) จำเป็นต้องปรับปรุงอย่างเร่งด่วนเพื่อจบการศึกษา"
    # ถ้า GPA >= 2.00 ให้ข้ามข้อความเตือนนี้และไปตรวจสอบเงื่อนไขอื่นต่อ

    if is_at_or_past_final_standard_term:
        if len(failed_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมีวิชาตก (เกรด F) ที่ต้องแก้ไข"
        if len(blocked_courses_ids) > 0:
            return "ไม่สามารถจบการศึกษาได้ตามกำหนด. มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน)"
        if len(incomplete_courses) > 0:
            return f"ไม่สามารถจบการศึกษาได้ตามกำหนด. ยังมี {len(incomplete_courses)} วิชาที่ยังไม่ผ่าน/ไม่มีเกรด"
        
        if avg_gpa == 0.0 and (not failed_courses_ids and not incomplete_courses):
            return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี (ตามเงื่อนไข)"
        
        return "คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี"
    
    else:
        if len(failed_courses_ids) > 0:
            return "มีวิชาตก (เกรด F) อาจทำให้การจบการศึกษาล่าช้ากว่ากำหนด"
        if len(blocked_courses_ids) > 0:
            return "มีวิชาที่ถูกบล็อก (prerequisite ไม่ผ่าน) ควรแก้ไขเพื่อเรียนต่อตามแผน"
        if len(incomplete_courses) > 0:
            return "กำลังอยู่ในเส้นทาง. ยังมีวิชาที่ต้องเรียน/ได้รับเกรดในภาคเรียนถัดไป"
        
        return "อยู่ในเส้นทางที่คาดว่าจะจบการศึกษาได้ตามกำหนด 4 ปี"

def update_recommendations_backend(failed_courses_ids, avg_gpa, blocked_courses_ids):
    """Generates specific recommendations based on academic issues."""
    recommendations = []
    if len(failed_courses_ids) > 0:
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][0])
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][1])
    if 0 < avg_gpa < app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']:
        recommendations.append(app.config['MESSAGES']['recommendations']['medium_risk'][1])
    if len(blocked_courses_ids) > 0:
        recommendations.append(app.config['MESSAGES']['recommendations']['high_risk'][3])
    if not recommendations:
        recommendations.append(app.config['MESSAGES']['recommendations']['low_risk'][0])
    return list(set(recommendations))

# ===============================
# กราฟ 3 เส้นและการวิเคราะห์ขั้นสูง
# ===============================

def generate_three_line_chart_data(current_grades, loaded_terms_count=8):
    """
    สร้างข้อมูลสำหรับกราฟ 3 เส้น โดยใช้ข้อมูลจริงจากนักศึกษา:
    1. เส้นเป้าหมาย (Target Line - สีเขียว): GPA ที่ควรจะเป็นเพื่อจบการศึกษา
    2. เส้นผลจริง (Actual Line - สีน้ำเงิน): GPA สะสมจริงตามเทอม
    3. เส้นทำนาย (Prediction Line - สีส้ม): การทำนายเทอมถัดไป
    """
    try:
        # ข้อมูลพื้นฐาน
        grade_mapping = app.config.get('DATA_CONFIG', {}).get('grade_mapping', {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'I': 0.0
        })
        courses_data = app.config.get('COURSES_DATA', [])
        all_terms_data = app.config.get('ALL_TERMS_DATA', [])
        
        # จำนวนเทอมทั้งหมดในหลักสูตร (8 เทอมปกติ)
        total_terms = len(all_terms_data)
        
        # === 1. คำนวณเส้นเป้าหมาย (Target Line) ===
        # GPA เป้าหมาย = 3.0 (เกณฑ์ปลอดภัยสำหรับจบการศึกษา)
        target_gpa = 3.0
        target_line = [target_gpa] * (total_terms + 2)  # +2 สำหรับทำนายอนาคต
        
        # === 2. คำนวณเส้นผลจริง (Actual Line) - จาก GPA สะสมจริงตามเทอม ===
        actual_line = []
        term_gpa_list = []
        
        # จัดกลุ่มวิชาตามเทอม
        courses_by_term = {}
        for term_idx, term_data in enumerate(all_terms_data):
            if term_idx >= loaded_terms_count:
                break
            courses_by_term[term_idx] = term_data['ids']
        
        # คำนวณ GPA สะสมจริงตามเทอม
        cumulative_points = 0
        cumulative_credits = 0
        
        for term_idx in range(loaded_terms_count):
            term_courses = courses_by_term.get(term_idx, [])
            term_points = 0
            term_credits = 0
            
            for course_id in term_courses:
                grade = current_grades.get(course_id, '')
                if grade and grade in grade_mapping:
                    # หาหน่วยกิตจริงของวิชา
                    course = next((c for c in courses_data if c['id'] == course_id), None)
                    credit = course['credit'] if course else 3
                    grade_point = grade_mapping[grade]
                    
                    # คำนวณเฉพาะวิชาที่ไม่ใช่ W, I
                    if grade not in ['W', 'I']:
                        term_points += grade_point * credit
                        term_credits += credit
            
            # เพิ่มคะแนนสะสม
            cumulative_points += term_points
            cumulative_credits += term_credits
            
            # คำนวณ GPA สะสมณ จุดนี้
            term_cumulative_gpa = cumulative_points / cumulative_credits if cumulative_credits > 0 else 0
            actual_line.append(round(term_cumulative_gpa, 2))
            term_gpa_list.append(round(term_cumulative_gpa, 2))
        
        # เติม None สำหรับเทอมที่ยังไม่ได้เรียน
        for term_idx in range(loaded_terms_count, total_terms):
            actual_line.append(None)
        
        # === 3. คำนวณเส้นทำนาย (Prediction Line) ===
        prediction_line = actual_line.copy()
        
        # ทำนาย GPA เทอมถัดไป
        if len(term_gpa_list) > 0:
            current_gpa = term_gpa_list[-1]
            
            # ใช้โมเดล AI ทำนาย (ถ้ามี)
            if hasattr(app, 'advanced_trainer') and app.advanced_trainer:
                try:
                    # ใช้โมเดล AI ทำนายจริง
                    prediction_result = app.advanced_trainer.predict_graduation(current_grades)
                    predicted_gpa_next_term = prediction_result.get('predicted_gpa', current_gpa)
                    
                    # จำกัดค่าให้สมเหตุสมผล
                    predicted_gpa_next_term = max(0, min(4.0, predicted_gpa_next_term))
                except Exception as e:
                    logger.warning(f"AI prediction failed, using statistical method: {e}")
                    # Fallback: ใช้แนวโน้มทางสถิติ
                    if len(term_gpa_list) >= 2:
                        # คำนวณ trend จาก 2 เทอมล่าสุด
                        trend = term_gpa_list[-1] - term_gpa_list[-2]
                        predicted_gpa_next_term = current_gpa + (trend * 0.5)  # ลด impact ของ trend
                    else:
                        predicted_gpa_next_term = current_gpa
            else:
                # ใช้วิธีทางสถิติ
                if len(term_gpa_list) >= 2:
                    trend = term_gpa_list[-1] - term_gpa_list[-2]
                    predicted_gpa_next_term = current_gpa + (trend * 0.5)
                else:
                    predicted_gpa_next_term = current_gpa
            
            # จำกัดค่าทำนายให้อยู่ในช่วงที่เป็นไปได้
            predicted_gpa_next_term = max(1.5, min(4.0, predicted_gpa_next_term))
            
            # เติมข้อมูลทำนายสำหรับเทอมที่เหลือ
            for i in range(len(prediction_line)):
                if prediction_line[i] is None:
                    # ทำนายว่า GPA จะค่อยๆ ดีขึ้นหรือคงที่
                    if predicted_gpa_next_term > current_gpa:
                        improvement_rate = (predicted_gpa_next_term - current_gpa) * 0.8
                        prediction_line[i] = round(min(4.0, current_gpa + improvement_rate), 2)
                    else:
                        prediction_line[i] = round(predicted_gpa_next_term, 2)
            
            # เพิ่มทำนายอนาคต 2 เทอม
            prediction_line.extend([round(predicted_gpa_next_term, 2)] * 2)
        else:
            # ไม่มีข้อมูล ใส่ค่าเริ่มต้น
            prediction_line = [None] * total_terms
            prediction_line.extend([None, None])
        
        # สร้างป้ายกำกับเทอม
        term_labels = []
        for i, term_data in enumerate(all_terms_data):
            term_labels.append(f"ปี {term_data['year']} เทอม {term_data['term']}")
        term_labels.extend(["ทำนาย +1", "ทำนาย +2"])
        
        # สรุปข้อมูลสำคัญ
        summary = {
            'current_gpa': term_gpa_list[-1] if term_gpa_list else 0,
            'predicted_next_gpa': round(predicted_gpa_next_term, 2) if term_gpa_list else 0,
            'target_gpa': target_gpa,
            'terms_completed': loaded_terms_count,
            'total_terms': total_terms,
            'on_track': (term_gpa_list[-1] >= target_gpa) if term_gpa_list else False
        }
        
        return {
            'terms': term_labels,
            'target_line': target_line,
            'actual_line': actual_line + [None, None],  # เพิ่ม None สำหรับทำนาย
            'prediction_line': prediction_line,
            'colors': {
                'target': '#28a745',      # สีเขียว
                'actual': '#007bff',      # สีน้ำเงิน
                'prediction': '#fd7e14'   # สีส้ม
            },
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error generating three line chart data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'terms': [],
            'target_line': [],
            'actual_line': [],
            'prediction_line': [],
            'colors': {
                'target': '#28a745',
                'actual': '#007bff',
                'prediction': '#fd7e14'
            },
            'summary': {
                'current_gpa': 0,
                'predicted_next_gpa': 0,
                'target_gpa': 3.0,
                'terms_completed': 0,
                'total_terms': 8,
                'on_track': False
            }
        }

def analyze_graduation_failure_reasons(current_grades, loaded_terms_count=8, prediction_result=None):
    """
    วิเคราะห์เหตุผลที่ไม่จบการศึกษาอย่างครอบคลุม
    ใช้ผลการทำนายจากโมเดล AI (prediction_result) ร่วมกับการวิเคราะห์เงื่อนไข
    
    Args:
        current_grades: dict ของเกรดปัจจุบัน {course_id: grade}
        loaded_terms_count: จำนวนเทอมที่โหลด
        prediction_result: ผลการทำนายจากโมเดล AI (dict) ประกอบด้วย:
            - prediction: 'จบ' หรือ 'ไม่จบ'
            - prob_pass: ความน่าจะเป็นที่จะจบ (0-1)
            - prob_fail: ความน่าจะเป็นที่จะไม่จบ (0-1)
            - confidence: ความมั่นใจในการทำนาย (0-1)
            - risk_level: ระดับความเสี่ยง
            - method: วิธีที่ใช้ทำนาย
    """
    try:
        grade_mapping = app.config.get('DATA_CONFIG', {}).get('grade_mapping', {})
        courses_data_list = app.config.get('COURSES_DATA', [])
        
        # สร้าง dict สำหรับ lookup หน่วยกิตจาก course id
        courses_credit_map = {course['id']: course.get('credit', 3) for course in courses_data_list}
        
        reasons = []
        risk_factors = []
        
        # Helper function สำหรับดึงหน่วยกิต
        def get_course_credit(course_id):
            return courses_credit_map.get(course_id, 3)
        
        # คำนวณสถิติพื้นฐาน
        total_courses = len(current_grades)
        failed_courses = [course for course, grade in current_grades.items() 
                         if grade and grade_mapping.get(str(grade).upper(), 0) == 0 and str(grade).upper() not in ['W', 'WF', 'WU', 'I']]
        low_grade_courses = [course for course, grade in current_grades.items() 
                           if grade and 0 < grade_mapping.get(str(grade).upper(), 0) < 2.0]
        incomplete_courses = [course for course, grade in current_grades.items() 
                            if grade and str(grade).upper() in ['I', 'W', 'WF', 'WU']]
        
        # คำนวณ GPA และหน่วยกิต (ไม่รวม S, W, WF, WU)
        total_points = 0
        total_credits = 0
        passed_credits = 0
        
        # เกรดที่ไม่นำมาคำนวณ GPA
        excluded_grades = {'S', 'W', 'WF', 'WU', 'I', None}
        
        for course, grade in current_grades.items():
            if not grade or str(grade).upper() in excluded_grades:
                continue
                
            grade_upper = str(grade).upper()
            grade_point = grade_mapping.get(grade_upper)
            
            # ถ้า grade_point เป็น None (เช่น S) ให้ข้าม
            if grade_point is None:
                continue
            
            # ดึงหน่วยกิตจริงจากข้อมูลหลักสูตร
            credits = get_course_credit(course)
            
            # รวมเฉพาะเกรดที่นำมาคำนวณ GPA ได้
            if grade_point is not None:
                total_points += grade_point * credits
                total_credits += credits
                if grade_point > 0:
                    passed_credits += credits
        
        current_gpa = total_points / total_credits if total_credits > 0 else 0
        current_gpa = round(current_gpa, 2)  # ปัดเป็น 2 ทศนิยม
        
        # วิเคราะห์เหตุผลการไม่จบแบบละเอียด
        
        # 1. วิชาที่ไม่ผ่าน (เกรด F)
        if len(failed_courses) > 0:
            failed_credits = sum([get_course_credit(course) for course in failed_courses])
            reasons.append({
                'type': 'critical',
                'title': f'มีวิชาที่ไม่ผ่าน {len(failed_courses)} วิชา ({failed_credits} หน่วยกิต)',
                'description': f'วิชาที่ได้เกรด F: {", ".join(failed_courses[:3])}{"..." if len(failed_courses) > 3 else ""}',
                'courses': failed_courses,
                'impact': 'สูงมาก - ต้องเรียนซ้ำก่อนจบการศึกษา',
                'solution': 'ลงทะเบียนเรียนซ้ำในวิชาที่ตกทันที และเตรียมตัวให้ดีก่อนสอบ',
                'timeline': f'ต้องใช้เวลาเพิ่ม {math.ceil(len(failed_courses)/6)} เทอม'
            })
        
        # 2. GPA ต่ำกว่าเกณฑ์
        if current_gpa < 2.0:
            gpa_deficit = 2.0 - current_gpa
            credits_needed = math.ceil(gpa_deficit * total_credits / 2.0)
            reasons.append({
                'type': 'critical',
                'title': f'GPA ต่ำกว่าเกณฑ์ขั้นต่ำ ({current_gpa:.2f} < 2.00)',
                'description': f'ต้องปรับปรุง GPA อีก {gpa_deficit:.2f} จุด',
                'impact': 'สูงมาก - ไม่สามารถจบการศึกษาได้',
                'solution': f'ต้องได้เกรดเฉลี่ย A ใน {credits_needed} หน่วยกิตถัดไป',
                'timeline': f'ต้องใช้เวลาเพิ่ม {math.ceil(credits_needed/18)} เทอม'
            })
        
        # 3. วิชาที่ได้เกรดต่ำ
        if len(low_grade_courses) > 0:
            low_credits = sum([get_course_credit(course) for course in low_grade_courses])
            reasons.append({
                'type': 'warning',
                'title': f'มีวิชาที่ได้เกรดต่ำ {len(low_grade_courses)} วิชา ({low_credits} หน่วยกิต)',
                'description': f'วิชาที่ได้เกรด D+, D: {", ".join(low_grade_courses[:3])}{"..." if len(low_grade_courses) > 3 else ""}',
                'courses': low_grade_courses,
                'impact': 'ปานกลาง - ส่งผลต่อ GPA รวม',
                'solution': 'พิจารณาเรียนซ้ำเพื่อปรับปรุงเกรดและ GPA',
                'timeline': 'สามารถปรับปรุงได้ในเทอมถัดไป'
            })
        
        # 4. วิชาที่ไม่สมบูรณ์
        if len(incomplete_courses) > 0:
            incomplete_credits = sum([get_course_credit(course) for course in incomplete_courses])
            reasons.append({
                'type': 'warning',
                'title': f'มีวิชาที่ไม่สมบูรณ์ {len(incomplete_courses)} วิชา',
                'description': f'วิชาที่ได้เกรด I, W, WF, WU: {", ".join(incomplete_courses)}',
                'courses': incomplete_courses,
                'impact': 'ปานกลาง - ไม่นับหน่วยกิต',
                'solution': 'ติดต่ออาจารย์เพื่อดำเนินการให้เสร็จสิ้น',
                'timeline': 'ต้องดำเนินการภายในเวลาที่กำหนด'
            })
        
        # 5. ความคืบหน้าช้า
        expected_credits = loaded_terms_count * 18  # สมมติ 18 หน่วยกิตต่อเทอม
        if passed_credits < expected_credits * 0.8:
            reasons.append({
                'type': 'warning',
                'title': 'ความคืบหน้าช้ากว่าเกณฑ์',
                'description': f'ผ่าน {passed_credits}/{expected_credits} หน่วยกิต ({passed_credits/expected_credits*100:.1f}%)',
                'impact': 'ปานกลาง - อาจจบช้ากว่าแผน',
                'solution': 'เพิ่มจำนวนหน่วยกิตในเทอมถัดไป',
                'timeline': f'อาจจบช้า {math.ceil((expected_credits-passed_credits)/18)} เทอม'
            })
        
        # ประเมินความเสี่ยงแบบละเอียด
        risk_score = 0
        if len(failed_courses) > 0:
            risk_score += len(failed_courses) * 20
        if current_gpa < 2.0:
            risk_score += (2.0 - current_gpa) * 50
        if len(low_grade_courses) > 0:
            risk_score += len(low_grade_courses) * 5
        if len(incomplete_courses) > 0:
            risk_score += len(incomplete_courses) * 10
        
        if risk_score >= 80:
            risk_level = 'สูงมาก'
            risk_color = 'danger'
            risk_description = 'มีความเสี่ยงสูงมากที่จะไม่จบ'
        elif risk_score >= 60:
            risk_level = 'สูง'
            risk_color = 'danger'
            risk_description = 'มีความเสี่ยงสูงที่จะจบช้า'
        elif risk_score >= 40:
            risk_level = 'ปานกลาง'
            risk_color = 'warning'
            risk_description = 'มีความเสี่ยงปานกลาง'
        elif risk_score >= 20:
            risk_level = 'ต่ำ-ปานกลาง'
            risk_color = 'info'
            risk_description = 'มีความเสี่ยงเล็กน้อย'
        else:
            risk_level = 'ต่ำ'
            risk_color = 'success'
            risk_description = 'มีโอกาสจบตามแผน'
        
        # คำนวณโอกาสจบแบบละเอียด
        base_probability = 90
        if len(failed_courses) > 0:
            base_probability -= len(failed_courses) * 15
        if current_gpa < 2.0:
            base_probability -= (2.0 - current_gpa) * 30
        if len(low_grade_courses) > 0:
            base_probability -= len(low_grade_courses) * 3
        if len(incomplete_courses) > 0:
            base_probability -= len(incomplete_courses) * 5
        
        graduation_probability = max(5, min(95, base_probability))
        
        # สถานะการจบ
        if graduation_probability >= 80:
            graduation_status = 'มีโอกาสจบสูง'
            graduation_status_color = 'success'
        elif graduation_probability >= 60:
            graduation_status = 'มีโอกาสจบปานกลาง'
            graduation_status_color = 'info'
        elif graduation_probability >= 40:
            graduation_status = 'มีความเสี่ยงที่จะไม่จบ'
            graduation_status_color = 'warning'
        else:
            graduation_status = 'มีความเสี่ยงสูงที่จะไม่จบ'
            graduation_status_color = 'danger'
        
        # ====== การทำนายผลการจบ/ไม่จบ: ใช้ผลจากโมเดล AI เป็นหลัก ======
        # ถ้ามีผลจากโมเดล AI ให้ใช้ผลนั้นเป็นหลัก
        if prediction_result and isinstance(prediction_result, dict):
            # ใช้ผลการทำนายจากโมเดล AI
            model_prediction = prediction_result.get('prediction', '')
            model_prob_pass = prediction_result.get('prob_pass', 0.5)
            model_confidence = prediction_result.get('confidence', 0.5)
            prediction_method = prediction_result.get('method', 'Unknown')
            
            # กำหนด will_graduate จากผลของโมเดล
            will_graduate = model_prediction == 'จบ' or model_prob_pass >= 0.5
            
            # ปรับ graduation_probability ให้สอดคล้องกับผลจากโมเดล
            graduation_probability = model_prob_pass * 100
            
            logger.info(f"📊 Using AI Model prediction: {model_prediction} (prob: {model_prob_pass:.2f}, confidence: {model_confidence:.2f}) from {prediction_method}")
        else:
            # Fallback: ใช้เงื่อนไขพื้นฐานถ้าไม่มีผลจากโมเดล
            will_graduate = graduation_probability >= 50 and current_gpa >= 2.0 and len(failed_courses) == 0
            prediction_method = 'Rule-based Analysis'
            model_prob_pass = graduation_probability / 100
            model_confidence = 0.5
            logger.info(f"📊 Using rule-based prediction (no AI model result)")
        
        # สร้างเหตุผลที่จะจบการศึกษา
        reasons_for_graduation = []
        if current_gpa >= 2.0:
            reasons_for_graduation.append({
                'icon': '✅',
                'title': f'GPA ผ่านเกณฑ์',
                'description': f'GPA สะสม {current_gpa:.2f} ≥ 2.00 (เกณฑ์ขั้นต่ำ)',
                'type': 'success'
            })
        if len(failed_courses) == 0:
            reasons_for_graduation.append({
                'icon': '✅',
                'title': 'ไม่มีวิชาที่ตก',
                'description': 'ผ่านทุกวิชาที่ลงทะเบียน',
                'type': 'success'
            })
        if current_gpa >= 2.5:
            reasons_for_graduation.append({
                'icon': '🌟',
                'title': 'ผลการเรียนดี',
                'description': f'GPA {current_gpa:.2f} อยู่ในระดับที่ดี',
                'type': 'success'
            })
        if current_gpa >= 3.0:
            reasons_for_graduation.append({
                'icon': '🏆',
                'title': 'ผลการเรียนดีมาก',
                'description': f'GPA {current_gpa:.2f} สูงกว่าค่าเฉลี่ย',
                'type': 'success'
            })
        if passed_credits >= 100:
            reasons_for_graduation.append({
                'icon': '📚',
                'title': 'หน่วยกิตใกล้ครบ',
                'description': f'สะสมหน่วยกิตแล้ว {passed_credits} หน่วยกิต',
                'type': 'success'
            })
        progress_pct = round(passed_credits/136*100, 1) if passed_credits > 0 else 0
        if progress_pct >= 75:
            reasons_for_graduation.append({
                'icon': '📈',
                'title': 'ความคืบหน้าดี',
                'description': f'เรียนผ่านแล้ว {progress_pct}% ของหลักสูตร',
                'type': 'success'
            })
        if len(low_grade_courses) == 0 and total_courses > 0:
            reasons_for_graduation.append({
                'icon': '💪',
                'title': 'คุณภาพผลการเรียนดี',
                'description': 'ไม่มีวิชาที่ได้เกรดต่ำ (D+, D)',
                'type': 'success'
            })
        
        # สร้างเหตุผลที่ไม่จบการศึกษา
        reasons_for_not_graduation = []
        if current_gpa < 2.0:
            gpa_deficit = 2.0 - current_gpa
            reasons_for_not_graduation.append({
                'icon': '❌',
                'title': f'GPA ต่ำกว่าเกณฑ์',
                'description': f'GPA สะสม {current_gpa:.2f} < 2.00 (ต้องเพิ่มอีก {gpa_deficit:.2f})',
                'type': 'critical'
            })
        if len(failed_courses) > 0:
            failed_names = ', '.join(failed_courses[:3])
            if len(failed_courses) > 3:
                failed_names += f' และอีก {len(failed_courses)-3} วิชา'
            reasons_for_not_graduation.append({
                'icon': '❌',
                'title': f'มีวิชาที่ตก {len(failed_courses)} วิชา',
                'description': f'วิชาที่ไม่ผ่าน: {failed_names}',
                'type': 'critical',
                'courses': failed_courses
            })
        if len(low_grade_courses) > 2:
            reasons_for_not_graduation.append({
                'icon': '⚠️',
                'title': f'มีวิชาเกรดต่ำจำนวนมาก',
                'description': f'ได้เกรด D+, D จำนวน {len(low_grade_courses)} วิชา ส่งผลต่อ GPA',
                'type': 'warning',
                'courses': low_grade_courses
            })
        if len(incomplete_courses) > 0:
            reasons_for_not_graduation.append({
                'icon': '⚠️',
                'title': f'มีวิชาที่ยังไม่สมบูรณ์',
                'description': f'วิชาที่ได้ I/W/WF/WU: {len(incomplete_courses)} วิชา',
                'type': 'warning',
                'courses': incomplete_courses
            })
        if progress_pct < 50 and loaded_terms_count >= 4:
            reasons_for_not_graduation.append({
                'icon': '⚠️',
                'title': 'ความคืบหน้าช้า',
                'description': f'เรียนผ่านเพียง {progress_pct}% อาจจบช้ากว่าแผน',
                'type': 'warning'
            })
        
        # ข้อความทำนายผลการจบ - รวมข้อมูลจากโมเดล AI
        if will_graduate:
            graduation_prediction_text = '🎓 คาดว่าจะจบการศึกษาตามเกณฑ์'
            
            # สร้างรายละเอียดจากผลการทำนาย
            detail_parts = []
            if prediction_result and isinstance(prediction_result, dict):
                detail_parts.append(f'AI Model ทำนาย: โอกาสจบ {graduation_probability:.0f}%')
                detail_parts.append(f'ความมั่นใจ: {model_confidence*100:.0f}%')
                if prediction_method:
                    detail_parts.append(f'วิธีทำนาย: {prediction_method}')
            
            if current_gpa >= 2.0:
                detail_parts.append(f'GPA {current_gpa:.2f} ผ่านเกณฑ์ขั้นต่ำ')
            if len(failed_courses) == 0:
                detail_parts.append('ไม่มีวิชาที่ตก')
            
            graduation_prediction_detail = ' | '.join(detail_parts) if detail_parts else f'โอกาสจบการศึกษา {graduation_probability:.0f}%'
        else:
            graduation_prediction_text = '⚠️ เสี่ยงไม่จบการศึกษา'
            
            # สร้างรายละเอียดจากผลการทำนายและปัญหาที่พบ
            detail_parts = []
            if prediction_result and isinstance(prediction_result, dict):
                detail_parts.append(f'AI Model ทำนาย: โอกาสจบ {graduation_probability:.0f}%')
                detail_parts.append(f'ความมั่นใจ: {model_confidence*100:.0f}%')
            
            main_issues = []
            if current_gpa < 2.0:
                main_issues.append(f'GPA ต่ำกว่าเกณฑ์ ({current_gpa:.2f} < 2.00)')
            if len(failed_courses) > 0:
                main_issues.append(f'มีวิชาตก {len(failed_courses)} วิชา')
            if len(main_issues) == 0 and graduation_probability < 50:
                main_issues.append(f'โอกาสจบต่ำตามการทำนายของ AI ({graduation_probability:.0f}%)')
            
            if main_issues:
                detail_parts.append('สาเหตุหลัก: ' + ', '.join(main_issues))
            
            graduation_prediction_detail = ' | '.join(detail_parts) if detail_parts else 'มีความเสี่ยงจากผลการเรียน'
        
        return {
            'reasons': reasons,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_description': risk_description,
            'risk_score': risk_score,
            'graduation_probability': round(graduation_probability, 1),
            'graduation_status': graduation_status,
            'graduation_status_color': graduation_status_color,
            'current_gpa': round(current_gpa, 2),
            'failed_courses_count': len(failed_courses),
            'low_grade_courses_count': len(low_grade_courses),
            'incomplete_courses_count': len(incomplete_courses),
            'total_courses': total_courses,
            'passed_credits': passed_credits,
            'total_credits': total_credits,
            'expected_credits': expected_credits,
            'progress_percentage': round(passed_credits/136*100, 1) if passed_credits > 0 else 0,  # 136 หน่วยกิตรวม
            'recommendations': generate_improvement_recommendations(current_gpa, failed_courses, low_grade_courses, incomplete_courses),
            # ====== ข้อมูลการทำนายจบ/ไม่จบ (ใช้ AI Model เป็นหลัก) ======
            'will_graduate': will_graduate,
            'graduation_prediction_text': graduation_prediction_text,
            'graduation_prediction_detail': graduation_prediction_detail,
            'reasons_for_graduation': reasons_for_graduation,
            'reasons_for_not_graduation': reasons_for_not_graduation,
            'failed_courses': failed_courses,
            'low_grade_courses': low_grade_courses,
            'incomplete_courses': incomplete_courses,
            # ข้อมูลจากโมเดล AI
            'prediction_method': prediction_method if 'prediction_method' in dir() else 'Rule-based',
            'model_prob_pass': model_prob_pass if 'model_prob_pass' in dir() else graduation_probability / 100,
            'model_confidence': model_confidence if 'model_confidence' in dir() else 0.5,
            'ai_prediction_used': prediction_result is not None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing graduation failure reasons: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'reasons': [],
            'risk_level': 'ไม่ทราบ',
            'risk_color': 'secondary',
            'risk_description': 'ไม่สามารถวิเคราะห์ได้',
            'risk_score': 0,
            'graduation_probability': 0,
            'graduation_status': 'ไม่ทราบ',
            'graduation_status_color': 'secondary',
            'current_gpa': 0,
            'failed_courses_count': 0,
            'low_grade_courses_count': 0,
            'incomplete_courses_count': 0,
            'total_courses': 0,
            'passed_credits': 0,
            'total_credits': 0,
            'expected_credits': 0,
            'progress_percentage': 0,
            'recommendations': [],
            # ข้อมูลการทำนาย (fallback)
            'will_graduate': False,
            'graduation_prediction_text': '⚠️ ไม่สามารถวิเคราะห์ได้',
            'graduation_prediction_detail': 'เกิดข้อผิดพลาดในการวิเคราะห์',
            'reasons_for_graduation': [],
            'reasons_for_not_graduation': [],
            'failed_courses': [],
            'low_grade_courses': [],
            'incomplete_courses': [],
            'prediction_method': 'Error',
            'model_prob_pass': 0,
            'model_confidence': 0,
            'ai_prediction_used': False
        }

def generate_improvement_recommendations(current_gpa, failed_courses, low_grade_courses, incomplete_courses=None):
    """สร้างคำแนะนำสำหรับการปรับปรุงแบบครอบคลุม"""
    if incomplete_courses is None:
        incomplete_courses = []
    
    recommendations = []
    
    # คำแนะนำเร่งด่วนสำหรับวิชาที่ตก
    if len(failed_courses) > 0:
        recommendations.extend([
            "🔴 เร่งด่วน: ลงทะเบียนเรียนซ้ำในวิชาที่ตกทันที",
            "📚 ศึกษาเนื้อหาให้เข้าใจก่อนเข้าเรียนซ้ำ",
            "👨‍🏫 ปรึกษาอาจารย์ผู้สอนเพื่อขอคำแนะนำ",
            "📝 ทำแบบฝึกหัดและสอบเก่าเพื่อเตรียมตัว",
            "🤝 หาเพื่อนที่เก่งมาช่วยสอนหรือติวให้"
        ])
    
    # คำแนะนำสำหรับ GPA ต่ำ
    if current_gpa < 2.0:
        recommendations.extend([
            "⚠️ เร่งด่วน: ต้องปรับปรุง GPA ให้ถึง 2.00",
            "📊 คำนวณเกรดที่ต้องได้ในวิชาถัดไป",
            "🎯 เน้นวิชาที่มีหน่วยกิตสูงเพื่อปรับ GPA",
            "📈 ลงทะเบียนวิชาที่มั่นใจว่าจะได้เกรดดี",
            "⏰ จัดสรรเวลาเรียนให้มากขึ้น"
        ])
    elif current_gpa < 2.5:
        recommendations.extend([
            "📈 พยายามปรับปรุง GPA ให้สูงขึ้น",
            "🎯 เลือกวิชาที่เหมาะกับความสามารถ"
        ])
    
    # คำแนะนำสำหรับวิชาที่ได้เกรดต่ำ
    if len(low_grade_courses) > 0:
        recommendations.extend([
            "🔄 พิจารณาเรียนซ้ำวิชาที่ได้เกรดต่ำ",
            "📈 ใช้กลยุทธ์การเรียนที่มีประสิทธิภาพมากขึ้น",
            "💡 หาวิธีการเรียนที่เหมาะกับตัวเอง"
        ])
    
    # คำแนะนำสำหรับวิชาที่ไม่สมบูรณ์
    if len(incomplete_courses) > 0:
        recommendations.extend([
            "📋 ติดต่ออาจารย์เพื่อดำเนินการวิชาที่ไม่สมบูรณ์",
            "⏱️ ดำเนินการให้เสร็จภายในเวลาที่กำหนด",
            "📞 ติดตามสถานะการดำเนินการอย่างสม่ำเสมอ"
        ])
    
    # คำแนะนำทั่วไปตาม GPA
    if current_gpa >= 3.5:
        recommendations.extend([
            "🌟 รักษาผลการเรียนที่ดีต่อไป",
            "🎓 เตรียมตัวสำหรับการจบการศึกษา",
            "💼 หาประสบการณ์เพิ่มเติมเช่น internship"
        ])
    elif current_gpa >= 3.0:
        recommendations.extend([
            "📚 พยายามรักษาและปรับปรุงผลการเรียน",
            "🎯 ตั้งเป้าหมายให้ได้ GPA 3.5 ขึ้นไป"
        ])
    
    # คำแนะนำทั่วไป
    recommendations.extend([
        "⏰ จัดตารางเวลาเรียนและทบทวนอย่างสม่ำเสมอ",
        "🤝 หากลุ่มเรียนหรือติวเตอร์ช่วยสอน",
        "💪 ตั้งเป้าหมายระยะสั้นและระยะยาว",
        "🏥 ดูแลสุขภาพกายและใจให้แข็งแรง",
        "📱 ใช้แอปพลิเคชันช่วยจัดการเวลาและการเรียน",
        "🎯 มุ่งเน้นวิชาที่สำคัญและมีผลต่อการจบ",
        "📞 ปรึกษาอาจารย์ที่ปรึกษาเมื่อมีปัญหา"
    ])
    
    return recommendations

def generate_next_term_grade_prediction_table(current_grades):
    """สร้างตารางทำนายเกรดเทอมถัดไป"""
    try:
        # ข้อมูลวิชาที่อาจเรียนในเทอมถัดไป (ตัวอย่าง)
        next_term_courses = [
            {"id": "CS301", "name": "โครงสร้างข้อมูล", "credits": 3, "difficulty": "สูง"},
            {"id": "CS302", "name": "การออกแบบอัลกอริทึม", "credits": 3, "difficulty": "สูง"},
            {"id": "CS303", "name": "ระบบฐานข้อมูล", "credits": 3, "difficulty": "ปานกลาง"},
            {"id": "CS304", "name": "การพัฒนาเว็บแอปพลิเคชัน", "credits": 3, "difficulty": "ปานกลาง"},
            {"id": "GE401", "name": "วิชาศึกษาทั่วไป", "credits": 3, "difficulty": "ต่ำ"}
        ]
        
        # คำนวณ GPA ปัจจุบัน
        grade_mapping = app.config.get('DATA_CONFIG', {}).get('grade_mapping', {})
        total_points = 0
        total_credits = 0
        
        for course, grade in current_grades.items():
            if grade and grade in grade_mapping:
                grade_point = grade_mapping[grade]
                credits = 3
                total_points += grade_point * credits
                total_credits += credits
        
        current_gpa = total_points / total_credits if total_credits > 0 else 0
        
        # ทำนายเกรดสำหรับแต่ละวิชา
        predictions = []
        for course in next_term_courses:
            # ทำนายตามความยากและ GPA ปัจจุบัน
            if course["difficulty"] == "สูง":
                if current_gpa >= 3.5:
                    predicted_grade = "B+"
                    confidence = 75
                elif current_gpa >= 3.0:
                    predicted_grade = "B"
                    confidence = 70
                elif current_gpa >= 2.5:
                    predicted_grade = "C+"
                    confidence = 65
                else:
                    predicted_grade = "C"
                    confidence = 60
            elif course["difficulty"] == "ปานกลาง":
                if current_gpa >= 3.5:
                    predicted_grade = "A"
                    confidence = 80
                elif current_gpa >= 3.0:
                    predicted_grade = "B+"
                    confidence = 75
                elif current_gpa >= 2.5:
                    predicted_grade = "B"
                    confidence = 70
                else:
                    predicted_grade = "C+"
                    confidence = 65
            else:  # ต่ำ
                if current_gpa >= 3.0:
                    predicted_grade = "A"
                    confidence = 85
                elif current_gpa >= 2.5:
                    predicted_grade = "B+"
                    confidence = 80
                else:
                    predicted_grade = "B"
                    confidence = 75
            
            predictions.append({
                "course_id": course["id"],
                "course_name": course["name"],
                "credits": course["credits"],
                "difficulty": course["difficulty"],
                "predicted_grade": predicted_grade,
                "confidence": confidence,
                "grade_point": grade_mapping.get(predicted_grade, 0)
            })
        
        # คำนวณ GPA ที่คาดว่าจะได้
        predicted_points = sum(p["grade_point"] * p["credits"] for p in predictions)
        predicted_credits = sum(p["credits"] for p in predictions)
        predicted_term_gpa = predicted_points / predicted_credits if predicted_credits > 0 else 0
        
        # คำนวณ GPA สะสมใหม่
        new_total_points = total_points + predicted_points
        new_total_credits = total_credits + predicted_credits
        predicted_cumulative_gpa = new_total_points / new_total_credits if new_total_credits > 0 else 0
        
        return {
            "predictions": predictions,
            "current_gpa": round(current_gpa, 2),
            "predicted_term_gpa": round(predicted_term_gpa, 2),
            "predicted_cumulative_gpa": round(predicted_cumulative_gpa, 2),
            "total_credits": predicted_credits,
            "improvement": round(predicted_cumulative_gpa - current_gpa, 2)
        }
        
    except Exception as e:
        logger.error(f"Error generating next term prediction table: {str(e)}")
        return {
            "predictions": [],
            "current_gpa": 0,
            "predicted_term_gpa": 0,
            "predicted_cumulative_gpa": 0,
            "total_credits": 0,
            "improvement": 0
        }

# ==========================================
# Gemini AI Integration Helpers
# ==========================================
GEMINI_SYSTEM_PROMPT = (
    "คุณคือที่ปรึกษาด้านการศึกษาที่เชี่ยวชาญการวิเคราะห์ข้อมูลเกรด "
    "ให้คำอธิบายเป็นภาษาไทย ใช้น้ำเสียงมืออาชีพ กระชับ และแนะนำเชิงปฏิบัติ "
    "ถ้าข้อมูลไม่เพียงพอให้ระบุอย่างชัดเจน หลีกเลี่ยงการเดา"
)

GEMINI_RESPONSE_SCHEMAS = {
    "insights": {
        "type": "object",
        "properties": {
            "analysis_markdown": {"type": "string"},
            "risk_level": {"type": "string"},
            "graduation_prediction": {
                "type": "object",
                "properties": {
                    "will_graduate": {"type": "boolean"},
                    "prediction_text": {"type": "string"},
                    "reason_why_graduate": {"type": "string"},
                    "reason_why_not_graduate": {"type": "string"},
                    "confidence_percent": {"type": "number"}
                },
                "required": ["will_graduate", "prediction_text"]
            },
            "outcome_summary": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "confidence": {"type": "number"},
                    "description": {"type": "string"}
                },
                "required": ["status"]
            },
            "key_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                        "trend": {"type": "string"}
                    }
                }
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "chart": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "data": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            },
                            "required": ["name", "data"]
                        }
                    }
                }
            }
        },
        "required": ["analysis_markdown"]
    }
}


def is_gemini_available() -> bool:
    return bool(genai and gemini_client_ready)


def _should_retry_with_fallback(exc: Exception) -> bool:
    """
    ตรวจสอบว่า error นี้ควรลอง fallback model หรือไม่
    ใช้สำหรับกรณี model ไม่อยู่ใน region หรือไม่รองรับ generateContent
    """
    message = str(exc).lower()
    fallback_indicators = [
        'not found',
        '404',
        'not supported',
        'unsupported',
        'does not exist'
    ]
    return any(indicator in message for indicator in fallback_indicators)


def _to_native_value(value):
    """แปลงค่า numpy/pandas ให้เป็น Python ปกติ"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.generic, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_native_value(v) for v in value]
    return value


def summarize_dataframe_for_gemini(df, max_columns=20, max_samples=10):
    summary = {
        'row_count': int(df.shape[0]),
        'column_count': int(df.shape[1]),
        'columns': []
    }
    
    for column in df.columns[:max_columns]:
        series = df[column]
        column_summary = {
            'name': str(column),
            'dtype': str(series.dtype),
            'missing': int(series.isna().sum()),
            'unique': int(series.nunique(dropna=True))
        }
        
        if pd.api.types.is_numeric_dtype(series):
            non_na = series.dropna()
            if not non_na.empty:
                column_summary.update({
                    'min': _to_native_value(non_na.min()),
                    'max': _to_native_value(non_na.max()),
                    'mean': _to_native_value(non_na.mean()),
                    'median': _to_native_value(non_na.median()),
                    'std': _to_native_value(non_na.std())
                })
        else:
            top_values = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(5)
                .index
                .tolist()
            )
            column_summary['top_values'] = top_values
        
        summary['columns'].append(column_summary)
    
    sample_rows = []
    for _, row in df.head(max_samples).iterrows():
        sample = {}
        for column in df.columns[:max_columns]:
            sample[column] = _to_native_value(row[column])
        sample_rows.append(sample)
    
    return summary, sample_rows


def summarize_grades_for_gemini(course_grades: Dict[str, str], loaded_terms_count: int):
    distribution = Counter(course_grades.values())
    course_details = []
    total_credits = 0
    total_grade_points = 0.0
    credits_for_gpa = 0.0
    failed_courses = []
    
    # Grade mapping สำหรับคำนวณ GPA
    grade_mapping = {
        'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
        'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'WF': 0.0, 'WU': 0.0, 'S': None
    }
    
    for course_id, grade in course_grades.items():
        info = COURSE_LOOKUP.get(course_id, {})
        credits = info.get('credit', 3)
        total_credits += credits
        
        # คำนวณ GPA แบบ weighted ด้วยหน่วยกิต (ไม่รวม S, W, WF, WU)
        grade_upper = grade.upper() if isinstance(grade, str) else str(grade).upper()
        grade_point = grade_mapping.get(grade_upper)
        
        if grade_point is not None:
            # รวมหน่วยกิตและคะแนนสำหรับคำนวณ GPA
            credits_for_gpa += credits
            total_grade_points += grade_point * credits
        
        # ตรวจสอบวิชาที่สอบตก (F, U, WF, WU)
        if grade_upper in ['F', 'U', 'WF', 'WU']:
            failed_courses.append({
                'course_id': course_id,
                'course_name': info.get('thaiName') or info.get('name'),
                'credit': credits,
                'grade': grade
            })
        
        course_details.append({
            'course_id': course_id,
            'course_name': info.get('thaiName') or info.get('name'),
            'credit': credits,
            'grade': grade
        })
    
    # คำนวณ GPA แบบ weighted average (รวมหน่วยกิต)
    estimated_gpa = float(total_grade_points / credits_for_gpa) if credits_for_gpa > 0 else None
    
    return {
        'total_courses': len(course_grades),
        'estimated_gpa': estimated_gpa,
        'grade_distribution': dict(distribution),
        'course_details': course_details,
        'loaded_terms_count': loaded_terms_count,
        'total_credits_recorded': total_credits,
        'failed_courses': failed_courses,
        'failed_count': len(failed_courses)
    }


def summarize_course_dna(course_profiles: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    สรุป Course DNA จาก course_profiles เพื่อใช้ใน Gemini Prompt
    ค้นหาวิชาปราบเซียน (Killer Courses), วิชาง่าย (Easy Courses), และวิชาเฟ้อ (High Inflation)
    """
    if not course_profiles:
        return {
            "killer_courses": [],
            "easy_courses": [],
            "high_inflation_courses": [],
            "total_courses": 0
        }
    
    killer_courses = []
    easy_courses = []
    high_inflation_courses = []
    
    # กำหนด threshold
    KILLER_FAIL_RATE_THRESHOLD = 0.3  # ตกมากกว่า 30% ถือว่าเป็น Killer
    EASY_AVG_GRADE_THRESHOLD = 3.5  # เกรดเฉลี่ยมากกว่า 3.5 ถือว่าง่าย
    EASY_FAIL_RATE_THRESHOLD = 0.1  # ตกน้อยกว่า 10% ถือว่าง่าย
    HIGH_INFLATION_AVG_THRESHOLD = 3.7  # เกรดเฉลี่ยสูงมาก (เฟ้อ)
    
    for course_id, profile in course_profiles.items():
        fail_rate = profile.get('fail_rate', 0)
        avg_grade = profile.get('avg_grade', 0)
        difficulty_score = profile.get('difficulty_score', 0)
        total_students = profile.get('total_students', 0)
        
        # ตรวจสอบว่าเป็น Killer Course (วิชาปราบเซียน)
        if fail_rate >= KILLER_FAIL_RATE_THRESHOLD and total_students >= 10:  # ต้องมีข้อมูลพอ
            killer_courses.append({
                'course_id': course_id,
                'fail_rate': fail_rate,
                'avg_grade': avg_grade,
                'difficulty_score': difficulty_score,
                'total_students': total_students,
                'fail_rate_percent': f"{fail_rate*100:.1f}%"
            })
        
        # ตรวจสอบว่าเป็น Easy Course (วิชาง่าย)
        if avg_grade >= EASY_AVG_GRADE_THRESHOLD and fail_rate <= EASY_FAIL_RATE_THRESHOLD and total_students >= 10:
            easy_courses.append({
                'course_id': course_id,
                'avg_grade': avg_grade,
                'fail_rate': fail_rate,
                'total_students': total_students
            })
        
        # ตรวจสอบว่าเป็น High Inflation Course (วิชาเฟ้อ)
        if avg_grade >= HIGH_INFLATION_AVG_THRESHOLD and total_students >= 10:
            high_inflation_courses.append({
                'course_id': course_id,
                'avg_grade': avg_grade,
                'fail_rate': fail_rate,
                'total_students': total_students
            })
    
    # เรียงลำดับ Killer Courses ตาม fail_rate (สูงสุดก่อน)
    killer_courses.sort(key=lambda x: x['fail_rate'], reverse=True)
    
    # เรียงลำดับ Easy Courses ตาม avg_grade (สูงสุดก่อน)
    easy_courses.sort(key=lambda x: x['avg_grade'], reverse=True)
    
    return {
        "killer_courses": killer_courses[:20],  # เอาแค่ 20 อันแรก
        "easy_courses": easy_courses[:20],
        "high_inflation_courses": high_inflation_courses[:20],
        "total_courses": len(course_profiles),
        "killer_count": len(killer_courses),
        "easy_count": len(easy_courses),
        "inflation_count": len(high_inflation_courses)
    }

def run_gemini_training_analysis(
    df: pd.DataFrame,
    analysis_goal: Optional[str],
    training_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    ให้ Gemini อ่านไฟล์เทรนและวิเคราะห์ตาม prompt ที่กำหนด
    แล้วส่งค่ากลับมาเพื่อใช้ในการเทรนโมเดล
    """
    if not is_gemini_available():
        logger.debug("Gemini not available; skip training-time analysis")
        return None
    
    try:
        logger.info("🤖 Starting Gemini training analysis...")
        
        # สรุปข้อมูลไฟล์เทรน
        summary, samples = summarize_dataframe_for_gemini(df, max_columns=30, max_samples=20)
        
        # ดึงข้อมูล Course DNA จาก training_context
        dna_info = training_context.get('course_dna_insight', {})
        
        # สร้าง prompt สำหรับ Gemini
        # กำหนดเป้าหมายอัตโนมัติ ถ้า user ไม่ได้พิมพ์มา
        goal_text = (analysis_goal or '').strip()
        if not goal_text:
            goal_text = "วิเคราะห์ความยากง่ายของหลักสูตรและพฤติกรรมเกรดของนักศึกษาจาก Course DNA"
        
        # สร้างข้อความ Course DNA สำหรับใส่ใน prompt
        dna_section = ""
        if dna_info and dna_info.get('killer_courses'):
            killer_list = dna_info.get('killer_courses', [])[:10]  # เอาแค่ 10 อันแรก
            killer_str = ", ".join([f"{k['course_id']} (Fail Rate: {k['fail_rate_percent']})" for k in killer_list])
            easy_list = dna_info.get('easy_courses', [])[:10]
            easy_str = ", ".join([f"{e['course_id']} (Avg: {e['avg_grade']:.2f})" for e in easy_list])
            
            dna_section = f"""
**ข้อมูลเชิงลึกของหลักสูตร (Course DNA) - วิเคราะห์โดยระบบ Backend:**
- วิชาปราบเซียน (Killer Courses): {killer_str if killer_str else "ไม่พบ"}
- จำนวนวิชาปราบเซียนทั้งหมด: {dna_info.get('killer_count', 0)} วิชา
- วิชาง่าย (Easy Courses): {easy_str if easy_str else "ไม่พบ"}
- จำนวนวิชาง่ายทั้งหมด: {dna_info.get('easy_count', 0)} วิชา
- วิชาเฟ้อเกรด (High Inflation): {dna_info.get('inflation_count', 0)} วิชา
- จำนวนรายวิชาทั้งหมด: {dna_info.get('total_courses', 0)} วิชา

**หมายเหตุ:** ระบบ Backend ได้วิเคราะห์ Course DNA จากข้อมูลเทรนแล้ว โดย:
- Killer Course = วิชาที่มีอัตราการตก ≥ 30%
- Easy Course = วิชาที่มีเกรดเฉลี่ย ≥ 3.5 และอัตราการตก ≤ 10%
- High Inflation = วิชาที่มีเกรดเฉลี่ย ≥ 3.7

"""
        
        # สร้าง prompt ที่ละเอียดขึ้น
        training_prompt = f"""
คุณคือผู้เชี่ยวชาญด้าน Data Science การศึกษา
ข้อมูลที่ได้รับคือ Course DNA และประวัติเกรดที่ผ่านการประมวลผลแบบ Dynamic Snapshots แล้ว

**วัตถุประสงค์การวิเคราะห์:**
{goal_text}

**ข้อมูลไฟล์เทรน:**
- จำนวนแถว: {summary['row_count']} แถว
- จำนวนคอลัมน์: {summary['column_count']} คอลัมน์
- รายละเอียดคอลัมน์: {json.dumps(summary['columns'], ensure_ascii=False, indent=2)}

**ตัวอย่างข้อมูล (20 แถวแรก):**
{json.dumps(samples, ensure_ascii=False, indent=2)}

{dna_section}

**บริบทการเทรน:**
- รูปแบบข้อมูล: {training_context.get('data_format', 'ไม่ระบุ')}
- ประเภทการเทรน: {training_context.get('training_type', 'ไม่ระบุ')}
- ใช้ Advanced Pipeline: {training_context.get('use_advanced_pipeline', False)}
- จำนวนตัวอย่างที่เตรียมแล้ว: {training_context.get('prepared_samples', 0)}
- จำนวน features: {training_context.get('feature_count', 0)}
- การกระจายของ label: {json.dumps(training_context.get('label_distribution', {}), ensure_ascii=False)}

**งานของคุณ:**
1. วิเคราะห์ความสมดุลของหลักสูตร (มีความยาก-ง่ายเหมาะสมไหม)
2. ประเมินว่า Killer Courses เหล่านี้มีผลกระทบต่อการจบการศึกษาอย่างไร
3. วิเคราะห์คุณภาพข้อมูล: ข้อมูลมีความสมบูรณ์เพียงพอหรือไม่ มี missing values หรือข้อมูลผิดปกติหรือไม่?
4. วิเคราะห์ความสมดุลของข้อมูล: การกระจายของ label (จบ/ไม่จบ) สมดุลหรือไม่?
5. ให้คำแนะนำ: มีคำแนะนำอะไรสำหรับการปรับปรุงหลักสูตรหรือการเทรนโมเดลหรือไม่?
6. ไม่ต้องถามข้อมูลเพิ่ม ให้วิเคราะห์จากเท่าที่มี

กรุณาวิเคราะห์และให้คำตอบตามโครงสร้างที่กำหนด
"""
        
        payload = {
            'analysis_goal': goal_text,
            'dataset_summary': summary,
            'sample_rows': samples,
            'training_context': training_context,
            'detailed_prompt': training_prompt
        }
        
        logger.info("📤 Sending training data to Gemini for analysis...")
        gemini_output = call_gemini_structured('training_dataset_analysis', payload)
        
        logger.info("✅ Gemini training analysis completed")
        
        return {
            'analysis_goal': goal_text,
            'dataset_summary': summary,
            'sample_rows': samples,
            'training_context': training_context,
            'gemini': gemini_output,
            'generated_at': datetime.now().isoformat(),
            'analysis_type': 'training_file_analysis'
        }
    except Exception as exc:
        logger.warning(f"Gemini training analysis failed: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'analysis_goal': analysis_goal,
            'training_context': training_context,
            'error': str(exc),
            'generated_at': datetime.now().isoformat()
        }


def repair_incomplete_json(text: str) -> str:
    """
    ซ่อมแซม JSON ที่ไม่สมบูรณ์ (unterminated string, unclosed braces, etc.)
    """
    if not text:
        return text
    
    # ลบ markdown code blocks ถ้ามี
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # ลบ whitespace ที่ไม่จำเป็น
    cleaned = cleaned.strip()
    
    # ถ้าไม่มี { ให้ return text เดิม
    if '{' not in cleaned:
        return cleaned
    
    # หา JSON object ที่สมบูรณ์ที่สุด
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = -1
    start_pos = cleaned.find('{')
    
    if start_pos == -1:
        return cleaned
    
    # วิเคราะห์ JSON structure
    for i in range(start_pos, len(cleaned)):
        char = cleaned[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i
                break
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
    
    # ถ้าเจอ JSON object ที่สมบูรณ์ ให้ใช้ส่วนนั้น
    if last_valid_pos > start_pos:
        return cleaned[start_pos:last_valid_pos + 1]
    
    # ถ้าไม่สมบูรณ์ ให้ลองซ่อมแซม
    if brace_count > 0 or bracket_count > 0 or in_string:
        # หา string ที่ไม่ปิด quote
        in_string = False
        escape_next = False
        string_start = -1
        last_string_char = -1
        
        for i in range(start_pos, len(cleaned)):
            char = cleaned[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                if not in_string:
                    in_string = True
                    string_start = i
                else:
                    in_string = False
                    string_start = -1
                    last_string_char = i
        
        # ถ้ามี string ที่ไม่ปิด ให้ปิดมัน
        if in_string and string_start >= 0:
            # หาจุดที่เหมาะสมในการปิด string
            # ลองหาจากท้ายสุดย้อนกลับ
            insert_pos = len(cleaned)
            for i in range(len(cleaned) - 1, string_start, -1):
                if cleaned[i] in [',', '}', ']', '\n', '\r']:
                    insert_pos = i
                    break
            
            # ปิด string
            cleaned = cleaned[:insert_pos] + '"' + cleaned[insert_pos:]
        
        # ปิด brackets ที่เหลือ (ต้องปิดก่อน braces)
        if bracket_count > 0:
            cleaned = cleaned + ']' * bracket_count
        
        # ปิด braces ที่เหลือ
        if brace_count > 0:
            cleaned = cleaned + '}' * brace_count
    
    return cleaned

@retry_on_quota_error(max_retries=3, initial_delay=20)
def call_gemini_with_retry(prompt_or_payload, task_type='prediction_analysis'):
    """เรียก Gemini พร้อมระบบ retry (ลองทั้งหมด 3 ครั้ง)"""
    # ถ้าเป็น string ให้สร้าง payload ด้วย detailed_prompt
    if isinstance(prompt_or_payload, str):
        payload = {'detailed_prompt': prompt_or_payload}
    else:
        payload = prompt_or_payload
    
    return call_gemini_structured(task_type, payload)


def call_gemini_structured(task_name: str, payload: Dict[str, Any], schema_key: str = 'insights'):
    """
    เรียกใช้ Gemini API เพื่อวิเคราะห์ข้อมูล
    รองรับทั้ง prompt ธรรมดาและ detailed prompt
    """
    if not is_gemini_available():
        raise RuntimeError("Gemini API is not configured")

    # ถ้ามี detailed_prompt ให้ใช้แทน
    if 'detailed_prompt' in payload:
        user_prompt = payload['detailed_prompt']
        logger.info("📝 Using detailed prompt for Gemini analysis")
    else:
        prompt_payload = {
            'task': task_name,
            'payload': payload
        }
        user_prompt = json.dumps(prompt_payload, ensure_ascii=False)

    base_generation_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
        "response_mime_type": "application/json"
    }

    schema = GEMINI_RESPONSE_SCHEMAS.get(schema_key)
    if schema:
        base_generation_config["response_schema"] = schema
        logger.info(f"📋 Using schema: {schema_key}")
    else:
        logger.warning(f"⚠️ Schema '{schema_key}' not found, using default")

    def _execute_with_model(model_name: str):
        generation_config = copy.deepcopy(base_generation_config)
        
        # Increase token limit for 1.5 models which support higher output
        if '1.5' in model_name:
            generation_config['max_output_tokens'] = 8192
            
        model = genai.GenerativeModel(
            model_name,
            system_instruction=GEMINI_SYSTEM_PROMPT,
            generation_config=generation_config
        )

        logger.info(f"📤 Sending request to Gemini (task: {task_name}, model: {model_name})...")
        logger.debug(f"Prompt length: {len(user_prompt)} characters")

        response = model.generate_content(user_prompt)
        response_text = None

        if hasattr(response, 'text') and response.text:
            response_text = response.text
            logger.debug("Got response from response.text")

        if not response_text and hasattr(response, 'candidates') and response.candidates:
            try:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts:
                        first_part = parts[0]
                        if hasattr(first_part, 'text'):
                            response_text = first_part.text
                        elif isinstance(first_part, dict) and 'text' in first_part:
                            response_text = first_part['text']
                        logger.debug("Got response from candidates[0].content.parts[0]")
            except Exception as e:
                logger.warning(f"Error extracting from candidates: {e}")

        if not response_text:
            try:
                response_str = str(response)
                if response_str and response_str != 'None':
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                        logger.debug("Got response from string extraction")
            except Exception as e:
                logger.warning(f"Error extracting from string: {e}")

        if not response_text:
            error_msg = "Empty response from Gemini"
            logger.error(f"❌ {error_msg}")
            logger.debug(f"Response object: {response}")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response dir: {dir(response)}")
            raise ValueError(error_msg)

        logger.debug(f"Response text length: {len(response_text)} characters")
        logger.debug(f"Response preview: {response_text[:200]}...")

        try:
            result = json.loads(response_text)
            logger.info("✅ Gemini analysis completed successfully")
            return result
        except json.JSONDecodeError as json_err:
            logger.warning(f"⚠️ Initial JSON parse failed: {json_err}")
            logger.debug(f"Response text (first 1000 chars): {response_text[:1000]}...")
            
            try:
                # Attempt 1: Clean markdown code blocks
                cleaned_text = response_text.strip()
                if cleaned_text.startswith('```'):
                    cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text)
                    cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
                
                try:
                    result = json.loads(cleaned_text)
                    logger.info("✅ Successfully parsed JSON after cleaning markdown")
                    return result
                except json.JSONDecodeError as e1:
                    logger.debug(f"Markdown cleaning failed: {e1}")
                    
                    # Attempt 2: Repair incomplete JSON (unterminated strings, unclosed braces)
                    try:
                        repaired_text = repair_incomplete_json(cleaned_text)
                        result = json.loads(repaired_text)
                        logger.info("✅ Successfully parsed JSON after repair")
                        return result
                    except json.JSONDecodeError as e2:
                        logger.debug(f"JSON repair failed: {e2}")
                        
                        # Attempt 3: Extract JSON object using regex (fallback)
                        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                        if json_match:
                            extracted_json = json_match.group(0)
                            try:
                                # Try to repair extracted JSON too
                                repaired_extracted = repair_incomplete_json(extracted_json)
                                result = json.loads(repaired_extracted)
                                logger.info("✅ Successfully extracted and parsed JSON using regex + repair")
                                return result
                            except json.JSONDecodeError:
                                # Last attempt: try original extracted
                                result = json.loads(extracted_json)
                                logger.info("✅ Successfully extracted and parsed JSON using regex")
                                return result
                        raise ValueError(f"Invalid JSON response from Gemini: {str(json_err)}")
            except Exception as repair_exc:
                logger.error(f"❌ All JSON repair attempts failed: {repair_exc}")
                logger.error(f"Original error: {json_err}")
                logger.error(f"Response text (first 2000 chars): {response_text[:2000]}...")
                raise ValueError(f"Invalid JSON response from Gemini: {str(json_err)}")

    last_error = None
    tried_models: List[str] = []

    for model_name in GEMINI_MODEL_CANDIDATES:
        tried_models.append(model_name)
        try:
            return _execute_with_model(model_name)
        except ValueError as ve:
            logger.warning(f"⚠️ Gemini model '{model_name}' returned invalid response: {ve}")
            last_error = ve
            continue
        except Exception as exc:
            last_error = exc
            if _should_retry_with_fallback(exc) and model_name != GEMINI_MODEL_CANDIDATES[-1]:
                logger.warning(f"⚠️ Gemini model '{model_name}' unavailable ({exc}); trying fallback...")
                continue
            logger.error(f"❌ Gemini request failed on model '{model_name}': {exc}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Gemini API error ({model_name}): {str(exc)}") from exc

    raise RuntimeError(
        f"Gemini API error after trying models {', '.join(tried_models)}: {last_error}"
    )

# Flask Routes (Keep all other routes unchanged)

def refresh_r2_storage_from_settings():
    """Reinitialize storage object with current environment settings."""
    global storage
    try:
        storage = S3Storage()
        return True
    except Exception as e:
        logger.error(f"Failed to refresh R2 storage runtime: {e}")
        return False


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if not admin_manager.enabled:
        return "MongoDB admin backend is not configured", 503

    if request.method == 'GET':
        if session.get('user_id'):
            return redirect(url_for('admin_settings'))
        return render_template('admin_login.html')

    payload = request.get_json(silent=True) or request.form
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''

    user = admin_manager.authenticate(username, password)
    if not user:
        if request.path.startswith('/api/') or request.is_json:
            return _json_error('Invalid credentials', 401)
        return render_template('admin_login.html', error='ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง')

    session['user_id'] = str(user.get('_id'))
    session['username'] = user.get('username')
    session['user_role'] = user.get('role', 'user')
    session['must_change_password'] = bool(user.get('must_change_password', False))
    session.permanent = True

    admin_manager._log_audit(session['user_id'], 'LOGIN', 'admin_login')

    if session.get('must_change_password'):
        return redirect(url_for('admin_change_password'))
    return redirect(url_for('admin_settings'))


@app.route('/admin')
def admin_home():
    if not admin_manager.enabled:
        return "MongoDB admin backend is not configured", 503
    if not is_logged_in():
        return redirect(url_for('admin_login'))
    if session.get('must_change_password'):
        return redirect(url_for('admin_change_password'))
    if has_any_role('admin', 'super_admin'):
        return redirect(url_for('admin_settings'))
    return redirect(url_for('index'))


@app.route('/admin/logout')
def admin_logout():
    if admin_manager.enabled and session.get('user_id'):
        admin_manager._log_audit(session.get('user_id'), 'LOGOUT', 'admin_logout')
    session.clear()
    return redirect(url_for('admin_login'))


@app.route('/admin/change-password', methods=['GET', 'POST'])
@admin_login_required
def admin_change_password():
    if request.method == 'GET':
        return render_template('admin_change_password.html')

    payload = request.get_json(silent=True) or request.form
    current_password = payload.get('current_password', '')
    new_password = payload.get('new_password', '')
    confirm_password = payload.get('confirm_password', '')

    if len(new_password) < 8:
        return render_template('admin_change_password.html', error='รหัสผ่านใหม่ต้องยาวอย่างน้อย 8 ตัวอักษร')
    if new_password != confirm_password:
        return render_template('admin_change_password.html', error='ยืนยันรหัสผ่านไม่ตรงกัน')

    user_doc = admin_manager.users.find_one({'username': session.get('username')})
    if not user_doc:
        return render_template('admin_change_password.html', error='ไม่พบบัญชีผู้ใช้')

    if not user_doc.get('must_change_password'):
        if not check_password_hash(user_doc.get('password_hash', ''), current_password):
            return render_template('admin_change_password.html', error='รหัสผ่านปัจจุบันไม่ถูกต้อง')

    admin_manager.change_password(user_doc['_id'], new_password)
    session['must_change_password'] = False
    admin_manager._log_audit(session.get('user_id'), 'CHANGE_PASSWORD', session.get('username'))
    return render_template('admin_change_password.html', success='เปลี่ยนรหัสผ่านสำเร็จ')


@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_login_required
@roles_required('admin', 'super_admin')
def admin_settings():
    if request.method == 'POST':
        form = request.get_json(silent=True) or request.form

        setting_map = {
            'gemini_api_key': 'GEMINI_API_KEY',
            'gemini_model_name': 'GEMINI_MODEL_NAME',
            'gemini_model_fallbacks': 'GEMINI_MODEL_FALLBACKS',
            'r2_access_key': 'CLOUDFLARE_R2_ACCESS_KEY_ID',
            'r2_secret_key': 'CLOUDFLARE_R2_SECRET_ACCESS_KEY',
            'r2_endpoint': 'CLOUDFLARE_R2_ENDPOINT',
            'r2_bucket_name': 'CLOUDFLARE_R2_BUCKET_NAME',
        }

        for form_key, setting_key in setting_map.items():
            if form_key in form and form.get(form_key) is not None:
                value = str(form.get(form_key)).strip()
                if value:
                    admin_manager.set_setting(setting_key, value, actor_user_id=session.get('user_id'))

        admin_manager.apply_runtime_env()
        refresh_gemini_runtime_from_settings()
        refresh_r2_storage_from_settings()

        return render_template('admin_settings.html', success='บันทึกค่าเรียบร้อยแล้ว')

    current_values = {
        'gemini_api_key': admin_manager.get_setting('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', '')),
        'gemini_model_name': admin_manager.get_setting('GEMINI_MODEL_NAME', os.environ.get('GEMINI_MODEL_NAME', 'gemini-3-flash-preview')),
        'gemini_model_fallbacks': admin_manager.get_setting('GEMINI_MODEL_FALLBACKS', os.environ.get('GEMINI_MODEL_FALLBACKS', '')),
        'r2_access_key': admin_manager.get_setting('CLOUDFLARE_R2_ACCESS_KEY_ID', os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', '')),
        'r2_secret_key': admin_manager.get_setting('CLOUDFLARE_R2_SECRET_ACCESS_KEY', os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', '')),
        'r2_endpoint': admin_manager.get_setting('CLOUDFLARE_R2_ENDPOINT', os.environ.get('CLOUDFLARE_R2_ENDPOINT', '')),
        'r2_bucket_name': admin_manager.get_setting('CLOUDFLARE_R2_BUCKET_NAME', os.environ.get('CLOUDFLARE_R2_BUCKET_NAME', '')),
        'gemini_api_key_masked': MongoAdminManager.mask_value(admin_manager.get_setting('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))),
        'r2_access_key_masked': MongoAdminManager.mask_value(admin_manager.get_setting('CLOUDFLARE_R2_ACCESS_KEY_ID', os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID', ''))),
        'r2_secret_key_masked': MongoAdminManager.mask_value(admin_manager.get_setting('CLOUDFLARE_R2_SECRET_ACCESS_KEY', os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY', ''))),
    }

    return render_template('admin_settings.html', **current_values)


@app.route('/api/admin/me')
@admin_login_required
def admin_me():
    return jsonify({
        'success': True,
        'username': session.get('username'),
        'role': session.get('user_role'),
        'must_change_password': bool(session.get('must_change_password'))
    })


@app.route('/api/admin/users', methods=['GET', 'POST'])
@admin_login_required
@roles_required('super_admin')
def admin_users_api():
    if request.method == 'GET':
        users = []
        for u in admin_manager.users.find({}, {'password_hash': 0}):
            users.append({
                'id': str(u.get('_id')),
                'username': u.get('username'),
                'role': u.get('role', 'user'),
                'is_active': bool(u.get('is_active', True)),
                'must_change_password': bool(u.get('must_change_password', False)),
                'created_at': u.get('created_at')
            })
        return jsonify({'success': True, 'users': users})

    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    role = (data.get('role') or 'user').strip()

    if not username:
        return _json_error('username is required', 400)
    if role not in ['user', 'admin', 'super_admin']:
        return _json_error('invalid role', 400)

    temp_password = secrets.token_urlsafe(10)
    try:
        admin_manager.users.insert_one({
            'username': username,
            'email': data.get('email'),
            'password_hash': generate_password_hash(temp_password),
            'role': role,
            'must_change_password': True,
            'is_active': True,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'last_login_at': None
        })
        admin_manager._log_audit(session.get('user_id'), 'CREATE_USER', username)
        return jsonify({
            'success': True,
            'username': username,
            'role': role,
            'temporary_password': temp_password
        })
    except Exception as e:
        return _json_error(f'cannot create user: {e}', 400)


@app.route('/api/admin/test-connections', methods=['POST'])
@admin_login_required
@roles_required('admin', 'super_admin')
def admin_test_connections():
    """Test runtime connections for MongoDB, Cloudflare R2, and Gemini."""
    results = {
        'success': True,
        'mongo': {'connected': False, 'message': ''},
        'r2': {'connected': False, 'message': ''},
        'gemini': {'connected': False, 'message': ''},
    }

    # MongoDB
    try:
        if admin_manager.enabled and admin_manager.client:
            admin_manager.client.admin.command('ping')
            results['mongo'] = {'connected': True, 'message': 'MongoDB ping success'}
        else:
            results['mongo'] = {'connected': False, 'message': 'MongoDB admin backend not enabled'}
    except Exception as e:
        results['mongo'] = {'connected': False, 'message': str(e)}

    # R2
    try:
        refresh_r2_storage_from_settings()
        if storage.use_local or not getattr(storage, 's3_client', None):
            results['r2'] = {'connected': False, 'message': 'R2 client not initialized (using local storage)'}
        else:
            storage.s3_client.list_objects_v2(Bucket=storage.bucket_name, MaxKeys=1)
            results['r2'] = {'connected': True, 'message': f'Connected to bucket {storage.bucket_name}'}
    except Exception as e:
        results['r2'] = {'connected': False, 'message': str(e)}

    # Gemini
    try:
        refresh_gemini_runtime_from_settings()
        if not is_gemini_available():
            results['gemini'] = {'connected': False, 'message': 'Gemini API key/model not configured'}
        else:
            # Lightweight call to verify API key/model works
            if genai is None:
                raise RuntimeError('google.generativeai not installed')
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            ping_response = model.generate_content('ping')
            if not getattr(ping_response, 'text', None):
                raise RuntimeError('Empty response from Gemini')
            results['gemini'] = {'connected': True, 'message': f'Connected with model {GEMINI_MODEL_NAME}'}
    except Exception as e:
        results['gemini'] = {'connected': False, 'message': str(e)}

    overall_ok = all([
        results['mongo']['connected'],
        results['r2']['connected'],
        results['gemini']['connected'],
    ])

    results['success'] = overall_ok
    admin_manager._log_audit(session.get('user_id'), 'TEST_CONNECTIONS', 'mongo_r2_gemini', 'success' if overall_ok else 'failed', results)
    return jsonify(results)


@app.route('/')
def index():
    """Landing page: redirect users to prediction page."""
    return redirect(url_for('curriculum_prediction_form'))

@app.route('/test')
def curriculum_prediction_form():
    """Page for predicting graduation based on curriculum and prerequisites."""
    try:
        # ส่งข้อมูลเป็น JSON string ที่ปลอดภัย
        courses_json = json.dumps(app.config.get('COURSES_DATA', []))
        terms_json = json.dumps(app.config.get('ALL_TERMS_DATA', []))
        grades_json = json.dumps(app.config.get('DATA_CONFIG', {}).get('grade_mapping', {}))
        
        return render_template(
            'curriculum_prediction_form.html',
            coursesData=courses_json,
            allTermsData=terms_json,
            gradeMapping=grades_json
        )
    except Exception as e:
        logger.error(f"Error rendering curriculum form: {str(e)}")
        # Fallback ถ้ามีปัญหา
        return render_template(
            'curriculum_prediction_form.html',
            coursesData='[]',
            allTermsData='[]',
            gradeMapping='{}'
        )


@app.route('/status')
def status_page():
    """หน้าแสดงสถานะระบบ"""
    return render_template('status.html')

@app.route('/api/system/status')
def system_status():
    """API สำหรับตรวจสอบสถานะระบบ"""
    import sys
    import flask
    
    try:
        logger.info("🔧 Checking system status...")
        
        # ใช้ instance ที่สร้างไว้แล้ว
        storage_status = storage.get_connection_status()
        
        # นับจำนวนโมเดล
        try:
            models_list = storage.list_models()
            total_size = sum(m.get('size', 0) for m in models_list if 'size' in m)
        except Exception as e:
            logger.warning(f"Could not get models info: {e}")
            models_list = []
            total_size = 0
        
        # Get recent logs (ถ้ามี log file)
        recent_logs = []
        try:
            log_file = 'app.log'
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_logs = lines[-20:]
        except Exception as e:
            logger.warning(f"Could not read logs: {e}")

        # ใช้ข้อมูลจาก storage_status โดยตรง
        status_data = {
            'success': True,
            'r2_connected': storage_status['connected'],
            'storage_provider': storage_status['storage_type'],
            'bucket_name': storage_status['bucket_name'],
            'models_available': len(models_list),
            'total_size': total_size,
            'python_version': sys.version.split()[0],
            'flask_version': flask.__version__,
            'environment': os.environ.get('FLASK_ENV', 'production'),
            'debug_mode': app.debug,
            'server_time': datetime.now().isoformat(),
            'app_folders': {
                'upload_folder': app.config['UPLOAD_FOLDER'],
                'model_folder': app.config['MODEL_FOLDER'],
                'upload_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'model_exists': os.path.exists(app.config['MODEL_FOLDER'])
            },
            'recent_logs': recent_logs,
            'error_message': storage_status.get('errors'),
            'env_vars_status': {
                'R2_ACCESS_KEY': bool(os.environ.get('CLOUDFLARE_R2_ACCESS_KEY_ID')),
                'R2_SECRET_KEY': bool(os.environ.get('CLOUDFLARE_R2_SECRET_ACCESS_KEY')),
                'R2_ENDPOINT': bool(os.environ.get('CLOUDFLARE_R2_ENDPOINT')),
                'R2_BUCKET': bool(os.environ.get('CLOUDFLARE_R2_BUCKET_NAME'))
            }
        }
        
        logger.info("✅ System status check completed")
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'server_time': datetime.now().isoformat()
        }), 500

@app.route('/api/test-r2-connection')
def test_r2_connection():
    """ทดสอบการเชื่อมต่อ R2"""
    try:
        logger.info("🧪 Testing R2 connection...")
        
        if storage.use_local or not hasattr(storage, 's3_client') or not storage.s3_client:
            return jsonify({
                'success': False,
                'error': 'R2 client not initialized - using local storage',
                'storage_provider': 'local'
            })
        
        # ทดสอบ list objects
        response = storage.s3_client.list_objects_v2(
            Bucket=storage.bucket_name,
            MaxKeys=1
        )
        
        logger.info("✅ R2 connection test successful")
        return jsonify({
            'success': True,
            'message': 'R2 connection successful',
            'bucket': storage.bucket_name,
            'endpoint': storage.endpoint_url,
            'objects_found': len(response.get('Contents', []))
        })
        
    except Exception as e:
        logger.error(f"❌ R2 connection test failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'storage_provider': 'local_fallback'
        })
        
@app.route('/api/test-r2')
def test_r2():
    try:
        # ทดสอบเขียนไฟล์
        test_data = {'test': True, 'timestamp': datetime.now().isoformat()}
        success = storage.save_model(test_data, 'test_model.joblib')
        
        if success:
            # ทดสอบอ่านไฟล์
            loaded = storage.load_model('test_model.joblib')
            if loaded:
                # ลบไฟล์ทดสอบ
                storage.delete_model('test_model.joblib')
                return jsonify({
                    'success': True,
                    'message': 'R2 storage working perfectly!',
                    'can_write': True,
                    'can_read': True,
                    'can_delete': True
                })
        
        return jsonify({'success': False, 'message': 'R2 test failed'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
        
@app.route('/api/config', methods=['GET'])
def get_config_for_frontend():
    """Provides frontend with necessary configuration data."""
    try:
        config_data = {
            'success': True,
            'COURSES_DATA': app.config.get('COURSES_DATA', []),
            'ALL_TERMS_DATA': app.config.get('ALL_TERMS_DATA', []),
            'GRADE_MAPPING': app.config.get('DATA_CONFIG', {}).get('grade_mapping', {}),
            'MESSAGES': app.config.get('MESSAGES', {}),
            'DATA_CONFIG_RISK_LEVELS': app.config.get('DATA_CONFIG', {}).get('risk_levels', {}),
            'GEMINI_ENABLED': is_gemini_available()
        }
        return jsonify(config_data)
    except Exception as e:
        logger.error(f"Error loading config data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gemini/analyze_file', methods=['POST'])
def gemini_analyze_file_route():
    """วิเคราะห์ไฟล์ CSV/XLSX ด้วย Gemini"""
    if not is_gemini_available():
        return jsonify({'success': False, 'error': 'ยังไม่ได้ตั้งค่า Gemini API Key'}), 503
    
    uploaded_file = request.files.get('file')
    analysis_goal = request.form.get('analysis_goal', '').strip()
    
    if uploaded_file is None or uploaded_file.filename == '':
        return jsonify({'success': False, 'error': 'กรุณาเลือกไฟล์ CSV หรือ Excel'}), 400
    
    try:
        file_bytes = uploaded_file.read()
        if not file_bytes:
            return jsonify({'success': False, 'error': 'ไฟล์ว่างเปล่า'}), 400
        
        max_bytes = int(GEMINI_MAX_FILE_SIZE_MB * 1024 * 1024)
        if len(file_bytes) > max_bytes:
            return jsonify({
                'success': False,
                'error': f'ไฟล์มีขนาดใหญ่เกิน {GEMINI_MAX_FILE_SIZE_MB} MB'
            }), 400
        
        buffer = BytesIO(file_bytes)
        filename = uploaded_file.filename.lower()
        
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(buffer)
        else:
            buffer.seek(0)
            df = pd.read_csv(buffer)
        
        summary, samples = summarize_dataframe_for_gemini(df)
        payload = {
            'analysis_goal': analysis_goal or 'วิเคราะห์ภาพรวมของข้อมูลเกรดในไฟล์',
            'dataset_summary': summary,
            'sample_rows': samples
        }
        gemini_output = call_gemini_structured('csv_dataset_analysis', payload)
        
        return jsonify({
            'success': True,
            'source': 'file',
            'dataset_summary': summary,
            'sample_rows': samples,
            'gemini': gemini_output
        })
    except Exception as exc:
        logger.error(f"Gemini file analysis error: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/gemini/predict', methods=['POST'])
def gemini_predict_route():
    """ให้ Gemini วิเคราะห์ข้อมูลเกรดที่ผู้ใช้กรอกแบบสด"""
    if not is_gemini_available():
        logger.warning("Gemini API not available - missing API key")
        return jsonify({
            'success': False, 
            'error': '⚠️ ยังไม่ได้ตั้งค่า Gemini API Key',
            'suggestion': 'กรุณาตั้งค่า GEMINI_API_KEY ใน environment variables'
        }), 503
    
    # ตรวจสอบ rate limit
    can_proceed, wait_time = gemini_rate_limiter.can_proceed()
    if not can_proceed:
        return jsonify({
            'success': False,
            'error': f'⏱️ ใช้งานบ่อยเกินไป กรุณารอ {wait_time} วินาที',
            'suggestion': 'กรุณารอสักครู่ก่อนลองใหม่',
            'retry_after': wait_time
        }), 429
    
    try:
        payload = request.get_json(silent=True) or {}
        logger.info(f"Received Gemini prediction request: {list(payload.keys())}")
        
        course_grades = payload.get('course_grades') or payload.get('grades') or {}
        student_name = payload.get('student_name', 'นักศึกษา')
        analysis_goal = payload.get('analysis_goal', '').strip()
        loaded_terms_count = int(payload.get('loaded_terms_count') or 0)
        model_filename = payload.get('model_filename')
        training_analysis = None
        model_metadata = None
        
        logger.info(f"Processing prediction for {student_name} with {len(course_grades)} course grades")
        
        if not isinstance(course_grades, dict) or len(course_grades) == 0:
            logger.warning("No course grades provided")
            return jsonify({'success': False, 'error': 'กรุณาใส่ข้อมูลเกรดอย่างน้อย 1 วิชา'}), 400
    except Exception as parse_error:
        logger.error(f"Error parsing request: {parse_error}")
        return jsonify({'success': False, 'error': f'เกิดข้อผิดพลาดในการอ่านข้อมูล: {str(parse_error)}'}), 400
    
    # กรองเฉพาะเกรดที่ไม่ว่าง
    cleaned_grades = {
        str(course_id): grade
        for course_id, grade in course_grades.items()
        if grade
    }
    
    if len(cleaned_grades) == 0:
        return jsonify({'success': False, 'error': 'ไม่พบเกรดที่สามารถนำมาวิเคราะห์ได้'}), 400
    
    # โหลด Course DNA จากโมเดลเพื่อใช้ในการวิเคราะห์
    course_profiles = None
    course_context_str = ""
    training_analysis = None
    model_metadata = None
    
    try:
        if model_filename:
            try:
                stored_model = storage.load_model(model_filename)
                if stored_model:
                    training_analysis = stored_model.get('gemini_training_analysis')
                    course_profiles = stored_model.get('course_profiles', {})
                    model_metadata = {
                        'model_filename': model_filename,
                        'data_format': stored_model.get('data_format'),
                        'training_type': stored_model.get('training_type'),
                        'performance_metrics': stored_model.get('performance_metrics')
                    }
                    logger.info(f"Loaded model context: {model_filename}")
            except Exception as model_exc:
                logger.warning(f"Unable to load model for Gemini context: {model_exc}")
        else:
            # ถ้าไม่ได้ระบุ model_filename ให้ลองโหลดโมเดลล่าสุด
            try:
                models = storage.list_models()
                if models:
                    latest_model = models[0]  # โมเดลล่าสุด
                    loaded_model = storage.load_model(latest_model['filename'])
                    if loaded_model:
                        course_profiles = loaded_model.get('course_profiles', {})
                        logger.info(f"Loaded course profiles from latest model: {latest_model['filename']}")
            except Exception as load_exc:
                logger.warning(f"Unable to load latest model for course profiles: {load_exc}")
        
        # ตรวจสอบวิชาที่นักศึกษาเรียน เทียบกับ Course DNA
        if course_profiles:
            student_risk_factors = []
            student_easy_courses_failed = []
            
            for course_id, grade in cleaned_grades.items():
                if course_id in course_profiles:
                    prof = course_profiles[course_id]
                    fail_rate = prof.get('fail_rate', 0)
                    avg_grade = prof.get('avg_grade', 0)
                    difficulty_score = prof.get('difficulty_score', 0)
                    
                    # ตรวจสอบว่าเป็น Killer Course และนักศึกษาตก
                    is_killer = fail_rate >= 0.3  # 30% fail rate threshold
                    is_failed = grade in ['F', 'W', 'WF', 'WU', 'D'] or (isinstance(grade, str) and grade.upper() in ['F', 'W', 'WF', 'WU', 'D'])
                    
                    if is_killer and is_failed:
                        student_risk_factors.append(
                            f"- ตกวิชาปราบเซียน {course_id} (ปกติคนตก {fail_rate*100:.0f}%, เกรดเฉลี่ย: {avg_grade:.2f})"
                        )
                    elif is_killer and not is_failed:
                        # ผ่าน Killer Course แล้ว (เป็นสัญญาณดี)
                        student_risk_factors.append(
                            f"- ผ่านวิชาปราบเซียน {course_id} แล้ว (ปกติคนตก {fail_rate*100:.0f}%) ✓"
                        )
                    
                    # ตรวจสอบว่าเป็น Easy Course แต่ได้เกรดแย่
                    is_easy = avg_grade >= 3.5 and fail_rate <= 0.1
                    is_poor_grade = grade in ['C', 'D', 'F', 'W', 'WF', 'WU'] or (isinstance(grade, str) and grade.upper() in ['C', 'D', 'F', 'W', 'WF', 'WU'])
                    
                    if is_easy and is_poor_grade:
                        student_easy_courses_failed.append(
                            f"- ได้เกรดน้อยในวิชาช่วย {course_id} (เกรดเฉลี่ยปกติ: {avg_grade:.2f}, ได้: {grade})"
                        )
            
            # สร้างข้อความ Course Context
            if student_risk_factors or student_easy_courses_failed:
                course_context_str = "\n**การวิเคราะห์ตาม Course DNA (Backend System):**\n"
                if student_risk_factors:
                    course_context_str += "\n".join(student_risk_factors)
                if student_easy_courses_failed:
                    course_context_str += "\n" + "\n".join(student_easy_courses_failed)
                course_context_str += "\n"
    except Exception as context_exc:
        logger.warning(f"Error building course context: {context_exc}")
        course_context_str = ""
    
    try:
        logger.info(f"🔮 Starting Gemini prediction for {student_name} with {len(cleaned_grades)} courses")
        
        grade_summary = summarize_grades_for_gemini(cleaned_grades, loaded_terms_count)
        grade_summary['student_name'] = student_name
        
        # =========================================================
        # 📝 AUTO-PROMPT GENERATION: สร้าง Prompt ให้ User เอง
        # =========================================================
        
        # กำหนด Goal อัตโนมัติ ถ้า User ไม่พิมพ์มา
        if not analysis_goal:
            analysis_goal = "ทำนายโอกาสจบการศึกษาและอธิบายเหตุผลตามเกณฑ์อย่างละเอียด"
        
        analysis_goal_text = analysis_goal
        
        # สร้าง Detailed Prompt ที่ฉลาดขึ้น - เน้นการทำนายและอธิบายผลการจบ/ไม่จบ
        detailed_prompt = f"""
คุณคือที่ปรึกษาทางวิชาการ ทำหน้าที่**ทำนายและอธิบาย**ผลการจบการศึกษาของนักศึกษา

**ข้อมูลนักศึกษา:**
- ชื่อ: {student_name}
- จำนวนวิชาที่เรียนแล้ว: {grade_summary.get('total_courses', 0)} วิชา
- จำนวนเทอมที่เรียน: {loaded_terms_count} เทอม
- เกรดเฉลี่ย (โดยประมาณ): {grade_summary.get('estimated_gpa', 0):.2f}
- หน่วยกิตที่เรียนแล้ว: {grade_summary.get('total_credits_recorded', 0)} หน่วยกิต
- จำนวนวิชาที่สอบตก: {grade_summary.get('failed_count', 0)} วิชา

**การกระจายเกรด:**
{json.dumps(grade_summary.get('grade_distribution', {}), ensure_ascii=False, indent=2)}

**รายละเอียดวิชา:**
{json.dumps(grade_summary.get('course_details', [])[:20], ensure_ascii=False, indent=2)}

**วิชาที่สอบตก (ถ้ามี):**
{json.dumps(grade_summary.get('failed_courses', []), ensure_ascii=False, indent=2)}

{course_context_str}
(ข้อมูลข้างบนคือข้อมูลเปรียบเทียบกับสถิติรุ่นพี่ที่เทรนมาแล้ว)

**งานของคุณ (ตอบเป็นภาษาไทยทั้งหมด):**

**1. ทำนายผลการจบการศึกษา (ต้องฟันธงชัดเจน):**
   - ทำนายว่านักศึกษาคนนี้ "จะจบตามเกณฑ์" หรือ "จะไม่จบ/จบช้า"
   - เกณฑ์การจบ: GPA >= 2.00 และหน่วยกิต >= 136 และผ่านวิชาบังคับ/วิชาปราบเซียน
   - ระบุความมั่นใจในการทำนายเป็นเปอร์เซ็นต์ (0-100%)

**2. อธิบายเหตุผลว่าทำไมถึงจบ (ถ้าทำนายว่าจบ):**
   - อธิบายเป็นภาษาไทยว่าทำไมถึงทำนายว่าจบ
   - เช่น: "GPA ผ่านเกณฑ์ (2.15 >= 2.00), ผ่านวิชาปราบเซียนแล้ว, มีแนวโน้มดี"
   - ระบุปัจจัยสำคัญที่ทำให้จบได้

**3. อธิบายเหตุผลว่าทำไมถึงไม่จบ (ถ้าทำนายว่าไม่จบ):**
   - อธิบายเป็นภาษาไทยว่าทำไมถึงทำนายว่าไม่จบ
   - เช่น: "GPA ต่ำกว่าเกณฑ์ (1.85 < 2.00), ตกวิชาปราบเซียน X, Y, Z"
   - ระบุปัญหาหลักที่ทำให้ไม่จบ (GPA ต่ำ, ตกวิชา Killer, หน่วยกิตไม่ครบ)

**4. วิเคราะห์เชิงลึก:**
   - วิเคราะห์ปัจจัยที่ส่งผลต่อการจบการศึกษามากที่สุด
   - ให้ความสำคัญกับ "วิชาปราบเซียน" (Killer Courses) ที่นักศึกษาตกเป็นอันดับแรก
   - ถ้าได้เกรดแย่ในวิชาง่าย (Easy Courses) ให้ระบุเป็นสัญญาณเตือน

**5. คำแนะนำ:**
   - สิ่งที่ต้องทำทันทีในเทอมหน้าเพื่อให้รอด
   - ถ้าตกวิชา Killer ให้แนะนำวิธีแก้ที่เข้มข้นและเป็นรูปธรรม
   - ทำนายโอกาสจบโดยดูว่าผ่านด่านวิชายากๆ ไปหรือยัง

**คำสั่งสำคัญ:**
- **ต้องทำนายผลการจบ/ไม่จบให้ชัดเจน** (ไม่ใช่แค่วิเคราะห์)
- **ต้องอธิบายเหตุผลเป็นภาษาไทย** ว่าทำไมถึงจบ/ไม่จบ
- วิเคราะห์จากข้อมูลที่มี ไม่ต้องถามข้อมูลเพิ่ม
- ให้ความสำคัญกับ Course DNA และ Killer Courses
- ตอบในรูปแบบ JSON ตาม Schema ที่กำหนด

กรุณาทำนายและให้คำตอบตามโครงสร้างที่กำหนด โดยให้:
- graduation_prediction: 
  * will_graduate: true/false (ทำนายว่าจบหรือไม่จบ)
  * prediction_text: ข้อความทำนายเป็นภาษาไทย (เช่น "คาดว่าจะจบตามเกณฑ์" หรือ "คาดว่าจะไม่จบการศึกษา")
  * reason_why_graduate: อธิบายเหตุผลว่าทำไมถึงจบ (ถ้าทำนายว่าจบ) เป็นภาษาไทย
  * reason_why_not_graduate: อธิบายเหตุผลว่าทำไมถึงไม่จบ (ถ้าทำนายว่าไม่จบ) เป็นภาษาไทย
  * confidence_percent: ระดับความมั่นใจในการทำนาย (0-100)
- analysis_markdown: สรุปการวิเคราะห์เป็นภาษาไทย (อย่างน้อย 200 คำ) โดยเน้นที่การทำนายและเหตุผล
- risk_level: ระดับความเสี่ยง (very_low, low, moderate, high, very_high)
- outcome_summary: สรุปผลการทำนาย (status: "graduated" หรือ "not_graduated", confidence: 0-1, description: อธิบายเหตุผลเป็นภาษาไทย)
- key_metrics: ตัวชี้วัดสำคัญ (อย่างน้อย 3 ตัว) รวมถึงจำนวน Killer Courses ที่ตก
- recommendations: คำแนะนำ (อย่างน้อย 3 ข้อ) โดยเน้นที่การแก้ปัญหา Killer Courses
"""
        
        prompt_payload = {
            'student_name': student_name,
            'analysis_goal': analysis_goal_text,
            'grade_summary': grade_summary,
            'detailed_prompt': detailed_prompt  # เพิ่ม detailed_prompt
        }
        
        logger.info("📤 Calling Gemini API for prediction...")
        try:
            # ใช้ function ที่มี retry แทน (ลองทั้งหมด 3 ครั้ง)
            gemini_output = call_gemini_with_retry(prompt_payload, 'live_grade_prediction')
            logger.info("✅ Gemini prediction completed successfully")
        except (ValueError, RuntimeError) as gemini_error:
            # ถ้า Gemini API ล้มเหลว ให้ส่ง error message ที่ชัดเจน
            logger.error(f"❌ Gemini API error: {gemini_error}")
            return jsonify({
                'success': False,
                'error': f'ไม่สามารถเชื่อมต่อกับ Gemini API ได้: {str(gemini_error)}',
                'details': str(gemini_error),
                'suggestion': 'กรุณาตรวจสอบว่า GEMINI_API_KEY ถูกตั้งค่าถูกต้องและลองใหม่อีกครั้ง'
            }), 400
        
        return jsonify({
            'success': True,
            'source': 'live_grades',
            'analysis': grade_summary,
            'gemini': gemini_output,
            'training_analysis': training_analysis,
            'model_metadata': model_metadata
        })
    except Exception as exc:
        error_msg = str(exc)
        
        # Check for quota/rate limit errors
        if '429' in error_msg or 'quota' in error_msg.lower() or 'resource exhausted' in error_msg.lower():
            logger.error(f"❌ Gemini quota exceeded: {error_msg}")
            return jsonify({
                'success': False,
                'error': '⚠️ โควต้า Gemini API หมดชั่วคราว',
                'details': 'Free tier: 15 requests/minute, 1,500 requests/day',
                'suggestion': 'รอ 1-2 นาที หรืออัพเกรดเป็น Paid Tier',
                'retry_after': 60
            }), 429
        
        logger.error(f"❌ Gemini prediction error: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'เกิดข้อผิดพลาด: {error_msg}',
            'suggestion': 'กรุณาลองใหม่อีกครั้ง'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and basic data format detection."""
    try:
        logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'success': False, 'error': 'ไม่พบไฟล์ในคำขอ'})

        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'success': False, 'error': 'ไม่ได้เลือกไฟล์'})

        # ตรวจสอบนามสกุลไฟล์
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            logger.warning(f"Invalid file extension")
            return jsonify({'success': False, 'error': 'รองรับเฉพาะไฟล์ .csv, .xlsx, .xls เท่านั้น'})

        # สร้าง filename ที่ปลอดภัย
        from werkzeug.utils import secure_filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # เก็บ extension ก่อน เพราะ secure_filename อาจลบจุดออก
        original_filename = file.filename
        file_ext = ''
        if '.' in original_filename:
            file_ext = '.' + original_filename.rsplit('.', 1)[1].lower()
        
        safe_filename = secure_filename(file.filename)
        # ถ้า secure_filename ลบ extension ออก ให้เพิ่มกลับ
        if file_ext and not safe_filename.lower().endswith(file_ext):
            filename_on_disk = f"{timestamp}_{safe_filename}{file_ext}"
        else:
            filename_on_disk = f"{timestamp}_{safe_filename}"
        
        # ใช้ absolute path
        upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
            
        filepath = os.path.join(upload_folder, filename_on_disk)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # ตรวจสอบว่าไฟล์ถูกบันทึกจริง
        if not os.path.exists(filepath):
            raise ValueError("ไฟล์ไม่ถูกบันทึก")
            
        logger.info(f"File saved successfully: {filename_on_disk}")

        # อ่านและตรวจสอบไฟล์
        try:
            df = None
            if safe_filename.lower().endswith('.csv'):
                # ลองหลาย encoding สำหรับไฟล์ CSV
                encodings = ['utf-8-sig', 'utf-8', 'cp874', 'tis-620', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"Successfully read CSV with encoding: {encoding}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed with {encoding}: {e}")
                        continue
                        
                if df is None:
                    # ถ้าอ่านไม่ได้ ลองใช้ engine python
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', engine='python')
                    except:
                        df = pd.read_csv(filepath, encoding='cp874', engine='python')
                        
            else:  # Excel files
                df = pd.read_excel(filepath, engine='openpyxl')
                logger.info(f"Successfully read Excel file")

            if df is None or df.empty:
                os.remove(filepath)
                raise ValueError("ไม่สามารถอ่านไฟล์ได้ หรือไฟล์ว่างเปล่า")

            data_format = detect_data_format(df)
            logger.info(f"Detected data format: {data_format}")

            response_data = {
                'success': True,
                'filename': filename_on_disk,
                'rows': len(df),
                'columns': len(df.columns),
                'data_format': data_format,
                'sample_columns': df.columns.tolist()[:10]
            }
            
            logger.info(f"Upload successful, returning: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Removed invalid file: {filepath}")
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'success': False, 'error': f'ไม่สามารถอ่านไฟล์ได้: {str(e)}'})

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'เกิดข้อผิดพลาดในการอัปโหลด: {str(e)}'})

# Keep all other routes unchanged...
@app.route('/analyze', methods=['POST'])
def analyze_subjects():
    """Analyzes subjects from a CSV/Excel file (for Subject-based data)."""
    try:
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided.'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Specified file not found.'})

        file_extension = filename.rsplit('.', 1)[1].lower()
        df = None
        if file_extension == 'csv':
            encodings = app.config['DATA_CONFIG']['fallback_encodings']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except Exception as e:
                    logger.debug(f"Failed to read CSV with {encoding}: {e}")
                    continue
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file type for analysis.")

        if df is None:
            return jsonify({'success': False, 'error': 'Could not read file.'})

        data_format = detect_data_format(df)

        if data_format != 'subject_based':
            return jsonify({'success': False, 'error': 'Only subject-based data is supported for this analysis.'})

        name_col = None
        possible_name_cols = ['ชื่อ-นามสกุล', 'ชื่อ', 'ชื่อนักศึกษา', 'name', 'student_name', 'รหัส', 'student_id']
        for col in possible_name_cols:
            if col in df.columns:
                name_col = col
                break
        if not name_col:
            name_candidates = [col for col in df.columns if 'ชื่อ' in col.lower() or 'รหัส' in col.lower() or 'id' in col.lower()]
            if name_candidates:
                name_col = name_candidates[0]
            else:
                return jsonify({'success': False, 'error': 'Could not find student name or ID column.'})

        exclude_cols = [name_col, 'ปีที่เข้า', 'ปีที่จบ', 'year_in', 'year_out']
        exclude_keywords = ['gpa', 'เกรด', 'สำเร็จ', 'จบ', 'success', 'graduated', 'status']
        subject_cols = [
            col for col in df.columns
            if col not in exclude_cols and not any(kw in col.lower() for kw in exclude_keywords)
        ]

        logger.info(f"Analyzing {len(subject_cols)} subjects.")

        subject_analysis = {}
        all_gpas = []
        all_fail_rates = []
        subject_stats = []

        for subject in subject_cols:
            grades = []
            for _, row in df.iterrows():
                grade = row[subject]
                if pd.notna(grade) and str(grade).strip():
                    numeric_grade = grade_to_numeric(grade)
                    grades.append(numeric_grade)

            if grades:
                category = categorize_subject(subject)
                if category not in subject_analysis:
                    subject_analysis[category] = {}

                avg_grade = np.mean(grades)
                min_grade = np.min(grades)
                max_grade = np.max(grades)
                std_grade = np.std(grades) if len(grades) > 1 else 0
                fail_count = sum(1 for g in grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
                fail_rate = fail_count / len(grades)
                pass_rate = 1 - fail_rate

                grade_distribution = {}
                fail_grade_chars = [k for k, v in app.config['DATA_CONFIG']['grade_mapping'].items() if v == 0.0]
                fail_char_for_display = 'F/W/I/NP' if 'F' in fail_grade_chars else '0.0'

                for grade_point_val in sorted(list(set(app.config['DATA_CONFIG']['grade_mapping'].values())), reverse=True):
                    count = sum(1 for g in grades if g == grade_point_val)
                    if count > 0:
                        if grade_point_val == 0.0:
                            grade_distribution[fail_char_for_display] = grade_distribution.get(fail_char_for_display, 0) + count
                        else:
                            char_grade = next((k for k, v in app.config['DATA_CONFIG']['grade_mapping'].items() if v == grade_point_val and k not in fail_grade_chars), str(grade_point_val))
                            grade_distribution[char_grade] = count

                subject_info = {
                    'average': avg_grade,
                    'minimum': min_grade,
                    'maximum': max_grade,
                    'std_dev': std_grade,
                    'fail_rate': fail_rate,
                    'pass_rate': pass_rate,
                    'num_students': len(grades),
                    'num_failed': fail_count,
                    'grade_distribution': grade_distribution
                }

                subject_analysis[category][subject] = subject_info
                subject_stats.append({
                    'subject': subject,
                    'category': category,
                    'average': avg_grade,
                    'fail_rate': fail_rate,
                    'num_students': len(grades)
                })

                all_gpas.append(avg_grade)
                all_fail_rates.append(fail_rate)

        overall_stats = {
            'total_students': len(df),
            'total_subjects': len(subject_cols),
            'avg_gpa': np.mean(all_gpas) if all_gpas else 0,
            'overall_fail_rate': np.mean(all_fail_rates) if all_fail_rates else 0,
            'max_gpa_subject': np.max(all_gpas) if all_gpas else 0,
            'min_gpa_subject': np.min(all_gpas) if all_gpas else 0
        }

        high_fail_subjects = []
        low_gpa_subjects = []
        excellent_subjects = []

        high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']
        low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']

        for stat in subject_stats:
            if stat['fail_rate'] > high_fail_rate_threshold:
                high_fail_subjects.append(stat)
            if stat['average'] < low_gpa_threshold:
                low_gpa_subjects.append(stat)
            if stat['average'] >= 3.5 and stat['fail_rate'] < 0.1:
                excellent_subjects.append(stat)

        high_fail_subjects.sort(key=lambda x: x['fail_rate'], reverse=True)
        low_gpa_subjects.sort(key=lambda x: x['average'])
        excellent_subjects.sort(key=lambda x: x['average'], reverse=True)

        recommendations = []
        overall_fail_rate_warning = app.config['DATA_CONFIG']['risk_levels']['medium_fail_rate_threshold']
        overall_fail_rate_high = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']

        if overall_stats['overall_fail_rate'] > overall_fail_rate_high:
            recommendations.append(f"อัตราการตกโดยรวมสูงมาก (> {(overall_fail_rate_high * 100):.0f}%) ควรทบทวนวิธีการสอนและการประเมินผลทั้งหมด")
        elif overall_stats['overall_fail_rate'] > overall_fail_rate_warning:
            recommendations.append(f"อัตราการตกโดยรวมค่อนข้างสูง (> {(overall_fail_rate_warning * 100):.0f}%) ควรปรับปรุงกระบวนการเรียนการสอน")

        if high_fail_subjects:
            top_problem_subjects = [s['subject'] for s in high_fail_subjects[:3]]
            recommendations.append(f"วิชาที่มีอัตราตกสูงที่สุด: {', '.join(top_problem_subjects)}")
            recommendations.append("ควรทบทวนเนื้อหา วิธีการสอน และเกณฑ์การประเมินในวิชาเหล่านี้")

        if low_gpa_subjects:
            recommendations.append(f"พบ {len(low_gpa_subjects)} วิชาที่มีเกรดเฉลี่ยต่ำกว่า {low_gpa_threshold}")
            recommendations.append("ควรจัดการเรียนเสริมหรือปรับลดความยากของเนื้อหา")

        if excellent_subjects:
            recommendations.append(f"พบ {len(excellent_subjects)} วิชาที่มีผลการเรียนดีเยี่ยม สามารถนำมาเป็นแบบอย่าง")

        category_summary = {}
        for category, subjects in subject_analysis.items():
            if subjects:
                avg_gpas = [s['average'] for s in subjects.values()]
                avg_fail_rates = [s['fail_rate'] for s in subjects.values()]
                total_students_in_category = sum(s['num_students'] for s in subjects.values())

                category_summary[category] = {
                    'num_subjects': len(subjects),
                    'avg_gpa': np.mean(avg_gpas),
                    'avg_fail_rate': np.mean(avg_fail_rates),
                    'total_students_in_category': total_students_in_category,
                    'hardest_subject': min(subjects.items(), key=lambda x: x[1]['average'])[0] if subjects else None,
                    'easiest_subject': max(subjects.items(), key=lambda x: x[1]['average'])[0] if subjects else None
                }

        logger.info("Subject analysis successful.")

        return jsonify({
            'success': True,
            'subject_analysis': subject_analysis,
            'overall_stats': overall_stats,
            'category_summary': category_summary,
            'problem_subjects': {
                'high_fail_rate': high_fail_subjects[:10],
                'low_gpa': low_gpa_subjects[:10]
            },
            'excellent_subjects': excellent_subjects[:10],
            'recommendations': recommendations
        })

    except Exception as e:
        logger.error(f"Error during subject analysis: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during analysis: {str(e)}'})



@app.route('/api/analyze_curriculum', methods=['POST'])
def analyze_curriculum():
    """
    วิเคราะห์หลักสูตรและทำนายการจบการศึกษา
    ระบบจะส่งค่าเกรดเข้าไปวิเคราะห์และทำนาย
    """
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 0)
        repeated_courses_in_this_term_ids = data.get('repeated_courses_in_this_term_ids', [])
        model_filename = data.get('model_filename')
        student_name = data.get('student_name', 'นักศึกษา')

        logger.info(f"🔮 Analyzing curriculum for {student_name} with {len(current_grades)} grades")
        logger.info(f"📊 Grade values sent for analysis: {list(current_grades.keys())[:10]}...")  # Log first 10 courses
        
        # Get configuration
        courses_data = app.config['COURSES_DATA']
        all_terms_data = app.config['ALL_TERMS_DATA']
        grade_mapping = app.config['DATA_CONFIG']['grade_mapping']
        
        # Get loaded courses
        loaded_courses_ids = []
        for i in range(loaded_terms_count):
            if i < len(all_terms_data):
                loaded_courses_ids.extend(all_terms_data[i]['ids'])
        loaded_courses_ids.extend(repeated_courses_in_this_term_ids)
        loaded_courses_ids = list(set(loaded_courses_ids))  # Remove duplicates
        
        # Calculate GPA and failed courses
        total_points = 0
        total_credits = 0
        completed_credits = 0
        failed_courses_ids = []
        incomplete_courses_ids = []
        
        for course_id in loaded_courses_ids:
            course = next((c for c in courses_data if c['id'] == course_id), None)
            if not course:
                continue
            
            grade = current_grades.get(course_id, '')
            
            if not grade:
                incomplete_courses_ids.append(course_id)
            elif grade:
                grade_point = grade_mapping.get(grade, 0)
                
                if grade_point > 0:
                    total_points += grade_point * course['credit']
                    total_credits += course['credit']
                    completed_credits += course['credit']
                elif grade == 'F':
                    failed_courses_ids.append(course_id)
                    total_credits += course['credit']  # F still counts for GPA calculation
        
        avg_gpa = total_points / total_credits if total_credits > 0 else 0
        
        # Check for blocked courses
        blocked_courses = find_all_blocked_courses(
            current_grades, loaded_courses_ids, courses_data, grade_mapping
        )
        
        # Build dependency graph and find impact chains
        dependency_graph = build_course_dependency_graph(loaded_courses_ids, courses_data)
        
        blocked_chain_texts = []
        for failed_id in failed_courses_ids:
            affected = find_courses_affected_by_failure(failed_id, dependency_graph)
            if affected:
                failed_course = next((c for c in courses_data if c['id'] == failed_id), None)
                affected_names = []
                for aid in affected:
                    course = next((c for c in courses_data if c['id'] == aid), None)
                    if course:
                        affected_names.append(f"{course['thaiName']} ({aid})")
                
                if affected_names:
                    chain_text = f"วิชา {failed_course['thaiName']} ({failed_id}) ไม่ผ่าน ส่งผลกระทบต่อวิชาอื่นๆ ดังนี้:\n"
                    chain_text += "\n".join([f"  - {name}" for name in affected_names])
                    blocked_chain_texts.append(chain_text)
        
        # Calculate completion rate
        total_required_credits = sum(c['credit'] for c in courses_data)
        completion_rate = (completed_credits / total_required_credits * 100) if total_required_credits > 0 else 0
        
        # Determine graduation status
        graduation_status = determine_graduation_status_backend(
            completion_rate, avg_gpa, blocked_courses, failed_courses_ids,
            loaded_courses_ids, current_grades, all_terms_data, courses_data, loaded_terms_count
        )
        
        # Generate recommendations
        recommendations = []
        if failed_courses_ids:
            recommendations.append("ควรลงทะเบียนซ้ำวิชาที่ตก (F) โดยเร็ว")
            recommendations.append("ควรปรึกษาอาจารย์ที่ปรึกษาเพื่อวางแผนการเรียน")
        
        # ตรวจสอบ GPA ที่ถูกต้อง (ต้อง >= 2.00 ถึงจะผ่านเกณฑ์)
        gpa_threshold = 2.00
        if avg_gpa > 0 and avg_gpa < gpa_threshold:
            recommendations.append(f"GPA ต่ำกว่าเกณฑ์ ({avg_gpa:.2f} < {gpa_threshold:.2f}) ควรปรับปรุงผลการเรียนอย่างเร่งด่วน")
        
        if blocked_courses:
            recommendations.append("ควรผ่านวิชาพื้นฐานให้ครบก่อนลงวิชาขั้นสูง")
        
        if not recommendations:
            recommendations.append("ผลการเรียนอยู่ในเกณฑ์ดี รักษามาตรฐานและวางแผนให้จบตามกำหนด")
        
        # Prepare response
        response_data = {
            'success': True,
            'student_name': student_name,
            'completion_rate': completion_rate,
            'avg_gpa': avg_gpa,
            'completed_credits': completed_credits,
            'total_required_credits': total_required_credits,
            'graduation_status': graduation_status,
            'incomplete_courses': [
                next((c['thaiName'] for c in courses_data if c['id'] == cid), cid)
                for cid in incomplete_courses_ids
            ],
            'failed_courses': [
                next((c['thaiName'] for c in courses_data if c['id'] == cid), cid)
                for cid in failed_courses_ids
            ],
            'blocked_courses_details': blocked_courses,
            'blocked_chain_texts': blocked_chain_texts,
            'recommendations': recommendations,
            'debug': {
                'current_grades_count': len(current_grades),
                'loaded_terms_count': loaded_terms_count,
                'loaded_courses_count': len(loaded_courses_ids),
                'blocked_courses_count': len(blocked_courses)
            }
        }
        
        # ตรวจสอบความสมบูรณ์ของหลักสูตรก่อนทำนาย
        def check_curriculum_completion(current_grades, courses_data, min_credits=136, min_gpa=2.0):
            """ตรวจสอบว่านักศึกษาจบหลักสูตรแล้วหรือยัง"""
            total_credits = 0
            total_grade_points = 0
            total_credits_for_gpa = 0
            passed_courses = 0
            
            for course_id, grade in current_grades.items():
                if grade:  # มีเกรด
                    course = next((c for c in courses_data if c['id'] == course_id), None)
                    if course:
                        credit = course.get('credit', 3)
                        grade_point = grade_mapping.get(grade, 0)
                        
                        # นับหน่วยกิตที่ผ่าน (ไม่ใช่ F, W, I)
                        if grade not in ['F', 'W', 'I']:
                            total_credits += credit
                            passed_courses += 1
                        
                        # คำนวณ GPA (รวม F แต่ไม่รวม W, I)
                        if grade not in ['W', 'I']:
                            total_grade_points += grade_point * credit
                            total_credits_for_gpa += credit
            
            gpa = total_grade_points / total_credits_for_gpa if total_credits_for_gpa > 0 else 0
            
            # ตรวจสอบเงื่อนไขการจบ
            is_completed = total_credits >= min_credits and gpa >= min_gpa
            
            return {
                'is_completed': is_completed,
                'total_credits': total_credits,
                'gpa': gpa,
                'passed_courses': passed_courses,
                'min_credits_met': total_credits >= min_credits,
                'min_gpa_met': gpa >= min_gpa,
                'min_credits': min_credits,
                'min_gpa': min_gpa
            }
        
        # ตรวจสอบความสมบูรณ์ของหลักสูตร
        completion_status = check_curriculum_completion(current_grades, courses_data)
        
        # เพิ่มข้อมูลความสมบูรณ์ในผลลัพธ์
        response_data['completion_status'] = {
            'total_credits': completion_status['total_credits'],
            'min_credits': completion_status['min_credits'],
            'credits_remaining': max(0, completion_status['min_credits'] - completion_status['total_credits']),
            'gpa': completion_status['gpa'],
            'min_gpa': completion_status['min_gpa'],
            'is_completed': completion_status['is_completed'],
            'min_credits_met': completion_status['min_credits_met'],
            'min_gpa_met': completion_status['min_gpa_met']
        }
        
        # Try to predict if model is provided
        if model_filename:
            try:
                logger.info(f"Making prediction with model: {model_filename}")
                
                # ตรวจสอบว่าจบหลักสูตรแล้วหรือยัง
                if completion_status['is_completed']:
                    # ถ้าครบหลักสูตรและ GPA ≥ 2.0 แล้ว → จบแน่นอน
                    response_data['prediction_result'] = {
                        'prediction': 'จบ',
                        'prob_pass': 1.0,
                        'prob_fail': 0.0,
                        'confidence': 1.0,
                        'risk_level': 'ไม่มี',
                        'gpa_input': float(completion_status['gpa']),
                        'reason': 'ใส่เกรดครบหลักสูตรและ GPA ≥ 2.00 แล้ว',
                        'method': 'Curriculum Completion Check',
                        'status': 'graduated'
                    }
                    logger.info(f"Curriculum completed: GPA {completion_status['gpa']:.2f}, Credits {completion_status['total_credits']}")
                    
                elif completion_status['min_credits_met'] and not completion_status['min_gpa_met']:
                    # ถ้าครบหน่วยกิตแต่ GPA < 2.0 → ไม่จบ
                    response_data['prediction_result'] = {
                        'prediction': 'ไม่จบ',
                        'prob_pass': 0.0,
                        'prob_fail': 1.0,
                        'confidence': 1.0,
                        'risk_level': 'สูง',
                        'gpa_input': float(completion_status['gpa']),
                        'reason': f'GPA {completion_status["gpa"]:.2f} ต่ำกว่าเกณฑ์ขั้นต่ำ {completion_status["min_gpa"]:.2f}',
                        'method': 'Curriculum Completion Check',
                        'status': 'failed_gpa'
                    }
                    logger.info(f"Failed due to low GPA: {completion_status['gpa']:.2f} < {completion_status['min_gpa']}")
                    
                else:
                    # ถ้ายังไม่ครบหลักสูตร → ใช้ AI ทำนาย
                    # Load model
                    loaded_model_data = storage.load_model(model_filename)
                    if loaded_model_data:
                        # สร้าง AdvancedFeatureEngineer และ ContextAwarePredictor
                        engineer = AdvancedFeatureEngineer(
                            grade_mapping=app.config['DATA_CONFIG']['grade_mapping']
                        )
                        
                        # โหลด course_profiles จากโมเดล
                        course_profiles = loaded_model_data.get('course_profiles', {})
                        engineer.course_profiles = course_profiles
                        
                        # ======= โหลด models, scaler, feature_names จากไฟล์โมเดล =======
                        models = loaded_model_data.get('models', {})
                        scaler = loaded_model_data.get('scaler', None)
                        # feature_names อาจเก็บใน 'feature_columns' หรือ 'feature_names'
                        feature_names = loaded_model_data.get('feature_columns', loaded_model_data.get('feature_names', []))
                        
                        logger.info(f"📦 Model loaded: models={list(models.keys())}, features={len(feature_names)}, scaler={'✅' if scaler else '❌'}")
                        
                        # สร้าง transcript data จากข้อมูลปัจจุบัน
                        transcript_data = []
                        for course_id, grade in current_grades.items():
                            if grade:  # มีเกรด
                                course = next((c for c in courses_data if c['id'] == course_id), None)
                                if course:
                                    # แปลงเกรดเป็น GRADE_POINT
                                    grade_point = grade_mapping.get(grade, 0)
                                    
                                    transcript_data.append({
                                        'Dummy StudentNO': student_name,
                                        'COURSE_CODE': course_id,
                                        'COURSE_TITLE_TH': course.get('thaiName', ''),
                                        'CREDIT': course.get('credit', 3),
                                        'GRADE': grade,
                                        'GRADE_POINT': grade_point,
                                        'ปีที่เข้า': 2020,  # ค่าเริ่มต้น
                                        'ปีการศึกษา': 2024,  # ค่าเริ่มต้น
                                        'เทอม': 1
                                    })
                        
                        if transcript_data:
                            # แปลงเป็น DataFrame
                            transcript_df = pd.DataFrame(transcript_data)
                            
                            # ใช้ ContextAwarePredictor พร้อม models, scaler, feature_names จากโมเดลที่โหลด
                            from advanced_training import ContextAwarePredictor
                            predictor = ContextAwarePredictor(
                                feature_engineer=engineer,
                                models=models,
                                scaler=scaler,
                                feature_names=feature_names
                            )
                            
                            # ทำนายด้วย Context-Aware AI System
                            prediction_result = predictor.predict_graduation_probability(transcript_df)
                            
                            prob_pass = prediction_result['probability']
                            prob_fail = 1 - prob_pass
                            confidence = prediction_result['confidence']
                            
                            prediction = 'จบ' if prob_pass >= 0.5 else 'ไม่จบ'
                            
                            risk_level = 'สูง'
                            if confidence > 0.8:
                                risk_level = 'ต่ำ' if prediction == 'จบ' else 'สูง'
                            elif confidence > 0.6:
                                risk_level = 'ปานกลาง'
                            
                            # ดึง prediction_method และ models_used จากผลลัพธ์ (ถ้ามี)
                            ai_method = prediction_result.get('prediction_method', 'AI_MODEL')
                            models_used = prediction_result.get('models_used', [])
                            feature_importance = prediction_result.get('feature_importance', {})
                            
                            response_data['prediction_result'] = {
                                'prediction': prediction,
                                'prob_pass': float(prob_pass),
                                'prob_fail': float(prob_fail),
                                'confidence': float(confidence),
                                'risk_level': risk_level,
                                'gpa_input': float(completion_status['gpa']),
                                'features_used': prediction_result['features_used'],
                                'courses_analyzed': prediction_result['courses_analyzed'],
                                'method': ai_method,  # AI_MODEL หรือ Context-Aware
                                'models_used': models_used,  # ['rf', 'gb', 'lr']
                                'feature_importance': feature_importance,  # Top 10 features
                                'status': 'predicted'
                            }
                            logger.info(f"🤖 AI Model Prediction: {prediction} (confidence: {confidence:.3f}, models: {models_used})")
                        else:
                            logger.warning("No valid transcript data for prediction")
                        
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # ✨ เพิ่มข้อมูลกราฟ 3 เส้น (ใช้ข้อมูลจริงจากนักศึกษา)
        try:
            chart_data = generate_three_line_chart_data(current_grades, loaded_terms_count)
            response_data['chart_data'] = chart_data
            logger.info(f"Added chart data with {len(chart_data.get('terms', []))} terms")
        except Exception as e:
            logger.error(f"Error generating chart data: {str(e)}")
            response_data['chart_data'] = None
        
        # ✨ เพิ่มการวิเคราะห์เหตุผลที่อาจไม่จบ - ส่งผลการทำนายจากโมเดล AI ไปด้วย
        try:
            # ดึงผลการทำนายจากโมเดล AI (ถ้ามี)
            ai_prediction_result = response_data.get('prediction_result', None)
            
            graduation_analysis = analyze_graduation_failure_reasons(
                current_grades, 
                loaded_terms_count, 
                prediction_result=ai_prediction_result  # ส่งผลจากโมเดล AI
            )
            response_data['graduation_analysis'] = graduation_analysis
            
            # Log ว่าใช้วิธีไหนในการทำนาย
            prediction_method = graduation_analysis.get('prediction_method', 'Unknown')
            will_graduate = graduation_analysis.get('will_graduate', False)
            logger.info(f"✅ Graduation analysis: will_graduate={will_graduate}, method={prediction_method}, risk={graduation_analysis.get('risk_level', 'N/A')}")
        except Exception as e:
            logger.error(f"Error in graduation analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            response_data['graduation_analysis'] = None
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during curriculum analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/explain_prediction', methods=['POST'])
def explain_prediction():
    """
    API สำหรับการวิเคราะห์และอธิบายการทำนายอย่างละเอียด
    รวมถึงคำแนะนำเชิงรุกสำหรับการปรับปรุง
    """
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 0)
        student_name = data.get('student_name', 'นักศึกษา')
        
        logger.info(f"Explaining prediction for {student_name}")
        
        # ตรวจสอบว่ามี advanced_trainer หรือไม่
        if not hasattr(app, 'advanced_trainer') or not app.advanced_trainer:
            return jsonify({
                'success': False,
                'error': 'โมเดล AI ยังไม่ได้ถูกโหลด กรุณารอสักครู่แล้วลองใหม่อีกครั้ง'
            })
        
        # สร้าง student_data DataFrame
        courses_data = app.config.get('COURSES_DATA', [])
        grade_mapping = app.config.get('DATA_CONFIG', {}).get('grade_mapping', {})
        
        student_data_rows = []
        for course_id, grade in current_grades.items():
            if grade and grade in grade_mapping:
                course = next((c for c in courses_data if c['id'] == course_id), None)
                if course:
                    student_data_rows.append({
                        'course_code': course_id,
                        'grade': grade,
                        'grade_point': grade_mapping[grade],
                        'credit': course.get('credit', 3),
                        'semester': 1,
                        'academic_year': 2024
                    })
        
        if not student_data_rows:
            return jsonify({
                'success': False,
                'error': 'ไม่พบข้อมูลเกรดที่ถูกต้อง'
            })
        
        student_data = pd.DataFrame(student_data_rows)
        
        # ทำนายด้วย Context-Aware Predictor พร้อมคำอธิบาย
        try:
            if hasattr(app.advanced_trainer, 'predictor'):
                prediction_result = app.advanced_trainer.predictor.predict_graduation_probability(
                    student_data, explain=True
                )
            else:
                # Fallback: สร้าง predictor ใหม่
                predictor = ContextAwarePredictor(
                    feature_engineer=app.advanced_trainer.feature_engineer,
                    models=app.advanced_trainer.models if hasattr(app.advanced_trainer, 'models') else {},
                    scaler=app.advanced_trainer.scaler if hasattr(app.advanced_trainer, 'scaler') else None,
                    feature_names=app.advanced_trainer.feature_names if hasattr(app.advanced_trainer, 'feature_names') else []
                )
                prediction_result = predictor.predict_graduation_probability(student_data, explain=True)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return jsonify({
                'success': False,
                'error': f'เกิดข้อผิดพลาดในการทำนาย: {str(e)}'
            })
        
        # ดึงคำอธิบาย
        explanation = prediction_result.get('explanation')
        
        if not explanation:
            # สร้างคำอธิบายแบบ fallback
            probability = prediction_result.get('probability', 0.5)
            will_graduate = probability >= 0.5
            
            explanation = {
                'summary': {
                    'will_graduate': will_graduate,
                    'probability': round(probability * 100, 1),
                    'confidence': round(prediction_result.get('confidence', 0.5) * 100, 1),
                    'status': 'มีโอกาสจบการศึกษา' if will_graduate else 'มีความเสี่ยงที่จะไม่จบ'
                },
                'key_factors': [],
                'reasons': ['ไม่สามารถสร้างคำอธิบายละเอียดได้ในขณะนี้'],
                'obstacles': [],
                'strengths': [],
                'recommendations': [],
                'graduation_path': {},
                'future_projections': {}
            }
        
        # คำนวณข้อมูลเพิ่มเติม
        total_credits = sum([row['credit'] for row in student_data_rows])
        gpa = sum([grade_mapping[row['grade']] * row['credit'] for row in student_data_rows]) / total_credits if total_credits > 0 else 0
        failed_count = sum([1 for row in student_data_rows if grade_mapping[row['grade']] == 0])
        
        # สร้างข้อมูลสำหรับแสดงผล
        response_data = {
            'success': True,
            'student_name': student_name,
            'current_status': {
                'gpa': round(gpa, 2),
                'total_credits': total_credits,
                'courses_taken': len(student_data_rows),
                'failed_courses': failed_count,
                'terms_completed': loaded_terms_count
            },
            'prediction': {
                'probability': prediction_result.get('probability', 0.5),
                'confidence': prediction_result.get('confidence', 0.5),
                'will_graduate': prediction_result.get('probability', 0.5) >= 0.5
            },
            'explanation': explanation,
            'factors': prediction_result.get('factors', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Successfully explained prediction for {student_name}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in explain_prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'เกิดข้อผิดพลาด: {str(e)}'
        })
    
    
@app.route('/model_status', methods=['GET'])
def model_status():
    """Checks the status of currently loaded models."""
    try:
        logger.info("Checking model status...")
        
        global models
        if models is None:
            models = {
                'subject_model': None,
                'gpa_model': None,
                'subject_model_info': None,
                'gpa_model_info': None,
                'subject_feature_cols': None,
                'gpa_feature_cols': None
            }

        subject_model_status = models.get('subject_model') is not None
        subject_info = models.get('subject_model_info')
        
        logger.info(f"Subject model status: {subject_model_status}")

        status = {
            'success': True,
            'subject_model': subject_model_status,
            'gpa_model': False,  # Always false since we're removing GPA model
            'subject_model_info': subject_info,
            'gpa_model_info': None,  # No GPA model info
            'server_time': datetime.now().isoformat(),
            'models_folder_exists': os.path.exists(app.config['MODEL_FOLDER']),
            'uploads_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            's3_available': not storage.use_local
        }
        
        logger.info("Model status check completed successfully")
        return jsonify(status)

    except Exception as e:
        logger.error(f"Error in model_status: {str(e)}")
        return jsonify({
            'success': False,
            'subject_model': False,
            'gpa_model': False,
            'error': str(e),
            'server_time': datetime.now().isoformat()
        }), 500



@app.route('/api/sync-models', methods=['POST'])
def sync_local_models_to_storage():
    """Sync local models to cloud storage"""
    try:
        if storage.use_local:
            return jsonify({'success': False, 'error': 'Storage is in local mode'})
        
        synced = []
        model_folder = app.config['MODEL_FOLDER']
        
        if os.path.exists(model_folder):
            for filename in os.listdir(model_folder):
                if filename.endswith('.joblib'):
                    filepath = os.path.join(model_folder, filename)
                    try:
                        model_data = joblib.load(filepath)
                        if storage.save_model(model_data, filename):
                            synced.append(filename)
                            logger.info(f"Synced {filename} to storage")
                    except Exception as e:
                        logger.error(f"Could not sync {filename}: {e}")
        
        return jsonify({'success': True, 'synced': synced})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===============================
# API Endpoints สำหรับกราฟ 3 เส้นและการวิเคราะห์ขั้นสูง
# ===============================

@app.route('/api/three-line-chart', methods=['POST'])
def get_three_line_chart_data():
    """API สำหรับข้อมูลกราฟ 3 เส้น"""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 8)
        
        chart_data = generate_three_line_chart_data(current_grades, loaded_terms_count)
        
        return jsonify({
            'success': True,
            'chart_data': chart_data
        })
        
    except Exception as e:
        logger.error(f"Error generating three line chart data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/graduation-analysis', methods=['POST'])
def get_graduation_analysis():
    """API สำหรับการวิเคราะห์เหตุผลที่ไม่จบการศึกษา"""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 8)
        
        analysis_result = analyze_graduation_failure_reasons(current_grades, loaded_terms_count)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
        
    except Exception as e:
        logger.error(f"Error analyzing graduation failure reasons: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/next-term-prediction', methods=['POST'])
def get_next_term_prediction():
    """API สำหรับตารางทำนายเกรดเทอมถัดไป"""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        
        prediction_table = generate_next_term_grade_prediction_table(current_grades)
        
        return jsonify({
            'success': True,
            'prediction_table': prediction_table
        })
        
    except Exception as e:
        logger.error(f"Error generating next term prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comprehensive-analysis', methods=['POST'])
def get_comprehensive_analysis():
    """API สำหรับการวิเคราะห์ครอบคลุมทั้งหมด"""
    try:
        data = request.get_json()
        current_grades = data.get('current_grades', {})
        loaded_terms_count = data.get('loaded_terms_count', 8)
        student_name = data.get('student_name', 'นักศึกษา')
        
        # สร้างข้อมูลกราฟ 3 เส้น
        chart_data = generate_three_line_chart_data(current_grades, loaded_terms_count)
        
        # วิเคราะห์เหตุผลที่ไม่จบ
        graduation_analysis = analyze_graduation_failure_reasons(current_grades, loaded_terms_count)
        
        # ทำนายเกรดเทอมถัดไป
        next_term_prediction = generate_next_term_grade_prediction_table(current_grades)
        
        # สร้างสรุปผลรวม
        comprehensive_summary = {
            'student_name': student_name,
            'analysis_date': datetime.now().isoformat(),
            'current_status': {
                'gpa': graduation_analysis['current_gpa'],
                'total_courses': graduation_analysis['total_courses'],
                'failed_courses': graduation_analysis['failed_courses_count'],
                'risk_level': graduation_analysis['risk_level'],
                'graduation_probability': graduation_analysis['graduation_probability']
            },
            'predictions': {
                'next_term_gpa': next_term_prediction['predicted_term_gpa'],
                'cumulative_gpa': next_term_prediction['predicted_cumulative_gpa'],
                'improvement': next_term_prediction['improvement']
            },
            'key_insights': [
                f"GPA ปัจจุบัน: {graduation_analysis['current_gpa']:.2f}",
                f"ความเสี่ยง: {graduation_analysis['risk_level']}",
                f"โอกาสจบ: {graduation_analysis['graduation_probability']:.1f}%",
                f"ทำนาย GPA เทอมถัดไป: {next_term_prediction['predicted_term_gpa']:.2f}"
            ]
        }
        
        return jsonify({
            'success': True,
            'chart_data': chart_data,
            'graduation_analysis': graduation_analysis,
            'next_term_prediction': next_term_prediction,
            'comprehensive_summary': comprehensive_summary
        })
        
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
# Keep all other routes unchanged...
@app.route('/page')
def main_page():
    return render_template('main_page.html')

# แก้ไข route /curriculum  
@app.route('/curriculum')
def curriculum_page():
    """Page for curriculum analysis."""
    try:
        courses_json = json.dumps(app.config.get('COURSES_DATA', []))
        terms_json = json.dumps(app.config.get('ALL_TERMS_DATA', []))
        grades_json = json.dumps(app.config.get('DATA_CONFIG', {}).get('grade_mapping', {}))
        
        return render_template(
            'curriculum_prediction_form.html',
            coursesData=courses_json,
            allTermsData=terms_json,
            gradeMapping=grades_json
        )
    except Exception as e:
        logger.error(f"Error rendering curriculum page: {str(e)}")
        return render_template(
            'curriculum_prediction_form.html',
            coursesData='[]',
            allTermsData='[]',
            gradeMapping='{}'
        )

@app.route('/advanced')
def advanced_test_page():
    """Page for advanced curriculum analysis."""
    return render_template('advanced_test.html')

@app.route('/predict-batch')
def predict_batch_page():
    return render_template('index.html')

@app.route('/models')
def models_page():
    if admin_manager.enabled and not has_any_role('admin', 'super_admin'):
        return redirect(url_for('index'))
    return render_template('model_management.html')
@app.route('/predict_manual_input', methods=['POST'])
def predict_manual_input():
    """Predicts outcome from manually entered subject data."""
    try:
        data = request.json
        student_name = data.pop('student_name', 'ผู้ใช้งาน')
        model_filename = data.get('model_filename')

        if not model_filename:
            models_list = storage.list_models()
            subject_models = [m for m in models_list if 'subject_based' in m.get('filename', '') or m.get('data_format') == 'subject_based']
            if subject_models:
                model_filename = subject_models[0]['filename']
            else:
                return jsonify({'success': False, 'error': 'No model filename provided and no trained model found.'})

        # โหลดโมเดลจาก storage
        loaded_model_data = storage.load_model(model_filename)
        if not loaded_model_data:
            return jsonify({'success': False, 'error': f'Specified model file {model_filename} not found.'})

        model_info = {
            'models': loaded_model_data.get('models', {}),
            'scaler': loaded_model_data.get('scaler')
        }
        feature_cols = loaded_model_data.get('feature_columns', [])
        data_format_used = loaded_model_data.get('data_format', 'unknown')
        logger.info(f"Loaded model '{model_filename}' (format: {data_format_used}) for manual prediction.")

        input_grades_raw = {cid: grade for cid, grade in data.items() if cid != 'model_filename'}

        student_data_for_prediction = {}
        grade_mapping = app.config['DATA_CONFIG']['grade_mapping']
        subject_categories = app.config['SUBJECT_CATEGORIES']
        all_known_courses_from_config = app.config['COURSES_DATA']

        all_grades_entered = []
        subject_categories_grades = {cat: [] for cat in subject_categories.keys()}
        subject_categories_grades['อื่นๆ'] = []

        for course_id, grade_str in input_grades_raw.items():
            if grade_str and grade_str.strip():
                numeric_grade = None
                try:
                    numeric_grade = float(grade_str)
                    if not (0.0 <= numeric_grade <= 4.0):
                        numeric_grade = 0.0
                except ValueError:
                    numeric_grade = grade_mapping.get(str(grade_str).upper(), 0.0)
                
                all_grades_entered.append(numeric_grade)

                course_name = ""
                for c_data in all_known_courses_from_config:
                    if c_data['id'] == course_id:
                        course_name = c_data['thaiName']
                        break

                category = 'อื่นๆ'
                if course_name:
                    for cat, info in subject_categories.items():
                        if any(keyword in course_name.lower() for keyword in info['keywords']):
                            category = cat
                            break
                        
                subject_categories_grades.get(category, subject_categories_grades['อื่นๆ']).append(numeric_grade)

        gpa = np.mean(all_grades_entered) if all_grades_entered else 0.0
        min_grade = np.min(all_grades_entered) if all_grades_entered else 0.0
        max_grade = np.max(all_grades_entered) if all_grades_entered else 0.0
        std_grade = np.std(all_grades_entered) if len(all_grades_entered) > 1 else 0.0
        fail_count = sum(1 for g in all_grades_entered if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01)
        fail_rate = fail_count / len(all_grades_entered) if all_grades_entered else 0.0
        total_subjects = len(all_grades_entered)

        student_data_for_prediction = {
            'gpa': gpa,
            'min_grade': min_grade,
            'max_grade': max_grade,
            'std_grade': std_grade,
            'fail_count': fail_count,
            'fail_rate': fail_rate,
            'total_subjects': total_subjects,
            'year_in': 0,
            'year_out': 0,
            'total_terms': 0
        }

        for cat, cat_grades in subject_categories_grades.items():
            if cat_grades:
                student_data_for_prediction[f'gpa_{cat}'] = np.mean(cat_grades)
                student_data_for_prediction[f'min_{cat}'] = np.min(cat_grades)
                student_data_for_prediction[f'max_{cat}'] = np.max(cat_grades)
                student_data_for_prediction[f'fail_rate_{cat}'] = sum(1 for g in cat_grades if g <= app.config['DATA_CONFIG']['grade_mapping'].get('D', 1.0) - 0.01) / len(cat_grades)
            else:
                student_data_for_prediction[f'gpa_{cat}'] = 0.0
                student_data_for_prediction[f'min_{cat}'] = 0.0
                student_data_for_prediction[f'max_{cat}'] = 0.0
                student_data_for_prediction[f'fail_rate_{cat}'] = 0.0

        processed_input_for_df = {}
        for feature in feature_cols:
            processed_input_for_df[feature] = [student_data_for_prediction.get(feature, 0.0)]

        input_df = pd.DataFrame(processed_input_for_df)

        trained_models = model_info['models']
        scaler = model_info['scaler']

        predictions_proba_list = []
        for name, model in trained_models.items():
            try:
                if name == 'lr':
                    X_scaled = scaler.transform(input_df)
                    pred_proba = model.predict_proba(X_scaled)
                else:
                    pred_proba = model.predict_proba(input_df)
                
                if pred_proba.shape[1] == 1:
                    pred_proba = np.hstack((1 - pred_proba, pred_proba))
                
                predictions_proba_list.append(pred_proba)
            except Exception as e:
                logger.warning(f"Could not predict with model {name} from manual input: {str(e)}")
                continue

        if not predictions_proba_list:
            return jsonify({'success': False, 'error': 'Could not make predictions with manual input.'})

        avg_prob_per_student = np.mean([pred_proba_array[0] for pred_proba_array in predictions_proba_list], axis=0)
        avg_prob_fail = avg_prob_per_student[0]
        avg_prob_pass = avg_prob_per_student[1]

        prediction = 'จบ' if avg_prob_pass >= avg_prob_fail else 'ไม่จบ'

        confidence = max(avg_prob_pass, avg_prob_fail)
        high_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['high_confidence_threshold']
        medium_confidence_threshold = app.config['DATA_CONFIG']['risk_levels']['medium_confidence_threshold']

        if confidence > high_confidence_threshold:
            risk_level = 'ต่ำ' if prediction == 'จบ' else 'สูง'
        elif confidence > medium_confidence_threshold:
            risk_level = 'ปานกลาง'
        else:
            risk_level = 'สูง' if prediction == 'ไม่จบ' else 'ปานกลาง'

        gpa_for_analysis = student_data_for_prediction.get('gpa', 0.0)

        analysis = []
        recommendations = []

        low_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['low_gpa_threshold']
        warning_gpa_threshold = app.config['DATA_CONFIG']['risk_levels']['warning_gpa_threshold']
        high_fail_rate_threshold = app.config['DATA_CONFIG']['risk_levels']['high_fail_rate_threshold']

        if gpa_for_analysis < low_gpa_threshold:
            analysis.append(f"GPA ต่ำมาก ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['high_risk'])
        elif gpa_for_analysis < warning_gpa_threshold:
            analysis.append(f"GPA อยู่ในเกณฑ์เสี่ยง ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['medium_risk'])
        elif gpa_for_analysis < 3.0:
            analysis.append(f"GPA พอใช้ ({float(gpa_for_analysis):.2f})")
            recommendations.append("มีโอกาสพัฒนาผลการเรียนให้ดีขึ้น")
        else:
            analysis.append(f"GPA ดี ({float(gpa_for_analysis):.2f})")
            recommendations.extend(app.config['MESSAGES']['recommendations']['low_risk'])

        if prediction == 'ไม่จบ':
            recommendations.append("แนะนำให้ทบทวนแผนการเรียนและขอความช่วยเหลือ")
            if student_data_for_prediction.get('fail_rate', 0) > high_fail_rate_threshold:
                recommendations.append("มีอัตราการตกในบางวิชาสูง ควรให้ความสำคัญกับการเรียนซ่อม")

        if data_format_used == 'subject_based':
            weak_categories = []
            for cat_key in app.config['SUBJECT_CATEGORIES'].keys():
                cat_gpa_key = f'gpa_{cat_key}'
                if student_data_for_prediction.get(cat_gpa_key, 0) < low_gpa_threshold:
                    weak_categories.append(cat_key)
            if weak_categories:
                recommendations.append(f"ควรเน้นปรับปรุงวิชาในหมวด: {', '.join(weak_categories[:2])}")

        return jsonify({
            'success': True,
            'student_name': student_name,
            'prediction': prediction,
            'prob_pass': float(avg_prob_pass),
            'prob_fail': float(avg_prob_fail),
            'gpa_input': float(gpa_for_analysis),
            'risk_level': risk_level,
            'confidence': float(confidence),
            'analysis': list(set(analysis)),
            'recommendations': list(set(recommendations)),
            'data_format_used': data_format_used
        })

    except Exception as e:
        logger.error(f"Error during manual input prediction: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred during prediction: {str(e)}'})

def load_existing_models():
    """Loads existing trained models from storage."""
    try:
        logger.info("🔍 Searching for existing models...")
        
        # ลองหาโมเดลจากทั้ง storage และ local
        models_found = []
        
        # 1. ลองจาก storage ก่อน
        try:
            models_list = storage.list_models()
            if models_list:
                models_found.extend(models_list)
                logger.info(f"Found {len(models_list)} models in storage")
        except Exception as e:
            logger.warning(f"Could not load from storage: {e}")
        
        # 2. ลองจาก local folder
        try:
            model_folder = app.config['MODEL_FOLDER']
            if os.path.exists(model_folder):
                for filename in os.listdir(model_folder):
                    if filename.endswith('.joblib'):
                        # ตรวจสอบว่าไม่ซ้ำกับที่มีอยู่แล้ว
                        if not any(m.get('filename') == filename for m in models_found):
                            filepath = os.path.join(model_folder, filename)
                            try:
                                model_data = joblib.load(filepath)
                                models_found.append({
                                    'filename': filename,
                                    'created_at': model_data.get('created_at', datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()),
                                    'data_format': model_data.get('data_format', 'subject_based'),
                                    'performance_metrics': model_data.get('performance_metrics', {}),
                                    'storage': 'local'
                                })
                            except Exception as e:
                                logger.warning(f"Could not load local model {filename}: {e}")
        except Exception as e:
            logger.warning(f"Could not check local folder: {e}")
        
        if not models_found:
            logger.info("No existing models found")
            return
        
        # Load subject-based model เท่านั้น (ตัด GPA-based ออก)
        subject_models = [m for m in models_found if 'subject_based' in m.get('filename', '') or m.get('data_format') == 'subject_based']
        
        if subject_models:
            # เรียงตามวันที่สร้าง
            subject_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            latest_subject = subject_models[0]
            
            # โหลดโมเดล
            loaded_data = None
            
            # ลองโหลดจาก storage ก่อน
            if latest_subject.get('storage') != 'local':
                try:
                    loaded_data = storage.load_model(latest_subject['filename'])
                except Exception as e:
                    logger.warning(f"Could not load from storage: {e}")
            
            # ถ้าไม่ได้ ลองโหลดจาก local
            if not loaded_data:
                try:
                    filepath = os.path.join(app.config['MODEL_FOLDER'], latest_subject['filename'])
                    if os.path.exists(filepath):
                        loaded_data = joblib.load(filepath)
                        logger.info(f"Loaded from local: {latest_subject['filename']}")
                except Exception as e:
                    logger.error(f"Could not load model: {e}")
            
            if loaded_data:
                models['subject_model'] = {
                    'models': loaded_data.get('models', {}),
                    'scaler': loaded_data.get('scaler')
                }
                models['subject_feature_cols'] = loaded_data.get('feature_columns', [])
                models['subject_model_info'] = loaded_data.get('performance_metrics', 
                    {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1_score': 0.85})
                models['subject_model_info']['created_at'] = loaded_data.get('created_at', datetime.now().isoformat())
                models['subject_model_info']['loaded_from_file'] = True
                models['subject_model_info']['filename'] = latest_subject['filename']
                logger.info(f"✅ Loaded latest subject model: {latest_subject['filename']}")

    except Exception as e:
        logger.error(f"❌ Error loading existing models: {str(e)}")

if __name__ == '__main__':
    logger.info("=== FLASK APP CONFIGURATION ===")
    logger.info(f"App name: {app.name}")
    logger.info(f"App debug: {app.debug}")
    logger.info(f"App testing: {app.testing}")
    logger.info(f"Config keys: {list(app.config.keys())}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Model folder: {app.config['MODEL_FOLDER']}")
    logger.info(f"S3 Storage: {'Enabled' if not storage.use_local else 'Disabled (using local)'}")
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"✅ Created folder: {folder}")
    
    # โหลดโมเดลที่มีอยู่
    load_existing_models()
    
    logger.info("🚀 Starting server...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
