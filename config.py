import os
from datetime import timedelta
from typing import Dict, Any, List, Optional, Union

class Config:
    """
    การตั้งค่าหลักของระบบ
    รวบรวมการตั้งค่าทั้งหมดสำหรับแอปพลิเคชัน
    """

    # --- การตั้งค่าพื้นฐานของแอปพลิเคชัน ---
    SECRET_KEY: str = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    DEBUG: bool = True

    # --- การตั้งค่าโฟลเดอร์และพาธ ---
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, 'uploads')
    MODEL_FOLDER: str = os.path.join(BASE_DIR, 'models')
    TEMPLATE_FOLDER: str = os.path.join(BASE_DIR, 'templates')
    STATIC_FOLDER: str = os.path.join(BASE_DIR, 'static')

    # --- การตั้งค่าไฟล์อัพโหลด ---
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: set = {'csv', 'xlsx', 'xls'}

    # --- การตั้งค่า Session ---
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(hours=24)
    SESSION_COOKIE_SECURE: bool = False  # ใช้ True ใน production กับ HTTPS
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = 'Lax'

    # --- การตั้งค่าการประมวลผลข้อมูลและโครงสร้างหลักสูตร ---
    DATA_CONFIG: Dict[str, Any] = {
        'min_students_for_training': 5,
        'min_subjects_per_student': 3,
        'default_encoding': 'utf-8-sig',
        'fallback_encodings': ['utf-8', 'cp874', 'iso-8859-1'],
        'grade_mapping': {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'I': 0.0,
            'WF': 0.0, 'WU': 0.0, 'AU': 0.0, 'P': 2.0, 'NP': 0.0
        },
        'risk_levels': {
            'high_confidence_threshold': 0.8,
            'medium_confidence_threshold': 0.6,
            'high_fail_rate_threshold': 0.3,
            'medium_fail_rate_threshold': 0.15,
            'low_gpa_threshold': 2.0,
            'warning_gpa_threshold': 2.5,
            'credit_penalty_per_blocked_course': 2,
            'low_gpa_penalty': 5,
            'high_fail_count_penalty': 10,
            'medium_fail_count_penalty': 5,
            'core_fail_penalty': 3,
            'repeat_course_penalty': 5
        }
    }

    SUBJECT_CATEGORIES: Dict[str, Any] = {
        'คณิตศาสตร์และวิทยาศาสตร์': {
            'keywords': [
                'คณิต', 'แคลคูลัส', 'สถิติ', 'พีชคณิต', 'เรขาคณิต', 'ตรีโกณมิติ',
                'ฟิสิกส์', 'เคมี', 'ชีววิทยา', 'คณิตศาสตร์', 'วิทยาศาสตร์',
                'calculus', 'statistics', 'mathematics', 'physics', 'chemistry', 'biology'
            ],
            'color': '#007bff',
            'icon': 'calculator'
        },
        'การเขียนโปรแกรม': {
            'keywords': [
                'โปรแกรม', 'คอมพิวเตอร์', 'ซอฟต์แวร์', 'ข้อมูล', 'อัลกอริทึม',
                'โครงสร้างข้อมูล', 'ฐานข้อมูล', 'เว็บ', 'แอป', 'เกม',
                'programming', 'computer', 'software', 'data', 'algorithm', 'database'
            ],
            'color': '#28a745',
            'icon': 'code'
        },
        'วิศวกรรมพื้นฐาน': {
            'keywords': [
                'วิศวกรรม', 'วงจร', 'อิเล็กทรอนิกส์', 'ไฟฟ้า', 'กลศาสตร์',
                'วัสดุ', 'เขียนแบบ', 'ออกแบบ', 'การวัด', 'เครื่องมือวัด',
                'engineering', 'circuit', 'electronics', 'electrical', 'mechanical'
            ],
            'color': '#17a2b8',
            'icon': 'cogs'
        },
        'ระบบคอมพิวเตอร์': {
            'keywords': [
                'ระบบ', 'เครือข่าย', 'ปฏิบัติการ', 'ไมโครโปรเซสเซอร์', 'ดิจิตอล',
                'vlsi', 'สื่อสาร', 'องค์ประกอบ', 'แอสแซมบลี', 'สถาปัตยกรรม',
                'system', 'network', 'operating', 'microprocessor', 'digital'
            ],
            'color': '#ffc107',
            'icon': 'microchip'
        },
        'ศึกษาทั่วไป': {
            'keywords': [
                'ภาษา', 'อังกฤษ', 'ไทย', 'การสื่อสาร', 'สังคม', 'กีฬา', 'นันทนาการ',
                'สุขภาพ', 'คุณภาพชีวิต', 'สารสนเทศ', 'ทักษะ', 'พัฒนา', 'ชีวิต', 'สิ่งแวดล้อม',
                'english', 'thai', 'communication', 'social', 'sport', 'health'
            ],
            'color': '#6c757d',
            'icon': 'book'
        },
        'วิชาขั้นสูง': {
            'keywords': [
                'ปัญญาประดิษฐ์', 'ขั้นสูง', 'การรู้ภาพ', 'วิเคราะห์', 'หัวข้อ', 'มองเห็น',
                'เหมืองข้อมูล', 'ขนาดใหญ่', 'พกพา',
                'artificial', 'intelligence', 'advanced', 'vision', 'analysis', 'mining'
            ],
            'color': '#dc3545',
            'icon': 'brain'
        },
        'โครงงานและสหกิจ': {
            'keywords': [
                'โครงงาน', 'สหกิจ', 'เตรียม', 'ปฏิบัติการ', 'ฝึก',
                'project', 'cooperative', 'internship', 'practicum'
            ],
            'color': '#fd7e14',
            'icon': 'project-diagram'
        }
    }

    COURSES_DATA: List[Dict[str, Any]] = [
        # ปี 1 เทอม 1
        { "id": "00-000-012-002", "thaiName": "พลวัตทางสังคมและการอยู่ร่วมอย่างมีความสุข", "credit": 3, "prereq": [] },
        { "id": "00-000-032-101", "thaiName": "การสื่อสารภาษาไทย", "credit": 3, "prereq": [] },
        { "id": "00-000-031-101", "thaiName": "ภาษาอังกฤษเพื่อพัฒนาทักษะการเรียน", "credit": 3, "prereq": [] },
        { "id": "02-005-011-109", "thaiName": "แคลคูลัส 1 สำหรับวิศวกร", "credit": 3, "prereq": [] },
        { "id": "02-005-030-101", "thaiName": "ฟิสิกส์ 1", "credit": 3, "prereq": [] },
        { "id": "02-005-030-102", "thaiName": "ปฏิบัติการฟิสิกส์ 1", "credit": 1, "prereq": ["02-005-030-101"] },
        { "id": "03-407-100-101", "thaiName": "การเขียนโปรแกรมคอมพิวเตอร์", "credit": 3, "prereq": [] },
        { "id": "03-407-100-102", "thaiName": "การฝึกพื้นฐานทางวิศวกรรมคอมพิวเตอร์", "credit": 3, "prereq": ["03-407-100-101"] },

        # ปี 1 เทอม 2
        { "id": "00-000-041-004", "thaiName": "เทคโนโลยีสารสนเทศเพื่อการดำเนินชีวิตอัจฉริยะ", "credit": 3, "prereq": [] },
        { "id": "00-000-031-102", "thaiName": "ภาษาอังกฤษเพื่อการสื่อสาร", "credit": 3, "prereq": [] },
        { "id": "00-000-022-003", "thaiName": "คุณค่ามนุษย์ : ศิลปะและวิทย์ในชีวิตประจำวัน", "credit": 3, "prereq": [] },
        { "id": "02-005-011-110", "thaiName": "แคลคูลัส 2 สำหรับวิศวกร", "credit": 3, "prereq": ["02-005-011-109"] },
        { "id": "02-005-030-103", "thaiName": "ฟิสิกส์ 2", "credit": 3, "prereq": ["02-005-030-101"] },
        { "id": "02-005-030-104", "thaiName": "ปฏิบัติการฟิสิกส์ 2", "credit": 1, "prereq": ["02-005-030-103"] },
        { "id": "03-407-000-100", "thaiName": "บทนำสู่ทางวิศวกรรม", "credit": 2, "prereq": [] },
        { "id": "03-407-100-103", "thaiName": "อุปกรณ์อิเล็กทรอนิกส์สำหรับวิศวกรรมคอมพิวเตอร์", "credit": 3, "prereq": [] },

        # ปี 2 เทอม 1
        { "id": "03-407-102-201", "thaiName": "โครงสร้างข้อมูลและอัลกอริทึม", "credit": 3, "prereq": ["03-407-100-101"] },
        { "id": "03-407-101-201", "thaiName": "สถิติเพื่อวิศวกรรมคอมพิวเตอร์", "credit": 3, "prereq": [] },
        { "id": "03-407-102-202", "thaiName": "การเขียนโปรแกรมเชิงวัตถุ", "credit": 3, "prereq": ["03-407-100-101"] },
        { "id": "03-407-100-204", "thaiName": "การออกแบบวงจรดิจิทัล", "credit": 4, "prereq": [] },
        { "id": "03-407-100-205", "thaiName": "วงจรไฟฟ้าสำหรับวิศวกรรมคอมพิวเตอร์", "credit": 3, "prereq": ["02-005-030-101"] },

        # ปี 2 เทอม 2
        { "id": "00-000-041-001", "thaiName": "ชีวิตและสิ่งแวดล้อม", "credit": 3, "prereq": [] },
        { "id": "03-407-103-201", "thaiName": "คณิตศาสตร์เชิงการจัดระเบียบสำหรับวิศวกรรม", "credit": 3, "prereq": [] },
        { "id": "03-407-104-201", "thaiName": "การวัดและเครื่องมือสัญญาณดิจิทัล", "credit": 3, "prereq": [] },
        { "id": "03-407-104-202", "thaiName": "วงจรอิเล็กทรอนิกส์สำหรับวิศวกรรมคอมพิวเตอร์", "credit": 3, "prereq": ["03-407-100-103"] },
        { "id": "03-407-104-203", "thaiName": "สถาปัตยกรรมคอมพิวเตอร์และระบบ", "credit": 3, "prereq": [] },
        { "id": "03-407-104-204", "thaiName": "ปฏิบัติการสถาปัตยกรรมคอมพิวเตอร์และระบบ", "credit": 1, "prereq": ["03-407-104-203"] },
        { "id": "03-407-102-303", "thaiName": "ระบบฐานข้อมูล", "credit": 3, "prereq": [] },

        # ปี 3 เทอม 1
        { "id": "00-000-031-204", "thaiName": "การสนทนาภาษาอังกฤษในชีวิตประจำวัน", "credit": 3, "prereq": ["00-000-031-101", "00-000-031-102"] },
        { "id": "03-407-104-305", "thaiName": "ไมโครคอนโทรลเลอร์", "credit": 3, "prereq": ["03-407-100-204"] },
        { "id": "03-407-103-202", "thaiName": "การสื่อสารข้อมูล", "credit": 3, "prereq": [] },
        { "id": "03-407-102-404", "thaiName": "วิศวกรรมซอฟต์แวร์", "credit": 3, "prereq": ["03-407-102-201"] },
        { "id": "03-407-106-303", "thaiName": "การเขียนโปรแกรมเว็บ", "credit": 3, "prereq": ["03-407-100-101"] },
        { "id": "03-407-106-409", "thaiName": "การค้นหาความรู้และเหมืองข้อมูล", "credit": 3, "prereq": ["03-407-103-201"] },

        # ปี 3 เทอม 2
        { "id": "00-000-031-203", "thaiName": "การอ่านภาษาอังกฤษเพื่อวัตถุประสงค์ทางวิชาการ", "credit": 3, "prereq": ["00-000-031-101", "00-000-031-102"] },
        { "id": "03-407-103-304", "thaiName": "ระบบปฏิบัติการ", "credit": 3, "prereq": ["03-407-104-203"] },
        { "id": "03-407-101-302", "thaiName": "โครงงานวิศวกรรมคอมพิวเตอร์ 1", "credit": 3, "prereq": [] },
        { "id": "03-407-106-407", "thaiName": "วิทยาการมองเห็นด้วยคอมพิวเตอร์", "credit": 3, "prereq": ["03-407-102-201"] },
        { "id": "03-407-105-301", "thaiName": "อินเทอร์เน็ตของสรรพสิ่ง", "credit": 3, "prereq": ["03-407-104-305"] },
        { "id": "03-407-103-303", "thaiName": "เครือข่ายคอมพิวเตอร์", "credit": 3, "prereq": ["03-407-103-202"] },

        # ปี 4 เทอม 1
        { "id": "00-000-031-205", "thaiName": "การเขียนภาษาอังกฤษในชีวิตประจำวัน", "credit": 3, "prereq": ["00-000-031-101", "00-000-031-102"] },
        { "id": "03-407-108-301", "thaiName": "เตรียมความพร้อมสหกิจศึกษา", "credit": 1, "prereq": [] },
        { "id": "03-407-101-403", "thaiName": "โครงงานวิศวกรรมคอมพิวเตอร์ 2", "credit": 3, "prereq": ["03-407-101-302"] },
        { "id": "03-407-106-411", "thaiName": "การวิเคราะห์ข้อมูลขนาดใหญ่", "credit": 3, "prereq": ["03-407-106-409"] },
        { "id": "03-407-105-404", "thaiName": "การพัฒนาแอปพลิเคชันสำหรับอุปกรณ์พกพา", "credit": 3, "prereq": ["03-407-108-301"] },
        { "id": "03-407-106-408", "thaiName": "ปัญญาประดิษฐ์", "credit": 3, "prereq": ["03-407-103-201"] },

        # ปี 4 เทอม 2
        { "id": "03-407-108-402", "thaiName": "สหกิจศึกษาทางวิศวกรรมคอมพิวเตอร์", "credit": 6, "prereq": ["03-407-108-301"] },
    ]

    ALL_TERMS_DATA: List[Dict[str, Any]] = [
        { "year": 1, "term": 1, "ids": ["00-000-012-002","00-000-032-101","00-000-031-101","02-005-011-109","02-005-030-101","02-005-030-102","03-407-100-101","03-407-100-102"]},
        { "year": 1, "term": 2, "ids": ["00-000-041-004","00-000-031-102","00-000-022-003","02-005-011-110","02-005-030-103","02-005-030-104","03-407-000-100","03-407-100-103"]},
        { "year": 2, "term": 1, "ids": ["03-407-102-201","03-407-101-201","03-407-102-202","03-407-100-204","03-407-100-205"]},
        { "year": 2, "term": 2, "ids": ["00-000-041-001","03-407-103-201","03-407-104-201","03-407-104-202","03-407-104-203","03-407-104-204","03-407-102-303"]},
        { "year": 3, "term": 1, "ids": ["00-000-031-204","03-407-104-305","03-407-103-202","03-407-102-404","03-407-106-303","03-407-106-409"]},
        { "year": 3, "term": 2, "ids": ["00-000-031-203","03-407-103-304","03-407-101-302","03-407-106-407","03-407-105-301","03-407-103-303"]},
        { "year": 4, "term": 1, "ids": ["00-000-031-205","03-407-108-301","03-407-101-403","03-407-106-411","03-407-105-404","03-407-106-408"]},
        { "year": 4, "term": 2, "ids": ["03-407-108-402"]},
    ]

    CORE_SUBJECTS_IDS: List[str] = [
        "02-005-011-109",  # แคลคูลัส 1 สำหรับวิศวกร
        "02-005-030-101",  # ฟิสิกส์ 1
        "03-407-100-101",  # การเขียนโปรแกรมคอมพิวเตอร์
        "03-407-102-201",  # โครงสร้างข้อมูลและอัลกอริทึม
        "03-407-104-203",  # สถาปัตยกรรมคอมพิวเตอร์และระบบ
        "03-407-102-303",  # ระบบฐานข้อมูล
        "03-407-102-404",  # วิศวกรรมซอฟต์แวร์
        "03-407-103-304"   # ระบบปฏิบัติการ
    ]

    # --- การตั้งค่า Machine Learning และการทำนาย ---
    ML_CONFIG: Dict[str, Any] = {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'verbose': 0
    }

    ADVANCED_PREDICTION_CONFIG: Dict[str, Any] = {
        'enable_pattern_analysis': True,
        'pattern_window_size': 3,  # จำนวนเทอมที่ใช้วิเคราะห์รูปแบบ
        'confidence_threshold_high': 0.8,
        'confidence_threshold_medium': 0.6,
        'prerequisite_weight': 0.6,  # น้ำหนักของ prerequisite ในการทำนาย
        'trend_weight': 0.3,         # น้ำหนักของแนวโน้ม
        'category_weight': 0.1,      # น้ำหนักของหมวดวิชา
        'subject_correlation_threshold': 0.5,  # เกณฑ์ความสัมพันธ์ระหว่างวิชา
        'grade_variance_threshold': 1.0,       # เกณฑ์ความแปรปรวนของเกรด
    }

    PATTERN_ANALYSIS_CONFIG: Dict[str, Any] = {
        'min_grades_for_trend': 3,
        'trend_smoothing_factor': 0.3,
        'category_correlation_threshold': 0.5,
        'improvement_detection_window': 2,
        'decline_detection_threshold': 0.2,
        'stability_threshold': 0.15,
        'seasonal_adjustment': True,
        'weighted_recent_performance': True,
        'recent_performance_weight': 0.4
    }

    ADVANCED_ML_CONFIG: Dict[str, Any] = {
        'enable_ensemble_methods': True,
        'enable_deep_learning': False,  # ปิดไว้เพื่อประสิทธิภาพ
        'enable_transfer_learning': True,
        'enable_incremental_learning': True,
        'auto_hyperparameter_tuning': True,
        'cross_validation_folds': 5,
        'early_stopping': True,
        'model_comparison_enabled': True,
        'feature_importance_analysis': True,
        'model_interpretability': True
    }

    MODEL_HYPERPARAMETERS: Dict[str, Dict[str, List[Any]]] = {
        'RandomForest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'max_iter': [1000, 2000, 3000, 5000],
            'class_weight': ['balanced', None]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4],
            'class_weight': ['balanced', None]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
    }

    VALIDATION_CONFIG: Dict[str, Any] = {
        'cross_validation_enabled': True,
        'stratified_cv': True,
        'time_series_cv': False,
        'validation_split': 0.2,
        'test_split': 0.2,
        'shuffle_data': True,
        'random_seed': 42,
        'validation_metrics': [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'balanced_accuracy'
        ]
    }

    # --- การตั้งค่าการวิเคราะห์ประสิทธิภาพและการประเมิน ---
    PERFORMANCE_ANALYTICS_CONFIG: Dict[str, bool] = {
        'track_learning_patterns': True,
        'analyze_study_habits': True,
        'predict_future_performance': True,
        'identify_at_risk_students': True,
        'generate_personalized_recommendations': True,
        'benchmark_against_peers': True,
        'seasonal_performance_analysis': True,
        'prerequisite_impact_analysis': True
    }

    EVALUATION_METRICS: Dict[str, Any] = {
        'primary_metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
        'secondary_metrics': ['auc_roc', 'balanced_accuracy', 'matthews_corrcoef'],
        'custom_metrics': {
            'early_warning_precision': 0.8,  # ความแม่นยำในการเตือนล่วงหน้า
            'intervention_effectiveness': 0.7,  # ประสิทธิภาพของการแทรกแซง
            'prediction_stability': 0.75      # ความเสถียรของการทำนาย
        },
        'benchmark_thresholds': {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'needs_improvement': 0.6
        }
    }

    # --- การตั้งค่าการแจ้งเตือนและการรายงาน ---
    ALERT_SETTINGS: Dict[str, Any] = {
        'high_risk_threshold': 0.3,
        'improvement_needed_threshold': 0.6,
        'excellent_performance_threshold': 0.85,
        'grade_drop_threshold': 0.5,  # ลดลงครึ่งเกรด
        'consecutive_fail_threshold': 2,  # ตกติดต่อกัน 2 วิชา
        'gpa_drop_threshold': 0.3,  # GPA ลดลง 0.3
        'notification_preferences': {
            'email_alerts': True,
            'dashboard_alerts': True,
            'mobile_push': False,
            'weekly_reports': True,
            'semester_summary': True
        },
        'monitoring_intervals': {
            'daily_check': False,
            'weekly_check': True,
            'monthly_check': True,
            'semester_check': True
        }
    }

    REPORTING_CONFIG: Dict[str, Any] = {
        'auto_report_generation': True,
        'report_formats': ['pdf', 'excel', 'csv', 'json'],
        'report_types': {
            'individual_student': True,
            'class_summary': True,
            'department_overview': True,
            'comparative_analysis': True,
            'trend_analysis': True
        },
        'scheduling': {
            'daily_reports': False,
            'weekly_reports': True,
            'monthly_reports': True,
            'semester_reports': True
        },
        'customization': {
            'logo_inclusion': True,
            'custom_headers': True,
            'branded_templates': True,
            'multilingual_support': True
        }
    }

    # --- การตั้งค่า Dashboard และการแสดงผล ---
    DASHBOARD_CONFIG: Dict[str, Any] = {
        'default_charts': [
            'performance_trend',
            'category_breakdown',
            'risk_distribution',
            'improvement_suggestions'
        ],
        'interactive_features': {
            'drill_down': True,
            'time_range_selection': True,
            'comparative_analysis': True,
            'export_functionality': True
        },
        'visualization_preferences': {
            'color_scheme': 'professional',
            'animation_enabled': True,
            'responsive_design': True,
            'accessibility_compliant': True
        }
    }

    # --- การตั้งค่าความปลอดภัยและความเป็นส่วนตัว ---
    SECURITY_CONFIG: Dict[str, Any] = {
        'data_encryption': True,
        'secure_file_upload': True,
        'user_authentication': False,  # ปิดไว้สำหรับ demo
        'access_logging': True,
        'data_anonymization': True,
        'gdpr_compliance': True,
        'data_retention_policy': {
            'raw_data': '2 years',
            'processed_data': '5 years',
            'model_data': '3 years',
            'log_data': '1 year'
        }
    }

    # --- การตั้งค่า Logging ---
    LOGGING_CONFIG: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s'
            },
            'json': {
                'format': '{"time":"%(asctime)s", "level":"%(levelname)s", "logger":"%(name)s", "message":"%(message)s"}'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': 'INFO',
                'formatter': 'detailed',
                'class': 'logging.FileHandler',
                'filename': 'app.log',
                'mode': 'a',
                'encoding': 'utf-8',
            },
            'error_file': {
                'level': 'ERROR',
                'formatter': 'json',
                'class': 'logging.FileHandler',
                'filename': 'errors.log',
                'mode': 'a',
                'encoding': 'utf-8',
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'error': {
                'handlers': ['error_file'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }

    # --- ข้อความและคำแนะนำต่างๆ ---
    MESSAGES: Dict[str, Any] = {
        'success': {
            'upload': 'อัพโหลดไฟล์สำเร็จ',
            'train': 'ฝึกโมเดลสำเร็จ',
            'predict': 'ทำนายผลสำเร็จ',
            'analyze': 'วิเคราะห์ข้อมูลสำเร็จ',
            'model_save': 'บันทึกโมเดลสำเร็จ',
            'report_generate': 'สร้างรายงานสำเร็จ'
        },
        'error': {
            'no_file': 'กรุณาเลือกไฟล์',
            'invalid_format': 'รูปแบบไฟล์ไม่ถูกต้อง',
            'no_model': 'ไม่พบโมเดล กรุณาฝึกโมเดลก่อน',
            'insufficient_data': 'ข้อมูลไม่เพียงพอสำหรับการฝึกโมเดล',
            'processing_error': 'เกิดข้อผิดพลาดในการประมวลผล',
            'model_load_error': 'ไม่สามารถโหลดโมเดลได้',
            'prediction_error': 'เกิดข้อผิดพลาดในการทำนาย',
            'connection_error': 'เกิดข้อผิดพลาดในการเชื่อมต่อ'
        },
        'warning': {
            'low_performance': 'ประสิทธิภาพโมเดลต่ำ',
            'insufficient_features': 'จำนวน Features ไม่เพียงพอ',
            'data_quality': 'คุณภาพข้อมูลอาจไม่ดี',
            'overfitting': 'โมเดลอาจมีปัญหา Overfitting'
        },
        'info': {
            'processing': 'กำลังประมวลผล...',
            'training': 'กำลังฝึกโมเดล...',
            'analyzing': 'กำลังวิเคราะห์...',
            'generating_report': 'กำลังสร้างรายงาน...'
        },
        'recommendations': {
            'high_risk': [
                "ควรติดตามนักศึกษาใกล้ชิดและให้คำปรึกษาเป็นประจำ",
                "จัดให้มีการปรึกษาเฉพาะรายและวางแผนการเรียนใหม่",
                "แนะนำให้เข้าร่วมกิจกรรมการเรียนเสริมและติวเสริม",
                "ทบทวนแผนการเรียนและพิจารณาปรับลดภาระวิชา",
                "ประเมินสาเหตุของปัญหาและแก้ไขที่ต้นเหตุ"
            ],
            'medium_risk': [
                "ติดตามผลการเรียนอย่างสม่ำเสมอและให้กำลังใจ",
                "แนะนำเทคนิคการเรียนและการบริหารเวลา",
                "สร้างแรงจูงใจและช่วยวางเป้าหมายที่ชัดเจน",
                "จัดกลุ่มการเรียนร่วมกับเพื่อนที่มีผลการเรียนดี",
                "ให้คำแนะนำเฉพาะด้านที่ต้องปรับปรุง"
            ],
            'low_risk': [
                "รักษาผลการเรียนให้คงเดิมและต่อยอดให้ดีขึ้น",
                "สนับสนุนให้พัฒนาทักษะเพิ่มเติมนอกหลักสูตร",
                "แนะนำให้เป็นพี่เลี้ยงช่วยเหลือเพื่อนที่เรียนไม่ดี",
                "เตรียมความพร้อมสำหรับการฝึกงานและการทำงาน",
                "สนับสนุนให้เข้าร่วมการแข่งขันหรือโครงการพิเศษ"
            ]
        }
    }

    DETAILED_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
        'gpa_based': {
            'excellent': [
                "GPA ยอดเยี่ยม! พิจารณาเข้าร่วมกิจกรรมวิชาการเพิ่มเติม",
                "ช่วยเหลือเพื่อนที่เรียนไม่ดี เพื่อพัฒนาทักษะการสื่อสาร",
                "เตรียมตัวสำหรับการฝึกงานหรือการทำงาน",
                "พิจารณาทำวิจัยหรือโครงงานพิเศษ",
                "สมัครเป็นติวเตอร์หรือผู้ช่วยสอน"
            ],
            'good': [
                "รักษาระดับการเรียนให้อยู่ในเกณฑ์ดี",
                "มุ่งเน้นการพัฒนาในวิชาที่สนใจ",
                "วางแผนการเรียนระยะยาวให้ชัดเจน",
                "เพิ่มการมีส่วนร่วมในชั้นเรียน",
                "หาประสบการณ์เพิ่มเติมนอกห้องเรียน"
            ],
            'fair': [
                "ควรเพิ่มความพยายามในการเรียน",
                "หาเทคนิคการเรียนที่เหมาะสมกับตัวเอง",
                "ปรึกษาอาจารย์เมื่อมีปัญหาการเรียน",
                "จัดเวลาเรียนให้มีประสิทธิภาพมากขึ้น",
                "หากลุ่มเรียนที่มีคุณภาพ"
            ],
            'poor': [
                "ต้องปรับปรุงวิธีการเรียนอย่างเร่งด่วน",
                "พบอาจารย์ที่ปรึกษาเพื่อวางแผนแก้ไข",
                "พิจารณาลดภาระการเรียนในเทอมถัดไป",
                "ขอความช่วยเหลือจากศูนย์การเรียนรู้",
                "ประเมินปัจจัยที่ส่งผลต่อการเรียน"
            ]
        },
        'subject_based': {
            'math_weak': [
                "จัดเวลาทบทวนคณิตศาสตร์เป็นพิเศษ",
                "หาติวเตอร์หรือกลุ่มเรียนคณิตศาสตร์",
                "ทำแบบฝึกหัดเพิ่มเติมอย่างสม่ำเสมอ",
                "ใช้แอปพลิเคชันช่วยเรียนคณิตศาสตร์",
                "ทบทวนพื้นฐานที่อาจจะขาดหาย"
            ],
            'programming_weak': [
                "ฝึกเขียนโปรแกรมทุกวันสักเล็กน้อย",
                "ศึกษาจากตัวอย่างโค้ดออนไลน์",
                "เข้าร่วมชุมชนโปรแกรมเมอร์",
                "หาโปรเจ็กต์เล็กๆ มาฝึกทำ",
                "เรียนรู้จากการแก้ปัญหาใน Coding Platform"
            ],
            'engineering_weak': [
                "ทบทวนหลักการทางวิศวกรรมพื้นฐาน",
                "ฝึกการแก้ปัญหาเชิงเลข",
                "ศึกษาตัวอย่างการประยุกต์ในชีวิตจริง",
                "หาเพื่อนที่เก่งมาอธิบาย",
                "ใช้โปรแกรมจำลองช่วยในการเรียน"
            ],
            'system_weak': [
                "ศึกษาหลักการระบบคอมพิวเตอร์เพิ่มเติม",
                "ลองติดตั้งและใช้ระบบปฏิบัติการต่างๆ",
                "ฝึกเขียนโปรแกรมระดับต่ำ",
                "ศึกษาสถาปัตยกรรมฮาร์ดแวร์",
                "ทำแลปเสริมเพื่อเข้าใจลึกขึ้น"
            ],
            'general_weak': [
                "อ่านหนังสือและบทความเพิ่มเติม",
                "พัฒนาทักษะการเขียนและการนำเสนอ",
                "เข้าร่วมกิจกรรมเสริมความรู้ทั่วไป",
                "ฝึกการสื่อสารภาษาอังกฤษ",
                "หาความรู้จากสื่อต่างๆ เช่น podcast, วิดีโอ"
           ]
       },
       'improvement_strategies': {
           'time_management': [
               "จัดทำตารางเวลาเรียนและทบทวน",
               "ใช้เทคนิค Pomodoro ในการเรียน",
               "กำหนดเป้าหมายรายวันและรายสัปดาห์",
               "ลำดับความสำคัญของงานที่ต้องทำ"
           ],
           'study_methods': [
               "หาวิธีเรียนที่เหมาะกับรูปแบบการเรียนรู้",
               "ใช้เทคนิค Mind Mapping",
               "สร้างบันทึกย่อที่มีประสิทธิภาพ",
               "ฝึกการอธิบายให้ผู้อื่นฟัง"
           ],
           'collaboration': [
               "จัดตั้งกลุ่มเรียนร่วมกับเพื่อน",
               "แลกเปลี่ยนโน้ตและแนวคิด",
               "ช่วยเหลือกันในวิชาที่แต่ละคนถนัด",
               "เข้าร่วมชุมชนการเรียนรู้ออนไลน์"
           ]
       }
    }

    # --- การตั้งค่า API สำหรับการขยายในอนาคต ---
    API_CONFIG: Dict[str, Any] = {
        'version': 'v2',
        'rate_limit': '1000/hour',
        'authentication': False,
        'cors_origins': ['http://localhost:3000', 'http://localhost:8080', 'http://127.0.0.1:5000'],
        'max_request_size': '32MB',
        'timeout': 600,
        'pagination': {
            'default_page_size': 50,
            'max_page_size': 200
        },
        'caching': {
            'enabled': True,
            'ttl': 300,  # 5 minutes
            'redis_url': None
        }
    }


class DevelopmentConfig(Config):
    """
    การตั้งค่าสำหรับการพัฒนา
    Override การตั้งค่าบางส่วนเพื่อความสะดวกในการพัฒนา
    """
    DEBUG: bool = True
    TESTING: bool = False
    
    # Override สำหรับ Development
    ADVANCED_ML_CONFIG: Dict[str, Any] = Config.ADVANCED_ML_CONFIG.copy()
    ADVANCED_ML_CONFIG.update({
        'auto_hyperparameter_tuning': False,  # ปิดเพื่อความเร็วในการพัฒนา
        'cross_validation_folds': 3
    })


class ProductionConfig(Config):
    """
    การตั้งค่าสำหรับ Production
    ปรับการตั้งค่าให้เหมาะสมกับการใช้งานจริง (ความปลอดภัย, ประสิทธิภาพ)
    """
    DEBUG: bool = False
    TESTING: bool = False
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(hours=2)

    # การตั้งค่า ML สำหรับ Production
    ML_CONFIG: Dict[str, Any] = Config.ML_CONFIG.copy()
    ML_CONFIG.update({
        'n_estimators': 300,
        'cv_folds': 10,
        'verbose': 0,
        'n_jobs': -1
    })
    
    # การตั้งค่าความปลอดภัยสำหรับ Production
    SECURITY_CONFIG: Dict[str, Any] = Config.SECURITY_CONFIG.copy()
    SECURITY_CONFIG.update({
        'user_authentication': True,
        'rate_limiting': True,
        'input_validation': True,
        'sql_injection_protection': True
    })

    # การตั้งค่า Logging สำหรับ Production
    LOGGING_CONFIG: Dict[str, Any] = Config.LOGGING_CONFIG.copy()
    LOGGING_CONFIG['handlers']['file']['level'] = 'WARNING'
    LOGGING_CONFIG['loggers']['']['level'] = 'WARNING'


class TestingConfig(Config):
    """
    การตั้งค่าสำหรับการทดสอบ
    ปรับการตั้งค่าให้เหมาะสมกับการทดสอบโค้ด
    """
    TESTING: bool = True
    DEBUG: bool = True
    UPLOAD_FOLDER: str = os.path.join(Config.BASE_DIR, 'test_uploads')
    MODEL_FOLDER: str = os.path.join(Config.BASE_DIR, 'test_models')

    # การตั้งค่า ML สำหรับการทดสอบ
    ML_CONFIG: Dict[str, Any] = Config.ML_CONFIG.copy()
    ML_CONFIG.update({
        'n_estimators': 10,
        'cv_folds': 3,
        'verbose': 1,
        'random_state': 42
    })
    
    # การตั้งค่าข้อมูลสำหรับการทดสอบ
    DATA_CONFIG: Dict[str, Any] = Config.DATA_CONFIG.copy()
    DATA_CONFIG.update({
        'min_students_for_training': 3,
        'min_subjects_per_student': 2
    })


# การเลือก Configuration ตาม Environment
config: Dict[str, type[Config]] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config() -> Config:
    """
    ดึงการตั้งค่าตาม Environment Variable
    หากไม่พบ FLASK_ENV จะใช้ 'default' (DevelopmentConfig)
    """
    env: str = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])()

def get_feature_config(feature_name: str) -> Dict[str, Any]:
    """
    ดึงการตั้งค่าของฟีเจอร์เฉพาะ
    คืนค่าเป็น dictionary ว่างหากไม่พบฟีเจอร์นั้น
    """
    config_instance: Config = get_config()
    return getattr(config_instance, feature_name, {})

def is_feature_enabled(feature_name: str) -> bool:
    """
    ตรวจสอบว่าฟีเจอร์เปิดใช้งานหรือไม่
    จะคืนค่า True หากฟีเจอร์นั้นมีคีย์ 'enabled' เป็น True
    """
    config_instance: Config = get_config()
    feature_config: Dict[str, Any] = getattr(config_instance, feature_name, {})
    return feature_config.get('enabled', False)

def get_model_hyperparameters(model_name: str) -> Dict[str, List[Any]]:
    """
    ดึง hyperparameters สำหรับโมเดลที่ระบุ
    คืนค่าเป็น dictionary ว่างหากไม่พบโมเดลนั้น
    """
    config_instance: Config = get_config()
    return config_instance.MODEL_HYPERPARAMETERS.get(model_name, {})

def get_recommendation_by_type(rec_type: str, risk_level: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
    """
    ดึงคำแนะนำตามประเภทและระดับความเสี่ยง
    หากระบุ risk_level จะคืนค่าเป็น list ของคำแนะนำสำหรับระดับความเสี่ยงนั้น
    หากไม่ระบุ risk_level จะคืนค่าเป็น dictionary ของคำแนะนำทั้งหมดในประเภทนั้น
    """
    config_instance: Config = get_config()
    recommendations: Dict[str, Any] = config_instance.DETAILED_RECOMMENDATIONS
    
    if risk_level:
        return recommendations.get(rec_type, {}).get(risk_level, [])
    else:
        return recommendations.get(rec_type, {})
