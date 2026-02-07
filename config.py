import os
from datetime import timedelta
from typing import Dict, Any, List, Optional, Union

class Config:
    """
    การตั้งค่าหลักของระบบ - Optimized for High Performance
    รวบรวมการตั้งค่าทั้งหมดสำหรับแอปพลิเคชัน
    """

    # --- การตั้งค่าพื้นฐานของแอปพลิเคชัน ---
    SECRET_KEY: str = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    DEBUG: bool = False  # ปิด Debug ใน production

    # --- การตั้งค่าโฟลเดอร์และพาธ ---
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, 'uploads')
    MODEL_FOLDER: str = os.path.join(BASE_DIR, 'models')
    TEMPLATE_FOLDER: str = os.path.join(BASE_DIR, 'templates')
    STATIC_FOLDER: str = os.path.join(BASE_DIR, 'static')

    # --- การตั้งค่าไฟล์อัพโหลด ---
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # เพิ่มเป็น 100MB
    ALLOWED_EXTENSIONS: set = {'csv', 'xlsx', 'xls'}

    # --- การตั้งค่า Session ---
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(hours=24)
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = 'Lax'

    # --- การตั้งค่าการประมวลผลข้อมูลและโครงสร้างหลักสูตร ---
    DATA_CONFIG: Dict[str, Any] = {
        'min_students_for_training': 5,
        'min_subjects_per_student': 3,
        'default_encoding': 'utf-8-sig',
        'fallback_encodings': ['utf-8', 'cp874', 'iso-8859-1', 'windows-1252'],
        'grade_mapping': {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'I': 0.0,
            'WF': 0.0, 'WU': 0.0, 'AU': 0.0, 'P': 2.0, 'NP': 0.0,
            'S': 0.0, 'U': 0.0
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
                'เหมืองข้อมูล', 'ขนาดใหญ่', 'พกพา', 'machine learning', 'deep learning',
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

    # --- Course Code Mapping: รหัสเก่า (3 ส่วน) → รหัสใหม่ (4 ส่วน) ---
    # ใช้สำหรับ normalize รหัสวิชาจากข้อมูลเก่าให้ตรงกับหลักสูตรปัจจุบัน
    COURSE_CODE_MAPPING: Dict[str, str] = {
        # คณิตศาสตร์ & วิทยาศาสตร์
        '02-011-109': '02-005-011-109',   # แคลคูลัส 1 สำหรับวิศวกร
        '02-011-110': '02-005-011-110',   # แคลคูลัส 2 สำหรับวิศวกร
        '02-030-101': '02-005-030-101',   # ฟิสิกส์ 1
        '02-030-102': '02-005-030-102',   # ปฏิบัติการฟิสิกส์ 1
        '02-030-103': '02-005-030-103',   # ฟิสิกส์ 2
        '02-030-104': '02-005-030-104',   # ปฏิบัติการฟิสิกส์ 2
        '02-020-105': '02-005-020-105',   # เคมีพื้นฐาน
        '02-020-106': '02-005-020-106',   # ปฏิบัติการเคมีพื้นฐาน
        # ภาษา & ศึกษาทั่วไป
        '00-031-101': '00-000-031-101',   # ภาษาอังกฤษเพื่อพัฒนาทักษะการเรียน
        '00-031-102': '00-000-031-102',   # ภาษาอังกฤษเพื่อการสื่อสาร
        '00-031-203': '00-000-031-203',   # การอ่านภาษาอังกฤษเพื่อวัตถุประสงค์ทางวิชาการ
        '00-031-204': '00-000-031-204',   # การสนทนาภาษาอังกฤษในชีวิตประจำวัน
        '00-031-205': '00-000-031-205',   # การเขียนภาษาอังกฤษในชีวิตประจำวัน
        '00-032-101': '00-000-032-101',   # การสื่อสารภาษาไทย
        '00-041-001': '00-000-041-001',   # ชีวิตและสิ่งแวดล้อม
        '00-041-004': '00-000-041-004',   # เทคโนโลยีสารสนเทศเพื่อการดำเนินชีวิตอัจฉริยะ
        '00-012-001': '00-000-012-001',   # การพัฒนาคุณภาพชีวิตและสังคม
        '00-012-002': '00-000-012-002',   # พลวัตทางสังคมฯ
        '00-022-001': '00-000-022-001',   # คุณค่าของมนุษย์ฯ
        '00-022-003': '00-000-022-003',   # คุณค่ามนุษย์ : ศิลปะและวิทย์ฯ
        '00-023-001': '00-000-023-001',   # กีฬาและนันทนาการเพื่อสุขภาพ
        # วิศวกรรมคอมพิวเตอร์ (หลักสูตรเก่า → ใหม่ จับจากชื่อวิชา)
        '04-060-101': '03-407-100-101',   # การโปรแกรมคอมพิวเตอร์ → การเขียนโปรแกรมคอมพิวเตอร์
        '04-061-201': '03-407-100-103',   # อุปกรณ์อิเล็กทรอนิกส์ฯ
        '04-061-211': '03-407-102-201',   # โครงสร้างข้อมูลและอัลกอริทึม
        '04-061-213': '03-407-102-202',   # การเขียนโปรแกรมเชิงวัตถุ
        '04-061-102': '03-407-100-204',   # การออกแบบวงจรดิจิทัล
        '04-061-304': '03-407-102-303',   # ระบบฐานข้อมูล
        '04-061-310': '03-407-101-302',   # โครงงานวิศวกรรมคอมพิวเตอร์ 1
        '04-061-404': '03-407-101-403',   # โครงงานวิศวกรรมคอมพิวเตอร์ 2
        '04-040-102': '03-407-000-100',   # แนะนำวิศวกรรมศาสตร์ → บทนำสู่ทางวิศวกรรม
    }

    # --- Course Name Matching: ชื่อวิชาไทย (keyword) → รหัสใหม่ ---
    # ใช้สำหรับจับคู่วิชาเมื่อรหัสไม่ match (fallback จากชื่อ)
    COURSE_NAME_KEYWORDS: Dict[str, str] = {
        'แคลคูลัส 1': '02-005-011-109',
        'แคลคูลัส 2': '02-005-011-110',
        'ฟิสิกส์ 1': '02-005-030-101',
        'ปฏิบัติการฟิสิกส์ 1': '02-005-030-102',
        'ฟิสิกส์ 2': '02-005-030-103',
        'ปฏิบัติการฟิสิกส์ 2': '02-005-030-104',
        'โปรแกรมคอมพิวเตอร์': '03-407-100-101',
        'การเขียนโปรแกรมคอมพิวเตอร์': '03-407-100-101',
        'โครงสร้างข้อมูล': '03-407-102-201',
        'เขียนโปรแกรมเชิงวัตถุ': '03-407-102-202',
        'วงจรดิจิทัล': '03-407-100-204',
        'วงจรไฟฟ้า': '03-407-100-205',
        'สถาปัตยกรรมคอมพิวเตอร์': '03-407-104-203',
        'ระบบฐานข้อมูล': '03-407-102-303',
        'ระบบปฏิบัติการ': '03-407-103-304',
        'เครือข่ายคอมพิวเตอร์': '03-407-103-303',
        'วิศวกรรมซอฟต์แวร์': '03-407-102-404',
        'โครงงานวิศวกรรมคอมพิวเตอร์ 1': '03-407-101-302',
        'โครงงานวิศวกรรมคอมพิวเตอร์ 2': '03-407-101-403',
        'ไมโครคอนโทรลเลอร์': '03-407-104-305',
        'ปัญญาประดิษฐ์': '03-407-106-408',
        'สื่อสารข้อมูล': '03-407-103-202',
        'เขียนโปรแกรมเว็บ': '03-407-106-303',
        'สหกิจศึกษา': '03-407-108-402',
        'เตรียมความพร้อมสหกิจ': '03-407-108-301',
        'อินเทอร์เน็ตของสรรพสิ่ง': '03-407-105-301',
        'สถิติ': '03-407-101-201',
        'เหมืองข้อมูล': '03-407-106-409',
        'วิเคราะห์ข้อมูลขนาดใหญ่': '03-407-106-411',
        'แอปพลิเคชัน': '03-407-105-404',
        'คอมพิวเตอร์วิชัน': '03-407-106-407',
        'มองเห็นด้วยคอมพิวเตอร์': '03-407-106-407',
    }

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

    # --- การตั้งค่า Machine Learning และการทำนาย (HIGH PERFORMANCE) ---
    ML_CONFIG: Dict[str, Any] = {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 10,  # เพิ่มเป็น 10 folds สำหรับ accuracy สูงสุด
        'n_estimators': 500,  # เพิ่มจำนวน trees
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': 1,  # Single thread to prevent ShutdownExecutorError with Gunicorn workers
        'verbose': 0  # ปิด verbose เพื่อหลีกเลี่ยง rate limit
    }

    # Hyperparameters สำหรับ GridSearchCV - FULL RANGE สำหรับ High Spec
    MODEL_HYPERPARAMETERS: Dict[str, Dict[str, List[Any]]] = {
        'RandomForest': {
            'n_estimators': [100, 200],  # ลดลง
            'max_depth': [5, 10, None],  # ลดลง
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None], # ลดลง
            'bootstrap': [True],
            'class_weight': ['balanced', None]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200], # ลดลง
            'learning_rate': [0.05, 0.1, 0.2], # ลดลง
            'max_depth': [3, 5, 7], # ลดลง
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 1.0],
            'max_features': ['sqrt', None] # ลดลง
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10], # ลดลง
            'penalty': ['l1', 'l2'], # ลดลง
            'solver': ['liblinear'], # ลดลง
            'max_iter': [1000, 2000], # ลดลง
            'class_weight': ['balanced', None]
        },
        'SVM': {
            'C': [1, 10], # ลดลง
            'kernel': ['linear', 'rbf'], # ลดลง
            'gamma': ['scale'], # ลดลง
            'class_weight': ['balanced', None]
        },
        'XGBoost': {  # ลดลง
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
        }
    }

    # Advanced Prediction Config สำหรับ High Performance
    ADVANCED_PREDICTION_CONFIG: Dict[str, Any] = {
        'enable_pattern_analysis': True,
        'pattern_window_size': 5,  # เพิ่มจาก 3
        'confidence_threshold_high': 0.85,  # เพิ่ม threshold
        'confidence_threshold_medium': 0.65,
        'prerequisite_weight': 0.6,
        'trend_weight': 0.3,
        'category_weight': 0.1,
        'subject_correlation_threshold': 0.4,  # ลดลงเพื่อจับ correlation มากขึ้น
        'grade_variance_threshold': 0.8,
        'enable_deep_analysis': True,  # เพิ่ม option สำหรับ deep analysis
        'enable_ensemble_voting': True,  # ใช้ ensemble voting
        'enable_feature_selection': True,  # เพิ่ม feature selection
        'max_features_to_select': 50  # จำนวน features สูงสุด
    }

    # Pattern Analysis Config - Enhanced
    PATTERN_ANALYSIS_CONFIG: Dict[str, Any] = {
        'min_grades_for_trend': 3,
        'trend_smoothing_factor': 0.3,
        'category_correlation_threshold': 0.4,
        'improvement_detection_window': 3,
        'decline_detection_threshold': 0.15,
        'stability_threshold': 0.1,
        'seasonal_adjustment': True,
        'weighted_recent_performance': True,
        'recent_performance_weight': 0.5,
        'enable_clustering': True,  # เพิ่ม clustering analysis
        'n_clusters': 5,
        'enable_anomaly_detection': True,  # เพิ่ม anomaly detection
        'anomaly_threshold': 0.95
    }

    # Advanced ML Config - FULL FEATURES
    ADVANCED_ML_CONFIG: Dict[str, Any] = {
        'enable_ensemble_methods': True,
        'enable_deep_learning': False,  # ปิดไว้ก่อน ถ้าต้องการใช้ต้องติดตั้ง tensorflow
        'enable_transfer_learning': True,
        'enable_incremental_learning': True,
        'auto_hyperparameter_tuning': True,
        'cross_validation_folds': 10,  # เพิ่ม folds
        'early_stopping': True,
        'model_comparison_enabled': True,
        'feature_importance_analysis': True,
        'model_interpretability': True,
        'enable_stacking': True,  # เพิ่ม stacking
        'enable_voting': True,  # เพิ่ม voting classifier
        'enable_calibration': True,  # เพิ่ม probability calibration
        'parallel_processing': True,  # ใช้ parallel processing
        'cache_models': True,  # cache models สำหรับความเร็ว
        'optimize_memory': False,  # ไม่ต้อง optimize memory เพราะมี spec สูง
    }

    # Validation Config - Enhanced
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
            'roc_auc', 'balanced_accuracy', 'matthews_corrcoef',
            'cohen_kappa', 'log_loss', 'brier_score'  # เพิ่ม metrics
        ],
        'bootstrap_confidence': True,  # คำนวณ confidence intervals
        'n_bootstrap_samples': 1000
    }

    # Performance Analytics Config - FULL FEATURES
    PERFORMANCE_ANALYTICS_CONFIG: Dict[str, bool] = {
        'track_learning_patterns': True,
        'analyze_study_habits': True,
        'predict_future_performance': True,
        'identify_at_risk_students': True,
        'generate_personalized_recommendations': True,
        'benchmark_against_peers': True,
        'seasonal_performance_analysis': True,
        'prerequisite_impact_analysis': True,
        'course_difficulty_analysis': True,  # เพิ่ม
        'instructor_effect_analysis': True,  # เพิ่ม
        'time_series_forecasting': True,  # เพิ่ม
        'cohort_analysis': True,  # เพิ่ม
        'retention_prediction': True,  # เพิ่ม
        'intervention_effectiveness': True  # เพิ่ม
    }

    # Enhanced Evaluation Metrics
    EVALUATION_METRICS: Dict[str, Any] = {
        'primary_metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
        'secondary_metrics': ['auc_roc', 'balanced_accuracy', 'matthews_corrcoef', 'cohen_kappa'],
        'advanced_metrics': ['log_loss', 'brier_score', 'calibration_error', 'expected_calibration_error'],
        'custom_metrics': {
            'early_warning_precision': 0.85,
            'intervention_effectiveness': 0.75,
            'prediction_stability': 0.8,
            'fairness_metric': 0.9,  # เพิ่ม fairness analysis
            'explainability_score': 0.85  # เพิ่ม explainability
        },
        'benchmark_thresholds': {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.75,
            'needs_improvement': 0.65
        }
    }

    # Cache Configuration สำหรับ Performance
    CACHE_CONFIG: Dict[str, Any] = {
        'enable_caching': True,
        'cache_type': 'simple',  # หรือ 'redis' ถ้ามี Redis
        'cache_timeout': 3600,  # 1 hour
        'cache_key_prefix': 'ml_model_',
        'max_cache_size': 1000,  # MB
        'cache_predictions': True,
        'cache_features': True,
        'cache_models': True
    }

    # Parallel Processing Config
    PARALLEL_CONFIG: Dict[str, Any] = {
        'enable_parallel': True,
        'n_jobs': -1,  # ใช้ CPU ทั้งหมด
        'backend': 'threading',  # หรือ 'multiprocessing'
        'batch_size': 1000,
        'chunk_size': 100,
        'max_workers': None,  # None = auto detect
        'prefer': 'threads'  # threads หรือ processes
    }

    # Optimization Config
    OPTIMIZATION_CONFIG: Dict[str, Any] = {
        'enable_optimization': True,
        'optimization_method': 'bayesian',  # bayesian, grid, random
        'n_iter': 100,  # สำหรับ bayesian optimization
        'scoring': 'roc_auc',
        'refit': True,
        'return_train_score': True,
        'enable_pruning': True,  # pruning สำหรับ decision trees
        'enable_quantization': False,  # model quantization
        'enable_distillation': False  # knowledge distillation
    }

    # --- Logging Configuration (Optimized for Railway) ---
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
                'level': 'WARNING',  # ตั้งเป็น WARNING เพื่อลด logs
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
                'handlers': ['default'],  # ใช้แค่ default handler
                'level': 'WARNING',  # ตั้งเป็น WARNING
                'propagate': False
            },
            'error': {
                'handlers': ['error_file'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }

    # ส่วนที่เหลือคงเดิม...
    MESSAGES = {
        'success': {
            'upload': 'อัพโหลดไฟล์สำเร็จ',
            'train': 'ฝึกโมเดลสำเร็จ',
            'predict': 'ทำนายผลสำเร็จ',
            'analyze': 'วิเคราะห์ข้อมูลสำเร็จ'
        },
        'error': {
            'no_file': 'กรุณาเลือกไฟล์',
            'invalid_format': 'รูปแบบไฟล์ไม่ถูกต้อง',
            'no_model': 'ไม่พบโมเดล กรุณาฝึกโมเดลก่อน'
        },
        'recommendations': {
            'high_risk': [
                "ควรติดตามนักศึกษาใกล้ชิดและให้คำปรึกษาเป็นประจำ",
                "จัดให้มีการปรึกษาเฉพาะรายและวางแผนการเรียนใหม่"
            ],
            'medium_risk': [
                "ติดตามผลการเรียนอย่างสม่ำเสมอและให้กำลังใจ",
                "แนะนำเทคนิคการเรียนและการบริหารเวลา"
            ],
            'low_risk': [
                "รักษาผลการเรียนให้คงเดิมและต่อยอดให้ดีขึ้น",
                "สนับสนุนให้พัฒนาทักษะเพิ่มเติมนอกหลักสูตร"
            ]
        }
    }


class DevelopmentConfig(Config):
    """การตั้งค่าสำหรับการพัฒนา"""
    DEBUG: bool = True
    TESTING: bool = False
    ML_CONFIG: Dict[str, Any] = Config.ML_CONFIG.copy()
    ML_CONFIG.update({
        'cv_folds': 3,  # ลดลงเพื่อความเร็วในการพัฒนา
        'n_jobs': 1,  # ใช้ single thread เพื่อ debug
        'verbose': 1  # เปิด verbose เพื่อดู progress
    })


class ProductionConfig(Config):
    """การตั้งค่าสำหรับ Production - Optimized for Railway High Spec"""
    DEBUG: bool = False
    TESTING: bool = False
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(hours=2)
    
    # Override logging สำหรับ Railway
    LOGGING_CONFIG = Config.LOGGING_CONFIG.copy()
    LOGGING_CONFIG['loggers']['']['level'] = 'WARNING'  # ลด log level
    
    # ML Config for Production
    ML_CONFIG: Dict[str, Any] = Config.ML_CONFIG.copy()
    ML_CONFIG.update({
        'n_jobs': 1,  # Single thread to prevent ShutdownExecutorError with Gunicorn
        'cv_folds': 3, # Reduce for faster training and less memory usage
    })

    MODEL_HYPERPARAMETERS: Dict[str, Dict[str, List[Any]]] = {
        'RandomForest': {
            'n_estimators': [100], 
            'max_depth': [10],    
            'min_samples_split': [2],
            'min_samples_leaf': [2],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'class_weight': ['balanced']
        },
        'GradientBoosting': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3],
            'subsample': [0.8],
        },
        'LogisticRegression': {
            'C': [1],
            'penalty': ['l2'],
            'solver': ['liblinear'],
        },
        'XGBoost': {
            'n_estimators': [100],
            'max_depth': [3],
        }
    }
    
    # Parallel Processing Config
    PARALLEL_CONFIG: Dict[str, Any] = Config.PARALLEL_CONFIG.copy()
    PARALLEL_CONFIG.update({
        'n_jobs': 1,  # Single thread for Gunicorn worker compatibility
        'backend': 'threading',
    })

class TestingConfig(Config):
    """การตั้งค่าสำหรับการทดสอบ"""
    TESTING: bool = True
    DEBUG: bool = True
    ML_CONFIG: Dict[str, Any] = Config.ML_CONFIG.copy()
    ML_CONFIG.update({
        'n_estimators': 10,
        'cv_folds': 2,
        'verbose': 0
    })


# การเลือก Configuration ตาม Environment
config: Dict[str, type[Config]] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': ProductionConfig  # ใช้ Production เป็น default
}

def get_config() -> Config:
    """ดึงการตั้งค่าตาม Environment Variable"""
    env: str = os.environ.get('FLASK_ENV', 'production')  # default เป็น production
    return config.get(env, config['default'])()
