import os
import logging
import pandas as pd # Import pandas here for CSV reading

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_secret_key_that_should_be_replaced')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 # 16 MB limit

    # Path to the curriculum CSV file (assuming it's in the same directory as config.py)
    # The user provided 'แผนการเรียน รับเข้าปี 2567.csv'
    CSV_CURRICULUM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'แผนการเรียน รับเข้าปี 2567.csv')

    # Data Processing Configurations
    DATA_CONFIG = {
        'grade_mapping': { # Standard 4.0 scale
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0,
            'W': 0.0, 'I': 0.0, 'NP': 0.0, 'WF': 0.0, 'WU': 0.0, 'S': 2.5, 'U': 0.0, 'AU': 0.0, 'P': 2.5 # S/P are often pass/fail, assign mid-range GPA if needed for calculations
        },
        'min_subjects_per_student': 3, # Minimum number of grades to consider a student's record valid
        'min_students_for_training': 10, # Minimum number of processed student records to train a model
        'min_samples_for_stratify': 2, # Minimum samples per class for stratified split (sklearn requirement)
        'fallback_encodings': ['utf-8', 'tis-620', 'latin1', 'cp1252'], # Encodings for CSV
        'total_required_credits': 130, # Example total credits for graduation
        'default_credits_per_subject': 3, # Default credits if not specified in curriculum data
        'min_subject_columns_for_subject_based': 5, # Minimum number of grade columns to detect subject_based format

        # Risk Levels for Analysis and Prediction Interpretation
        'risk_levels': {
            'low_gpa_threshold': 1.5,     # Below this GPA is very high risk
            'warning_gpa_threshold': 2.0, # Below this GPA is warning risk (e.g., probation)
            'high_fail_rate_threshold': 0.25, # Percentage of failed subjects to be considered high risk
            'medium_fail_rate_threshold': 0.15,
            'high_confidence_threshold': 0.9, # Confidence above this is "high"
            'medium_confidence_threshold': 0.7 # Confidence between this and high_confidence_threshold is "medium"
        }
    }

    # Machine Learning Configurations
    ML_CONFIG = {
        'random_state': 42,
        'cv_folds': 5, # Cross-validation folds for GridSearchCV
        'n_jobs': -1, # Use all available CPU cores for parallel processing (-1 means all)
        'verbose': 0, # Verbosity level for GridSearchCV (0=silent, 1=info, 2=detailed)
        'apply_smote': True # Whether to apply SMOTE for imbalanced datasets
    }

    # Hyperparameters for GridSearchCV
    MODEL_HYPERPARAMETERS = {
        'RandomForest': {
            'n_estimators': [50, 100, 200, 300], # Expanded
            'max_depth': [5, 10, 15, None], # Expanded
            'min_samples_leaf': [1, 2, 4, 8]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200, 300], # Expanded
            'learning_rate': [0.01, 0.05, 0.1, 0.2], # Expanded
            'max_depth': [3, 5, 7, 10]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100], # Expanded
            'solver': ['liblinear', 'lbfgs', 'saga'], # Added 'saga' for more flexibility
            'max_iter': [1000, 2000, 5000] # Ensure convergence for 'lbfgs' and 'saga'
        }
    }

    # Placeholder for curriculum data - will be populated dynamically
    COURSES_DATA = []
    ALL_TERMS_DATA = []

    # Core subjects (example, can also be loaded from CSV if present)
    CORE_SUBJECTS_IDS = ['CS101', 'CS102', 'CS201', 'CS301', 'CS401', 'COOP499']

    # Subject Categorization (for feature engineering)
    SUBJECT_CATEGORIES = {
        'programming': {'keywords': ['เขียนโปรแกรม', 'โปรแกรมมิ่ง', 'programming', 'code', 'algorithm', 'โครงสร้างข้อมูล'], 'thaiName': 'การเขียนโปรแกรม'},
        'math_stat': {'keywords': ['คณิตศาสตร์', 'สถิติ', 'math', 'statistic', 'discrete'], 'thaiName': 'คณิตศาสตร์และสถิติ'},
        'database': {'keywords': ['ฐานข้อมูล', 'database', 'sql'], 'thaiName': 'ฐานข้อมูล'},
        'project_coop': {'keywords': ['โครงงาน', 'โปรเจค', 'สหกิจ', 'project', 'coop', 'capstone'], 'thaiName': 'โครงงานและสหกิจศึกษา'},
        'general_ed': {'keywords': ['ภาษาอังกฤษ', 'ไทย', 'general education', 'eng', 'thai'], 'thaiName': 'วิชาศึกษาทั่วไป'},
        'others': {'keywords': [], 'thaiName': 'อื่นๆ'} # Catch-all
    }

    # Messages for frontend display
    MESSAGES = {
        'recommendations': {
            'low_risk': [
                "รักษาระดับผลการเรียนที่ดีต่อไป",
                "หาโอกาสพัฒนาทักษะเพิ่มเติม",
                "พิจารณาการทำกิจกรรมเสริมหลักสูตร"
            ],
            'medium_risk': [
                "ควรปรับปรุง GPA ให้สูงขึ้น",
                "ปรึกษาอาจารย์ประจำวิชาที่ได้เกรดน้อย",
                "จัดสรรเวลาเรียนให้เหมาะสม"
            ],
            'high_risk': [
                "ปรึกษาอาจารย์ที่ปรึกษาอย่างเร่งด่วนเพื่อวางแผนแก้ไข",
                "เข้าพบศูนย์ให้คำปรึกษาด้านการเรียน",
                "พิจารณาลดจำนวนหน่วยกิตในภาคเรียนถัดไป",
                "ทบทวนวิชาพื้นฐานที่ไม่ผ่านเกณฑ์โดยเร็ว"
            ]
        },
        'confidence_levels': {
            'high': "โมเดลมีความเชื่อมั่นสูงมากในการทำนายผล (มากกว่า 90%)",
            'medium': "โมเดลมีความเชื่อมั่นปานกลางในการทำนายผล (ระหว่าง 70% - 90%)",
            'low': "โมเดลมีความเชื่อมั่นต่ำในการทำนายผล (น้อยกว่า 70%) ควรพิจารณาปัจจัยอื่นประกอบ"
        },
        'error': {
            'processing_error': "เกิดข้อผิดพลาดในการประมวลผลข้อมูล",
            'no_data_for_training': "ข้อมูลไม่เพียงพอต่อการฝึกโมเดล",
            'curriculum_load_error': "ไม่สามารถโหลดข้อมูลแผนการเรียนจากไฟล์ CSV ได้"
        }
    }

    # Logging Configuration (same as before)
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': 'app.log',
                'maxBytes': 10485760, # 10 MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard'
            },
            'startup_file': { # For app_startup.log
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': 'app_startup.log',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': { # root logger
                'handlers': ['file', 'console'],
                'level': 'INFO',
                'propagate': True
            },
            'werkzeug': { # Suppress werkzeug access logs if needed
                'handlers': ['file', 'console'],
                'level': 'WARNING',
                'propagate': False
            },
            '__main__': { # specific logger for our app.py
                'handlers': ['file', 'console', 'startup_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }

# --- New Function: load_curriculum_from_csv ---
def load_curriculum_from_csv(csv_path, fallback_encodings, header_row_index=None):
    """
    Loads curriculum data (courses and terms) from a CSV file.
    Assumes CSV contains columns like: 'รหัสวิชา', 'ชื่อวิชา', 'หน่วยกิต', 'Pre-requisite', 'ปี', 'เทอม'
    """
    _local_logger = logging.getLogger(__name__ + '.curriculum_loader')
    _local_logger.info(f"Attempting to load curriculum from: {csv_path}")

    df = None
    for encoding in fallback_encodings:
        try:
            if header_row_index is not None:
                # pandas read_csv 'header' parameter is 0-indexed. If user says '6 rows above header',
                # then header is at index 6 (7th row).
                df = pd.read_csv(csv_path, encoding=encoding, header=header_row_index)
            else:
                df = pd.read_csv(csv_path, encoding=encoding)
            _local_logger.info(f"Successfully read CSV with encoding: {encoding} and header at row {header_row_index + 1 if header_row_index is not None else 1}.")
            break
        except Exception as e:
            _local_logger.debug(f"Failed to read CSV with {encoding} (header_row_index={header_row_index}): {e}")
            df = None # Ensure df is None if current encoding fails

    if df is None:
        _local_logger.error(f"Could not read curriculum CSV file '{csv_path}' with any supported encoding or header setting.")
        return [], [] # Return empty if unable to load

    # Normalize column names for robust access
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Map possible column names to standard names
    col_map = {
        'id': next((c for c in df.columns if 'รหัสวิชา' in c or 'course_id' in c or 'รหัส' == c), None),
        'thaiName': next((c for c in df.columns if 'ชื่อวิชา' in c or 'course_name' in c or 'ชื่อ' == c), None),
        'credit': next((c for c in df.columns if 'หน่วยกิต' in c or 'credit' in c), None),
        'prereq_raw': next((c for c in df.columns if 'pre-requisite' in c or 'prereq' in c), None),
        'year': next((c for c in df.columns if 'ปี' in c or 'year' in c or 'academic_year' in c), None),
        'term': next((c for c in df.columns if 'เทอม' in c or 'term' in c or 'semester' in c), None),
    }
    
    _local_logger.info(f"Detected columns: {col_map}")

    # Validate essential columns
    if not col_map['id'] or not col_map['thaiName'] or not col_map['credit'] or not col_map['year'] or not col_map['term']:
        _local_logger.error(f"Missing essential columns in curriculum CSV: {col_map}")
        return [], []

    courses_data = {} # Use a dict for unique courses by ID
    all_terms_data_raw = {} # { (year, term): [course_ids] }

    for index, row in df.iterrows():
        try:
            course_id = str(row[col_map['id']]).strip().upper()
            thai_name = str(row[col_map['thaiName']]).strip()
            credit = pd.to_numeric(row[col_map['credit']], errors='coerce')
            
            # Handle prerequisites
            prereq_list = []
            if col_map['prereq_raw'] and pd.notna(row[col_map['prereq_raw']]):
                # Split by common delimiters like comma, semicolon, or "and"
                prereq_str = str(row[col_map['prereq_raw']]).replace(' ', '').replace('และ', ',').replace(';', ',')
                prereq_ids = [p.strip().upper() for p in prereq_str.split(',') if p.strip()]
                prereq_list = prereq_ids

            course_year = pd.to_numeric(row[col_map['year']], errors='coerce')
            course_term = pd.to_numeric(row[col_map['term']], errors='coerce')

            if pd.isna(credit) or credit <= 0:
                credit = Config.DATA_CONFIG['default_credits_per_subject']
                _local_logger.warning(f"Invalid credit for {course_id}, defaulting to {credit}.")

            if pd.isna(course_year) or pd.isna(course_term):
                _local_logger.warning(f"Skipping course {course_id} due to missing Year/Term information.")
                continue

            # Add to courses_data (handle duplicates by overwriting, or taking first encountered)
            if course_id not in courses_data:
                courses_data[course_id] = {
                    'id': course_id,
                    'thaiName': thai_name,
                    'credit': int(credit),
                    'prereq': prereq_list
                }
            else:
                # If course exists, update its prerequisites if new ones are found, or keep existing for simplicity
                # This logic depends on how robust the CSV is. For now, just ensure prereqs are consolidated.
                existing_prereqs = set(courses_data[course_id]['prereq'])
                courses_data[course_id]['prereq'] = list(existing_prereqs.union(prereq_list))


            # Add to all_terms_data_raw
            term_key = (int(course_year), int(course_term))
            if term_key not in all_terms_data_raw:
                all_terms_data_raw[term_key] = []
            all_terms_data_raw[term_key].append(course_id)

        except Exception as e:
            _local_logger.warning(f"Error processing row {index} in curriculum CSV: {e}. Skipping row.")
            continue

    # Convert to list format and sort terms
    all_terms_data = []
    for (year, term) in sorted(all_terms_data_raw.keys()):
        all_terms_data.append({
            'year': year,
            'term': term,
            'ids': sorted(list(set(all_terms_data_raw[(year, term)]))) # Ensure unique and sorted course IDs
        })
    
    _local_logger.info(f"Loaded {len(courses_data)} unique courses and {len(all_terms_data)} terms from CSV.")
    return list(courses_data.values()), all_terms_data

# --- Dynamic Loading of Curriculum Data ---
# Configure local logger for this section
_curriculum_loader_logger = logging.getLogger(__name__ + '.curriculum_init')
_curriculum_loader_logger.info("Attempting to dynamically load curriculum data...")

try:
    # Try with header at row 6 (0-indexed, so header=6 means 7th row) based on user's hint
    COURSES_DATA_FROM_CSV, ALL_TERMS_DATA_FROM_CSV = load_curriculum_from_csv(
        Config.CSV_CURRICULUM_PATH,
        Config.DATA_CONFIG['fallback_encodings'],
        header_row_index=6 # Assuming header is after 6 rows of meta-data
    )

    if not COURSES_DATA_FROM_CSV and not ALL_TERMS_DATA_FROM_CSV:
        _curriculum_loader_logger.warning("No curriculum data loaded from CSV with header_row_index=6. Trying with default header (row 1).")
        # Fallback to default header if the first attempt yields no data
        COURSES_DATA_FROM_CSV, ALL_TERMS_DATA_FROM_CSV = load_curriculum_from_csv(
            Config.CSV_CURRICULUM_PATH,
            Config.DATA_CONFIG['fallback_encodings'],
            header_row_index=0 # Default to first row as header
        )
        if not COURSES_DATA_FROM_CSV and not ALL_TERMS_DATA_FROM_CSV:
            _curriculum_loader_logger.error("Failed to load curriculum data from CSV with any tested header setting.")
            # Keep empty lists if loading completely fails
            Config.COURSES_DATA = []
            Config.ALL_TERMS_DATA = []
            Config.MESSAGES['error']['curriculum_load_error'] = "ไม่สามารถโหลดข้อมูลแผนการเรียนจากไฟล์ CSV ได้ (ไม่พบข้อมูลในไฟล์ หรือรูปแบบไม่ถูกต้อง)"
        else:
            Config.COURSES_DATA = COURSES_DATA_FROM_CSV
            Config.ALL_TERMS_DATA = ALL_TERMS_DATA_FROM_CSV
            _curriculum_loader_logger.info("Successfully loaded curriculum from CSV with default header.")
    else:
        Config.COURSES_DATA = COURSES_DATA_FROM_CSV
        Config.ALL_TERMS_DATA = ALL_TERMS_DATA_FROM_CSV
        _curriculum_loader_logger.info("Successfully loaded curriculum from CSV with header at row 7.")

except FileNotFoundError:
    _curriculum_loader_logger.warning(f"Curriculum CSV file not found at: {Config.CSV_CURRICULUM_PATH}. Using default empty curriculum.")
    Config.COURSES_DATA = []
    Config.ALL_TERMS_DATA = []
    Config.MESSAGES['error']['curriculum_load_error'] = "ไม่พบไฟล์แผนการเรียน CSV! (แผนการเรียน รับเข้าปี 2567.csv)"
except Exception as e:
    _curriculum_loader_logger.error(f"An unexpected error occurred while loading curriculum CSV: {e}")
    Config.COURSES_DATA = []
    Config.ALL_TERMS_DATA = []
    Config.MESSAGES['error']['curriculum_load_error'] = f"เกิดข้อผิดพลาดไม่คาดคิดขณะโหลดไฟล์แผนการเรียน: {str(e)}"

# Update CORE_SUBJECTS_IDS from loaded courses if they match any
# This is a simple heuristic; a more robust approach would be to have a 'core' column in the CSV
if Config.COURSES_DATA:
    inferred_core_subjects = []
    for course in Config.COURSES_DATA:
        if 'โครงงาน' in course['thaiName'] or 'สหกิจ' in course['thaiName'] or 'Project' in course['thaiName'] or 'Co-op' in course['thaiName']:
            inferred_core_subjects.append(course['id'])
    # Only update if some core subjects were inferred, otherwise keep default
    if inferred_core_subjects:
        Config.CORE_SUBJECTS_IDS = list(set(inferred_core_subjects).union(Config.CORE_SUBJECTS_IDS)) # Union with existing defaults


class DevelopmentConfig(Config):
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000
    LOGGING_CONFIG = {**Config.LOGGING_CONFIG} # Copy, then modify
    LOGGING_CONFIG['loggers']['']['level'] = 'DEBUG' # More verbose in dev
    LOGGING_CONFIG['loggers']['__main__']['level'] = 'DEBUG' # More verbose in dev
    _curriculum_loader_logger.setLevel(logging.DEBUG) # Set loader logger to DEBUG in dev

class ProductionConfig(Config):
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 80))
    # No changes to LOGGING_CONFIG, INFO level is good for production


def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig

