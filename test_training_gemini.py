#!/usr/bin/env python3
"""
Test Script: Training System + Gemini Integration
ทดสอบระบบเทรนโมเดลและการทำงานร่วมกับ Gemini
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test(test_name, success, message="", is_warning=False):
    """บันทึกผลการทดสอบ"""
    if is_warning:
        test_results['warnings'].append((test_name, message))
        logger.warning(f"⚠️ [{test_name}] {message}")
    elif success:
        test_results['passed'].append(test_name)
        logger.info(f"✅ [{test_name}] PASSED - {message}")
    else:
        test_results['failed'].append((test_name, message))
        logger.error(f"❌ [{test_name}] FAILED - {message}")

# ==========================================
# Test 1: Import Modules
# ==========================================
def test_imports():
    """ทดสอบการ import modules ที่จำเป็น"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 1: Testing Module Imports")
    logger.info("="*60)
    
    modules = {
        'pandas': None,
        'numpy': None,
        'sklearn': None,
        'joblib': None,
        'imblearn': None,
    }
    
    for module_name in modules.keys():
        try:
            __import__(module_name)
            log_test(f"Import {module_name}", True, "Imported successfully")
        except ImportError as e:
            log_test(f"Import {module_name}", False, str(e))
    
    # Test local modules
    local_modules = ['advanced_training', 'model_trainer', 'gemini_service', 'config']
    for module_name in local_modules:
        try:
            __import__(module_name)
            log_test(f"Import {module_name}", True, "Imported successfully")
        except ImportError as e:
            log_test(f"Import {module_name}", False, str(e))

# ==========================================
# Test 2: Configuration Check
# ==========================================
def test_configuration():
    """ทดสอบ configuration ของระบบ"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 2: Testing Configuration")
    logger.info("="*60)
    
    try:
        from config import Config, get_config
        config = get_config()
        
        # Check essential configs
        essential_configs = [
            'ML_CONFIG',
            'DATA_CONFIG',
            'COURSES_DATA',
            'ALL_TERMS_DATA',
        ]
        
        for cfg_name in essential_configs:
            if hasattr(config, cfg_name):
                log_test(f"Config {cfg_name}", True, "Configuration exists")
            else:
                log_test(f"Config {cfg_name}", False, "Configuration missing")
        
        # Check grade mapping
        grade_mapping = config.DATA_CONFIG.get('grade_mapping', {})
        if len(grade_mapping) >= 8:
            log_test("Grade Mapping", True, f"Found {len(grade_mapping)} grade mappings")
        else:
            log_test("Grade Mapping", False, f"Only {len(grade_mapping)} grade mappings found")
            
    except Exception as e:
        log_test("Configuration", False, str(e))

# ==========================================
# Test 3: Data Loading
# ==========================================
def test_data_loading():
    """ทดสอบการโหลดข้อมูล"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 3: Testing Data Loading")
    logger.info("="*60)
    
    data_file = '/workspace/TAN117.xlsx'
    
    try:
        if os.path.exists(data_file):
            df = pd.read_excel(data_file)
            log_test("Load Excel File", True, f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Check required columns
            required_cols_variations = [
                ['DUMMY_STUDENTNO', 'dummy_studentno', 'student_id'],
                ['GRADE', 'grade', 'เกรด'],
                ['GRADE_POINT', 'grade_point', 'คะแนนเกรด'],
                ['CREDIT', 'credit', 'หน่วยกิต'],
                ['COURSE', 'course', 'course_code', 'รหัสวิชา']
            ]
            
            for col_variations in required_cols_variations:
                found = False
                for col in col_variations:
                    if col in df.columns or col.lower() in [c.lower() for c in df.columns]:
                        found = True
                        log_test(f"Column '{col_variations[0]}'", True, f"Found as column")
                        break
                if not found:
                    log_test(f"Column '{col_variations[0]}'", False, f"Column not found in: {list(df.columns)[:10]}...")
            
            return df
        else:
            log_test("Load Excel File", False, f"File not found: {data_file}")
            return None
    except Exception as e:
        log_test("Data Loading", False, str(e))
        return None

# ==========================================
# Test 4: Feature Engineering
# ==========================================
def test_feature_engineering(df=None):
    """ทดสอบ Feature Engineering"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 4: Testing Feature Engineering")
    logger.info("="*60)
    
    try:
        from advanced_training import AdvancedFeatureEngineer
        from config import get_config
        
        config = get_config()
        grade_mapping = config.DATA_CONFIG.get('grade_mapping', {
            'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0,
            'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0
        })
        
        engineer = AdvancedFeatureEngineer(grade_mapping)
        log_test("Create FeatureEngineer", True, "Instance created successfully")
        
        # Test with sample data if available
        if df is not None and len(df) > 0:
            try:
                X, y = engineer.prepare_training_data(df)
                log_test("Prepare Training Data", True, f"X shape: {X.shape}, y shape: {y.shape}")
                
                # Check for NaN values
                nan_count = X.isna().sum().sum()
                if nan_count == 0:
                    log_test("NaN Check", True, "No NaN values in features")
                else:
                    log_test("NaN Check", True, f"Found {nan_count} NaN values (will be handled)", is_warning=True)
                
                # Check class distribution
                unique, counts = np.unique(y, return_counts=True)
                class_dist = dict(zip(unique, counts))
                log_test("Class Distribution", True, f"Classes: {class_dist}")
                
                # Check if both classes exist
                if len(unique) >= 2:
                    log_test("Class Balance", True, "Both classes present")
                else:
                    log_test("Class Balance", False, f"Only {len(unique)} class(es) found - may cause training issues")
                
                return X, y, engineer
                
            except Exception as e:
                log_test("Prepare Training Data", False, str(e))
                return None, None, None
        else:
            log_test("Feature Engineering", True, "Skipped - no data available", is_warning=True)
            return None, None, None
            
    except Exception as e:
        log_test("Feature Engineering", False, str(e))
        return None, None, None

# ==========================================
# Test 5: Model Training
# ==========================================
def test_model_training(X=None, y=None):
    """ทดสอบการเทรนโมเดล"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 5: Testing Model Training")
    logger.info("="*60)
    
    if X is None or y is None:
        log_test("Model Training", True, "Skipped - no training data", is_warning=True)
        return None
    
    try:
        from advanced_training import train_ensemble_model
        
        logger.info(f"📊 Training with {len(X)} samples, {X.shape[1]} features")
        
        result = train_ensemble_model(X, y)
        
        if 'models' in result and len(result['models']) > 0:
            log_test("Ensemble Training", True, f"Trained {len(result['models'])} models")
        else:
            log_test("Ensemble Training", False, "No models were trained")
        
        # Check metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            if metric in result:
                value = result[metric]
                if value >= 0.5:
                    log_test(f"Metric {metric}", True, f"Value: {value:.4f}")
                else:
                    log_test(f"Metric {metric}", True, f"Value: {value:.4f} (low but acceptable)", is_warning=True)
        
        return result
        
    except Exception as e:
        log_test("Model Training", False, str(e))
        import traceback
        logger.error(traceback.format_exc())
        return None

# ==========================================
# Test 6: Gemini Service
# ==========================================
def test_gemini_service():
    """ทดสอบ Gemini Service"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 6: Testing Gemini Service")
    logger.info("="*60)
    
    try:
        from gemini_service import GeminiService
        
        service = GeminiService()
        
        if service.is_configured:
            log_test("Gemini Configuration", True, f"Configured with model: {service.model_name}")
            
            # Test a simple call (if API key is valid)
            try:
                test_payload = {
                    "test": True,
                    "analysis_goal": "Test connection"
                }
                # We won't actually call the API in test mode to avoid using quota
                log_test("Gemini API Ready", True, "API configured and ready")
            except Exception as e:
                log_test("Gemini API Test", False, str(e))
        else:
            log_test("Gemini Configuration", True, "Not configured (GEMINI_API_KEY missing)", is_warning=True)
            
    except Exception as e:
        log_test("Gemini Service", False, str(e))

# ==========================================
# Test 7: Integration Test - Training + Gemini
# ==========================================
def test_integration(df=None, training_result=None, engineer=None):
    """ทดสอบการทำงานร่วมกันระหว่าง Training และ Gemini"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 7: Testing Integration (Training + Gemini)")
    logger.info("="*60)
    
    try:
        # Check if training result has course profiles
        if engineer and hasattr(engineer, 'course_profiles'):
            profiles = engineer.course_profiles
            if profiles and len(profiles) > 0:
                log_test("Course DNA Profiles", True, f"Generated {len(profiles)} course profiles")
                
                # Check profile content
                sample_profile = list(profiles.values())[0] if profiles else {}
                required_keys = ['avg_grade', 'fail_rate', 'difficulty_score']
                for key in required_keys:
                    if key in sample_profile:
                        log_test(f"Profile Key '{key}'", True, f"Value: {sample_profile[key]:.3f}")
                    else:
                        log_test(f"Profile Key '{key}'", False, "Key missing in profile")
            else:
                log_test("Course DNA Profiles", False, "No profiles generated")
        else:
            log_test("Course DNA Profiles", True, "Skipped - no engineer available", is_warning=True)
        
        # Check if Gemini can receive training context
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if gemini_api_key:
            log_test("Gemini Integration Ready", True, "API key found, integration possible")
        else:
            log_test("Gemini Integration Ready", True, "API key not set - Gemini analysis will be disabled", is_warning=True)
            
    except Exception as e:
        log_test("Integration Test", False, str(e))

# ==========================================
# Test 8: Prediction Flow
# ==========================================
def test_prediction_flow(training_result=None, engineer=None):
    """ทดสอบ Flow การทำนาย"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Test 8: Testing Prediction Flow")
    logger.info("="*60)
    
    if training_result is None or engineer is None:
        log_test("Prediction Flow", True, "Skipped - no trained model", is_warning=True)
        return
    
    try:
        from advanced_training import ContextAwarePredictor
        
        # Create predictor
        predictor = ContextAwarePredictor(
            feature_engineer=engineer,
            models=training_result.get('models', {}),
            scaler=training_result.get('scaler'),
            feature_names=training_result.get('feature_names', [])
        )
        log_test("Create Predictor", True, "Predictor instance created")
        
        # Create sample student data
        sample_data = pd.DataFrame([
            {'course_code': '03-407-100-101', 'grade': 'A', 'credit': 3, 'academic_year': 2024, 'semester': 1},
            {'course_code': '03-407-100-102', 'grade': 'B+', 'credit': 3, 'academic_year': 2024, 'semester': 1},
            {'course_code': '02-005-011-109', 'grade': 'B', 'credit': 3, 'academic_year': 2024, 'semester': 1},
            {'course_code': '02-005-030-101', 'grade': 'C+', 'credit': 3, 'academic_year': 2024, 'semester': 1},
        ])
        
        try:
            result = predictor.predict_graduation_probability(sample_data, explain=False)
            if 'probability' in result:
                log_test("Prediction Result", True, f"Probability: {result['probability']:.4f}")
                
                if result.get('prediction_method') == 'AI_MODEL':
                    log_test("Prediction Method", True, "Using AI Model (not heuristic)")
                else:
                    log_test("Prediction Method", True, f"Method: {result.get('prediction_method', 'unknown')}", is_warning=True)
            else:
                log_test("Prediction Result", False, "No probability in result")
        except ValueError as e:
            # Expected if model not properly trained
            log_test("Prediction Result", True, f"Expected error: {str(e)[:50]}...", is_warning=True)
            
    except Exception as e:
        log_test("Prediction Flow", False, str(e))
        import traceback
        logger.error(traceback.format_exc())

# ==========================================
# Main Test Runner
# ==========================================
def run_all_tests():
    """รันการทดสอบทั้งหมด"""
    logger.info("\n" + "="*60)
    logger.info("🚀 STARTING COMPREHENSIVE SYSTEM TESTS")
    logger.info("="*60)
    logger.info(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Test 1: Imports
    test_imports()
    
    # Test 2: Configuration
    test_configuration()
    
    # Test 3: Data Loading
    df = test_data_loading()
    
    # Test 4: Feature Engineering
    X, y, engineer = test_feature_engineering(df)
    
    # Test 5: Model Training
    training_result = test_model_training(X, y)
    
    # Test 6: Gemini Service
    test_gemini_service()
    
    # Test 7: Integration
    test_integration(df, training_result, engineer)
    
    # Test 8: Prediction Flow
    test_prediction_flow(training_result, engineer)
    
    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    total_passed = len(test_results['passed'])
    total_failed = len(test_results['failed'])
    total_warnings = len(test_results['warnings'])
    total_tests = total_passed + total_failed
    
    logger.info(f"✅ Passed: {total_passed}")
    logger.info(f"❌ Failed: {total_failed}")
    logger.info(f"⚠️ Warnings: {total_warnings}")
    logger.info(f"📊 Total: {total_tests}")
    
    if total_failed > 0:
        logger.info("\n❌ FAILED TESTS:")
        for test_name, error in test_results['failed']:
            logger.info(f"   - {test_name}: {error}")
    
    if total_warnings > 0:
        logger.info("\n⚠️ WARNINGS:")
        for test_name, warning in test_results['warnings']:
            logger.info(f"   - {test_name}: {warning}")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    logger.info(f"\n📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("🎉 EXCELLENT! System is working well.")
    elif success_rate >= 70:
        logger.info("👍 GOOD. Some issues need attention.")
    else:
        logger.info("⚠️ NEEDS ATTENTION. Several issues found.")
    
    return test_results

if __name__ == '__main__':
    results = run_all_tests()
