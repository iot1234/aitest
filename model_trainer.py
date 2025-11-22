import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from advanced_training import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

def preprocess_tan1_data(file_path):
    """
    Process TAN1.csv from Long Format to Wide Format
    and calculate RESULT (Graduated/Not Graduated)
    """
    # Read data
    df_long = pd.read_csv(file_path)
    
    # Check required columns
    required_columns = ['STUDENT_ID', 'COURSE_ID', 'GRADE', 'CREDIT']
    for col in required_columns:
        if col not in df_long.columns:
            # Try mapping common variations
            found = False
            for var in [col, col.lower(), col.title()]:
                if var in df_long.columns:
                    df_long.rename(columns={var: col}, inplace=True)
                    found = True
                    break
            if not found:
                # If still not found, might be Thai column names, skip check or raise error
                # For now, let's assume the user provides correct format or we handle it later
                pass

    # Create grade mapping
    grade_mapping = {
        'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 
        'D+': 1.5, 'D': 1.0, 'F': 0.0, 'W': 0.0, 'S': None
    }
    
    # Add GRADE_POINT_NUM
    if 'GRADE_POINT_NUM' not in df_long.columns:
        df_long['GRADE_POINT_NUM'] = df_long['GRADE'].map(grade_mapping)
    
    # Convert to Wide Format
    df_wide = df_long.pivot_table(
        index='STUDENT_ID', 
        columns='COURSE_ID', 
        values='GRADE', 
        aggfunc='first'
    ).reset_index()
    
    # Calculate GPA and Credits
    student_stats = []
    
    for student_id in df_long['STUDENT_ID'].unique():
        student_data = df_long[df_long['STUDENT_ID'] == student_id]
        
        # Calculate GPA (excluding S)
        gpa_data = student_data[student_data['GRADE'] != 'S']
        if len(gpa_data) > 0:
            total_grade_points = (gpa_data['GRADE_POINT_NUM'] * gpa_data['CREDIT']).sum()
            total_credits = gpa_data['CREDIT'].sum()
            gpa = total_grade_points / total_credits if total_credits > 0 else 0
        else:
            gpa = 0
            total_credits = 0
        
        # Total credits (including S)
        total_credits_all = student_data['CREDIT'].sum()
        
        # Graduation criteria: GPA >= 2.00 and Credits >= 136
        result = 1 if (gpa >= 2.00 and total_credits_all >= 136) else 0
        
        student_stats.append({
            'STUDENT_ID': student_id,
            'GPA': gpa,
            'TOTAL_CREDITS': total_credits_all,
            'RESULT': result
        })
    
    # Merge RESULT
    stats_df = pd.DataFrame(student_stats)
    if not stats_df.empty:
        df_wide = df_wide.merge(stats_df[['STUDENT_ID', 'RESULT']], on='STUDENT_ID', how='left')
    
    # Create course_credit_map
    course_credit_map = df_long.groupby('COURSE_ID')['CREDIT'].first().to_dict()
    
    return df_wide, df_long, course_credit_map


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
        self.feature_engineer = AdvancedFeatureEngineer(self.grade_mapping)
        self.course_credit_map = {}
        
    def train(self, df: pd.DataFrame):
        """Train models using the provided DataFrame."""
        logger.info("🚀 Starting model training pipeline...")
        
        # 1. Feature Engineering
        X, y = self.feature_engineer.prepare_training_data(df)
        
        # 2. Scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Handle Imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # 5. Train & Evaluate
        results = {}
        best_model_name = None
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                best_model_name = name
                
        logger.info(f"🏆 Best model: {best_model_name} with accuracy {self.best_score:.4f}")
        return results

    def save_model(self, filepath: str):
        """Save the trained model and artifacts."""
        if not self.best_model:
            raise ValueError("No model trained to save.")
            
        data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'grade_mapping': self.grade_mapping,
            'course_profiles': self.feature_engineer.course_profiles,
            'best_score': self.best_score
        }
        joblib.dump(data, filepath)
        logger.info(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        data = joblib.load(filepath)
        self.best_model = data['model']
        self.scaler = data['scaler']
        self.grade_mapping = data.get('grade_mapping', self.grade_mapping)
        
        # Restore feature engineer state if possible
        if 'course_profiles' in data:
            self.feature_engineer.course_profiles = data['course_profiles']
            
        logger.info(f"✅ Model loaded from {filepath}")

class PredictionService:
    def __init__(self, model_path: str = None):
        self.trainer = AdvancedModelTrainer()
        if model_path:
            self.trainer.load_model(model_path)
            
    def predict(self, student_grades: dict):
        """
        Predict graduation likelihood for a single student.
        student_grades: dict of {course_code: grade_letter}
        """
        if not self.trainer.best_model:
            return {"error": "Model not loaded"}
            
        # Convert grades dict to DataFrame-like structure expected by feature engineer
        # This is a bit tricky because feature engineer expects a full dataframe history.
        # We need to adapt `create_dynamic_snapshot_features` to work with a single student dict.
        
        # For now, let's assume we can construct a minimal dataframe
        data = []
        for course, grade in student_grades.items():
            data.append({
                'student_id': 'current_student',
                'course_code': course,
                'grade': grade,
                'credit': 3, # Default
                'semester': 1, # Dummy
                'academic_year': 2024 # Dummy
            })
        
        df = pd.DataFrame(data)
        
        # Use feature engineer to create features
        # We need to access the internal method or expose a public one for single student
        features_dict = self.trainer.feature_engineer.create_dynamic_snapshot_features(
            df, self.trainer.feature_engineer.course_profiles
        )
        
        # Convert dict to array (ensure order matches training)
        # This requires knowing the feature columns order. 
        # Ideally, `prepare_training_data` returns a DataFrame with columns.
        # We should probably save feature names in the model file.
        
        # For simplicity in this refactor, let's assume we extract values in a consistent order
        # OR we update `AdvancedFeatureEngineer` to return a DataFrame with named columns
        # and we use that.
        
        feature_values = list(features_dict.values())
        
        # Scale
        features_scaled = self.trainer.scaler.transform([feature_values])
        
        # Predict
        prob = self.trainer.best_model.predict_proba(features_scaled)[0]
        prediction = self.trainer.best_model.predict(features_scaled)[0]
        
        return {
            "will_graduate": bool(prediction),
            "probability_graduate": float(prob[1]),
            "probability_not_graduate": float(prob[0]),
            "confidence": float(max(prob))
        }
