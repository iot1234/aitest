# explainable_ai.py - Explainable AI Module for Student Prediction System
"""
โมดูลสำหรับอธิบายการทำนายของระบบ AI
- Feature Importance (ปัจจัยที่มีผลต่อการทำนาย)
- SHAP Values (อธิบายการทำนายแต่ละคน)
- Decision Path (เส้นทางการตัดสินใจของโมเดล)
- Recommendation Engine (แนะนำการปรับปรุง)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExplainablePredictor:
    """
    คลาสสำหรับอธิบายการทำนาย และแนะนำแนวทางปรับปรุง
    """
    
    def __init__(self, models: Dict = None, feature_names: List[str] = None, 
                 course_profiles: Dict = None, courses_data: List = None):
        """
        Initialize Explainable Predictor
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            course_profiles: Course DNA profiles
            courses_data: Course information
        """
        self.models = models or {}
        self.feature_names = feature_names or []
        self.course_profiles = course_profiles or {}
        self.courses_data = courses_data or []
        
        # Feature importance from Random Forest (if available)
        self.feature_importance = None
        if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.models['rf'].feature_importances_
            ))
    
    def explain_prediction(self, student_features: Dict, prediction_result: Dict) -> Dict:
        """
        อธิบายผลการทำนายอย่างละเอียด
        
        Args:
            student_features: Features ของนักศึกษา
            prediction_result: ผลการทำนายจากโมเดล
        
        Returns:
            Dict ที่มีคำอธิบายละเอียด
        """
        try:
            probability = prediction_result.get('probability', 0.5)
            will_graduate = probability >= 0.5
            
            # === 1. สรุปผลการทำนาย ===
            prediction_summary = {
                'will_graduate': will_graduate,
                'probability': round(probability * 100, 1),
                'confidence': round(prediction_result.get('confidence', 0.5) * 100, 1),
                'risk_level': self._calculate_risk_level(probability),
                'status': self._get_graduation_status(probability)
            }
            
            # === 2. วิเคราะห์ปัจจัยหลัก (Key Factors) ===
            key_factors = self._analyze_key_factors(student_features, probability)
            
            # === 3. ทำไมถึงทำนายแบบนี้ (Why This Prediction) ===
            reasons = self._explain_why_this_prediction(
                student_features, probability, will_graduate
            )
            
            # === 4. อุปสรรคและความเสี่ยง (Obstacles & Risks) ===
            obstacles = self._identify_obstacles(student_features, will_graduate)
            
            # === 5. จุดแข็งและโอกาส (Strengths & Opportunities) ===
            strengths = self._identify_strengths(student_features)
            
            # === 6. คำแนะนำเชิงรุก (Actionable Recommendations) ===
            recommendations = self._generate_recommendations(
                student_features, probability, obstacles
            )
            
            # === 7. เส้นทางสู่การจบการศึกษา (Graduation Path) ===
            graduation_path = self._suggest_graduation_path(
                student_features, probability, will_graduate
            )
            
            # === 8. ทำนายผลลัพธ์ในอนาคต (Future Projections) ===
            future_projections = self._project_future_scenarios(
                student_features, probability
            )
            
            return {
                'summary': prediction_summary,
                'key_factors': key_factors,
                'reasons': reasons,
                'obstacles': obstacles,
                'strengths': strengths,
                'recommendations': recommendations,
                'graduation_path': graduation_path,
                'future_projections': future_projections,
                'detailed_breakdown': self._create_detailed_breakdown(student_features)
            }
            
        except Exception as e:
            logger.error(f"Error in explain_prediction: {e}")
            return self._get_default_explanation()
    
    def _calculate_risk_level(self, probability: float) -> str:
        """คำนวณระดับความเสี่ยง"""
        if probability >= 0.80:
            return 'very_low'  # เสี่ยงต่ำมาก
        elif probability >= 0.65:
            return 'low'       # เสี่ยงต่ำ
        elif probability >= 0.50:
            return 'moderate'  # เสี่ยงปานกลาง
        elif probability >= 0.35:
            return 'high'      # เสี่ยงสูง
        else:
            return 'very_high' # เสี่ยงสูงมาก
    
    def _get_graduation_status(self, probability: float) -> str:
        """แปลงความน่าจะเป็นเป็นสถานะ"""
        if probability >= 0.85:
            return 'จะจบการศึกษาแน่นอน'
        elif probability >= 0.70:
            return 'มีแนวโน้มจบการศึกษา'
        elif probability >= 0.55:
            return 'มีโอกาสจบการศึกษา'
        elif probability >= 0.40:
            return 'มีความเสี่ยงที่จะไม่จบ'
        else:
            return 'มีความเสี่ยงสูงที่จะไม่จบ'
    
    def _analyze_key_factors(self, features: Dict, probability: float) -> List[Dict]:
        """
        วิเคราะห์ปัจจัยหลักที่ส่งผลต่อการทำนาย
        """
        key_factors = []
        
        # 1. GPA (ปัจจัยสำคัญที่สุด)
        gpa = features.get('GPAX_so_far', 0)
        gpa_impact = self._calculate_gpa_impact(gpa)
        key_factors.append({
            'factor': 'เกรดเฉลี่ยสะสม (GPA)',
            'value': round(gpa, 2),
            'impact': gpa_impact['level'],
            'impact_score': gpa_impact['score'],
            'description': gpa_impact['description'],
            'icon': '📊'
        })
        
        # 2. วิชาที่ตก (Failed Courses)
        failed_count = features.get('Total_F_Count_so_far', 0)
        fail_impact = self._calculate_fail_impact(failed_count)
        key_factors.append({
            'factor': 'วิชาที่ไม่ผ่าน (เกรด F)',
            'value': int(failed_count),
            'impact': fail_impact['level'],
            'impact_score': fail_impact['score'],
            'description': fail_impact['description'],
            'icon': '❌'
        })
        
        # 3. Killer Courses Performance
        killer_passed = features.get('Passed_Killer_Courses', 0)
        killer_impact = self._calculate_killer_course_impact(killer_passed)
        key_factors.append({
            'factor': 'วิชายากที่ผ่าน',
            'value': int(killer_passed),
            'impact': killer_impact['level'],
            'impact_score': killer_impact['score'],
            'description': killer_impact['description'],
            'icon': '💪'
        })
        
        # 4. แนวโน้ม GPA
        gpa_trend = features.get('GPA_trend', 0)
        trend_impact = self._calculate_trend_impact(gpa_trend)
        key_factors.append({
            'factor': 'แนวโน้มการเรียน',
            'value': 'ดีขึ้น' if gpa_trend > 0 else 'แย่ลง' if gpa_trend < 0 else 'คงที่',
            'impact': trend_impact['level'],
            'impact_score': trend_impact['score'],
            'description': trend_impact['description'],
            'icon': '📈' if gpa_trend > 0 else '📉'
        })
        
        # 5. Performance vs Course Average
        vs_avg = features.get('Avg_vs_Course_Avg', 0)
        vs_avg_impact = self._calculate_vs_avg_impact(vs_avg)
        key_factors.append({
            'factor': 'เทียบกับค่าเฉลี่ยวิชา',
            'value': f"{'+' if vs_avg > 0 else ''}{round(vs_avg, 2)}",
            'impact': vs_avg_impact['level'],
            'impact_score': vs_avg_impact['score'],
            'description': vs_avg_impact['description'],
            'icon': '🎯'
        })
        
        # 6. Consistency Score
        consistency = features.get('Consistency_Score', 0)
        consistency_impact = self._calculate_consistency_impact(consistency)
        key_factors.append({
            'factor': 'ความสม่ำเสมอผลการเรียน',
            'value': round(consistency, 2),
            'impact': consistency_impact['level'],
            'impact_score': consistency_impact['score'],
            'description': consistency_impact['description'],
            'icon': '🎲'
        })
        
        # เรียงตาม impact_score จากมากไปน้อย
        key_factors.sort(key=lambda x: abs(x['impact_score']), reverse=True)
        
        return key_factors
    
    def _explain_why_this_prediction(self, features: Dict, probability: float, 
                                     will_graduate: bool) -> List[str]:
        """อธิบายว่าทำไมถึงทำนายแบบนี้"""
        reasons = []
        
        gpa = features.get('GPAX_so_far', 0)
        failed_count = features.get('Total_F_Count_so_far', 0)
        killer_passed = features.get('Passed_Killer_Courses', 0)
        gpa_trend = features.get('GPA_trend', 0)
        vs_avg = features.get('Avg_vs_Course_Avg', 0)
        consistency = features.get('Consistency_Score', 0)
        
        if will_graduate:
            # เหตุผลที่คาดว่าจะจบ
            if gpa >= 3.5:
                reasons.append(f"✅ มี GPA สูงมาก ({gpa:.2f}) แสดงถึงความสามารถทางการเรียนที่ดีเยี่ยม")
            elif gpa >= 3.0:
                reasons.append(f"✅ มี GPA อยู่ในเกณฑ์ดี ({gpa:.2f}) เพียงพอต่อการจบการศึกษา")
            elif gpa >= 2.5:
                reasons.append(f"⚠️ GPA อยู่ในเกณฑ์ผ่าน ({gpa:.2f}) แต่ควรปรับปรุงเพิ่มเติม")
            
            if failed_count == 0:
                reasons.append("✅ ไม่มีวิชาที่ไม่ผ่าน แสดงถึงความมุ่งมั่นและความสามารถที่ดี")
            elif failed_count <= 2:
                reasons.append(f"⚠️ มีวิชาที่ไม่ผ่านเพียง {failed_count} วิชา ซึ่งยังสามารถแก้ไขได้")
            
            if killer_passed >= 3:
                reasons.append(f"✅ ผ่านวิชายาก {killer_passed} วิชา แสดงถึงความสามารถในการเรียนวิชาที่ท้าทาย")
            
            if gpa_trend > 0.1:
                reasons.append(f"✅ ผลการเรียนมีแนวโน้มดีขึ้นเรื่อยๆ แสดงถึงการพัฒนาที่ดี")
            
            if vs_avg > 0.5:
                reasons.append("✅ ผลการเรียนสูงกว่าค่าเฉลี่ยของวิชาส่วนใหญ่ แสดงถึงความสามารถเหนือกว่า")
            
            if consistency > 0.7:
                reasons.append("✅ ผลการเรียนมีความสม่ำเสมอสูง แสดงถึงการเรียนที่มั่นคง")
        else:
            # เหตุผลที่คาดว่าไม่จบ
            if gpa < 2.0:
                reasons.append(f"❌ GPA ต่ำกว่าเกณฑ์ขั้นต่ำ ({gpa:.2f} < 2.00) จำเป็นต้องปรับปรุงทันที")
            elif gpa < 2.5:
                reasons.append(f"⚠️ GPA อยู่ในเกณฑ์ต่ำ ({gpa:.2f}) มีความเสี่ยงต่อการไม่จบ")
            
            if failed_count >= 5:
                reasons.append(f"❌ มีวิชาที่ไม่ผ่านมากถึง {failed_count} วิชา ส่งผลรุนแรงต่อการจบการศึกษา")
            elif failed_count >= 3:
                reasons.append(f"⚠️ มีวิชาที่ไม่ผ่าน {failed_count} วิชา ต้องเรียนซ้ำและใช้เวลาเพิ่ม")
            
            if killer_passed == 0:
                reasons.append("❌ ยังไม่ผ่านวิชายากเลย อาจมีปัญหาในการเรียนวิชาท้าทาย")
            
            if gpa_trend < -0.1:
                reasons.append("❌ ผลการเรียนมีแนวโน้มแย่ลงเรื่อยๆ จำเป็นต้องหาสาเหตุและแก้ไข")
            
            if vs_avg < -0.5:
                reasons.append("❌ ผลการเรียนต่ำกว่าค่าเฉลี่ยของวิชาส่วนใหญ่ อาจต้องปรับวิธีการเรียน")
            
            if consistency < 0.5:
                reasons.append("⚠️ ผลการเรียนไม่สม่ำเสมอ แสดงถึงปัญหาในการบริหารเวลาหรือแรงจูงใจ")
        
        return reasons[:5]  # คืนแค่ 5 เหตุผลแรก
    
    def _identify_obstacles(self, features: Dict, will_graduate: bool) -> List[Dict]:
        """ระบุอุปสรรคและความเสี่ยง"""
        obstacles = []
        
        gpa = features.get('GPAX_so_far', 0)
        failed_count = features.get('Total_F_Count_so_far', 0)
        low_grade_rate = features.get('Low_Grade_Rate', 0)
        risk_score = features.get('Risk_Score', 0)
        struggled_easy = features.get('Struggled_Easy_Courses', 0)
        
        # อุปสรรค 1: GPA ต่ำ
        if gpa < 2.0:
            obstacles.append({
                'type': 'critical',
                'title': 'GPA ต่ำกว่าเกณฑ์ขั้นต่ำ',
                'description': f'GPA ปัจจุบัน {gpa:.2f} ต่ำกว่า 2.00 ซึ่งเป็นเกณฑ์ขั้นต่ำในการจบการศึกษา',
                'severity': 10,
                'time_to_fix': f'{math.ceil((2.0 - gpa) * 2)} เทอม',
                'icon': '🚨'
            })
        elif gpa < 2.5:
            obstacles.append({
                'type': 'warning',
                'title': 'GPA ใกล้เคียงเกณฑ์ขั้นต่ำ',
                'description': f'GPA ปัจจุบัน {gpa:.2f} อยู่ในระดับที่มีความเสี่ยง',
                'severity': 7,
                'time_to_fix': '1-2 เทอม',
                'icon': '⚠️'
            })
        
        # อุปสรรค 2: วิชาที่ตก
        if failed_count >= 5:
            obstacles.append({
                'type': 'critical',
                'title': 'มีวิชาที่ไม่ผ่านจำนวนมาก',
                'description': f'มีวิชาที่ได้เกรด F ถึง {failed_count} วิชา ต้องเรียนซ้ำทั้งหมด',
                'severity': 9,
                'time_to_fix': f'{math.ceil(failed_count / 6)} เทอม',
                'icon': '❌'
            })
        elif failed_count >= 3:
            obstacles.append({
                'type': 'warning',
                'title': 'มีวิชาที่ไม่ผ่านหลายวิชา',
                'description': f'มีวิชาที่ได้เกรด F จำนวน {failed_count} วิชา ควรเรียนซ้ำโดยเร็ว',
                'severity': 6,
                'time_to_fix': '1-2 เทอม',
                'icon': '⚠️'
            })
        
        # อุปสรรค 3: เกรดต่ำในหลายวิชา
        if low_grade_rate > 0.3:
            obstacles.append({
                'type': 'warning',
                'title': 'มีวิชาที่ได้เกรดต่ำหลายวิชา',
                'description': f'{low_grade_rate*100:.0f}% ของวิชาที่เรียนได้เกรด D หรือต่ำกว่า',
                'severity': 5,
                'time_to_fix': '2-3 เทอม',
                'icon': '📉'
            })
        
        # อุปสรรค 4: ปัญหาในวิชาง่าย
        if struggled_easy >= 2:
            obstacles.append({
                'type': 'warning',
                'title': 'ประสบปัญหาในวิชาที่ถือว่าง่าย',
                'description': f'มีปัญหาในวิชาที่นักศึกษาส่วนใหญ่ผ่านได้ง่าย {struggled_easy} วิชา',
                'severity': 6,
                'time_to_fix': 'ต้องปรับวิธีการเรียน',
                'icon': '🤔'
            })
        
        # อุปสรรค 5: Risk Score สูง
        if risk_score > 2.0:
            obstacles.append({
                'type': 'critical',
                'title': 'คะแนนความเสี่ยงสูงมาก',
                'description': 'มีปัจจัยเสี่ยงหลายอย่างที่ส่งผลต่อการจบการศึกษา',
                'severity': 8,
                'time_to_fix': 'ต้องปรับปรุงหลายด้าน',
                'icon': '🚨'
            })
        
        # เรียงตาม severity
        obstacles.sort(key=lambda x: x['severity'], reverse=True)
        
        return obstacles
    
    def _identify_strengths(self, features: Dict) -> List[Dict]:
        """ระบุจุดแข็งและโอกาส"""
        strengths = []
        
        gpa = features.get('GPAX_so_far', 0)
        killer_passed = features.get('Passed_Killer_Courses', 0)
        above_avg_rate = features.get('Above_Avg_Rate', 0)
        consistency = features.get('Consistency_Score', 0)
        gpa_trend = features.get('GPA_trend', 0)
        high_grade_rate = features.get('High_Grade_Rate', 0)
        
        # จุดแข็ง 1: GPA สูง
        if gpa >= 3.5:
            strengths.append({
                'title': 'GPA สูงเยี่ยม',
                'description': f'มี GPA {gpa:.2f} ซึ่งอยู่ในระดับเกียรตินิยม',
                'impact': 'สูงมาก',
                'icon': '🌟'
            })
        elif gpa >= 3.0:
            strengths.append({
                'title': 'GPA ดี',
                'description': f'มี GPA {gpa:.2f} ซึ่งอยู่ในเกณฑ์ดี',
                'impact': 'สูง',
                'icon': '⭐'
            })
        
        # จุดแข็ง 2: ผ่านวิชายาก
        if killer_passed >= 3:
            strengths.append({
                'title': 'ผ่านวิชายากได้',
                'description': f'สามารถผ่านวิชายาก {killer_passed} วิชา แสดงถึงความสามารถสูง',
                'impact': 'สูง',
                'icon': '💪'
            })
        
        # จุดแข็ง 3: ผลการเรียนดีกว่าค่าเฉลี่ย
        if above_avg_rate > 0.6:
            strengths.append({
                'title': 'เรียนดีกว่าค่าเฉลี่ย',
                'description': f'{above_avg_rate*100:.0f}% ของวิชาได้เกรดสูงกว่าค่าเฉลี่ย',
                'impact': 'ปานกลาง',
                'icon': '🎯'
            })
        
        # จุดแข็ง 4: ความสม่ำเสมอสูง
        if consistency > 0.7:
            strengths.append({
                'title': 'ผลการเรียนสม่ำเสมอ',
                'description': 'มีความมั่นคงในผลการเรียน แสดงถึงความมุ่งมั่น',
                'impact': 'ปานกลาง',
                'icon': '🎲'
            })
        
        # จุดแข็ง 5: แนวโน้มดีขึ้น
        if gpa_trend > 0.2:
            strengths.append({
                'title': 'ผลการเรียนดีขึ้นเรื่อยๆ',
                'description': 'มีการพัฒนาที่ดีขึ้นอย่างต่อเนื่อง',
                'impact': 'สูง',
                'icon': '📈'
            })
        
        # จุดแข็ง 6: ได้เกรดสูงบ่อย
        if high_grade_rate > 0.4:
            strengths.append({
                'title': 'ได้เกรดสูงบ่อย',
                'description': f'{high_grade_rate*100:.0f}% ของวิชาได้เกรด A',
                'impact': 'สูง',
                'icon': '🏆'
            })
        
        return strengths
    
    def _generate_recommendations(self, features: Dict, probability: float, 
                                 obstacles: List[Dict]) -> List[Dict]:
        """สร้างคำแนะนำเชิงรุก"""
        recommendations = []
        
        gpa = features.get('GPAX_so_far', 0)
        failed_count = features.get('Total_F_Count_so_far', 0)
        low_grade_rate = features.get('Low_Grade_Rate', 0)
        consistency = features.get('Consistency_Score', 0)
        gpa_trend = features.get('GPA_trend', 0)
        
        # === คำแนะนำเร่งด่วน (Critical) ===
        if gpa < 2.0:
            recommendations.append({
                'priority': 'critical',
                'category': 'GPA',
                'title': 'ปรับปรุง GPA ทันที',
                'description': 'GPA ต่ำกว่าเกณฑ์ขั้นต่ำ จำเป็นต้องปรับปรุงอย่างเร่งด่วน',
                'actions': [
                    'เลือกเรียนวิชาที่มีโอกาสได้เกรดสูง (เช่น วิชาเลือกที่ถนัด)',
                    'ลดจำนวนวิชาในแต่ละเทอมเพื่อให้มีเวลาเรียนมากขึ้น',
                    'หาติวเตอร์หรือเพื่อนช่วยสอนในวิชาที่อ่อน',
                    'พบอาจารย์ที่ปรึกษาเพื่อขอคำแนะนำเพิ่มเติม',
                    'ใช้เวลาเรียนอย่างน้อย 3-4 ชม./วัน',
                    'เข้าร่วมกิจกรรมเสริมสร้างทักษะการเรียนรู้'
                ],
                'expected_result': f'ต้องการ GPA เฉลี่ย 3.5-4.0 ใน {math.ceil((2.0-gpa)*2)} เทอมถัดไป',
                'timeline': f'{math.ceil((2.0-gpa)*2)} เทอม',
                'icon': '🚨'
            })
        
        if failed_count >= 3:
            recommendations.append({
                'priority': 'critical',
                'category': 'วิชาตก',
                'title': 'เรียนซ้ำวิชาที่ไม่ผ่าน',
                'description': f'มีวิชาที่ไม่ผ่าน {failed_count} วิชา ต้องเรียนซ้ำทันที',
                'actions': [
                    'จัดลำดับความสำคัญของวิชาที่ต้องเรียนซ้ำ (วิชาพื้นฐานก่อน)',
                    'ลงทะเบียนเรียนซ้ำ 2-3 วิชาต่อเทอม (อย่าเรียนพร้อมกันมากเกินไป)',
                    'หาสาเหตุที่ไม่ผ่านในแต่ละวิชาและแก้ไข',
                    'ใช้ทรัพยากรเพิ่มเติม: ห้องสมุด, online courses, YouTube',
                    'ตั้งเป้าหมายว่าจะได้เกรดอะไรในการเรียนซ้ำ (อย่างน้อย C+)',
                    'พูดคุยกับอาจารย์ผู้สอนเพื่อขอคำแนะนำ'
                ],
                'expected_result': f'ใช้เวลา {math.ceil(failed_count/2)} เทอมในการเรียนซ้ำ',
                'timeline': f'{math.ceil(failed_count/2)}-{math.ceil(failed_count/1.5)} เทอม',
                'icon': '🔄'
            })
        
        # === คำแนะนำสำคัญ (High Priority) ===
        if low_grade_rate > 0.25:
            recommendations.append({
                'priority': 'high',
                'category': 'เกรดต่ำ',
                'title': 'ปรับปรุงวิชาที่ได้เกรดต่ำ',
                'description': 'มีวิชาที่ได้เกรด D หลายวิชา ควรพิจารณาเรียนซ้ำเพื่อเพิ่ม GPA',
                'actions': [
                    'เลือกวิชาที่มีหน่วยกิตสูงเพื่อเรียนซ้ำ (ผลต่อ GPA มากกว่า)',
                    'ใช้กลยุทธ์การเรียนที่แตกต่างจากครั้งก่อน',
                    'จดโน้ตอย่างละเอียดและทบทวนทุกวัน',
                    'ทำแบบฝึกหัดและข้อสอบเก่าอย่างสม่ำเสมอ',
                    'ตั้งเป้าหมายว่าจะได้เกรด B+ ขึ้นไป'
                ],
                'expected_result': 'GPA เพิ่มขึ้น 0.3-0.5 จุดต่อเทอม',
                'timeline': '2-3 เทอม',
                'icon': '📈'
            })
        
        if consistency < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'ความสม่ำเสมอ',
                'title': 'สร้างความสม่ำเสมอในการเรียน',
                'description': 'ผลการเรียนขึ้นลงมาก แสดงถึงปัญหาในการบริหารเวลาหรือแรงจูงใจ',
                'actions': [
                    'สร้างตารางเรียนและตารางทบทวนที่แน่นอน',
                    'เรียนและทบทวนวันละ 2-3 ชม. ทุกวัน',
                    'หลีกเลี่ยงการอ่านหนังสือติดต่อกันหลายชั่วโมงก่อนสอบ',
                    'พักผ่อนให้เพียงพอและออกกำลังกายสม่ำเสมอ',
                    'หากิจกรรมที่เพิ่มแรงจูงใจในการเรียน',
                    'หาเพื่อนกลุ่มเรียนเพื่อช่วยกันเรียน'
                ],
                'expected_result': 'ผลการเรียนมีความสม่ำเสมอและมั่นคงขึ้น',
                'timeline': '1-2 เทอม',
                'icon': '🎯'
            })
        
        if gpa_trend < -0.1:
            recommendations.append({
                'priority': 'high',
                'category': 'แนวโน้ม',
                'title': 'แก้ไขแนวโน้มที่แย่ลง',
                'description': 'ผลการเรียนมีแนวโน้มแย่ลงเรื่อยๆ ต้องหาสาเหตุและแก้ไขทันที',
                'actions': [
                    'วิเคราะห์ว่าเกิดอะไรขึ้นในช่วง 1-2 เทอมที่ผ่านมา',
                    'หาปัจจัยที่ทำให้ผลการเรียนแย่ลง (เช่น กิจกรรมมากเกินไป, ปัญหาส่วนตัว)',
                    'ปรับลดกิจกรรมนอกหลักสูตรชั่วคราว',
                    'พูดคุยกับอาจารย์ที่ปรึกษาหรือนักจิตวิทยา',
                    'ตั้งเป้าหมายใหม่และสร้างแรงจูงใจ',
                    'พิจารณา Gap Semester หากจำเป็น'
                ],
                'expected_result': 'กลับมามีแนวโน้มที่ดีขึ้นภายใน 1 เทอม',
                'timeline': '1-2 เทอม',
                'icon': '🔄'
            })
        
        # === คำแนะนำทั่วไป (Medium Priority) ===
        recommendations.append({
            'priority': 'medium',
            'category': 'กลยุทธ์การเรียน',
            'title': 'เลือกวิชาอย่างชาญฉลาด',
            'description': 'วางแผนการลงทะเบียนเรียนให้เหมาะสมกับความสามารถ',
            'actions': [
                'เลือกวิชาที่มีความยากแตกต่างกันในแต่ละเทอม (อย่าลงแต่วิชายาก)',
                'ลงวิชาเลือกที่ตนถนัดหรือสนใจเพื่อเพิ่ม GPA',
                'หลีกเลี่ยงการลงวิชามากเกินไปในเทอมเดียว (ไม่เกิน 21 หน่วยกิต)',
                'ลงวิชาที่มี section ที่เหมาะกับตัวเอง (เช่น อาจารย์ที่สอนดี)',
                'เช็ค Course DNA / รีวิววิชาก่อนลงทะเบียน'
            ],
            'expected_result': 'เลือกวิชาที่เหมาะสมและมีโอกาสได้เกรดดี',
            'timeline': 'ทุกเทอม',
            'icon': '🎓'
        })
        
        recommendations.append({
            'priority': 'medium',
            'category': 'ทักษะการเรียน',
            'title': 'พัฒนาทักษะการเรียนรู้',
            'description': 'เพิ่มประสิทธิภาพในการเรียนรู้และจดจำ',
            'actions': [
                'ใช้เทคนิค Pomodoro (เรียน 25 นาที พัก 5 นาที)',
                'สร้าง Mind Map สำหรับเนื้อหาที่ซับซ้อน',
                'ใช้ Active Recall แทนการอ่านซ้ำๆ',
                'สอนเพื่อนเพื่อทบทวนความเข้าใจ',
                'ทำข้อสอบเก่าเพื่อเตรียมสอบ',
                'หาแหล่งเรียนรู้เพิ่มเติม: MOOC, YouTube, Coursera'
            ],
            'expected_result': 'เรียนรู้ได้เร็วขึ้นและจดจำได้ดีขึ้น',
            'timeline': 'ต่อเนื่อง',
            'icon': '🧠'
        })
        
        # เรียงตามความสำคัญ
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def _suggest_graduation_path(self, features: Dict, probability: float, 
                                will_graduate: bool) -> Dict:
        """แนะนำเส้นทางสู่การจบการศึกษา"""
        
        gpa = features.get('GPAX_so_far', 0)
        failed_count = features.get('Total_F_Count_so_far', 0)
        total_credits = features.get('Total_Credits_so_far', 0)
        
        # คำนวณหน่วยกิตที่เหลือ (สมมติว่าต้อง 132 หน่วยกิต)
        required_credits = 132
        remaining_credits = max(0, required_credits - total_credits)
        
        # คำนวณเทอมที่เหลือ (สมมติว่าเรียนได้ 18 หน่วยกิตต่อเทอม)
        remaining_terms = math.ceil(remaining_credits / 18)
        
        # คำนวณเทอมที่ต้องใช้เพิ่มสำหรับเรียนซ้ำ
        retake_terms = math.ceil(failed_count / 3)  # เรียนซ้ำ 3 วิชาต่อเทอม
        
        # รวมเทอมทั้งหมด
        total_remaining_terms = remaining_terms + retake_terms
        
        # คำนวณ GPA เป้าหมายที่ต้องการ
        if gpa < 2.0:
            target_gpa = 2.0
            gpa_to_improve = target_gpa - gpa
        else:
            target_gpa = max(gpa, 3.0)  # พยายามให้ได้ 3.0 ขึ้นไป
            gpa_to_improve = max(0, target_gpa - gpa)
        
        # คำนวณเกรดที่ต้องการในเทอมถัดไป
        if gpa_to_improve > 0:
            needed_term_gpa = min(4.0, gpa + gpa_to_improve * 1.5)
        else:
            needed_term_gpa = 3.5  # เป้าหมายที่ดี
        
        return {
            'summary': {
                'will_graduate': will_graduate,
                'probability': round(probability * 100, 1),
                'remaining_credits': remaining_credits,
                'remaining_terms': remaining_terms,
                'retake_terms': retake_terms,
                'total_remaining_terms': total_remaining_terms,
                'estimated_graduation': f'{total_remaining_terms} เทอม ({total_remaining_terms/2:.1f} ปี)',
                'current_gpa': round(gpa, 2),
                'target_gpa': round(target_gpa, 2),
                'needed_term_gpa': round(needed_term_gpa, 2)
            },
            'milestones': self._create_milestones(
                total_remaining_terms, failed_count, gpa, target_gpa
            ),
            'semester_plan': self._create_semester_plan(
                total_remaining_terms, failed_count, remaining_credits, needed_term_gpa
            )
        }
    
    def _create_milestones(self, remaining_terms: int, failed_count: int, 
                          current_gpa: float, target_gpa: float) -> List[Dict]:
        """สร้าง Milestones สำหรับการจบการศึกษา"""
        milestones = []
        
        # Milestone 1: เรียนซ้ำวิชาที่ตก
        if failed_count > 0:
            retake_terms = math.ceil(failed_count / 3)
            milestones.append({
                'term': f'1-{retake_terms}',
                'title': 'เรียนซ้ำวิชาที่ไม่ผ่าน',
                'description': f'เรียนซ้ำ {failed_count} วิชาที่ได้เกรด F',
                'target': f'ได้เกรดอย่างน้อย C+ ในทุกวิชา',
                'status': 'pending'
            })
        
        # Milestone 2: ปรับปรุง GPA
        if current_gpa < target_gpa:
            gpa_improvement_terms = math.ceil((target_gpa - current_gpa) * 2)
            start_term = retake_terms + 1 if failed_count > 0 else 1
            end_term = start_term + gpa_improvement_terms - 1
            milestones.append({
                'term': f'{start_term}-{end_term}',
                'title': 'ปรับปรุง GPA',
                'description': f'เพิ่ม GPA จาก {current_gpa:.2f} เป็น {target_gpa:.2f}',
                'target': f'GPA เฉลี่ย 3.5-4.0 ต่อเทอม',
                'status': 'pending'
            })
        
        # Milestone 3: เรียนครบหน่วยกิต
        milestones.append({
            'term': f'{remaining_terms-1}-{remaining_terms}',
            'title': 'เรียนครบหน่วยกิต',
            'description': 'เรียนวิชาที่เหลือให้ครบตามหลักสูตร',
            'target': 'ครบ 132 หน่วยกิต',
            'status': 'pending'
        })
        
        # Milestone 4: จบการศึกษา
        milestones.append({
            'term': remaining_terms,
            'title': 'จบการศึกษา',
            'description': 'ยื่นขอสำเร็จการศึกษา',
            'target': f'GPA ≥ {target_gpa:.2f}',
            'status': 'pending'
        })
        
        return milestones
    
    def _create_semester_plan(self, remaining_terms: int, failed_count: int, 
                             remaining_credits: int, needed_gpa: float) -> List[Dict]:
        """สร้างแผนการเรียนแต่ละเทอม"""
        semester_plan = []
        
        credits_per_term = math.ceil(remaining_credits / max(1, remaining_terms))
        
        for term in range(1, remaining_terms + 1):
            if term <= math.ceil(failed_count / 3):
                # เทอมเรียนซ้ำ
                retake_courses = min(3, failed_count - (term-1)*3)
                semester_plan.append({
                    'term': term,
                    'focus': 'เรียนซ้ำวิชาที่ไม่ผ่าน',
                    'courses': f'{retake_courses} วิชาซ้ำ + {credits_per_term//3 - retake_courses} วิชาใหม่',
                    'target_credits': min(18, credits_per_term),
                    'target_gpa': round(needed_gpa, 2),
                    'strategies': [
                        'โฟกัสที่วิชาซ้ำเป็นหลัก',
                        'ใช้เวลาเรียนอย่างน้อย 4 ชม./วัน',
                        'หาติวเตอร์หรือกลุ่มเรียนสำหรับวิชายาก'
                    ]
                })
            else:
                # เทอมปกติ
                semester_plan.append({
                    'term': term,
                    'focus': 'เรียนวิชาใหม่และปรับปรุง GPA',
                    'courses': f'{credits_per_term//3} วิชา',
                    'target_credits': min(21, credits_per_term),
                    'target_gpa': round(needed_gpa, 2),
                    'strategies': [
                        'เลือกวิชาที่มีความยากแตกต่างกัน',
                        'ลงวิชาเลือกที่ถนัดเพื่อเพิ่ม GPA',
                        'ทบทวนอย่างสม่ำเสมอ',
                        'เข้าร่วมกิจกรรมเสริมทักษะ'
                    ]
                })
        
        return semester_plan
    
    def _project_future_scenarios(self, features: Dict, probability: float) -> Dict:
        """ทำนายสถานการณ์ในอนาคต"""
        
        gpa = features.get('GPAX_so_far', 0)
        gpa_trend = features.get('GPA_trend', 0)
        
        # Scenario 1: Best Case (ทำได้ดีที่สุด)
        best_case_gpa = min(4.0, gpa + 0.5 + abs(gpa_trend) * 0.3)
        best_case_prob = min(0.95, probability + 0.2)
        
        # Scenario 2: Expected Case (คาดว่าจะเป็น)
        expected_case_gpa = gpa + gpa_trend * 0.5
        expected_case_prob = probability
        
        # Scenario 3: Worst Case (แย่ที่สุด)
        worst_case_gpa = max(0.5, gpa - 0.3)
        worst_case_prob = max(0.05, probability - 0.2)
        
        return {
            'best_case': {
                'scenario': 'กรณีที่ดีที่สุด',
                'description': 'หากทำตามคำแนะนำอย่างเต็มที่และเรียนได้ดีทุกวิชา',
                'predicted_gpa': round(best_case_gpa, 2),
                'graduation_probability': round(best_case_prob * 100, 1),
                'conditions': [
                    'ได้เกรด A หรือ B+ ในทุกวิชา',
                    'ไม่มีวิชาตกเพิ่ม',
                    'เรียนซ้ำได้เกรดดี',
                    'มีความมุ่งมั่นสูง'
                ]
            },
            'expected_case': {
                'scenario': 'กรณีที่คาดว่าจะเป็น',
                'description': 'หากเรียนตามปกติและปรับปรุงบางส่วน',
                'predicted_gpa': round(expected_case_gpa, 2),
                'graduation_probability': round(expected_case_prob * 100, 1),
                'conditions': [
                    'ได้เกรดปานกลาง-ดี',
                    'ปรับปรุงตามคำแนะนำบางส่วน',
                    'มีความพยายามปานกลาง'
                ]
            },
            'worst_case': {
                'scenario': 'กรณีที่แย่ที่สุด',
                'description': 'หากไม่ปรับปรุงและมีปัญหาต่อเนื่อง',
                'predicted_gpa': round(worst_case_gpa, 2),
                'graduation_probability': round(worst_case_prob * 100, 1),
                'conditions': [
                    'ผลการเรียนไม่ดีขึ้น',
                    'มีวิชาตกเพิ่ม',
                    'ไม่ปรับปรุงตามคำแนะนำ'
                ]
            }
        }
    
    def _create_detailed_breakdown(self, features: Dict) -> Dict:
        """สร้างการวิเคราะห์โดยละเอียด"""
        return {
            'academic_performance': {
                'GPA': round(features.get('GPAX_so_far', 0), 2),
                'Total_Credits': features.get('Total_Credits_so_far', 0),
                'Total_Courses': features.get('Total_Courses_so_far', 0),
                'Failed_Courses': features.get('Total_F_Count_so_far', 0),
                'Pass_Rate': round(features.get('Pass_Rate', 0) * 100, 1),
                'High_Grade_Rate': round(features.get('High_Grade_Rate', 0) * 100, 1)
            },
            'performance_trends': {
                'GPA_Trend': round(features.get('GPA_trend', 0), 3),
                'Recent_GPA': round(features.get('GPA_last_window', 0), 2),
                'Improvement_Potential': round(features.get('Improvement_potential', 0), 2),
                'Consistency_Score': round(features.get('Consistency_Score', 0), 2)
            },
            'course_analysis': {
                'Killer_Courses_Passed': features.get('Passed_Killer_Courses', 0),
                'Avg_vs_Course_Avg': round(features.get('Avg_vs_Course_Avg', 0), 2),
                'Above_Avg_Rate': round(features.get('Above_Avg_Rate', 0) * 100, 1),
                'Better_Than_Avg_Count': features.get('Better_Than_Avg_Count', 0),
                'Worse_Than_Avg_Count': features.get('Worse_Than_Avg_Count', 0)
            },
            'risk_assessment': {
                'Risk_Score': round(features.get('Risk_Score', 0), 2),
                'At_Risk_Flag': features.get('At_Risk_Flag', 0),
                'Low_Grade_Rate': round(features.get('Low_Grade_Rate', 0) * 100, 1),
                'Struggled_Easy_Courses': features.get('Struggled_Easy_Courses', 0)
            }
        }
    
    # === Helper Methods for Impact Calculation ===
    
    def _calculate_gpa_impact(self, gpa: float) -> Dict:
        """คำนวณผลกระทบของ GPA"""
        if gpa >= 3.5:
            return {'level': 'positive_high', 'score': 0.9, 'description': 'GPA สูงมาก เป็นปัจจัยบวกที่สำคัญ'}
        elif gpa >= 3.0:
            return {'level': 'positive', 'score': 0.6, 'description': 'GPA ดี ส่งผลบวกต่อการจบการศึกษา'}
        elif gpa >= 2.5:
            return {'level': 'neutral', 'score': 0.3, 'description': 'GPA ปานกลาง ควรปรับปรุง'}
        elif gpa >= 2.0:
            return {'level': 'negative', 'score': -0.5, 'description': 'GPA ใกล้เกณฑ์ขั้นต่ำ มีความเสี่ยง'}
        else:
            return {'level': 'negative_high', 'score': -0.9, 'description': 'GPA ต่ำกว่าเกณฑ์ เป็นปัจจัยลบที่สำคัญ'}
    
    def _calculate_fail_impact(self, failed_count: int) -> Dict:
        """คำนวณผลกระทบของวิชาที่ตก"""
        if failed_count == 0:
            return {'level': 'positive', 'score': 0.7, 'description': 'ไม่มีวิชาตก เป็นสัญญาณที่ดี'}
        elif failed_count <= 2:
            return {'level': 'neutral', 'score': -0.3, 'description': 'มีวิชาตกเล็กน้อย สามารถแก้ไขได้'}
        elif failed_count <= 5:
            return {'level': 'negative', 'score': -0.6, 'description': 'มีวิชาตกหลายวิชา ส่งผลลบอย่างมาก'}
        else:
            return {'level': 'negative_high', 'score': -0.9, 'description': 'มีวิชาตกมาก เป็นปัจจัยลบร้ายแรง'}
    
    def _calculate_killer_course_impact(self, killer_passed: int) -> Dict:
        """คำนวณผลกระทบของการผ่านวิชายาก"""
        if killer_passed >= 5:
            return {'level': 'positive_high', 'score': 0.8, 'description': 'ผ่านวิชายากหลายวิชา แสดงความสามารถสูง'}
        elif killer_passed >= 3:
            return {'level': 'positive', 'score': 0.5, 'description': 'ผ่านวิชายากได้ดี'}
        elif killer_passed >= 1:
            return {'level': 'neutral', 'score': 0.2, 'description': 'ผ่านวิชายากบางวิชา'}
        else:
            return {'level': 'negative', 'score': -0.4, 'description': 'ยังไม่ผ่านวิชายากเลย อาจมีปัญหา'}
    
    def _calculate_trend_impact(self, gpa_trend: float) -> Dict:
        """คำนวณผลกระทบของแนวโน้ม"""
        if gpa_trend > 0.3:
            return {'level': 'positive_high', 'score': 0.7, 'description': 'ผลการเรียนดีขึ้นอย่างชัดเจน'}
        elif gpa_trend > 0.1:
            return {'level': 'positive', 'score': 0.4, 'description': 'ผลการเรียนดีขึ้น'}
        elif gpa_trend > -0.1:
            return {'level': 'neutral', 'score': 0, 'description': 'ผลการเรียนคงที่'}
        elif gpa_trend > -0.3:
            return {'level': 'negative', 'score': -0.4, 'description': 'ผลการเรียนแย่ลง'}
        else:
            return {'level': 'negative_high', 'score': -0.7, 'description': 'ผลการเรียนแย่ลงอย่างมาก'}
    
    def _calculate_vs_avg_impact(self, vs_avg: float) -> Dict:
        """คำนวณผลกระทบของการเทียบกับค่าเฉลี่ย"""
        if vs_avg > 1.0:
            return {'level': 'positive_high', 'score': 0.8, 'description': 'เรียนดีกว่าค่าเฉลี่ยมาก'}
        elif vs_avg > 0.5:
            return {'level': 'positive', 'score': 0.5, 'description': 'เรียนดีกว่าค่าเฉลี่ย'}
        elif vs_avg > -0.5:
            return {'level': 'neutral', 'score': 0, 'description': 'ใกล้เคียงค่าเฉลี่ย'}
        elif vs_avg > -1.0:
            return {'level': 'negative', 'score': -0.5, 'description': 'ต่ำกว่าค่าเฉลี่ย'}
        else:
            return {'level': 'negative_high', 'score': -0.8, 'description': 'ต่ำกว่าค่าเฉลี่ยมาก'}
    
    def _calculate_consistency_impact(self, consistency: float) -> Dict:
        """คำนวณผลกระทบของความสม่ำเสมอ"""
        if consistency > 0.8:
            return {'level': 'positive', 'score': 0.6, 'description': 'ผลการเรียนสม่ำเสมอสูง'}
        elif consistency > 0.6:
            return {'level': 'neutral', 'score': 0.3, 'description': 'ผลการเรียนค่อนข้างสม่ำเสมอ'}
        elif consistency > 0.4:
            return {'level': 'neutral', 'score': 0, 'description': 'ผลการเรียนขึ้นลงปานกลาง'}
        else:
            return {'level': 'negative', 'score': -0.4, 'description': 'ผลการเรียนไม่สม่ำเสมอ'}
    
    def _get_default_explanation(self) -> Dict:
        """คืนค่าเริ่มต้นเมื่อเกิดข้อผิดพลาด"""
        return {
            'summary': {
                'will_graduate': True,
                'probability': 50.0,
                'confidence': 50.0,
                'risk_level': 'moderate',
                'status': 'ไม่สามารถวิเคราะห์ได้'
            },
            'key_factors': [],
            'reasons': ['ไม่สามารถวิเคราะห์ได้ในขณะนี้'],
            'obstacles': [],
            'strengths': [],
            'recommendations': [],
            'graduation_path': {},
            'future_projections': {},
            'detailed_breakdown': {}
        }


# สำหรับ import ภายนอก
import math
