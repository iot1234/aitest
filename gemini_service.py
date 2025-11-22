import os
import logging
import json
import time
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
        self.is_configured = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.is_configured = True
                logger.info(f"✅ Gemini Service configured with model: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to configure Gemini Service: {e}")
        else:
            logger.warning("⚠️ GEMINI_API_KEY not found. Gemini features will be disabled.")

    def call_gemini_structured(self, task_name: str, payload: Dict[str, Any], schema_key: str = 'insights') -> Dict[str, Any]:
        """
        Calls Gemini API with a structured prompt and expects a JSON response.
        """
        if not self.is_configured:
            return {"error": "Gemini API is not configured"}

        try:
            prompt = self._construct_prompt(task_name, payload)
            
            # Safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Generate content
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            # Parse response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini response as JSON: {response.text}")
                # Fallback: try to extract JSON from text
                import re
                match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                return {"error": "Invalid JSON response from Gemini"}

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {"error": str(e)}

    def _construct_prompt(self, task_name: str, payload: Dict[str, Any]) -> str:
        """Constructs the prompt based on the task."""
        if task_name == 'training_dataset_analysis':
            return f"""
            Analyze the following training dataset summary for a student graduation prediction system.
            
            Dataset Summary:
            {json.dumps(payload, indent=2)}
            
            Your goal is to: {payload.get('analysis_goal', 'Analyze data quality and balance')}
            
            Return a JSON object with the following structure:
            {{
                "analysis_markdown": "Detailed markdown analysis...",
                "risk_level": "low|medium|high",
                "key_metrics": [
                    {{"label": "Metric Name", "value": "Value", "trend": "up|down|stable"}}
                ],
                "recommendations": ["Rec 1", "Rec 2"]
            }}
            """
        elif task_name == 'student_prediction_analysis':
            return f"""
            Analyze the following student grade data and predict graduation likelihood.
            
            Student Data:
            {json.dumps(payload, indent=2)}
            
            Goal: {payload.get('analysis_goal', 'Predict graduation and provide advice')}
            
            Return a JSON object with the following structure:
            {{
                "analysis_markdown": "Detailed markdown analysis...",
                "risk_level": "very_low|low|moderate|high|very_high",
                "outcome_summary": {{
                    "status": "Likely to Graduate|At Risk",
                    "confidence": 0.0-1.0,
                    "description": "Short summary"
                }},
                "key_metrics": [
                    {{"label": "GPA", "value": "...", "trend": "..."}}
                ],
                "recommendations": ["Rec 1", "Rec 2"]
            }}
            """
        else:
            return f"Task: {task_name}\nPayload: {json.dumps(payload)}"

    def analyze_dataset(self, df_summary: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Wrapper for dataset analysis."""
        payload = {
            "summary": df_summary,
            "analysis_goal": goal
        }
        return self.call_gemini_structured('training_dataset_analysis', payload)

    def predict_student_outcome(self, student_data: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Wrapper for student prediction."""
        payload = {
            "student_data": student_data,
            "analysis_goal": goal
        }
        return self.call_gemini_structured('student_prediction_analysis', payload)
