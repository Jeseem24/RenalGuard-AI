"""
RenalGuard AI - LLM Chat Assistant Module
Powered by Google Gemini API for conversational AI
"""

import os
import os
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
from typing import Dict, List, Optional
import json


class GeminiChatAssistant:
    """
    LLM-powered conversational assistant using Google Gemini API
    Helps doctors and patients understand CKD results
    """
    
    SYSTEM_PROMPT = """You are RenalGuard AI Assistant, a specialized medical AI helper for kidney health screening.

Your role:
- Explain CKD (Chronic Kidney Disease) screening results in simple, understandable language
- Provide general health guidance based on KDIGO clinical guidelines
- Help interpret blood and urine test results
- Suggest lifestyle modifications for kidney health
- Recommend when to see a doctor

Important guidelines:
1. Always clarify that you are an AI assistant, not a doctor
2. Recommend consulting a nephrologist for medical decisions
3. Use simple language that patients can understand
4. Be empathetic and supportive
5. Provide evidence-based information
6. Do not make definitive diagnoses
7. Focus on education and prevention

When explaining results:
- Reference the specific test values provided
- Explain what normal ranges are
- Describe what elevated or low values might indicate
- Suggest next steps and lifestyle changes
- Emphasize the importance of follow-up testing

CKD Stages reference:
- Stage 1: eGFR ≥90 (Kidney damage with normal function)
- Stage 2: eGFR 60-89 (Mild decrease)
- Stage 3: eGFR 30-59 (Moderate decrease)
- Stage 4: eGFR 15-29 (Severe decrease)
- Stage 5: eGFR <15 (Kidney failure)

Always include a medical disclaimer in your responses."""

    def __init__(self, api_key: str = None):
        """
        Initialize Gemini Chat Assistant
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.chat = None
        self.context = {}
        
        if self.api_key and HAS_GENAI:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat = self.model.start_chat(history=[])
            print("Gemini model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.model = None
    
    def set_context(self, patient_data: Dict, prediction_result: Dict, explanation: Dict):
        """
        Set the context for the chat based on patient data and prediction
        
        Args:
            patient_data: Dictionary of patient test values
            prediction_result: Dictionary with prediction results
            explanation: Dictionary with SHAP explanation
        """
        self.context = {
            'patient_data': patient_data,
            'prediction': prediction_result,
            'explanation': explanation
        }
        
        # Format context string
        context_str = f"""
Current Patient Context:
- CKD Risk Level: {prediction_result.get('risk_level', 'Unknown')}
- Risk Score: {prediction_result.get('risk_score', 0)}/100
- Predicted CKD Stage: {prediction_result.get('stage', 'Unknown')}
- eGFR: {prediction_result.get('egfr', 'Unknown')} mL/min/1.73m²

Key Test Values:
"""
        # Add key test values
        key_tests = ['sc', 'bu', 'hemo', 'bp', 'bgr', 'age']
        for test in key_tests:
            if test in patient_data:
                context_str += f"- {test.upper()}: {patient_data[test]}\n"
        
        # Add top risk factors from explanation
        if explanation and 'top_risk_factors' in explanation:
            context_str += "\nTop Contributing Factors:\n"
            for factor in explanation['top_risk_factors'][:3]:
                context_str += f"- {factor['feature']}: {factor['feature_value']} ({factor['contribution_pct']:.1f}% contribution)\n"
        
        # Reinitialize chat with context
        if self.model:
            self.chat = self.model.start_chat(history=[
                {
                    "role": "user",
                    "parts": [self.SYSTEM_PROMPT + "\n\n" + context_str]
                },
                {
                    "role": "model",
                    "parts": ["I understand. I'm ready to help explain the kidney health screening results. What would you like to know?"]
                }
            ])
    
    def chat(self, message: str) -> str:
        """
        Send a message and get response
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        if not self.model:
            return self._get_fallback_response(message)
        
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            print(f"Error getting response: {e}")
            return self._get_fallback_response(message)
    
    def send_message(self, message: str) -> str:
        """Alias for chat method"""
        return self.chat(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Provide fallback response when API is not available"""
        message_lower = message.lower()
        
        if 'creatinine' in message_lower:
            return """Serum creatinine is a waste product from muscle metabolism. 
Normal range is 0.6-1.2 mg/dL.

Elevated creatinine levels may indicate reduced kidney function, as kidneys filter creatinine from blood.

Please consult a nephrologist for proper evaluation of your creatinine levels.

Note: This is general information. Please consult a healthcare professional for medical advice."""
        
        elif 'stage' in message_lower or 'ckd stage' in message_lower:
            return """CKD stages are based on eGFR (estimated Glomerular Filtration Rate):

Stage 1: eGFR ≥90 - Normal kidney function with signs of damage
Stage 2: eGFR 60-89 - Mild decrease in kidney function
Stage 3: eGFR 30-59 - Moderate decrease
Stage 4: eGFR 15-29 - Severe decrease
Stage 5: eGFR <15 - Kidney failure

Each stage requires different management approaches. Please consult a nephrologist for personalized guidance.

Note: This is general information. Please consult a healthcare professional."""
        
        elif 'diet' in message_lower or 'food' in message_lower or 'eat' in message_lower:
            return """General dietary recommendations for kidney health:

1. Reduce sodium intake (<2g/day)
2. Control protein intake based on your stage
3. Limit processed foods
4. Stay hydrated (unless otherwise advised)
5. Limit potassium and phosphorus in advanced stages

Dietary needs vary by individual. Please consult a renal dietitian for personalized meal planning.

Note: This is general information. Please consult a healthcare professional."""
        
        elif 'next' in message_lower or 'do' in message_lower:
            return """Recommended next steps based on your screening:

1. Consult a nephrologist for clinical evaluation
2. Get confirmatory tests done
3. Monitor blood pressure regularly
4. Control blood sugar if diabetic
5. Follow up as recommended

Note: This is general guidance. Please consult a healthcare professional for personalized recommendations."""
        
        else:
            return """Thank you for your question. I'm here to help explain your kidney health screening results.

You can ask me about:
- Your test results and what they mean
- CKD stages and their implications
- Dietary recommendations
- Next steps to take

Please consult a nephrologist for medical advice and treatment.

Note: This AI assistant provides general information only and is not a substitute for professional medical advice."""
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for the user"""
        return [
            "What do my results mean?",
            "What should I do next?",
            "Explain my creatinine level",
            "What dietary changes should I make?",
            "When should I get tested again?",
            "What is CKD Stage 3?",
            "How can I improve my kidney health?",
            "What are the risk factors for CKD?"
        ]
    
    def reset_chat(self):
        """Reset the chat history"""
        if self.model:
            self.chat = self.model.start_chat(history=[])
            self.context = {}


class MockChatAssistant:
    """Mock chat assistant for demo when API key is not available"""
    
    def __init__(self):
        self.context = {}
    
    def set_context(self, patient_data: Dict, prediction_result: Dict, explanation: Dict):
        self.context = {
            'patient_data': patient_data,
            'prediction': prediction_result,
            'explanation': explanation
        }
    
    def send_message(self, message: str) -> str:
        # Return structured responses based on keywords
        message_lower = message.lower()
        
        if 'result' in message_lower or 'mean' in message_lower:
            risk = self.context.get('prediction', {}).get('risk_level', 'Unknown')
            return f"""Based on your screening results:

**Risk Level**: {risk}

Your kidney health screening indicates your current risk level. The AI analyzed multiple factors including serum creatinine, blood urea, hemoglobin, and other parameters to arrive at this assessment.

**Key Points:**
- Early detection is crucial for kidney health
- Follow up with a healthcare provider
- Lifestyle modifications can help slow progression

⚠️ **Disclaimer**: This is an AI-assisted screening. Please consult a nephrologist for proper medical evaluation."""
        
        elif 'diet' in message_lower or 'food' in message_lower:
            return """**Dietary Recommendations for Kidney Health:**

1. **Reduce Sodium**: Limit to less than 2g/day
   - Avoid processed foods, canned goods
   - Use herbs instead of salt for flavor

2. **Protein Intake**: Moderate protein consumption
   - Consult your doctor for specific limits
   - Choose high-quality protein sources

3. **Hydration**: Stay adequately hydrated
   - Water is best
   - Avoid sugary drinks

4. **Limit**: Potassium and phosphorus in advanced stages
   - Bananas, oranges, potatoes (potassium)
   - Dairy, nuts, colas (phosphorus)

⚠️ **Note**: Dietary needs vary by CKD stage. Please consult a renal dietitian."""
        
        elif 'next' in message_lower or 'do' in message_lower:
            return """**Recommended Next Steps:**

1. **Schedule a doctor visit** - Consult a nephrologist
2. **Confirmatory testing** - Get additional tests as recommended
3. **Monitor regularly** - Follow the suggested timeline
4. **Medication review** - Some medications affect kidneys
5. **Lifestyle changes** - Diet, exercise, quit smoking

**Urgency Level:** Based on your risk assessment, please seek medical consultation within the recommended timeframe.

⚠️ **Disclaimer**: This is general guidance. Please consult a healthcare professional."""
        
        else:
            return """Thank you for your question! I'm here to help you understand your kidney health screening results.

**You can ask me about:**
- Your test results and what they mean
- CKD stages and their implications
- Dietary recommendations
- Next steps to take
- Risk factors and prevention

**Important Note:**
This AI assistant provides general health information only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider.

⚠️ **Please consult a nephrologist for proper medical evaluation.**"""
    
    def get_suggested_questions(self) -> List[str]:
        return [
            "What do my results mean?",
            "What should I do next?",
            "What dietary changes should I make?",
            "How can I improve my kidney health?"
        ]
    
    def reset_chat(self):
        self.context = {}


def get_chat_assistant(api_key: str = None):
    """
    Factory function to get appropriate chat assistant
    
    Args:
        api_key: Optional Gemini API key
        
    Returns:
        GeminiChatAssistant if API key available, else MockChatAssistant
    """
    if HAS_GENAI and (api_key or os.getenv('GEMINI_API_KEY')):
        return GeminiChatAssistant(api_key)
    else:
        print("No Gemini API key found or google-generativeai module missing. Using mock assistant.")
        return MockChatAssistant()


if __name__ == '__main__':
    # Test the mock assistant
    assistant = MockChatAssistant()
    assistant.set_context(
        patient_data={'sc': 2.1, 'bu': 55, 'hemo': 10.5},
        prediction_result={'risk_level': 'HIGH', 'risk_score': 78, 'stage': 3},
        explanation={'top_risk_factors': [{'feature': 'sc', 'feature_value': 2.1, 'contribution_pct': 35}]}
    )
    
    print("Testing Chat Assistant:")
    print("-" * 50)
    print(assistant.send_message("What do my results mean?"))
