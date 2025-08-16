import google.generativeai as genai
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GeminiAPI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = None
        self.vision_model = None
        self.is_available = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.vision_model = genai.GenerativeModel('gemini-2.0-flash-vision')
                self.is_available = True
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Gemini API: {e}")
                self.is_available = False
        else:
            logger.warning("Google API key not provided. AI features will be limited.")
            self.is_available = False

    def _get_fallback_response(self, task_type: str, language: str = 'English') -> Dict[str, Any]:
        """Provide fallback responses when Gemini API is not available."""
        fallback_responses = {
            'analysis': {
                'summary': f'Medical report analysis is not available without API key. Please set GOOGLE_API_KEY environment variable.',
                'key_findings': ['API key required for AI analysis'],
                'recommendations': ['Contact administrator to configure API key'],
                'risk_level': 'Unknown',
                'language': language
            },
            'simplification': {
                'simplified_text': 'Medical term simplification requires API key. Please set GOOGLE_API_KEY environment variable.',
                'explained_terms': {},
                'language': language
            },
            'recommendations': {
                'lifestyle_recommendations': ['API key required for personalized recommendations'],
                'dietary_suggestions': ['Contact administrator to configure API key'],
                'exercise_recommendations': ['API key required for AI recommendations'],
                'follow_up_actions': ['Set up API key for full functionality'],
                'language': language
            }
        }
        return fallback_responses.get(task_type, {'error': 'Unknown task type'})

    def analyze_medical_report(self, report_text: str, relevant_context: List[Dict[str, Any]], language: str = 'English') -> Dict[str, Any]:
        """Analyze medical report using Gemini API."""
        if not self.is_available:
            logger.warning("Gemini API not available - using fallback response")
            return self._get_fallback_response('analysis', language)
        
        try:
            # Prepare context for the prompt
            context_text = ""
            if relevant_context:
                context_text = "\n\nRelevant medical context:\n"
                for ctx in relevant_context[:3]:  # Limit to top 3 contexts
                    context_text += f"- {ctx.get('text', '')}\n"
            
            prompt = f"""
            You are a patient-friendly medical translator. Your job is to explain this medical report in simple, everyday language that ANYONE can understand.
            
            IMPORTANT: Write as if you're explaining to a friend who has NO medical background. Use simple words, avoid medical jargon, and focus on what this means for the patient's daily life.
            
            Medical Report:
            {report_text}
            
            {context_text}
            
            Please provide your explanation in the following EXACT format:
            
            **1. Summary of the Report:**
            [In 2-3 simple sentences, explain what this report is about and what it found. Use everyday language.]
            
            **2. What This Means for You:**
            * [Explain finding 1 in simple terms - what it is and why it matters to the patient]
            * [Explain finding 2 in simple terms - what it is and why it matters to the patient]
            * [Continue with other important findings in simple language]
            
            **3. Should You Be Worried?:**
            * [Explain the risk level in simple terms - what it means for the patient's health]
            * [Mention any urgent concerns or if this is normal]
            
            **4. What You Should Do Next:**
            * [Give 2-3 simple, actionable steps the patient can take]
            * [Mention if they need to see a doctor and how soon]
            * [Suggest any lifestyle changes if relevant]
            
            **5. Risk Level:**
            [State the risk level as Low, Medium, or High]
            
            REMEMBER: 
            - Use simple, everyday words
            - Explain WHY each finding matters to the patient
            - Focus on what the patient can DO about it
            - Avoid medical terms unless absolutely necessary
            - Write as if explaining to a 12-year-old
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response to extract structured information
            response_text = response.text
            
            # Extract summary (first section)
            summary = response_text.split('**2. What This Means for You:**')[0] if '**2. What This Means for You:**' in response_text else response_text[:500]
            summary = summary.replace('## Medical Report Analysis:', '').replace('**1. Summary of the Report:**', '').strip()
            
            # Extract what this means for the patient (patient-friendly findings)
            key_findings = []
            if '**2. What This Means for You:**' in response_text and '**3. Should You Be Worried?:**' in response_text:
                findings_section = response_text.split('**2. What This Means for You:**')[1].split('**3. Should You Be Worried?:**')[0]
                # Extract bullet points
                for line in findings_section.split('\n'):
                    if line.strip().startswith('*') or line.strip().startswith('-'):
                        finding = line.strip().lstrip('*').lstrip('-').strip()
                        if finding:
                            key_findings.append(finding)
            
            if not key_findings:
                key_findings = ['Analysis completed via AI']
            
            # Extract worry level (patient-friendly implications)
            health_implications = []
            if '**3. Should You Be Worried?:**' in response_text and '**4. What You Should Do Next:**' in response_text:
                implications_section = response_text.split('**3. Should You Be Worried?:**')[1].split('**4. What You Should Do Next:**')[0]
                # Extract bullet points
                for line in implications_section.split('\n'):
                    if line.strip().startswith('*') or line.strip().startswith('-'):
                        implication = line.strip().lstrip('*').lstrip('-').strip()
                        if implication:
                            health_implications.append(implication)
            
            # Extract actionable next steps
            recommendations = []
            if '**4. What You Should Do Next:**' in response_text and '**5. Risk Level:**' in response_text:
                rec_section = response_text.split('**4. What You Should Do Next:**')[1].split('**5. Risk Level:**')[0]
                # Extract bullet points
                for line in rec_section.split('\n'):
                    if line.strip().startswith('*') or line.strip().startswith('-'):
                        rec = line.strip().lstrip('*').lstrip('-').strip()
                        if rec:
                            recommendations.append(rec)
            
            if not recommendations:
                recommendations = ['Follow up with healthcare provider']
            
            # Extract risk level
            risk_level = 'Medium'  # Default
            if '**5. Risk Level:**' in response_text:
                risk_section = response_text.split('**5. Risk Level:**')[1]
                if 'Low' in risk_section:
                    risk_level = 'Low'
                elif 'High' in risk_section:
                    risk_level = 'High'
                elif 'Medium' in risk_section:
                    risk_level = 'Medium'
            
            analysis = {
                'summary': summary[:500] + "..." if len(summary) > 500 else summary,
                'key_findings': key_findings,
                'health_implications': health_implications,
                'recommendations': recommendations,
                'risk_level': risk_level,
                'language': language,
                'full_response': response_text
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing medical report: {e}")
            return self._get_fallback_response('analysis', language)

    def simplify_medical_terms(self, medical_text: str, relevant_context: List[Dict[str, Any]], language: str = 'English') -> Dict[str, Any]:
        """Simplify medical terms using Gemini API."""
        if not self.is_available:
            logger.warning("Gemini API not available - using fallback response")
            return self._get_fallback_response('simplification', language)
        
        try:
            prompt = f"""
            You are a patient-friendly medical translator. Your job is to explain medical terms in simple, everyday language that ANYONE can understand.
            
            IMPORTANT: Write as if you're explaining to a friend who has NO medical background. Use simple words, avoid medical jargon, and focus on what this means for the patient's daily life.
            
            Medical Text:
            {medical_text}
            
            Please provide your explanation in the following EXACT format:
            
            **1. Simple Explanation:**
            [Rewrite the medical text in simple, everyday language that a 12-year-old could understand]
            
            **2. Medical Terms Explained:**
            * [Term 1]: [Simple explanation of what this means]
            * [Term 2]: [Simple explanation of what this means]
            [Continue with any other medical terms that need explanation]
            
            REMEMBER: 
            - Use simple, everyday words
            - Explain WHY each term matters to the patient
            - Focus on what the patient needs to know
            - Avoid medical jargon unless absolutely necessary
            - Write as if explaining to someone with no medical knowledge
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response to extract structured information
            response_text = response.text
            
            # Extract simplified text (first section)
            simplified_text = response_text.split('**2. Medical Terms Explained:**')[0] if '**2. Medical Terms Explained:**' in response_text else response_text[:1000]
            simplified_text = simplified_text.replace('**1. Simple Explanation:**', '').strip()
            
            # Extract explained terms
            explained_terms = {}
            if '**2. Medical Terms Explained:**' in response_text:
                terms_section = response_text.split('**2. Medical Terms Explained:**')[1]
                # Extract terms and explanations
                lines = terms_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line.startswith('*'):
                        # Remove the bullet point and split by colon
                        clean_line = line.lstrip('*').strip()
                        if ':' in clean_line:
                            parts = clean_line.split(':', 1)
                            if len(parts) == 2:
                                term = parts[0].strip()
                                explanation = parts[1].strip()
                                explained_terms[term] = explanation
            
            if not explained_terms:
                explained_terms = {'medical_terms': 'Simplified via AI'}
            
            simplification = {
                'simplified_text': simplified_text[:1000] + "..." if len(simplified_text) > 1000 else simplified_text,
                'explained_terms': explained_terms,
                'language': language,
                'full_response': response_text
            }
            
            return simplification
            
        except Exception as e:
            logger.error(f"Error simplifying medical terms: {e}")
            return self._get_fallback_response('simplification', language)

    def generate_health_recommendations(self, medical_data: str, patient_info: Dict[str, Any], relevant_context: List[Dict[str, Any]], language: str = 'English') -> Dict[str, Any]:
        """Generate personalized health recommendations using Gemini API."""
        if not self.is_available:
            logger.warning("Gemini API not available - using fallback response")
            return self._get_fallback_response('recommendations', language)
        
        try:
            # Prepare patient info
            patient_details = f"Age: {patient_info.get('age', 'Unknown')}, Gender: {patient_info.get('gender', 'Unknown')}"
            
            prompt = f"""
            You are a patient-friendly health advisor. Your job is to give simple, practical advice that ANYONE can follow.
            
            IMPORTANT: Write as if you're giving advice to a friend who has NO medical background. Use simple words, avoid medical jargon, and focus on what the patient can actually DO to improve their health.
            
            Patient Information:
            {patient_details}
            
            Medical Data:
            {medical_data}
            
            Please provide your advice in the following EXACT format:
            
            **1. Lifestyle Changes You Can Make:**
            * [Give 2-3 simple lifestyle changes the patient can start today]
            * [Focus on small, achievable steps]
            
            **2. What to Eat and Drink:**
            * [Give 2-3 simple dietary suggestions]
            * [Use everyday foods, not medical terms]
            
            **3. Exercise and Movement:**
            * [Give 2-3 simple exercise suggestions]
            * [Start with easy activities they can do at home]
            
            **4. What to Do Next:**
            * [Give 2-3 clear next steps]
            * [Mention if they need to see a doctor and when]
            
            REMEMBER: 
            - Use simple, everyday words
            - Give specific, actionable advice
            - Focus on what the patient can DO right now
            - Avoid medical jargon
            - Make it sound doable and not overwhelming
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response to extract structured information
            response_text = response.text
            
            # Extract recommendations by category
            lifestyle_recs = []
            dietary_sugs = []
            exercise_recs = []
            follow_up_acts = []
            
            # Parse based on the new structured format
            sections = response_text.split('\n')
            current_section = None
            
            for line in sections:
                line = line.strip()
                if '**1. Lifestyle Changes You Can Make:**' in line:
                    current_section = 'lifestyle'
                elif '**2. What to Eat and Drink:**' in line:
                    current_section = 'dietary'
                elif '**3. Exercise and Movement:**' in line:
                    current_section = 'exercise'
                elif '**4. What to Do Next:**' in line:
                    current_section = 'follow_up'
                elif line.startswith('*') or line.startswith('-'):
                    rec = line.strip().lstrip('*').lstrip('-').strip()
                    if rec and current_section:
                        if current_section == 'lifestyle':
                            lifestyle_recs.append(rec)
                        elif current_section == 'dietary':
                            dietary_sugs.append(rec)
                        elif current_section == 'exercise':
                            exercise_recs.append(rec)
                        elif current_section == 'follow_up':
                            follow_up_acts.append(rec)
            
            # Fallback if parsing didn't work
            if not lifestyle_recs:
                lifestyle_recs = ['Generated via AI analysis']
            if not dietary_sugs:
                dietary_sugs = ['Consult with healthcare provider']
            if not exercise_recs:
                exercise_recs = ['Based on medical data']
            if not follow_up_acts:
                follow_up_acts = ['Schedule follow-up appointment']
            
            recommendations = {
                'lifestyle_recommendations': lifestyle_recs,
                'dietary_suggestions': dietary_sugs,
                'exercise_recommendations': exercise_recs,
                'follow_up_actions': follow_up_acts,
                'language': language,
                'full_response': response_text
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating health recommendations: {e}")
            return self._get_fallback_response('recommendations', language)
