import re
import logging
from typing import List, Dict, Any
import string
from datetime import datetime

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Medical text preprocessing and cleaning utility
    Handles text normalization, noise removal, and medical-specific cleaning
    """
    
    def __init__(self):
        """Initialize the text cleaner with medical-specific patterns"""
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'Temp': 'temperature',
            'WBC': 'white blood cell count',
            'RBC': 'red blood cell count',
            'HGB': 'hemoglobin',
            'HCT': 'hematocrit',
            'PLT': 'platelet count',
            'Na': 'sodium',
            'K': 'potassium',
            'Cl': 'chloride',
            'CO2': 'carbon dioxide',
            'BUN': 'blood urea nitrogen',
            'Cr': 'creatinine',
            'Glucose': 'blood glucose',
            'CBC': 'complete blood count',
            'CMP': 'comprehensive metabolic panel',
            'LFT': 'liver function test',
            'PT': 'prothrombin time',
            'INR': 'international normalized ratio',
            'APTT': 'activated partial thromboplastin time',
            'ESR': 'erythrocyte sedimentation rate',
            'CRP': 'C-reactive protein',
            'Troponin': 'troponin I',
            'BNP': 'B-type natriuretic peptide',
            'D-dimer': 'D-dimer',
            'PSA': 'prostate-specific antigen',
            'TSH': 'thyroid-stimulating hormone',
            'T4': 'thyroxine',
            'T3': 'triiodothyronine',
            'HbA1c': 'hemoglobin A1c',
            'LDL': 'low-density lipoprotein',
            'HDL': 'high-density lipoprotein',
            'TG': 'triglycerides',
            'CT': 'computed tomography',
            'MRI': 'magnetic resonance imaging',
            'X-ray': 'X-ray',
            'ECG': 'electrocardiogram',
            'EKG': 'electrocardiogram',
            'EEG': 'electroencephalogram',
            'EMG': 'electromyography',
            'US': 'ultrasound',
            'CXR': 'chest X-ray',
            'ABG': 'arterial blood gas',
            'CBC': 'complete blood count',
            'UA': 'urinalysis',
            'CSF': 'cerebrospinal fluid',
            'IV': 'intravenous',
            'IM': 'intramuscular',
            'SC': 'subcutaneous',
            'PO': 'oral',
            'PRN': 'as needed',
            'BID': 'twice daily',
            'TID': 'three times daily',
            'QID': 'four times daily',
            'QD': 'daily',
            'QOD': 'every other day',
            'STAT': 'immediately',
            'NPO': 'nothing by mouth',
            'DNR': 'do not resuscitate',
            'DNR/DNI': 'do not resuscitate/do not intubate',
            'CPR': 'cardiopulmonary resuscitation',
            'ICU': 'intensive care unit',
            'CCU': 'coronary care unit',
            'ER': 'emergency room',
            'OR': 'operating room',
            'PACU': 'post-anesthesia care unit',
            'NICU': 'neonatal intensive care unit',
            'PICU': 'pediatric intensive care unit'
        }
        
        # Medical terms that should be preserved (not lowercased)
        self.preserve_case_terms = {
            'pH', 'Na+', 'K+', 'Cl-', 'Ca++', 'Mg++', 'Fe++', 'Fe+++',
            'CO2', 'O2', 'N2', 'H2O', 'H2O2', 'CO', 'NO', 'NO2',
            'HIV', 'AIDS', 'SARS', 'COVID-19', 'MERS', 'H1N1', 'H5N1',
            'DNA', 'RNA', 'mRNA', 'tRNA', 'rRNA', 'cDNA', 'PCR', 'RT-PCR',
            'ELISA', 'RIA', 'FISH', 'IHC', 'PCR', 'DNA', 'RNA',
            'CT', 'MRI', 'PET', 'SPECT', 'DEXA', 'ECG', 'EKG', 'EEG', 'EMG',
            'CBC', 'CMP', 'LFT', 'PT', 'INR', 'APTT', 'ESR', 'CRP',
            'BNP', 'PSA', 'TSH', 'T4', 'T3', 'HbA1c', 'LDL', 'HDL'
        }
        
        # Patterns for noise removal
        self.noise_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b',  # Time stamps
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',     # Dates
            r'\b\d{3}-\d{3}-\d{4}\b',           # Phone numbers
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email addresses
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]',  # Special characters (keep some)
        ]
        
        logger.info("Text cleaner initialized with medical patterns")
    
    def clean_medical_text(self, text: str) -> str:
        """
        Clean and preprocess medical text
        
        Args:
            text: Raw medical text
            
        Returns:
            Cleaned medical text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Step 1: Basic cleaning
            cleaned_text = self._basic_cleaning(text)
            
            # Step 2: Handle medical abbreviations
            cleaned_text = self._expand_medical_abbreviations(cleaned_text)
            
            # Step 3: Normalize medical terms
            cleaned_text = self._normalize_medical_terms(cleaned_text)
            
            # Step 4: Remove noise while preserving medical content
            cleaned_text = self._remove_noise(cleaned_text)
            
            # Step 5: Final formatting
            cleaned_text = self._final_formatting(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning medical text: {str(e)}")
            return text  # Return original text if cleaning fails
    
    def _basic_cleaning(self, text: str) -> str:
        """Perform basic text cleaning"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Normalize line breaks
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\r+', ' ', text)
            
            # Remove multiple spaces
            text = re.sub(r' +', ' ', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error in basic cleaning: {str(e)}")
            return text
    
    def _expand_medical_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        try:
            # Create a copy of text to work with
            expanded_text = text
            
            # Sort abbreviations by length (longest first) to avoid partial matches
            sorted_abbreviations = sorted(
                self.medical_abbreviations.items(),
                key=lambda x: len(x[0]),
                reverse=True
            )
            
            for abbreviation, expansion in sorted_abbreviations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(abbreviation) + r'\b'
                expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
            
            return expanded_text
            
        except Exception as e:
            logger.error(f"Error expanding medical abbreviations: {str(e)}")
            return text
    
    def _normalize_medical_terms(self, text: str) -> str:
        """Normalize medical terms while preserving important case"""
        try:
            # Split text into words
            words = text.split()
            normalized_words = []
            
            for word in words:
                # Check if word should preserve case
                if word.upper() in self.preserve_case_terms:
                    normalized_words.append(word)
                else:
                    # Normalize to lowercase
                    normalized_words.append(word.lower())
            
            return ' '.join(normalized_words)
            
        except Exception as e:
            logger.error(f"Error normalizing medical terms: {str(e)}")
            return text
    
    def _remove_noise(self, text: str) -> str:
        """Remove noise while preserving medical content"""
        try:
            cleaned_text = text
            
            # Remove noise patterns
            for pattern in self.noise_patterns:
                cleaned_text = re.sub(pattern, ' ', cleaned_text)
            
            # Remove excessive punctuation (keep some for medical context)
            cleaned_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]', ' ', cleaned_text)
            
            # Clean up extra spaces again
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error removing noise: {str(e)}")
            return text
    
    def _final_formatting(self, text: str) -> str:
        """Apply final formatting touches"""
        try:
            # Ensure proper sentence endings
            text = re.sub(r'\s+([.!?])', r'\1', text)
            
            # Remove multiple periods
            text = re.sub(r'\.+', '.', text)
            
            # Ensure single space after punctuation
            text = re.sub(r'([.!?])\s*', r'\1 ', text)
            
            # Final whitespace cleanup
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error in final formatting: {str(e)}")
            return text
    
    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential medical entities from text
        
        Args:
            text: Medical text
            
        Returns:
            List of medical entities with positions
        """
        try:
            entities = []
            
            # Pattern for medical terms (capitalized words, potential medical terms)
            medical_patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
                r'\b\d+(?:\.\d+)?\s*(?:mg|mEq|mmol|g|kg|ml|L|cm|mm|in|ft)\b',  # Measurements
                r'\b(?:normal|abnormal|elevated|decreased|positive|negative|present|absent)\b',  # Status terms
                r'\b(?:mild|moderate|severe|acute|chronic|subacute)\b',  # Severity terms
            ]
            
            for pattern in medical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'type': 'medical_term'
                    }
                    entities.append(entity)
            
            # Remove duplicates and sort by position
            unique_entities = []
            seen_positions = set()
            
            for entity in sorted(entities, key=lambda x: x['start']):
                position = (entity['start'], entity['end'])
                if position not in seen_positions:
                    unique_entities.append(entity)
                    seen_positions.add(position)
            
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {str(e)}")
            return []
    
    def segment_medical_text(self, text: str, max_segment_length: int = 1000) -> List[str]:
        """
        Segment medical text into manageable chunks
        
        Args:
            text: Medical text to segment
            max_segment_length: Maximum length of each segment
            
        Returns:
            List of text segments
        """
        try:
            if len(text) <= max_segment_length:
                return [text]
            
            segments = []
            sentences = re.split(r'[.!?]+', text)
            current_segment = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add sentence ending back
                sentence_with_ending = sentence + "."
                
                # Check if adding this sentence would exceed the limit
                if len(current_segment) + len(sentence_with_ending) <= max_segment_length:
                    current_segment += sentence_with_ending + " "
                else:
                    # Save current segment and start new one
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = sentence_with_ending + " "
            
            # Add the last segment
            if current_segment:
                segments.append(current_segment.strip())
            
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting medical text: {str(e)}")
            return [text]
    
    def validate_medical_text(self, text: str) -> Dict[str, Any]:
        """
        Validate medical text quality and content
        
        Args:
            text: Medical text to validate
            
        Returns:
            Validation results
        """
        try:
            validation_result = {
                'is_valid': True,
                'length': len(text),
                'word_count': len(text.split()),
                'medical_terms_count': 0,
                'abbreviations_count': 0,
                'issues': []
            }
            
            # Check for minimum content
            if len(text) < 10:
                validation_result['is_valid'] = False
                validation_result['issues'].append('Text too short')
            
            # Count medical terms
            medical_entities = self.extract_medical_entities(text)
            validation_result['medical_terms_count'] = len(medical_entities)
            
            # Count abbreviations
            abbreviation_count = 0
            for abbreviation in self.medical_abbreviations.keys():
                if re.search(r'\b' + re.escape(abbreviation) + r'\b', text, re.IGNORECASE):
                    abbreviation_count += 1
            validation_result['abbreviations_count'] = abbreviation_count
            
            # Check for common issues
            if not medical_entities:
                validation_result['issues'].append('No medical terms detected')
            
            if len(text.split()) < 5:
                validation_result['issues'].append('Insufficient word count')
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating medical text: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'issues': ['Validation error occurred']
            }
