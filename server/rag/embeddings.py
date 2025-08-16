import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class MedicalEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.nlp = None
        self.entity_linker = None
        
        # Try to load spaCy and scispacy, but make it optional
        try:
            import spacy
            import scispacy
            from scispacy.linking import EntityLinker
            self.nlp = spacy.load("en_core_sci_sm")
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            logger.info("✅ Medical entity recognition enabled with scispacy")
        except Exception as e:
            logger.warning(f"⚠️ Medical entity recognition disabled: {e}")
            logger.info("System will work with basic text processing")

    def generate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text using Sentence Transformers."""
        try:
            if isinstance(text, str):
                text = [text]
            embeddings = self.model.encode(text, convert_to_numpy=True)
            
            # Ensure we return a proper numpy array
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            # If we have a single text, return 1D array, otherwise 2D array
            if len(text) == 1:
                return embeddings.flatten()  # Return 1D array for single text
            else:
                return embeddings  # Return 2D array for multiple texts
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            if isinstance(text, str):
                return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
            else:
                return np.zeros((len(text), 384))

    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities using scispacy if available."""
        entities = []
        
        if self.nlp is None:
            logger.warning("Medical entity extraction not available - scispacy not loaded")
            return entities
            
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                
                # Add UMLS links if available
                if hasattr(ent, '_.umls_ents'):
                    entity_info["umls_links"] = [
                        {
                            "cui": umls_ent.cui,
                            "name": umls_ent.umls_name,
                            "score": umls_ent.score
                        }
                        for umls_ent in ent._.umls_ents
                    ]
                
                entities.append(entity_info)
                
        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            
        return entities

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for processing."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings in the overlap region
                for i in range(end - overlap, end):
                    if i < len(text) and text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts efficiently
        
        Args:
            texts: List of medical texts
            batch_size: Size of batches to process
            
        Returns:
            numpy array of embeddings
        """
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise e
