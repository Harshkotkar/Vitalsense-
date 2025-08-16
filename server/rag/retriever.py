import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .embeddings import MedicalEmbeddings
from .vector_store import VectorStore
import re

logger = logging.getLogger(__name__)

class MedicalRetriever:
    """
    Medical document retriever that combines embeddings and vector store
    for intelligent retrieval of relevant medical context
    """
    
    def __init__(self, vector_store: VectorStore, embeddings: MedicalEmbeddings):
        """
        Initialize the medical retriever
        
        Args:
            vector_store: Vector store instance
            embeddings: Medical embeddings instance
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        logger.info("Medical retriever initialized")
    
    def retrieve_relevant_context(self, query_text: str, top_k: int = 5, 
                                similarity_threshold: float = 0.7,
                                filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant medical context for a given query
        
        Args:
            query_text: Medical query text
            top_k: Number of top relevant documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant documents with context
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.generate_embeddings(query_text)
            
            # Search for similar documents
            similar_docs = self.vector_store.search_similar(
                query_embedding, 
                top_k=top_k * 2,  # Get more results for filtering
                filter_metadata=filter_metadata
            )
            
            # Filter by similarity threshold and extract medical entities
            relevant_context = []
            for doc in similar_docs:
                if doc['similarity_score'] >= similarity_threshold:
                    # Extract medical entities from the document
                    entities = self.embeddings.extract_medical_entities(doc['document'])
                    
                    # Create context entry
                    context_entry = {
                        'document': doc['document'],
                        'similarity_score': doc['similarity_score'],
                        'metadata': doc['metadata'],
                        'medical_entities': entities,
                        'relevant_snippets': self._extract_relevant_snippets(query_text, doc['document'])
                    }
                    
                    relevant_context.append(context_entry)
                    
                    # Stop if we have enough high-quality results
                    if len(relevant_context) >= top_k:
                        break
            
            # Sort by relevance
            relevant_context.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Retrieved {len(relevant_context)} relevant documents")
            return relevant_context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {str(e)}")
            return []
    
    def _extract_relevant_snippets(self, query_text: str, document_text: str, 
                                 snippet_length: int = 200) -> List[str]:
        """
        Extract relevant snippets from document based on query
        
        Args:
            query_text: Query text
            document_text: Document text
            snippet_length: Length of each snippet
            
        Returns:
            List of relevant snippets
        """
        try:
            # Extract medical entities from query
            query_entities = self.embeddings.extract_medical_entities(query_text)
            query_terms = [ent['text'].lower() for ent in query_entities]
            
            # If no entities found, use simple keyword matching
            if not query_terms:
                # Extract potential medical terms using regex
                medical_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
                query_terms = re.findall(medical_pattern, query_text.lower())
            
            snippets = []
            sentences = re.split(r'[.!?]+', document_text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance_score = 0
                
                # Calculate relevance based on entity matches
                for term in query_terms:
                    if term in sentence_lower:
                        relevance_score += 1
                
                # If sentence is relevant, add it as a snippet
                if relevance_score > 0:
                    # Clean and truncate snippet
                    snippet = sentence.strip()
                    if len(snippet) > snippet_length:
                        snippet = snippet[:snippet_length] + "..."
                    
                    snippets.append({
                        'text': snippet,
                        'relevance_score': relevance_score
                    })
            
            # Sort by relevance and return top snippets
            snippets.sort(key=lambda x: x['relevance_score'], reverse=True)
            return [s['text'] for s in snippets[:3]]  # Return top 3 snippets
            
        except Exception as e:
            logger.error(f"Error extracting relevant snippets: {str(e)}")
            return []
    
    def retrieve_by_medical_entities(self, entities: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on specific medical entities
        
        Args:
            entities: List of medical entity names
            top_k: Number of top documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Create a query from entities
            query_text = " ".join(entities)
            query_embedding = self.embeddings.generate_embeddings(query_text)
            
            # Search with entity-specific filters
            similar_docs = self.vector_store.search_similar(
                query_embedding,
                top_k=top_k
            )
            
            # Filter documents that contain the specified entities
            entity_filtered_docs = []
            for doc in similar_docs:
                doc_entities = self.embeddings.extract_medical_entities(doc['document'])
                doc_entity_texts = [ent['text'].lower() for ent in doc_entities]
                
                # Check if document contains any of the target entities
                entity_matches = sum(1 for entity in entities 
                                   if entity.lower() in doc_entity_texts)
                
                if entity_matches > 0:
                    doc['entity_matches'] = entity_matches
                    entity_filtered_docs.append(doc)
            
            # Sort by entity matches and similarity
            entity_filtered_docs.sort(
                key=lambda x: (x['entity_matches'], x['similarity_score']), 
                reverse=True
            )
            
            logger.info(f"Retrieved {len(entity_filtered_docs)} documents by entities")
            return entity_filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving by medical entities: {str(e)}")
            return []
    
    def retrieve_similar_reports(self, report_text: str, report_type: str = None, 
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar medical reports based on content and type
        
        Args:
            report_text: Medical report text
            report_type: Type of medical report (e.g., 'xray', 'blood_test', 'mri')
            top_k: Number of similar reports to retrieve
            
        Returns:
            List of similar reports
        """
        try:
            # Prepare filter metadata
            filter_metadata = None
            if report_type:
                filter_metadata = {'report_type': report_type}
            
            # Retrieve relevant context
            similar_reports = self.retrieve_relevant_context(
                report_text,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Add report-specific analysis
            for report in similar_reports:
                report['analysis'] = self._analyze_report_similarity(report_text, report['document'])
            
            return similar_reports
            
        except Exception as e:
            logger.error(f"Error retrieving similar reports: {str(e)}")
            return []
    
    def _analyze_report_similarity(self, query_report: str, similar_report: str) -> Dict[str, Any]:
        """
        Analyze similarity between two medical reports
        
        Args:
            query_report: Original report text
            similar_report: Similar report text
            
        Returns:
            Analysis of similarities
        """
        try:
            # Extract entities from both reports
            query_entities = self.embeddings.extract_medical_entities(query_report)
            similar_entities = self.embeddings.extract_medical_entities(similar_report)
            
            # Find common entities
            query_entity_texts = set(ent['text'].lower() for ent in query_entities)
            similar_entity_texts = set(ent['text'].lower() for ent in similar_entities)
            common_entities = query_entity_texts.intersection(similar_entity_texts)
            
            # Calculate similarity metrics
            total_entities = len(query_entity_texts.union(similar_entity_texts))
            entity_similarity = len(common_entities) / total_entities if total_entities > 0 else 0
            
            return {
                'common_entities': list(common_entities),
                'entity_similarity': entity_similarity,
                'query_entity_count': len(query_entities),
                'similar_entity_count': len(similar_entities)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing report similarity: {str(e)}")
            return {}
    
    def get_medical_knowledge_base(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Build a knowledge base from relevant medical documents
        
        Args:
            query_text: Query to build knowledge base for
            top_k: Number of documents to include
            
        Returns:
            Structured knowledge base
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_context(query_text, top_k=top_k)
            
            # Build knowledge base
            knowledge_base = {
                'query': query_text,
                'total_documents': len(relevant_docs),
                'medical_entities': set(),
                'common_conditions': [],
                'treatment_patterns': [],
                'diagnostic_insights': []
            }
            
            # Extract and aggregate information
            for doc in relevant_docs:
                # Collect medical entities
                for entity in doc['medical_entities']:
                    knowledge_base['medical_entities'].add(entity['text'])
                
                # Extract conditions, treatments, and diagnostics
                text_lower = doc['document'].lower()
                
                # Simple pattern matching for medical concepts
                if any(term in text_lower for term in ['diagnosis', 'diagnosed', 'condition']):
                    knowledge_base['common_conditions'].append(doc['document'][:200] + "...")
                
                if any(term in text_lower for term in ['treatment', 'therapy', 'medication']):
                    knowledge_base['treatment_patterns'].append(doc['document'][:200] + "...")
                
                if any(term in text_lower for term in ['test', 'examination', 'scan']):
                    knowledge_base['diagnostic_insights'].append(doc['document'][:200] + "...")
            
            # Convert set to list for JSON serialization
            knowledge_base['medical_entities'] = list(knowledge_base['medical_entities'])
            
            return knowledge_base
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
            return {'error': str(e)}
