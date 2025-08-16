import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for medical documents using ChromaDB
    Handles storage, retrieval, and similarity search of medical embeddings
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        try:
            self.persist_directory = persist_directory
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection for medical documents
            self.collection = self.client.get_or_create_collection(
                name="medical_documents",
                metadata={"description": "Medical reports and clinical documents"}
            )
            
            logger.info(f"Vector store initialized at {persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise e
    
    def add_document(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """
        Add a medical document to the vector store
        
        Args:
            text: Medical document text
            embedding: Document embedding vector
            metadata: Additional metadata about the document
            
        Returns:
            Document ID
        """
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'embedding_dimension': len(embedding)
            })
            
            # Add document to collection
            self.collection.add(
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Document added to vector store with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise e
    
    def add_documents_batch(self, texts: List[str], embeddings: List[np.ndarray], 
                          metadata_list: List[Dict[str, Any]] = None) -> List[str]:
        """
        Add multiple documents to the vector store in batch
        
        Args:
            texts: List of medical document texts
            embeddings: List of document embedding vectors
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        try:
            if len(texts) != len(embeddings):
                raise ValueError("Number of texts must match number of embeddings")
            
            # Generate document IDs
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            
            # Prepare metadata
            if metadata_list is None:
                metadata_list = [{} for _ in texts]
            
            # Update metadata with common fields
            for i, metadata in enumerate(metadata_list):
                metadata.update({
                    'timestamp': datetime.now().isoformat(),
                    'text_length': len(texts[i]),
                    'embedding_dimension': len(embeddings[i])
                })
            
            # Add documents to collection
            self.collection.add(
                documents=texts,
                embeddings=[emb.tolist() for emb in embeddings],
                metadatas=metadata_list,
                ids=doc_ids
            )
            
            logger.info(f"Added {len(texts)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents batch to vector store: {str(e)}")
            raise e
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar documents to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    similar_docs.append({
                        'document': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity_score': 1 - distance  # Convert distance to similarity
                    })
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if results['documents']:
                return {
                    'id': doc_id,
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def update_document(self, doc_id: str, text: str = None, embedding: np.ndarray = None, 
                       metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing document in the vector store
        
        Args:
            doc_id: Document ID to update
            text: New document text (optional)
            embedding: New embedding vector (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing document
            existing_doc = self.get_document_by_id(doc_id)
            if not existing_doc:
                logger.warning(f"Document {doc_id} not found for update")
                return False
            
            # Prepare update data
            update_data = {}
            if text is not None:
                update_data['documents'] = [text]
            if embedding is not None:
                update_data['embeddings'] = [embedding.tolist()]
            if metadata is not None:
                # Merge with existing metadata
                existing_metadata = existing_doc['metadata']
                existing_metadata.update(metadata)
                existing_metadata['last_updated'] = datetime.now().isoformat()
                update_data['metadatas'] = [existing_metadata]
            
            # Update document
            self.collection.update(
                ids=[doc_id],
                **update_data
            )
            
            logger.info(f"Document {doc_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Document {doc_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample documents for analysis
            sample_results = self.collection.get(
                limit=min(100, count),
                include=['metadatas']
            )
            
            # Calculate statistics
            stats = {
                'total_documents': count,
                'sample_size': len(sample_results['metadatas']) if sample_results['metadatas'] else 0
            }
            
            if sample_results['metadatas']:
                # Analyze metadata
                text_lengths = [meta.get('text_length', 0) for meta in sample_results['metadatas']]
                stats.update({
                    'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                    'min_text_length': min(text_lengths) if text_lengths else 0,
                    'max_text_length': max(text_lengths) if text_lengths else 0
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(where={})
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
