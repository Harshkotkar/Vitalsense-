from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# Import our custom modules
from rag.embeddings import MedicalEmbeddings
from rag.vector_store import VectorStore
from rag.retriever import MedicalRetriever
from ai.gemini_api import GeminiAPI
from ai.text_cleaner import TextCleaner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
embeddings = MedicalEmbeddings()
vector_store = VectorStore()
retriever = MedicalRetriever(vector_store, embeddings)
gemini_api = GeminiAPI()
text_cleaner = TextCleaner()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page for testing"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_report():
    """Upload and process medical report"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        new_filename = f"{file_id}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Save file
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Process the report
        result = process_medical_report(file_path, file_extension)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'analysis': result['analysis'],
            'original_text': result.get('original_text', ''),
            'cleaned_text': result.get('cleaned_text', ''),
            'relevant_context': result.get('relevant_context', [])
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_medical_report(file_path, file_extension):
    """Process medical report and generate analysis"""
    try:
        # Extract text from document
        if file_extension == 'pdf':
            import pymupdf
            doc = pymupdf.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            # For images, use OCR
            import pytesseract
            from PIL import Image
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        
        # Clean and preprocess text
        cleaned_text = text_cleaner.clean_medical_text(text)
        
        # Generate embeddings and store in vector database
        embeddings_data = embeddings.generate_embeddings(cleaned_text)
        
        # Ensure embeddings are in the correct format
        if isinstance(embeddings_data, list):
            embeddings_data = np.array(embeddings_data)
        elif not isinstance(embeddings_data, np.ndarray):
            embeddings_data = np.array(embeddings_data)
        
        # Flatten if it's a nested array
        if embeddings_data.ndim > 1:
            embeddings_data = embeddings_data.flatten()
        
        logger.info(f"Embeddings shape: {embeddings_data.shape}, type: {type(embeddings_data)}")
        
        vector_store.add_document(cleaned_text, embeddings_data, metadata={
            'file_path': file_path,
            'upload_time': datetime.now().isoformat(),
            'file_type': file_extension
        })
        
        # Retrieve relevant medical context
        relevant_context = retriever.retrieve_relevant_context(cleaned_text, top_k=5)
        
        # Generate AI analysis
        analysis = gemini_api.analyze_medical_report(
            cleaned_text, 
            relevant_context,
            language=request.form.get('language', 'English')
        )
        
        return {
            'original_text': text[:500] + "..." if len(text) > 500 else text,
            'cleaned_text': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            'analysis': analysis,
            'relevant_context': relevant_context
        }
        
    except Exception as e:
        logger.error(f"Error processing medical report: {str(e)}")
        raise e

@app.route('/api/explain', methods=['POST'])
def explain_medical_terms():
    """Convert medical terms to simple language"""
    try:
        data = request.get_json()
        medical_text = data.get('text', '')
        target_language = data.get('language', 'English')
        
        if not medical_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Clean the text
        cleaned_text = text_cleaner.clean_medical_text(medical_text)
        
        # Get relevant medical context
        relevant_context = retriever.retrieve_relevant_context(cleaned_text, top_k=3)
        
        # Generate simplified explanation
        explanation = gemini_api.simplify_medical_terms(
            cleaned_text, 
            relevant_context,
            language=target_language
        )
        
        return jsonify({
            'success': True,
            'original_text': medical_text,
            'simplified_explanation': explanation
        })
        
    except Exception as e:
        logger.error(f"Error explaining medical terms: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_health_recommendations():
    """Generate personalized health recommendations"""
    try:
        data = request.get_json()
        medical_data = data.get('medical_data', '')
        patient_info = data.get('patient_info', {})
        target_language = data.get('language', 'English')
        
        if not medical_data:
            return jsonify({'error': 'No medical data provided'}), 400
        
        # Clean the medical data
        cleaned_data = text_cleaner.clean_medical_text(medical_data)
        
        # Get relevant medical context
        relevant_context = retriever.retrieve_relevant_context(cleaned_data, top_k=5)
        
        # Generate recommendations
        recommendations = gemini_api.generate_health_recommendations(
            cleaned_data,
            patient_info,
            relevant_context,
            language=target_language
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'embeddings': 'initialized',
            'vector_store': 'initialized',
            'retriever': 'initialized',
            'gemini_api': 'initialized'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
