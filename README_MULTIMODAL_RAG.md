# HackRx Multimodal RAG API

A high-performance multimodal RAG (Retrieval-Augmented Generation) application built with FastAPI that processes PDFs, DOCX, and email documents with advanced text, table, and image extraction capabilities. Optimized for sub-30-second response times.

## 🚀 Features

- **Advanced Document Parsing**: PDF, DOCX, and email processing with LlamaParse
- **Multimodal Content Extraction**: Text, tables, and images with OCR
- **High-Performance Embeddings**: Together AI BGE-large model
- **Intelligent Chunking**: Optimized text splitting with content quality scoring
- **Vector Storage**: Pinecone for semantic search and retrieval
- **Fast Response Generation**: Llama-3.1-8B-Instruct-Turbo
- **Vision Processing**: Llama-Vision-Free for image analysis
- **Sub-30-Second Response Time**: Aggressive parallel processing and optimization
- **Bearer Token Authentication**: Secure API access

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **Document Parsing** | LlamaParse |
| **Embeddings** | Together AI BGE-large |
| **Chat Model** | Meta-Llama-3.1-8B-Instruct-Turbo |
| **Vision Model** | Meta-Llama-Vision-Free |
| **Vector DB** | Pinecone |
| **PDF Processing** | PyMuPDF, PDFPlumber, Camelot |
| **OCR** | Tesseract |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |

## 📋 Prerequisites

1. **Python 3.9+**
2. **API Keys**:
   - Together AI API key
   - LlamaParse API key  
   - Pinecone API key
3. **System Dependencies**:
   - Tesseract OCR
   - OpenCV (for advanced table extraction)

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd hackrx-multimodal-rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install libgl1-mesa-glx libglib2.0-0  # For OpenCV
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to system PATH

### 4. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Required environment variables:
```env
TOGETHER_API_KEY=your_together_ai_api_key
LLAMAPARSE_API_KEY=your_llamaparse_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
BEARER_TOKEN=your_secure_bearer_token
```

### 5. Run the Application

```bash
python main.py
```

The API will be available at: `http://localhost:8000`

## 📡 API Endpoints

### Main RAG Endpoint

**POST `/hackrx/run`**

Process documents and answer questions with sub-30-second response time.

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your_bearer_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

**Request Format:**
```json
{
  "documents": "string or array of URLs",
  "questions": ["array of questions (max 20)"]
}
```

**Response Format:**
```json
{
  "answers": ["array of answers"],
  "success": true,
  "processing_info": {
    "documents_processed": 1,
    "chunks_generated": 156,
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "chat_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "performance_target_met": true
  },
  "processing_time": 24.5
}
```

### Quick Query Endpoint

**POST `/hackrx/query`**

Query existing indexed documents (faster for follow-up questions).

### Health Check

**GET `/hackrx/health`**

Check system health and component status.

### Statistics

**GET `/hackrx/stats`**

Get vector store statistics and system information.

### Cleanup

**DELETE `/hackrx/cleanup/{source_id}`**

Remove vectors from a specific document source.

## 🔧 Configuration Options

### Performance Tuning

```env
# Response time optimization (default: 28 seconds max)
TOTAL_PIPELINE_TIMEOUT=28.0
LLM_TIMEOUT=15.0
RETRIEVAL_TIMEOUT=5.0

# Parallel processing
MAX_CONCURRENT_EMBEDDINGS=10
MAX_CONCURRENT_LLM_CALLS=5
EMBEDDING_BATCH_SIZE=32

# Chunking optimization
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNKS_PER_QUERY=8
```

### Document Processing

```env
# Table extraction strategy
TABLE_EXTRACTION_STRATEGY=hybrid  # pdfplumber, camelot, hybrid

# OCR settings
ENABLE_OCR=true
OCR_CONFIDENCE_THRESHOLD=0.7
IMAGE_PREPROCESSING=true

# File size limits
MAX_FILE_SIZE=52428800  # 50MB
```

### Caching

```env
# Embedding cache for faster repeated processing
ENABLE_EMBEDDING_CACHE=true
CACHE_TTL=3600  # 1 hour
```

## 📊 Performance Optimization

The system is optimized for sub-30-second response times through:

1. **Parallel Processing**: Document parsing, embedding generation, and LLM calls run concurrently
2. **Intelligent Chunking**: Optimized chunk sizes and overlap for better retrieval
3. **Embedding Caching**: Reduce repeated embedding generation
4. **Batch Operations**: Efficient vector store operations
5. **Timeout Management**: Aggressive timeouts to prevent hanging
6. **Quality Scoring**: Filter low-quality chunks to improve relevance

### Performance Targets

| Stage | Target Time | Optimization |
|-------|-------------|--------------|
| Document Parsing | 8-10s | Parallel parsing, multiple extraction strategies |
| Chunking | 2-3s | Optimized splitters, quality filtering |
| Embedding Generation | 5-8s | Batch processing, caching |
| Vector Indexing | 2-3s | Parallel upserts |
| Query Processing | 8-12s | Parallel search and generation |
| **Total** | **< 30s** | **End-to-end optimization** |

## 🔒 Authentication

All endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer your_bearer_token" \
  "http://localhost:8000/hackrx/health"
```

## 📝 Supported Document Types

1. **PDF Files**: Advanced parsing with text, table, and image extraction
2. **DOCX Files**: Microsoft Word document processing
3. **Email Files**: .eml email parsing
4. **Image Content**: OCR processing for scanned documents

## 🎯 Use Cases

- **Insurance Policy Analysis**: Answer questions about policy terms, conditions, and coverage
- **Contract Processing**: Extract and query contract clauses and terms
- **Legal Document Review**: Analyze legal documents for specific information
- **Research Paper Analysis**: Process academic papers and answer research questions
- **Technical Documentation**: Query technical manuals and documentation

## 🐛 Troubleshooting

### Common Issues

1. **Timeout Errors**: Reduce document size or number of questions
2. **OCR Failures**: Ensure Tesseract is properly installed
3. **Memory Issues**: Reduce batch sizes in configuration
4. **API Key Errors**: Verify all API keys are correctly set

### Debug Mode

Set log level to DEBUG for detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor processing times and identify bottlenecks:

```bash
curl -H "Authorization: Bearer your_token" \
  "http://localhost:8000/hackrx/stats"
```

## 📈 Monitoring and Metrics

The API provides detailed processing metrics:

- Document parsing time
- Chunk generation statistics
- Embedding generation performance
- Vector store operations
- LLM response times
- Overall pipeline performance

## 🔄 Development

### Adding New Document Types

1. Extend `LlamaParseService` with new parsing logic
2. Update `EnhancedChunker` for new content types
3. Add appropriate metadata handling

### Improving Performance

1. Monitor bottlenecks using the stats endpoint
2. Adjust timeout and batch size settings
3. Implement additional caching strategies
4. Optimize chunking strategies for your use case

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Monitor system health at `/hackrx/health`