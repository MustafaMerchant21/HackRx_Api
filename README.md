# HackRx Multimodal RAG API

A high-performance multimodal RAG (Retrieval-Augmented Generation) application built with FastAPI, designed to process documents and answer questions with response times under 30 seconds.

## 🚀 Features

- **Multimodal Document Processing**: Extract text, tables, and images from PDFs, DOCX, and emails
- **Advanced RAG Pipeline**: LlamaParse + LlamaIndex + Together AI + Pinecone
- **High-Performance**: Optimized for sub-30-second response times
- **Secure**: Bearer token authentication
- **HTTPS Ready**: Production-ready with SSL support
- **Scalable**: Async processing with parallel operations

## 🛠️ Tech Stack

- **FastAPI**: Modern web framework for APIs
- **LlamaParse**: Advanced document parsing
- **LlamaIndex**: RAG framework
- **Together AI**: 
  - Embeddings: BGE-large
  - LLM: Meta-Llama-3.1-8B-Instruct-Turbo
  - Vision: Llama-Vision-Free
- **Pinecone**: Vector database
- **PyMuPDF**: PDF processing
- **aiohttp**: Async HTTP client

## 📋 Prerequisites

- Python 3.8+
- Together AI API key
- Pinecone API key and environment
- LlamaParse API key (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd hackrx-multimodal-rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:

```env
TOGETHER_API_KEY=your_together_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
BEARER_TOKEN=your_secure_bearer_token_here
```

### 3. Create Pinecone Index

Create a Pinecone index with the following specifications:

```python
import pinecone

pinecone.init(api_key="your_api_key", environment="your_environment")
pinecone.create_index(
    name="hackrx-multimodal",
    dimension=1024,  # BGE-large dimension
    metric="cosine"
)
```

### 4. Run the Application

```bash
# Development mode
python main.py

# Production mode with HTTPS
python main.py
```

The API will be available at:
- **Local**: http://localhost:8000
- **HTTPS**: https://localhost:8000 (with SSL certificates)

## 📚 API Documentation

### Main Endpoint: `/hackrx/run`

**POST** `/hackrx/run`

Process documents and answer questions with multimodal RAG.

#### Headers
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer <YOUR_BEARER_TOKEN>
```

#### Request Body
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

#### Response
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."
    ],
    "processing_time": 15.23,
    "document_type": "pdf",
    "extraction_method": "LlamaParse + LlamaIndex + Together AI"
}
```

### Health Check: `/hackrx/health`

**GET** `/hackrx/health`

Check the health status of all services.

#### Response
```json
{
    "status": "healthy",
    "services": {
        "together_ai": "healthy",
        "pinecone": "healthy",
        "llama_parse": "healthy"
    },
    "timestamp": 1703123456.789
}
```

## 🔧 Configuration

### Performance Tuning

Adjust these settings in `.env` for optimal performance:

```env
# Processing Settings
MAX_FILE_SIZE=52428800          # 50MB max file size
CHUNK_SIZE=1000                 # Text chunk size
CHUNK_OVERLAP=200              # Chunk overlap
MAX_CHUNKS_PER_QUERY=15        # Max chunks for retrieval
MAX_RESPONSE_TIME=30           # Max response time in seconds

# Performance Settings
EMBEDDING_BATCH_SIZE=50        # Batch size for embeddings
MAX_CONCURRENT_EMBEDDINGS=10   # Concurrent embedding operations
MAX_CONCURRENT_LLM_CALLS=5     # Concurrent LLM calls
```

### Multimodal Settings

```env
# Multimodal Features
ENABLE_IMAGE_EXTRACTION=true    # Extract images from documents
ENABLE_TABLE_EXTRACTION=true    # Extract tables from documents
ENABLE_VISION_ANALYSIS=true     # Analyze images with vision model
IMAGE_QUALITY_THRESHOLD=0.7     # Minimum image quality for analysis
```

## 🔄 Pipeline Flow

1. **Document Download**: Download document from provided URL
2. **Multimodal Parsing**: Extract text, tables, and images using LlamaParse
3. **Vision Analysis**: Analyze images using Llama-Vision-Free
4. **Content Chunking**: Split content into optimized chunks
5. **Embedding Generation**: Generate embeddings using BGE-large
6. **Vector Storage**: Store in Pinecone vector database
7. **Retrieval**: Find relevant chunks for each question
8. **Answer Generation**: Generate answers using Llama-3.1-8B-Instruct-Turbo
9. **Response**: Return structured answers within 30 seconds

## 🚀 Performance Optimization

### Response Time Optimization

- **Parallel Processing**: Questions are processed concurrently
- **Optimized Chunking**: Smart text chunking for better retrieval
- **Batch Embeddings**: Efficient embedding generation
- **Caching**: Vector store caching for repeated queries
- **Timeout Management**: Strict 30-second timeout enforcement

### Memory Optimization

- **Streaming Downloads**: Large files downloaded in chunks
- **Temporary File Management**: Automatic cleanup of temporary files
- **Memory-Efficient Parsing**: Optimized document parsing

## 🔒 Security

- **Bearer Token Authentication**: Secure API access
- **HTTPS Support**: SSL/TLS encryption
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses

## 🧪 Testing

### Test the API

```bash
# Test with curl
curl -X POST "https://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic of this document?"]
  }'
```

### Load Testing

```bash
# Install testing dependencies
pip install locust

# Run load test
locust -f load_test.py
```

## 📊 Monitoring

### Health Monitoring

- Service health checks
- Response time monitoring
- Error rate tracking
- Resource utilization

### Logging

Structured logging with different levels:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical errors
- DEBUG: Detailed debugging information

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Deployment

1. **Environment Setup**: Configure production environment variables
2. **SSL Certificates**: Install valid SSL certificates
3. **Load Balancer**: Set up load balancer for high availability
4. **Monitoring**: Configure monitoring and alerting
5. **Backup**: Set up database backups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API examples

## 🔄 Changelog

### v1.0.0
- Initial release
- Multimodal RAG pipeline
- Together AI integration
- Pinecone vector store
- FastAPI implementation
- Bearer token authentication
- HTTPS support