#!/bin/bash

# HackRx Multimodal RAG API Deployment Script
# This script helps set up and deploy the application

set -e  # Exit on any error

echo "🚀 HackRx Multimodal RAG API Deployment"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "📋 Python version: $python_version"

# Check if Python 3.9+ is available
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "❌ Python 3.9+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check for system dependencies
echo "🔍 Checking system dependencies..."

# Check for Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-eng"
    echo "   macOS: brew install tesseract"
    echo "   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
else
    tesseract_version=$(tesseract --version 2>&1 | head -n1)
    echo "✅ $tesseract_version"
fi

# Check for environment file
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the application"
    echo "   Required: TOGETHER_API_KEY, LLAMAPARSE_API_KEY, PINECONE_API_KEY"
else
    echo "✅ .env file exists"
fi

# Check API keys in .env
echo "🔑 Checking API key configuration..."
if grep -q "your_.*_api_key_here" .env; then
    echo "⚠️  Please update your API keys in .env file:"
    echo "   - TOGETHER_API_KEY"
    echo "   - LLAMAPARSE_API_KEY" 
    echo "   - PINECONE_API_KEY"
    echo "   - PINECONE_ENVIRONMENT"
    echo "   - BEARER_TOKEN (optional: change default)"
else
    echo "✅ API keys appear to be configured"
fi

echo ""
echo "🎯 Deployment Summary"
echo "===================="
echo "✅ Virtual environment ready"
echo "✅ Dependencies installed"
echo "✅ Configuration template created"

echo ""
echo "🚀 Next Steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run the application: python main.py"
echo "3. Test the API: python test_hackrx_api.py"
echo "4. Visit documentation: http://localhost:8000/docs"

echo ""
echo "📚 Quick Start Commands:"
echo "  # Start the server"
echo "  python main.py"
echo ""
echo "  # Run tests" 
echo "  python test_hackrx_api.py"
echo ""
echo "  # Check health"
echo "  curl -H \"Authorization: Bearer hackrx-secure-token-2024\" http://localhost:8000/hackrx/health"

echo ""
echo "🏁 Deployment script completed!"