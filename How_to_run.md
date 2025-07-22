# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="your-env"

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
