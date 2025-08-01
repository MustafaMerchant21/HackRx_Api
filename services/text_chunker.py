import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken

from config.settings import get_settings

settings = get_settings()

class TextChunker:
    """Enhanced text chunking service with multimodal content support"""
    
    def __init__(self):
        # Initialize LangChain text splitter with optimized settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ".",     # Sentence endings
                "!",     # Exclamation
                "?",     # Questions
                ";",     # Semicolons
                ",",     # Commas
                " ",     # Spaces
                "",      # Characters
            ]
        )
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except:
            self.tokenizer = None
    
    async def chunk_documents(self, parsed_documents: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
        """
        Main chunking function - chunks documents into smaller pieces with metadata
        Input: List of (text, metadata) tuples from document parsing
        Output: List of chunk dictionaries with enhanced metadata
        """
        all_chunks = []
        
        if not parsed_documents:
            return all_chunks
        
        # Process each document
        for doc_text, doc_metadata in parsed_documents:
            if not doc_text or not doc_text.strip():
                continue
                
            # Clean the text before chunking
            cleaned_text = self._clean_text(doc_text)
            
            # Chunk the document using LangChain
            doc_chunks = await self.chunk_text_with_langchain(cleaned_text, doc_metadata)
            
            # Add to all chunks
            all_chunks.extend(doc_chunks)
        
        # Optimize chunk boundaries for better semantic coherence
        optimized_chunks = self.optimize_chunk_boundaries([chunk["chunk_text"] for chunk in all_chunks])
        
        # Update chunks with optimized text and final metadata
        final_chunks = []
        for i, (chunk_data, optimized_text) in enumerate(zip(all_chunks, optimized_chunks)):
            chunk_data["chunk_text"] = optimized_text
            chunk_data["metadata"]["chunk_index"] = i
            chunk_data["metadata"]["total_chunks"] = len(optimized_chunks)
            chunk_data["metadata"]["char_count"] = len(optimized_text)
            chunk_data["metadata"]["token_count"] = self._count_tokens(optimized_text)
            final_chunks.append(chunk_data)
        
        return final_chunks
    
    async def chunk_text_with_langchain(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """
        Chunk single document using LangChain RecursiveCharacterTextSplitter
        Enhanced with metadata and quality scoring
        """
        if not text or not text.strip():
            return []
        
        # Create document for LangChain
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        
        # Split the document using LangChain
        try:
            split_docs = self.text_splitter.split_documents([doc])
        except Exception as e:
            print(f"Error splitting document: {e}")
            # Fallback to simple splitting if LangChain fails
            return await self._fallback_chunking(text, metadata)
        
        # Convert to our chunk format with enhanced metadata
        chunks = []
        for i, split_doc in enumerate(split_docs):
            chunk_text = split_doc.page_content.strip()
            if not chunk_text:
                continue
            
            # Calculate quality score
            quality_score = self._calculate_chunk_quality(chunk_text)
            
            # Skip very low-quality chunks
            if quality_score < 0.2:
                continue
            
            # Add enhanced metadata
            enhanced_metadata = self.add_chunk_metadata(
                chunk_text, 
                split_doc.metadata, 
                i
            )
            enhanced_metadata["quality_score"] = quality_score
            
            chunk_dict = {
                "chunk_text": chunk_text,
                "metadata": enhanced_metadata
            }
            chunks.append(chunk_dict)
        
        return chunks
    
    def add_chunk_metadata(self, chunk: str, base_metadata: Dict, chunk_index: int) -> Dict[str, Any]:
        """Add comprehensive metadata to text chunks"""
        
        # Start with base metadata
        enhanced_metadata = base_metadata.copy()
        
        # Add chunk-specific metadata
        enhanced_metadata.update({
            "chunk_index": chunk_index,
            "chunk_length": len(chunk),
            "token_count": self._count_tokens(chunk),
            "word_count": len(chunk.split()),
            "sentence_count": len(re.split(r'[.!?]+', chunk)),
            "chunk_type": "text",  # Default type
        })
        
        # Add content-based metadata
        enhanced_metadata.update({
            "has_numbers": bool(re.search(r'\d', chunk)),
            "has_dates": bool(re.search(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', chunk)),
            "has_currency": bool(re.search(r'[\$₹€£¥]\d+|\d+\s*(USD|INR|EUR|GBP|JPY)', chunk)),
            "has_percentages": bool(re.search(r'\d+%', chunk)),
            "language": "en",  # Could be enhanced with language detection
        })
        
        # Extract key phrases or keywords
        keywords = self._extract_keywords(chunk)
        enhanced_metadata["keywords"] = keywords[:10]  # Top 10 keywords
        
        # Determine content category
        content_category = self._categorize_content(chunk)
        enhanced_metadata["content_category"] = content_category
        
        return enhanced_metadata
    
    def optimize_chunk_boundaries(self, chunks: List[str]) -> List[str]:
        """Optimize chunk boundaries for better semantic coherence"""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            # Clean up the chunk
            optimized_chunk = self._optimize_single_chunk(chunk)
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def _optimize_single_chunk(self, chunk: str) -> str:
        """Optimize a single chunk for better readability and coherence"""
        
        # Remove excessive whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Fix broken sentences at the beginning
        if chunk and not chunk[0].isupper() and not chunk[0].isdigit():
            # Try to capitalize the first letter if it's likely the start of a sentence
            chunk = chunk[0].upper() + chunk[1:] if len(chunk) > 1 else chunk.upper()
        
        # Ensure the chunk ends properly
        if chunk and chunk[-1] not in '.!?':
            # If the chunk doesn't end with punctuation, try to find a good breaking point
            sentences = re.split(r'([.!?]+)', chunk)
            if len(sentences) > 1:
                # Keep complete sentences
                complete_part = ''
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        complete_part += sentences[i] + sentences[i + 1]
                if complete_part.strip():
                    chunk = complete_part.strip()
        
        return chunk.strip()
    
    async def _fallback_chunking(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Fallback chunking method if LangChain fails"""
        
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._count_tokens(potential_chunk) <= settings.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    enhanced_metadata = self.add_chunk_metadata(
                        current_chunk.strip(), 
                        metadata, 
                        chunk_index
                    )
                    
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "metadata": enhanced_metadata
                    })
                    chunk_index += 1
                
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            enhanced_metadata = self.add_chunk_metadata(
                current_chunk.strip(), 
                metadata, 
                chunk_index
            )
            
            chunks.append({
                "chunk_text": current_chunk.strip(),
                "metadata": enhanced_metadata
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before chunking"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)   # Add space between numbers and letters
        
        # Clean up multiple punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken if available"""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: approximate token count (1 token ≈ 4 characters)
        return len(text) // 4
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate quality score for a chunk (0-1)"""
        if not chunk.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor
        length = len(chunk)
        if 50 <= length <= 2000:  # Optimal length range
            score += 0.2
        elif length < 20:  # Too short
            score -= 0.3
        
        # Word count factor
        words = chunk.split()
        if len(words) >= 5:
            score += 0.1
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', chunk)
        if len(sentences) >= 2:
            score += 0.1
        
        # Check for meaningful content (not just punctuation/numbers)
        alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk) if chunk else 0
        if alpha_ratio > 0.6:
            score += 0.1
        
        # Penalize chunks with too many special characters
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in chunk) / len(chunk) if chunk else 0
        if special_char_ratio > 0.3:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when',
            'how', 'why', 'what', 'who', 'which', 'said', 'say', 'says'
        }
        
        # Filter and get unique keywords
        keywords = list(set([word for word in words if word not in stop_words and len(word) > 2]))
        
        # Return top keywords by frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = text.lower().count(word)
        
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:15]]  # Top 15 keywords
    
    def _categorize_content(self, text: str) -> str:
        """Categorize content based on keywords and patterns"""
        text_lower = text.lower()
        
        # Insurance/Policy related
        if any(word in text_lower for word in ['policy', 'premium', 'coverage', 'claim', 'insurance', 'benefit']):
            return 'insurance_policy'
        
        # Legal/Contract related
        elif any(word in text_lower for word in ['contract', 'agreement', 'clause', 'terms', 'conditions', 'legal']):
            return 'legal_contract'
        
        # Financial related
        elif any(word in text_lower for word in ['payment', 'amount', 'cost', 'price', 'fee', 'charge', 'financial']):
            return 'financial'
        
        # Medical related
        elif any(word in text_lower for word in ['medical', 'health', 'doctor', 'hospital', 'treatment', 'disease']):
            return 'medical'
        
        # Procedural
        elif any(word in text_lower for word in ['procedure', 'process', 'step', 'method', 'instructions']):
            return 'procedural'
        
        # Default
        else:
            return 'general'
 