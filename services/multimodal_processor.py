import asyncio
import aiohttp
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import base64
from PIL import Image
import io

from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.multi_modal_llms.together import TogetherMultiModalLLM
from config.settings import get_settings

settings = get_settings()

class MultimodalProcessor:
    """Advanced multimodal document processor for text, table, and image extraction"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize LlamaParse with advanced settings
        self.parser = LlamaParse(
            api_key=self.settings.llama_parse_api_key,
            result_type="markdown",
            include_metadata=True,
            include_page_break_separator=True
        )
        
        # Initialize vision model for image analysis
        self.vision_llm = TogetherMultiModalLLM(
            model=self.settings.together_vision_model,
            api_key=self.settings.together_api_key,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Initialize node parser for text chunking
        self.node_parser = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
    
    async def process_document_advanced(self, file_path: str) -> List[Document]:
        """Advanced document processing with multimodal extraction"""
        try:
            # Parse document with LlamaParse
            parsed_documents = self.parser.load_data(file_path)
            
            documents = []
            
            for doc in parsed_documents:
                # Extract text content
                text_content = doc.text
                
                # Extract tables if present
                tables = self._extract_tables_from_text(text_content)
                
                # Extract images if present
                images = await self._extract_images_from_document(file_path, doc)
                
                # Process images with vision model
                image_analysis = []
                if images and self.settings.enable_vision_analysis:
                    image_analysis = await self._analyze_images(images)
                
                # Combine all content
                combined_content = self._combine_multimodal_content(
                    text_content, tables, image_analysis
                )
                
                # Create LlamaIndex Document
                llama_doc = Document(
                    text=combined_content,
                    metadata={
                        "source": file_path,
                        "page": getattr(doc, 'page', 0),
                        "type": self._detect_document_type(file_path),
                        "has_tables": len(tables) > 0,
                        "has_images": len(images) > 0,
                        "table_count": len(tables),
                        "image_count": len(images)
                    }
                )
                
                documents.append(llama_doc)
            
            return documents
            
        except Exception as e:
            raise Exception(f"Advanced document processing failed: {str(e)}")
    
    def _extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from text content"""
        tables = []
        
        # Look for table patterns in markdown
        lines = text.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line and ('---' in line or line.strip().startswith('|')):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            elif in_table and '|' in line:
                current_table.append(line)
            elif in_table:
                # End of table
                if current_table:
                    table_data = self._parse_markdown_table(current_table)
                    if table_data:
                        tables.append({
                            "type": "markdown_table",
                            "data": table_data,
                            "raw": '\n'.join(current_table)
                        })
                in_table = False
                current_table = []
        
        # Handle last table
        if in_table and current_table:
            table_data = self._parse_markdown_table(current_table)
            if table_data:
                tables.append({
                    "type": "markdown_table",
                    "data": table_data,
                    "raw": '\n'.join(current_table)
                })
        
        return tables
    
    def _parse_markdown_table(self, table_lines: List[str]) -> List[Dict[str, str]]:
        """Parse markdown table into structured data"""
        if len(table_lines) < 2:
            return []
        
        # Extract headers
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split('|')[1:-1]]
        
        # Skip separator line
        data_lines = table_lines[2:]
        
        table_data = []
        for line in data_lines:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) == len(headers):
                    row = dict(zip(headers, cells))
                    table_data.append(row)
        
        return table_data
    
    async def _extract_images_from_document(self, file_path: str, doc) -> List[Dict[str, Any]]:
        """Extract images from document"""
        images = []
        
        try:
            # Check if document has image data
            if hasattr(doc, 'images') and doc.images:
                for i, img_data in enumerate(doc.images):
                    images.append({
                        "index": i,
                        "data": img_data,
                        "type": "embedded"
                    })
            
            # For PDFs, try to extract images using PyMuPDF
            if file_path.endswith('.pdf'):
                pdf_images = await self._extract_pdf_images(file_path)
                images.extend(pdf_images)
                
        except Exception as e:
            print(f"Warning: Failed to extract images: {str(e)}")
        
        return images
    
    async def _extract_pdf_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF using PyMuPDF"""
        try:
            import fitz  # PyMuPDF
            
            images = []
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append({
                            "page": page_num,
                            "index": img_index,
                            "data": img_data,
                            "type": "pdf_image",
                            "format": "png"
                        })
                    
                    pix = None
            
            doc.close()
            return images
            
        except ImportError:
            print("PyMuPDF not available for image extraction")
            return []
        except Exception as e:
            print(f"Failed to extract PDF images: {str(e)}")
            return []
    
    async def _analyze_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze images using vision model"""
        analyses = []
        
        for img in images:
            try:
                if img["type"] == "embedded" and hasattr(img["data"], "decode"):
                    # Convert bytes to base64
                    img_base64 = base64.b64encode(img["data"]).decode('utf-8')
                    
                    # Analyze with vision model
                    prompt = "Describe this image in detail, focusing on any text, tables, charts, or important visual information that might be relevant for document understanding."
                    
                    response = await self.vision_llm.acomplete(
                        prompt=prompt,
                        image=img_base64
                    )
                    
                    analyses.append({
                        "image_index": img["index"],
                        "analysis": response.text,
                        "confidence": 0.8  # Placeholder confidence
                    })
                    
            except Exception as e:
                print(f"Failed to analyze image {img.get('index', 'unknown')}: {str(e)}")
        
        return analyses
    
    def _combine_multimodal_content(
        self, 
        text: str, 
        tables: List[Dict[str, Any]], 
        image_analysis: List[Dict[str, Any]]
    ) -> str:
        """Combine text, tables, and image analysis into unified content"""
        combined = text
        
        # Add table information
        if tables:
            combined += "\n\n## Extracted Tables:\n"
            for i, table in enumerate(tables):
                combined += f"\n### Table {i+1}:\n"
                combined += table["raw"]
                combined += "\n"
        
        # Add image analysis
        if image_analysis:
            combined += "\n\n## Image Analysis:\n"
            for analysis in image_analysis:
                combined += f"\n### Image {analysis['image_index']}:\n"
                combined += analysis["analysis"]
                combined += "\n"
        
        return combined
    
    def _detect_document_type(self, file_path: str) -> str:
        """Detect document type from file path"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.docx':
            return 'docx'
        elif ext == '.doc':
            return 'doc'
        elif ext == '.eml':
            return 'email'
        else:
            return 'unknown'
    
    async def chunk_documents_optimized(self, documents: List[Document]) -> List[Document]:
        """Optimized document chunking for better retrieval"""
        chunked_docs = []
        
        for doc in documents:
            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([doc])
            
            # Convert nodes back to documents with enhanced metadata
            for i, node in enumerate(nodes):
                chunked_doc = Document(
                    text=node.text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "chunk_type": "text"
                    }
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs