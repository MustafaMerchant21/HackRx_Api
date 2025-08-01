import asyncio
import aiohttp
import aiofiles
import tempfile
import os
import io
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from llama_parse import LlamaParse
import pymupdf4llm
import pdfplumber
import camelot
from PIL import Image
import pytesseract

from config.settings import get_settings

settings = get_settings()

class LlamaParseService:
    """Advanced document parsing service using LlamaParse and multimodal extraction"""
    
    def __init__(self):
        self.llamaparse = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type=settings.llamaparse_result_type,
            verbose=settings.llamaparse_verbose,
            language=settings.llamaparse_language,
            num_workers=4  # Parallel processing
        )
        
    async def download_document(self, url: str) -> str:
        """Download document from URL to temporary file"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                # Create temporary file with proper extension
                file_extension = self._get_file_extension(url, response.headers.get('content-type'))
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=file_extension
                )
                
                async with aiofiles.open(temp_file.name, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                return temp_file.name
    
    async def parse_documents(self, document_urls: Union[str, List[str]]) -> Dict[str, Any]:
        """Parse multiple documents with advanced multimodal extraction"""
        if isinstance(document_urls, str):
            document_urls = [document_urls]
        
        # Download all documents in parallel
        download_tasks = [self.download_document(url) for url in document_urls]
        temp_files = await asyncio.gather(*download_tasks)
        
        try:
            # Parse all documents in parallel
            parse_tasks = [self._parse_single_document(file_path) for file_path in temp_files]
            parsed_results = await asyncio.gather(*parse_tasks)
            
            # Combine results
            combined_result = {
                'documents': [],
                'total_chunks': 0,
                'processing_info': {
                    'documents_processed': len(parsed_results),
                    'extraction_methods': ['llamaparse', 'pymupdf', 'table_extraction', 'ocr']
                }
            }
            
            for i, result in enumerate(parsed_results):
                result['source_url'] = document_urls[i]
                combined_result['documents'].append(result)
                combined_result['total_chunks'] += len(result.get('chunks', []))
            
            return combined_result
            
        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    async def _parse_single_document(self, file_path: str) -> Dict[str, Any]:
        """Parse a single document with multiple extraction strategies"""
        file_ext = Path(file_path).suffix.lower()
        
        result = {
            'file_path': file_path,
            'file_type': file_ext,
            'chunks': [],
            'tables': [],
            'images': [],
            'metadata': {}
        }
        
        try:
            if file_ext == '.pdf':
                result = await self._parse_pdf_multimodal(file_path)
            elif file_ext in ['.docx', '.doc']:
                result = await self._parse_docx(file_path)
            else:
                # Fallback to basic text extraction
                result = await self._parse_generic(file_path)
                
        except Exception as e:
            result['error'] = str(e)
            result['chunks'] = [f"Error parsing document: {str(e)}"]
        
        return result
    
    async def _parse_pdf_multimodal(self, file_path: str) -> Dict[str, Any]:
        """Advanced PDF parsing with text, tables, and images"""
        result = {
            'file_path': file_path,
            'file_type': '.pdf',
            'chunks': [],
            'tables': [],
            'images': [],
            'metadata': {}
        }
        
        # Strategy 1: LlamaParse for high-quality extraction
        try:
            llamaparse_result = await asyncio.wait_for(
                asyncio.to_thread(self.llamaparse.load_data, file_path),
                timeout=settings.pdf_processing_timeout
            )
            
            for doc in llamaparse_result:
                if hasattr(doc, 'text') and doc.text.strip():
                    result['chunks'].append({
                        'content': doc.text,
                        'method': 'llamaparse',
                        'metadata': getattr(doc, 'metadata', {})
                    })
        except Exception as e:
            print(f"LlamaParse failed: {e}")
        
        # Strategy 2: PyMuPDF for structured extraction
        try:
            pymupdf_result = await asyncio.to_thread(
                pymupdf4llm.to_markdown, 
                file_path
            )
            
            if pymupdf_result and len(result['chunks']) == 0:
                result['chunks'].append({
                    'content': pymupdf_result,
                    'method': 'pymupdf4llm',
                    'metadata': {}
                })
        except Exception as e:
            print(f"PyMuPDF4LLM failed: {e}")
        
        # Strategy 3: Table extraction
        result['tables'] = await self._extract_tables_from_pdf(file_path)
        
        # Strategy 4: Image and OCR extraction
        result['images'] = await self._extract_images_from_pdf(file_path)
        
        # Fallback: Basic PyMuPDF if nothing else worked
        if not result['chunks'] and not result['tables'] and not result['images']:
            doc = fitz.open(file_path)
            text_content = ""
            for page in doc:
                text_content += page.get_text()
            doc.close()
            
            if text_content.strip():
                result['chunks'].append({
                    'content': text_content,
                    'method': 'pymupdf_fallback',
                    'metadata': {}
                })
        
        return result
    
    async def _extract_tables_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using multiple strategies"""
        tables = []
        
        if settings.table_extraction_strategy in ['pdfplumber', 'hybrid']:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        for table_idx, table_data in enumerate(page_tables):
                            if table_data and len(table_data) > 1:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                tables.append({
                                    'content': df.to_string(),
                                    'csv_content': df.to_csv(index=False),
                                    'method': 'pdfplumber',
                                    'page': page_num + 1,
                                    'table_index': table_idx,
                                    'shape': df.shape
                                })
            except Exception as e:
                print(f"PDFPlumber table extraction failed: {e}")
        
        if settings.table_extraction_strategy in ['camelot', 'hybrid']:
            try:
                camelot_tables = camelot.read_pdf(file_path, pages='all')
                for i, table in enumerate(camelot_tables):
                    if not table.df.empty:
                        tables.append({
                            'content': table.df.to_string(),
                            'csv_content': table.df.to_csv(index=False),
                            'method': 'camelot',
                            'table_index': i,
                            'accuracy': table.accuracy,
                            'shape': table.df.shape
                        })
            except Exception as e:
                print(f"Camelot table extraction failed: {e}")
        
        return tables
    
    async def _extract_images_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract and OCR images from PDF"""
        images = []
        
        if not settings.enable_ocr:
            return images
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Perform OCR
                            img_pil = Image.open(io.BytesIO(img_data))
                            if settings.image_preprocessing:
                                img_pil = self._preprocess_image(img_pil)
                            
                            ocr_text = pytesseract.image_to_string(img_pil)
                            
                            if ocr_text.strip():
                                images.append({
                                    'content': ocr_text,
                                    'method': 'ocr',
                                    'page': page_num + 1,
                                    'image_index': img_index,
                                    'confidence': self._calculate_ocr_confidence(ocr_text)
                                })
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Image extraction failed for image {img_index}: {e}")
            
            doc.close()
            
        except Exception as e:
            print(f"PDF image extraction failed: {e}")
        
        return images
    
    async def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """Parse DOCX documents"""
        try:
            docs = await asyncio.to_thread(self.llamaparse.load_data, file_path)
            
            result = {
                'file_path': file_path,
                'file_type': '.docx',
                'chunks': [],
                'tables': [],
                'images': [],
                'metadata': {}
            }
            
            for doc in docs:
                if hasattr(doc, 'text') and doc.text.strip():
                    result['chunks'].append({
                        'content': doc.text,
                        'method': 'llamaparse',
                        'metadata': getattr(doc, 'metadata', {})
                    })
            
            return result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'file_type': '.docx',
                'chunks': [f"Error parsing DOCX: {str(e)}"],
                'tables': [],
                'images': [],
                'metadata': {},
                'error': str(e)
            }
    
    async def _parse_generic(self, file_path: str) -> Dict[str, Any]:
        """Generic file parsing fallback"""
        try:
            docs = await asyncio.to_thread(self.llamaparse.load_data, file_path)
            
            result = {
                'file_path': file_path,
                'file_type': Path(file_path).suffix,
                'chunks': [],
                'tables': [],
                'images': [],
                'metadata': {}
            }
            
            for doc in docs:
                if hasattr(doc, 'text') and doc.text.strip():
                    result['chunks'].append({
                        'content': doc.text,
                        'method': 'llamaparse',
                        'metadata': getattr(doc, 'metadata', {})
                    })
            
            return result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'file_type': Path(file_path).suffix,
                'chunks': [f"Error parsing file: {str(e)}"],
                'tables': [],
                'images': [],
                'metadata': {},
                'error': str(e)
            }
    
    def _get_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """Determine file extension from URL or content type"""
        if url.lower().endswith('.pdf'):
            return '.pdf'
        elif url.lower().endswith('.docx'):
            return '.docx'
        elif url.lower().endswith('.doc'):
            return '.doc'
        elif content_type:
            if 'pdf' in content_type:
                return '.pdf'
            elif 'word' in content_type or 'docx' in content_type:
                return '.docx'
        
        return '.pdf'  # Default to PDF
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR"""
        # Resize if too large
        if img.size[0] > settings.max_image_size[0] or img.size[1] > settings.max_image_size[1]:
            img.thumbnail(settings.max_image_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale for better OCR
        img = img.convert('L')
        
        return img
    
    def _calculate_ocr_confidence(self, text: str) -> float:
        """Simple confidence calculation based on text characteristics"""
        if not text.strip():
            return 0.0
        
        # Basic heuristics
        word_count = len(text.split())
        char_count = len(text)
        
        if word_count == 0:
            return 0.0
        
        avg_word_length = char_count / word_count
        
        # Reasonable word length suggests good OCR
        if 2 <= avg_word_length <= 10:
            return min(0.9, 0.5 + (word_count * 0.05))
        
        return 0.3