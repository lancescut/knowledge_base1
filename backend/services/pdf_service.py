"""
PDF processing service using unstructured.io for technical document extraction.
"""

import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from unstructured.documents.elements import Title, NarrativeText, Text, Table, Image

from core.config import settings
from core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class PDFExtractionService:
    """Service for extracting and processing PDF documents."""
    
    def __init__(self):
        """Initialize the PDF extraction service."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.supported_file_types = ['.pdf']
        
    def _extract_pdf_content(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract content from PDF using unstructured.io (synchronous)."""
        try:
            logger.info(f"Starting PDF extraction: {file_path}")
            
            # Partition PDF with enhanced settings for technical content
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",  # High resolution for better text extraction
                infer_table_structure=True,  # Extract table information
                extract_images_in_pdf=True,  # Extract images for technical diagrams
                include_page_breaks=True,  # Maintain page structure
                languages=["eng"],  # English language processing
                chunking_strategy="by_title",  # Chunk by titles for better structure
                max_characters=1200,  # Increased for technical content
                new_after_n_chars=1000,  # Larger chunks for technical docs
                combine_text_under_n_chars=150,  # Combine small fragments
                overlap=50,  # Add overlap for context preservation
                pdf_infer_table_structure=True,  # Enhanced table detection
                extract_image_block_types=["Image", "Table"],  # Extract specific block types
                extract_image_block_to_payload=False,  # Don't include raw image data
                extract_image_block_output_dir=None,  # Don't save images to disk
                model_name="yolox"  # Use YOLOX model for better layout detection
            )
            
            logger.info(f"Extracted {len(elements)} elements from PDF")
            
            # Process and clean elements
            processed_elements = []
            
            for element in elements:
                # Clean the text
                if hasattr(element, 'text') and element.text:
                    cleaned_text = clean_extra_whitespace(element.text)
                    cleaned_text = group_broken_paragraphs(cleaned_text)
                    
                    # Skip very short elements (likely noise)
                    if len(cleaned_text.strip()) < 10:
                        continue
                    
                    element_data = {
                        "text": cleaned_text,
                        "type": element.__class__.__name__,
                        "metadata": {
                            "page_number": getattr(element.metadata, 'page_number', None),
                            "filename": getattr(element.metadata, 'filename', None),
                            "file_directory": getattr(element.metadata, 'file_directory', None),
                            "coordinates": getattr(element.metadata, 'coordinates', None),
                        }
                    }
                    
                    # Enhanced handling for different element types with technical focus
                    if isinstance(element, Title):
                        element_data["metadata"]["is_title"] = True
                        element_data["metadata"]["importance"] = "high"
                        element_data["metadata"]["content_type"] = "heading"
                        # Detect technical title patterns
                        title_lower = cleaned_text.lower()
                        if any(keyword in title_lower for keyword in ['api', 'function', 'method', 'class', 'algorithm']):
                            element_data["metadata"]["technical_importance"] = "critical"
                    elif isinstance(element, Table):
                        element_data["metadata"]["is_table"] = True
                        element_data["metadata"]["importance"] = "high"  # Tables are crucial in technical docs
                        element_data["metadata"]["content_type"] = "structured_data"
                        # Extract enhanced table information
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
                            element_data["metadata"]["table_html"] = element.metadata.text_as_html
                        # Detect technical table content
                        if any(keyword in cleaned_text.lower() for keyword in ['parameter', 'argument', 'return', 'error', 'status']):
                            element_data["metadata"]["technical_importance"] = "critical"
                    elif isinstance(element, NarrativeText):
                        element_data["metadata"]["importance"] = "medium"
                        element_data["metadata"]["content_type"] = "prose"
                        # Detect code blocks or technical content in narrative text
                        if any(indicator in cleaned_text for indicator in ['```', 'def ', 'class ', 'import ', 'function(', 'return ']):
                            element_data["metadata"]["contains_code"] = True
                            element_data["metadata"]["importance"] = "high"
                        # Detect technical concepts
                        if any(keyword in cleaned_text.lower() for keyword in ['implementation', 'configuration', 'deployment', 'architecture']):
                            element_data["metadata"]["technical_importance"] = "high"
                    elif isinstance(element, Image):
                        element_data["metadata"]["is_image"] = True
                        element_data["metadata"]["content_type"] = "visual"
                        element_data["metadata"]["importance"] = "medium"
                        # Technical diagrams are often important
                        if any(keyword in cleaned_text.lower() for keyword in ['diagram', 'flowchart', 'architecture', 'workflow']):
                            element_data["metadata"]["technical_importance"] = "high"
                    else:
                        element_data["metadata"]["importance"] = "low"
                        element_data["metadata"]["content_type"] = "other"
                    
                    # Additional technical content detection
                    if any(pattern in cleaned_text.lower() for pattern in [
                        'example:', 'note:', 'warning:', 'important:', 'see also:',
                        'prerequisite', 'requirement', 'step 1', 'step 2'
                    ]):
                        element_data["metadata"]["is_instructional"] = True
                        element_data["metadata"]["importance"] = "high"
                    
                    processed_elements.append(element_data)
            
            logger.info(f"Processed {len(processed_elements)} valid elements")
            return processed_elements
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to extract PDF content: {str(e)}")
    
    async def extract_from_file(self, file_data: BinaryIO, filename: str) -> Dict[str, Any]:
        """Extract text from uploaded PDF file."""
        try:
            # Validate file type
            if not filename.lower().endswith('.pdf'):
                raise DocumentProcessingError(f"Unsupported file type. Expected PDF, got: {filename}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # Write uploaded file to temporary location
                content = file_data.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Extract content asynchronously
                loop = asyncio.get_event_loop()
                elements = await loop.run_in_executor(
                    self.executor, 
                    self._extract_pdf_content, 
                    temp_file_path
                )
                
                # Calculate statistics
                total_chars = sum(len(element["text"]) for element in elements)
                pages = set(element["metadata"].get("page_number") for element in elements)
                page_count = len([p for p in pages if p is not None])
                
                result = {
                    "document_id": str(uuid.uuid4()),
                    "filename": filename,
                    "file_size": len(content),
                    "page_count": page_count,
                    "element_count": len(elements),
                    "total_characters": total_chars,
                    "elements": elements,
                    "extraction_metadata": {
                        "extraction_time": datetime.utcnow().isoformat(),
                        "method": "unstructured.partition_pdf",
                        "strategy": "hi_res",
                        "language": "eng"
                    }
                }
                
                logger.info(f"Successfully extracted PDF: {filename} ({len(elements)} elements, {total_chars} chars)")
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
                    
        except Exception as e:
            logger.error(f"File extraction failed: {str(e)}")
            raise DocumentProcessingError(f"File extraction failed: {str(e)}")
    
    def get_element_by_type(self, elements: List[Dict[str, Any]], element_type: str) -> List[Dict[str, Any]]:
        """Filter elements by type."""
        return [element for element in elements if element["type"] == element_type]
    
    def get_elements_by_page(self, elements: List[Dict[str, Any]], page_number: int) -> List[Dict[str, Any]]:
        """Filter elements by page number."""
        return [
            element for element in elements 
            if element["metadata"].get("page_number") == page_number
        ]
    
    def get_text_content(self, elements: List[Dict[str, Any]]) -> str:
        """Extract plain text from all elements."""
        return "\n\n".join(element["text"] for element in elements)
    
    def get_structured_content(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize content by type and importance."""
        structured = {
            "titles": self.get_element_by_type(elements, "Title"),
            "tables": self.get_element_by_type(elements, "Table"),
            "narrative": self.get_element_by_type(elements, "NarrativeText"),
            "other": [
                element for element in elements 
                if element["type"] not in ["Title", "Table", "NarrativeText"]
            ]
        }
        
        # Sort by page number within each category
        for category in structured.values():
            category.sort(key=lambda x: x["metadata"].get("page_number", 0))
        
        return structured
    
    def extract_code_blocks(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract code blocks from document elements."""
        code_blocks = []
        
        for element in elements:
            text = element.get("text", "")
            
            # Look for code block patterns
            if (element["metadata"].get("contains_code") or 
                any(pattern in text for pattern in ['```', 'def ', 'class ', 'function ', 'import '])):
                
                code_blocks.append({
                    "text": text,
                    "metadata": {
                        **element["metadata"],
                        "content_type": "code",
                        "importance": "critical"
                    }
                })
        
        return code_blocks
    
    def extract_technical_terms(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Extract technical terms and concepts from elements."""
        import re
        
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',               # snake_case
            r'\b[A-Z_]{2,}\b',                   # CONSTANTS
            r'\b\w+\(\)',                        # function()
            r'\b\w+\.\w+\b',                     # object.method
        ]
        
        terms = set()
        
        for element in elements:
            text = element.get("text", "")
            
            for pattern in technical_patterns:
                matches = re.findall(pattern, text)
                terms.update(matches)
        
        return sorted(list(terms))
    
    def get_document_outline(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract document outline from title elements."""
        outline = []
        
        for element in elements:
            if element["metadata"].get("is_title"):
                outline.append({
                    "title": element["text"],
                    "page": element["metadata"].get("page_number"),
                    "importance": element["metadata"].get("technical_importance", "normal"),
                    "level": self._detect_heading_level(element["text"])
                })
        
        return outline
    
    def _detect_heading_level(self, title: str) -> int:
        """Detect heading level based on content and format."""
        title_lower = title.lower().strip()
        
        # Level 1: Major sections
        if any(keyword in title_lower for keyword in ['introduction', 'overview', 'getting started', 'conclusion']):
            return 1
        
        # Level 2: Main topics
        if any(keyword in title_lower for keyword in ['api', 'configuration', 'installation', 'examples']):
            return 2
        
        # Level 3: Subtopics
        if any(keyword in title_lower for keyword in ['method', 'function', 'parameter', 'example']):
            return 3
        
        # Default level
        return 2
    
    def get_advanced_statistics(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get advanced statistics about the document content."""
        stats = {
            "total_elements": len(elements),
            "content_types": {},
            "importance_levels": {},
            "technical_elements": 0,
            "code_elements": 0,
            "table_elements": 0,
            "image_elements": 0,
            "instructional_elements": 0
        }
        
        for element in elements:
            metadata = element["metadata"]
            
            # Count content types
            content_type = metadata.get("content_type", "unknown")
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            # Count importance levels
            importance = metadata.get("importance", "unknown")
            stats["importance_levels"][importance] = stats["importance_levels"].get(importance, 0) + 1
            
            # Count special elements
            if metadata.get("technical_importance") == "critical":
                stats["technical_elements"] += 1
            if metadata.get("contains_code"):
                stats["code_elements"] += 1
            if metadata.get("is_table"):
                stats["table_elements"] += 1
            if metadata.get("is_image"):
                stats["image_elements"] += 1
            if metadata.get("is_instructional"):
                stats["instructional_elements"] += 1
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on PDF service."""
        try:
            # Test with minimal PDF processing capability
            return {
                "status": "healthy",
                "supported_formats": self.supported_file_types,
                "features": [
                    "high_resolution_extraction",
                    "enhanced_table_detection",
                    "image_extraction",
                    "page_awareness",
                    "element_classification",
                    "text_cleaning",
                    "technical_content_detection",
                    "code_block_extraction",
                    "document_outline_generation",
                    "advanced_analytics"
                ],
                "model": "yolox",
                "strategy": "hi_res"
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global PDF service instance
pdf_service = PDFExtractionService() 