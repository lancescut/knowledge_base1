"""
Qwen generation service for AI-powered answer generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from core.config import settings
from core.exceptions import GenerationError

logger = logging.getLogger(__name__)


class QwenGenerationService:
    """Service for generating answers using Qwen2-7B-Instruct model."""
    
    def __init__(self):
        """Initialize the generation service."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Model configuration
        if settings.ENVIRONMENT == "development":
            # Use smaller model for development
            self.model_name = "microsoft/DialoGPT-medium"
            self.max_length = 512
        else:
            # Production: Use Qwen2-7B-Instruct
            self.model_name = "Qwen/Qwen2-7B-Instruct"
            self.max_length = 2048
        
        self.generation_config = None
        
    async def initialize(self):
        """Initialize the generation model and tokenizer."""
        try:
            logger.info(f"🚀 Initializing generation model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token="<|endoftext|>"
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                load_in_8bit=True if self.device == "cuda" else False  # Memory optimization
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set generation configuration optimized for technical content
            self.generation_config = GenerationConfig(
                max_new_tokens=768,  # Increased for detailed technical explanations
                temperature=0.3,     # Lower temperature for more focused responses
                top_p=0.85,          # Slightly more focused sampling
                top_k=40,            # Limit vocabulary for technical precision
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,  # Prevent repetition in technical explanations
                length_penalty=1.0,       # Balanced length penalty
                early_stopping=True       # Stop when appropriate
            )
            
            logger.info("✅ Generation model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize generation model: {e}")
            raise GenerationError(f"Model initialization failed: {str(e)}")
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of technical query to optimize prompt selection."""
        query_lower = query.lower()
        
        # Code-related queries
        if any(keyword in query_lower for keyword in ['code', 'function', 'class', 'method', 'implementation', 'syntax', 'debug', 'error']):
            return "code"
        
        # API/Integration queries
        elif any(keyword in query_lower for keyword in ['api', 'endpoint', 'integration', 'request', 'response', 'http', 'rest']):
            return "api"
        
        # Configuration/Setup queries
        elif any(keyword in query_lower for keyword in ['config', 'setup', 'install', 'configure', 'environment', 'deploy']):
            return "setup"
        
        # Conceptual/Explanation queries
        elif any(keyword in query_lower for keyword in ['what is', 'how does', 'explain', 'concept', 'difference', 'compare']):
            return "concept"
        
        # Troubleshooting queries
        elif any(keyword in query_lower for keyword in ['troubleshoot', 'problem', 'issue', 'fix', 'resolve', 'not working']):
            return "troubleshoot"
        
        # Default to general
        else:
            return "general"
    
    def _create_rag_prompt(self, query: str, contexts: List[str], max_contexts: int = 3) -> str:
        """Create an optimized RAG prompt based on query type and technical content."""
        # Limit contexts to avoid token overflow
        limited_contexts = contexts[:max_contexts]
        query_type = self._detect_query_type(query)
        
        # Format contexts with better structure
        context_text = "\n".join([f"--- Document {i+1} ---\n{ctx}" for i, ctx in enumerate(limited_contexts)])
        
        # Base system prompt
        system_prompt = "You are an expert technical assistant specializing in software documentation and technical content analysis."
        
        # Query-type specific prompts
        if query_type == "code":
            specific_instructions = """
- Provide concrete code examples when available in the context
- Explain syntax and implementation details clearly
- Include best practices and common pitfalls if mentioned
- Format code blocks properly with syntax highlighting hints
- Reference specific functions, classes, or methods from the context"""
        
        elif query_type == "api":
            specific_instructions = """
- Include specific endpoint URLs, methods, and parameters
- Show request/response examples from the context
- Explain authentication and headers if mentioned
- Provide curl examples or code snippets when available
- Mention rate limits, error codes, and status responses"""
        
        elif query_type == "setup":
            specific_instructions = """
- Provide step-by-step installation or configuration instructions
- Include specific commands, file paths, and settings
- Mention prerequisites and dependencies
- Highlight platform-specific considerations
- Include troubleshooting tips for common setup issues"""
        
        elif query_type == "concept":
            specific_instructions = """
- Provide clear, comprehensive explanations of concepts
- Use analogies or examples to clarify complex ideas
- Explain relationships between different components
- Include benefits, use cases, and limitations
- Reference related concepts mentioned in the context"""
        
        elif query_type == "troubleshoot":
            specific_instructions = """
- Identify potential root causes based on the context
- Provide specific diagnostic steps and commands
- Include common solutions and workarounds
- Mention relevant error messages or symptoms
- Suggest prevention strategies if available"""
        
        else:  # general
            specific_instructions = """
- Provide comprehensive and accurate information
- Include relevant technical details and examples
- Explain concepts clearly for technical audiences
- Reference specific sections or features mentioned
- Offer practical implementation guidance"""
        
        prompt = f"""{system_prompt}

Based on the following technical documentation, answer the user's question with precision and depth.

=== CONTEXT DOCUMENTS ===
{context_text}
=== END CONTEXT ===

USER QUESTION: {query}

RESPONSE GUIDELINES:{specific_instructions}
- Base your answer primarily on the provided context documents
- If context is insufficient, clearly state what additional information would be needed
- Cite specific documents when referencing information (e.g., "According to Document 2...")
- Maintain technical accuracy and provide actionable information
- Structure your response clearly with headers or bullet points when appropriate

ANSWER:"""
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the model (synchronous)."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            if prompt in response:
                generated_text = response[len(prompt):].strip()
            else:
                generated_text = response.strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise GenerationError(f"Generation failed: {str(e)}")
    
    def _analyze_context_quality(self, contexts: List[str]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of retrieved contexts."""
        if not contexts:
            return {"quality": "none", "total_length": 0, "avg_length": 0, "has_code": False}
        
        total_length = sum(len(ctx) for ctx in contexts)
        avg_length = total_length / len(contexts)
        
        # Check for code presence
        has_code = any(any(indicator in ctx.lower() for indicator in 
                          ['def ', 'class ', 'function', '```', 'import ', 'return ', '{', '}', 'const ', 'var ']) 
                     for ctx in contexts)
        
        # Quality assessment based on length and content diversity
        if avg_length > 500 and len(contexts) >= 3:
            quality = "high"
        elif avg_length > 200 and len(contexts) >= 2:
            quality = "medium"
        else:
            quality = "low"
        
        return {
            "quality": quality,
            "total_length": total_length,
            "avg_length": round(avg_length),
            "has_code": has_code,
            "context_count": len(contexts)
        }
    
    async def generate_answer(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """Generate an answer using RAG approach with adaptive optimization."""
        if not self.model or not self.tokenizer:
            raise GenerationError("Generation model not initialized")
        
        try:
            start_time = time.time()
            
            # Analyze context quality for adaptive generation
            context_analysis = self._analyze_context_quality(contexts)
            
            # Adjust max_contexts based on quality
            if context_analysis["quality"] == "high":
                max_contexts = 4  # Use more contexts for high quality
            elif context_analysis["quality"] == "medium":
                max_contexts = 3  # Standard amount
            else:
                max_contexts = 2  # Fewer contexts for low quality
            
            # Create optimized RAG prompt
            prompt = self._create_rag_prompt(query, contexts, max_contexts)
            logger.info(f"Generated {context_analysis['quality']} quality prompt with {len(contexts)} contexts")
            
            # Generate answer asynchronously
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(self.executor, self._generate_text, prompt)
            
            generation_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "query": query,
                "contexts_used": min(len(contexts), max_contexts),
                "context_analysis": context_analysis,
                "generation_time": round(generation_time, 2),
                "model": self.model_name,
                "timestamp": time.time(),
                "prompt_type": self._detect_query_type(query)
            }
            
            logger.info(f"Generated {context_analysis['quality']} quality answer in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise GenerationError(f"Answer generation failed: {str(e)}")
    
    async def generate_simple(self, prompt: str) -> str:
        """Generate a simple text response."""
        if not self.model or not self.tokenizer:
            raise GenerationError("Generation model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(self.executor, self._generate_text, prompt)
            return response
            
        except Exception as e:
            logger.error(f"Simple generation failed: {e}")
            raise GenerationError(f"Generation failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on generation service."""
        try:
            if not self.model or not self.tokenizer:
                return {"status": "unhealthy", "error": "Model not initialized"}
            
            # Test generation with simple prompt
            test_prompt = "Hello, how are you?"
            test_inputs = self.tokenizer.encode(test_prompt, return_tensors="pt")
            
            # Quick generation test
            with torch.no_grad():
                test_outputs = self.model.generate(
                    test_inputs.to(self.device),
                    max_new_tokens=10,
                    do_sample=False
                )
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "device": self.device,
                "max_length": self.max_length,
                "test_completed": True
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global generation service instance
generation_service = QwenGenerationService() 