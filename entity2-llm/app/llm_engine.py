#!/usr/bin/env python3
"""
LLM Engine Implementation for Entity 2 using Llama

This module provides the core LLM functionality using Llama models with
support for GPU acceleration, batching, and optimized inference.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import structlog

logger = structlog.get_logger()


class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for generation"""

    def __init__(self, stop_tokens: List[str], tokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.stop_token_ids = [
            tokenizer.encode(token, add_special_tokens=False)
            for token in stop_tokens
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Convert last few tokens to text and check for stop sequences
        last_tokens = input_ids[0][-10:].tolist()
        generated_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)

        for stop_token in self.stop_tokens:
            if stop_token in generated_text:
                return True

        return False


class LlamaEngine:
    """
    High-performance Llama model engine with GPU acceleration and optimization features
    """

    def __init__(
        self,
        model_path: str,
        max_context_length: int = 4096,
        max_response_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        gpu_enabled: bool = True,
        gpu_memory_fraction: float = 0.8,
        batch_size: int = 1,
        num_threads: Optional[int] = None,
        model_warmup: bool = True
    ):
        self.model_path = model_path
        self.max_context_length = max_context_length
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.gpu_enabled = gpu_enabled
        self.gpu_memory_fraction = gpu_memory_fraction
        self.batch_size = batch_size
        self.num_threads = num_threads or min(8, os.cpu_count() or 1)
        self.model_warmup = model_warmup

        # Model components
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time": 0.0,
            "cache_hits": 0,
            "warmup_completed": False
        }

        # Generation cache
        self.generation_cache = {}
        self.cache_lock = threading.Lock()

        # State tracking
        self._initialized = False
        self._ready = False

    async def initialize(self) -> None:
        """Initialize the LLM engine"""
        if self._initialized:
            logger.warning("LLM engine already initialized")
            return

        logger.info("Initializing Llama engine", model_path=self.model_path)

        try:
            # Setup device and memory
            await self._setup_device()

            # Load tokenizer
            await self._load_tokenizer()

            # Load model
            await self._load_model()

            # Setup generation configuration
            self._setup_generation_config()

            # Warmup if enabled
            if self.model_warmup:
                await self._warmup_model()

            self._initialized = True
            self._ready = True

            logger.info("Llama engine initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Llama engine", error=str(e))
            self._ready = False
            raise

    async def _setup_device(self) -> None:
        """Setup compute device and memory allocation"""
        if self.gpu_enabled and torch.cuda.is_available():
            self.device = torch.device("cuda")

            # Set GPU memory fraction
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Found {gpu_count} GPU(s)")

                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_memory = gpu_props.total_memory / (1024**3)  # GB
                    logger.info(f"GPU {i}: {gpu_props.name}, {total_memory:.1f}GB")

                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)

        else:
            self.device = torch.device("cpu")
            logger.warning("GPU not available or disabled, using CPU")

            # Optimize CPU usage
            torch.set_num_threads(self.num_threads)

    async def _load_tokenizer(self) -> None:
        """Load the tokenizer"""
        logger.info("Loading tokenizer")

        def load_tokenizer_sync():
            return AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left"  # Important for batch generation
            )

        loop = asyncio.get_event_loop()
        self.tokenizer = await loop.run_in_executor(self.executor, load_tokenizer_sync)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded successfully")

    async def _load_model(self) -> None:
        """Load the Llama model with optimizations"""
        logger.info("Loading Llama model")

        def load_model_sync():
            # Quantization configuration for memory efficiency
            quantization_config = None
            if self.gpu_enabled and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            # Model loading configuration
            model_kwargs = {
                "torch_dtype": torch.float16 if self.gpu_enabled else torch.float32,
                "device_map": "auto" if self.gpu_enabled else None,
                "quantization_config": quantization_config,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            # Move to device if not using device_map
            if not self.gpu_enabled or not model_kwargs.get("device_map"):
                model = model.to(self.device)

            # Set to evaluation mode
            model.eval()

            # Enable optimizations
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            return model

        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(self.executor, load_model_sync)

        logger.info("Model loaded successfully",
                   model_size=f"{sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B parameters")

    def _setup_generation_config(self) -> None:
        """Setup generation configuration"""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_response_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        logger.info("Generation configuration setup completed")

    async def _warmup_model(self) -> None:
        """Warm up the model with dummy input"""
        logger.info("Warming up model")

        warmup_prompts = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Can you help me with financial advice?",
            "Tell me about investment strategies.",
            "How should I manage my portfolio?"
        ]

        for prompt in warmup_prompts:
            try:
                await self.generate(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7
                )
            except Exception as e:
                logger.warning("Warmup generation failed", error=str(e))

        self.stats["warmup_completed"] = True
        logger.info("Model warmup completed")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using the Llama model

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences to stop generation

        Returns:
            Generated text
        """
        if not self._ready:
            raise RuntimeError("LLM engine not ready")

        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_response_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        stop_sequences = stop_sequences or ["</s>", "<|end|>", "\n\nHuman:", "\n\nUser:"]

        # Check cache
        cache_key = self._create_cache_key(prompt, max_tokens, temperature, top_p, top_k)
        with self.cache_lock:
            if cache_key in self.generation_cache:
                self.stats["cache_hits"] += 1
                return self.generation_cache[cache_key]

        start_time = time.time()

        try:
            # Generate text
            def generate_sync():
                return self._generate_sync(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences
                )

            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(self.executor, generate_sync)

            # Update statistics
            generation_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_inference_time"] += generation_time
            self.stats["total_tokens_generated"] += len(self.tokenizer.encode(generated_text))

            # Cache result
            with self.cache_lock:
                if len(self.generation_cache) < 1000:  # Limit cache size
                    self.generation_cache[cache_key] = generated_text

            logger.info("Text generated successfully",
                       generation_time=f"{generation_time:.2f}s",
                       input_length=len(prompt),
                       output_length=len(generated_text))

            return generated_text

        except Exception as e:
            logger.error("Failed to generate text", error=str(e))
            raise

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: List[str]
    ) -> str:
        """Synchronous text generation"""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length - max_tokens,
            padding=False
        ).to(self.device)

        # Setup stopping criteria
        stopping_criteria = StoppingCriteriaList([
            CustomStoppingCriteria(stop_sequences, self.tokenizer)
        ])

        # Update generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=False
            )

        # Decode output
        generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up output
        generated_text = self._post_process_output(generated_text, stop_sequences)

        return generated_text

    def _post_process_output(self, text: str, stop_sequences: List[str]) -> str:
        """Post-process generated text"""
        # Remove stop sequences
        for stop_seq in stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0]

        # Clean up whitespace
        text = text.strip()

        # Remove incomplete sentences at the end
        if text and not text.endswith(('.', '!', '?', ':')):
            sentences = text.split('.')
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'

        return text

    def _create_cache_key(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        """Create cache key for generation parameters"""
        import hashlib

        key_data = f"{prompt}|{max_tokens}|{temperature}|{top_p}|{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """Generate text for multiple prompts in batch"""
        if not self._ready:
            raise RuntimeError("LLM engine not ready")

        # For now, process sequentially
        # TODO: Implement true batch processing
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                results.append(result)
            except Exception as e:
                logger.error("Batch generation failed for prompt", error=str(e))
                results.append("")

        return results

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text (if model supports it)"""
        # This would require a different model or additional embedding model
        # For now, return empty embedding
        logger.warning("Text embedding not implemented")
        return []

    def is_ready(self) -> bool:
        """Check if the engine is ready"""
        return self._ready

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        avg_inference_time = (
            self.stats["total_inference_time"] / max(self.stats["total_requests"], 1)
        )

        avg_tokens_per_request = (
            self.stats["total_tokens_generated"] / max(self.stats["total_requests"], 1)
        )

        return {
            "total_requests": self.stats["total_requests"],
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "average_inference_time": avg_inference_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1),
            "warmup_completed": self.stats["warmup_completed"],
            "model_ready": self._ready,
            "device": str(self.device) if self.device else None,
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

    def clear_cache(self) -> None:
        """Clear generation cache"""
        with self.cache_lock:
            self.generation_cache.clear()
        logger.info("Generation cache cleared")

    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self._ready:
                return False

            # Simple generation test
            test_prompt = "Hello"
            result = await self.generate(test_prompt, max_tokens=5, temperature=0.1)

            return len(result) > 0

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up LLM engine")

        try:
            # Clear cache
            self.clear_cache()

            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Shutdown executor
            self.executor.shutdown(wait=True)

            # Reset state
            self._ready = False

            logger.info("LLM engine cleanup completed")

        except Exception as e:
            logger.error("Error during cleanup", error=str(e))

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False)
        except:
            pass
