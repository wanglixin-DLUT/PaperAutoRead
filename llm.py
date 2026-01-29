import os
import json
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Provider configurations: base_url and env_var for API key
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "base_url_env_var": "OPENROUTER_API_BASE_URL",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "QWEN_API_KEY",
        "base_url_env_var": "QWEN_API_BASE_URL",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_var": "DEEPSEEK_API_KEY",
        "base_url_env_var": "DEEPSEEK_API_BASE_URL",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "base_url_env_var": "OPENAI_API_BASE_URL",
    },
    "gemini": {
        "base_url": None,  # Uses native SDK
        "env_var": "GEMINI_API_KEY",
        "base_url_env_var": "GEMINI_API_BASE_URL",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_var": "ZHIPUAI_API_KEY",
        "base_url_env_var": "ZHIPUAI_API_BASE_URL",
    },
}


class TokenUsageTracker:
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.usage_records: List[Dict] = []
        self.total_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0,
        }
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
    def add_record(self, 
                   provider: str,
                   model: str,
                   prompt_tokens: int,
                   completion_tokens: int,
                   total_tokens: int,
                   agent_name: str = "unknown"):
        record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "agent_name": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.usage_records.append(record)
        self.total_stats["total_prompt_tokens"] += prompt_tokens
        self.total_stats["total_completion_tokens"] += completion_tokens
        self.total_stats["total_tokens"] += total_tokens
        self.total_stats["total_calls"] += 1
        
    def export_to_file(self, file_path: Optional[str] = None):
        output_file = file_path or self.log_file
        export_data = {
            "export_time": datetime.now().isoformat(),
            "summary": self.total_stats,
            "records": self.usage_records,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        return output_file
    
    def print_summary(self):
        print("\n" + "="*60)
        print("Token Usage Summary")
        print("="*60)
        print(f"Total API calls: {self.total_stats['total_calls']}")
        print(f"Total input tokens: {self.total_stats['total_prompt_tokens']:,}")
        print(f"Total output tokens: {self.total_stats['total_completion_tokens']:,}")
        print(f"Total tokens: {self.total_stats['total_tokens']:,}")
        print("="*60 + "\n")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        provider: str = "openrouter",
        base_url: Optional[str] = None,
        default_model: str = "google/gemini-3-flash-preview",
        request_timeout: int = 600,
        token_tracker: Optional[TokenUsageTracker] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.provider = provider.lower()
        self.default_model = default_model
        self.request_timeout = request_timeout
        self.token_tracker = token_tracker
        self.current_agent_name = "unknown"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get provider config
        config = PROVIDER_CONFIGS.get(self.provider, PROVIDER_CONFIGS["openrouter"])
        
        if self.provider == "gemini":
            # Use native Google Generative AI SDK
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = None
            self._http_client = None
        else:
            # Use OpenAI-compatible API
            import httpx
            from openai import OpenAI
            
            self._genai = None
            
            # Set up headers (OpenRouter needs special headers)
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": site_url or "http://localhost",
                    "X-Title": site_name or "PaperAutoRead Assistant",
                }

            self._http_client = httpx.Client(
                trust_env=True, 
                timeout=httpx.Timeout(request_timeout, connect=30.0),
            )
            

            base_url_env_var = config.get("base_url_env_var")
            env_base_url = os.environ.get(base_url_env_var) if base_url_env_var else None
            effective_base_url = base_url or env_base_url or config["base_url"]
            
            self._client = OpenAI(
                base_url=effective_base_url,
                api_key=api_key,
                http_client=self._http_client,
                default_headers=extra_headers,
            )

    def generate(
        self,
        instructions: Optional[str],
        input_text: str,
        model: Optional[str] = None,
        enable_reasoning: bool = True,
        temperature: float = 0.6,
        agent_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        model_name = model or self.default_model
        if agent_name:
            self.current_agent_name = agent_name

        final_text = ""
        reasoning_text = ""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "gemini":
                    # Use native Gemini SDK
                    final_text, reasoning_text = self._generate_gemini(
                        instructions, input_text, model_name, temperature
                    )
                else:
                    # Use OpenAI-compatible API
                    final_text, reasoning_text = self._generate_openai_compatible(
                        instructions, input_text, model_name, temperature
                    )
                

                rate_limit_keywords = ["并发", "rate limit", "too many requests", "quota exceeded", "限流"]
                if any(keyword in final_text.lower() for keyword in rate_limit_keywords):
                    raise Exception(f"Rate limit detected in response: {final_text[:100]}...")
                
                return final_text or "", reasoning_text or ""
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    time.sleep(wait_time)
        
        return f"Error calling {self.provider} after {self.max_retries + 1} attempts: {str(last_error)}", ""
    
    def _generate_gemini(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        """Generate using native Google Generative AI SDK with streaming"""
        # Create model with system instruction
        generation_config = {
            "temperature": temperature,
        }
        
        model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions or "You are a helpful AI assistant.",
            generation_config=generation_config,
        )
        
        # Use streaming response
        response_stream = model.generate_content(input_text, stream=True)
        
        final_text = ""
        accumulated_tokens = 0
        # Rough estimation: ~4 characters per token for Chinese/English mixed text
        chars_per_token = 4
        usage_metadata = None
        
        for chunk in response_stream:
            if chunk.text:
                content = chunk.text
                final_text += content
                
                # Estimate token count (rough approximation)
                accumulated_tokens += len(content) / chars_per_token
                
            
            # Try to get usage metadata from the last chunk
            if hasattr(chunk, "usage_metadata"):
                usage_metadata = chunk.usage_metadata
        
        
        # Track token usage if available
        if self.token_tracker and usage_metadata:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
            total_tokens = getattr(usage_metadata, "total_token_count", 0)
            
            self.token_tracker.add_record(
                provider="gemini",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                agent_name=self.current_agent_name
            )
        elif self.token_tracker:
            # Fallback: estimate tokens if usage_metadata not available
            prompt_tokens = len((instructions or "") + input_text) // chars_per_token
            completion_tokens = len(final_text) // chars_per_token
            total_tokens = prompt_tokens + completion_tokens
            
            self.token_tracker.add_record(
                provider="gemini",
                model=model_name,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                agent_name=self.current_agent_name
            )
        
        return final_text, ""
    
    def _generate_openai_compatible(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        """Generate using OpenAI-compatible API with streaming"""
        messages = [
            {"role": "system", "content": (instructions or "You are a helpful AI assistant.")},
            {"role": "user", "content": input_text},
        ]
        
        # Use streaming response
        stream = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        
        final_text = ""
        accumulated_tokens = 0
        # Rough estimation: ~4 characters per token for Chinese/English mixed text
        chars_per_token = 4
        usage_info = None
        
        for chunk in stream:
            if not chunk.choices:
                continue
            
            # Try to get usage information from chunk (some providers include it)
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage
                
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content = delta.content
                final_text += content
                
                # Estimate token count (rough approximation)
                accumulated_tokens += len(content) / chars_per_token
                
        

        # Track token usage
        if self.token_tracker:
            if usage_info:
                # Use actual usage information if available
                prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
                completion_tokens = getattr(usage_info, "completion_tokens", 0)
                total_tokens = getattr(usage_info, "total_tokens", 0)
            else:
                # Estimate tokens: prompt tokens from input, completion tokens from output
                prompt_tokens = len((instructions or "") + input_text) // chars_per_token
                completion_tokens = len(final_text) // chars_per_token
                total_tokens = prompt_tokens + completion_tokens
            
            self.token_tracker.add_record(
                provider=self.provider,
                model=model_name,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                agent_name=self.current_agent_name
            )

        return final_text, ""

