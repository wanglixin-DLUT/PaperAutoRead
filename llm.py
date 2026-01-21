import os
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Provider configurations: base_url and env_var for API key
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "QWEN_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": None,  # Uses native SDK
        "env_var": "GEMINI_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_var": "ZHIPUAI_API_KEY",
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
        print(f"Token usage statistics exported to: {output_file}")
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
    ) -> None:
        self.provider = provider.lower()
        self.default_model = default_model
        self.request_timeout = request_timeout
        self.token_tracker = token_tracker
        self.current_agent_name = "unknown"
        
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
                    "X-Title": site_name or "Rebuttal Assistant",
                }

            self._http_client = httpx.Client(
                trust_env=True, 
                timeout=request_timeout,
                headers=extra_headers
            )
            
            # Use provided base_url or default from config
            effective_base_url = base_url or config["base_url"]
            
            self._client = OpenAI(
                base_url=effective_base_url,
                api_key=api_key,
                http_client=self._http_client,
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
        except Exception as e:
            print(f"[{self.provider.upper()} Error] {type(e).__name__}: {e}")
            final_text = f"Error calling {self.provider}: {str(e)}"
            
        return final_text or "", reasoning_text or ""
    
    def _generate_gemini(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        """Generate using native Google Generative AI SDK"""
        # Create model with system instruction
        generation_config = {
            "temperature": temperature,
        }
        
        model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions or "You are a helpful AI assistant.",
            generation_config=generation_config,
        )
        
        response = model.generate_content(input_text)
        
        final_text = ""
        if response.text:
            final_text = response.text
        
        # Track token usage if available
        if self.token_tracker and hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", 0)
            
            self.token_tracker.add_record(
                provider="gemini",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                agent_name=self.current_agent_name
            )
            print(f"[Token] {self.current_agent_name}: in={prompt_tokens}, out={completion_tokens}")
        
        return final_text, ""
    
    def _generate_openai_compatible(
        self,
        instructions: Optional[str],
        input_text: str,
        model_name: str,
        temperature: float,
    ) -> Tuple[str, str]:
        """Generate using OpenAI-compatible API"""
        messages = [
            {"role": "system", "content": (instructions or "You are a helpful AI assistant.")},
            {"role": "user", "content": input_text},
        ]
        
        response = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        
        final_text = ""
        if getattr(response, "choices", None):
            choice0 = response.choices[0]
            message = getattr(choice0, "message", None)
            if message is not None:
                final_text = getattr(message, "content", None) or ""

        if self.token_tracker and hasattr(response, "usage"):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            
            self.token_tracker.add_record(
                provider=self.provider,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                agent_name=self.current_agent_name
            )
            print(f"[Token] {self.current_agent_name}: in={prompt_tokens}, out={completion_tokens}")

        return final_text, ""

