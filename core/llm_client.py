import json, time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: int = 120

    def chat_completion(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

        t0 = time.time()
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        dt_ms = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        data["_request_time_ms"] = dt_ms
        return data
