import json
import os
from typing import Any, Callable, Optional
from dotenv import load_dotenv


class GroqServiceError(RuntimeError):
    pass


class GroqService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client: Any = None):
        load_dotenv()
        resolved_key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
        if client is None and (not resolved_key or resolved_key == "your_groq_api_key_here"):
            raise GroqServiceError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        self.model = (model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")).strip()
        if client is not None:
            self.client = client
        else:
            from groq import Groq
            self.client = Groq(api_key=resolved_key)

    def generate_json(
        self,
        prompt: str,
        validator: Optional[Callable[[dict], Any]] = None,
        max_retries: int = 2,
    ) -> Any:
        """Generate JSON from Groq and optionally validate with controlled retries."""
        attempts = max(1, int(max_retries) + 1)
        last_error: Optional[Exception] = None
        current_prompt = prompt

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise educational content generator. "
                                "Return valid JSON only. Follow the requested schema exactly."
                            ),
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise GroqServiceError("Groq returned an empty response.")
                raw_data = json.loads(content)
                if not isinstance(raw_data, dict):
                    raise GroqServiceError("Groq response must be a JSON object.")
                if validator is not None:
                    return validator(raw_data)
                return raw_data
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"IMPORTANT RETRY NOTICE (Attempt {attempt + 1}/{attempts}): "
                        f"Your previous response failed validation with error: {exc}. "
                        f"Return strictly valid JSON matching all schema and count constraints."
                    )
                else:
                    break

        if isinstance(last_error, GroqServiceError):
            raise last_error
        raise GroqServiceError(
            f"Groq JSON generation/validation failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    def chat(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful tutor. Answer accurately and stay focused on the selected lesson. "
                            "If the supplied PDF passages do not support an answer, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                raise GroqServiceError("Groq returned an empty response.")
            return answer
        except GroqServiceError:
            raise
        except Exception as exc:
            raise GroqServiceError(f"Groq chat request failed: {exc}") from exc
