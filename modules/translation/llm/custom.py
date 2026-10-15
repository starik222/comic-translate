from typing import Any
import re
import numpy as np

from .gpt import GPTTranslation
from ...utils.translator_utils import remove_think_block


class CustomTranslation(GPTTranslation):
    """Translation engine using a custom OpenAI-compatible API."""

    # Context history per language pair:
    # {(source_lang, target_lang): [(user_prompt, assistant_response), ...]}
    _context_history: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def __init__(self):
        super().__init__()
        self.context_enabled = False
        self.context_pairs = 3

    @classmethod
    def add_history_pair(cls, source_lang: str, target_lang: str, user_prompt: str, assistant_response: str) -> None:
        """Add a (user_prompt, assistant_response) pair to history for this language pair."""
        if not source_lang or not target_lang:
            return
        key = (source_lang, target_lang)
        if key not in cls._context_history:
            cls._context_history[key] = []
        cls._context_history[key].append((user_prompt, assistant_response))
        # Keep recent history bounded to prevent unbounded memory growth
        if len(cls._context_history[key]) > 100:
            cls._context_history[key] = cls._context_history[key][-100:]

    @classmethod
    def get_history(cls, source_lang: str, target_lang: str, max_pairs: int) -> list[tuple[str, str]]:
        """Get the latest max_pairs from history for this language pair."""
        if max_pairs <= 0:
            return []
        key = (source_lang, target_lang)
        return list(cls._context_history.get(key, []))[-max_pairs:]

    @classmethod
    def clear_context(cls, source_lang: str = None, target_lang: str = None) -> None:
        """Clear context history for a specific language pair or all pairs."""
        if source_lang is not None and target_lang is not None:
            cls._context_history.pop((source_lang, target_lang), None)
        else:
            cls._context_history.clear()

    def add_context_pair(self, user_prompt: str, assistant_response: str) -> None:
        self.__class__.add_history_pair(self.source_lang, self.target_lang, user_prompt, assistant_response)

    def get_context_history(self, max_pairs: int) -> list[tuple[str, str]]:
        return self.__class__.get_history(self.source_lang, self.target_lang, max_pairs)

    def clear_history(self) -> None:
        self.__class__.clear_context(self.source_lang, self.target_lang)

    def initialize(self, settings: Any, source_lang: str, target_lang: str, tr_key: str, **kwargs) -> None:
        """
        Initialize custom translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
        """
        # Call BaseLLMTranslation's initialize, not GPTTranslation's
        # to avoid the GPT-specific credential loading
        super(GPTTranslation, self).initialize(settings, source_lang, target_lang, **kwargs)
        
        # Get custom credentials instead of OpenAI credentials
        credentials = settings.get_credentials(settings.ui.tr(tr_key))
        self.api_key = credentials.get('api_key', '')
        self.model = credentials.get('model', '')
        
        # Override the API base URL with the custom one
        self.api_base_url = credentials.get('api_url', '').rstrip('/')
        self.timeout = 3600  # Custom timeout for potentially slower custom LLMs

        llm_settings = settings.get_llm_settings()
        self.context_enabled = bool(llm_settings.get('context_enabled', False))
        self.context_pairs = int(llm_settings.get('context_pairs', 3))

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using direct REST API calls to OpenAI-compatible server.
        
        Args:
            user_prompt: Text prompt from user
            system_prompt: System instructions
            image: Image as numpy array
            
        Returns:
            Translated text
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
        ]

        # Inject context history (previous user & assistant pairs)
        if getattr(self, "context_enabled", False) and getattr(self, "context_pairs", 0) > 0:
            history = self.get_context_history(self.context_pairs)
            for prev_user, prev_assistant in history:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prev_user}]
                })
                messages.append({
                    "role": "assistant",
                    "content": prev_assistant
                })

        # Append current user prompt
        if self.supports_images and self.img_as_llm_input and image is not None:
            encoded_image, mime_type = self.encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        response_content = self._make_api_request(payload, headers)

        # Store successfully translated request/response into history
        if getattr(self, "context_enabled", False) and response_content:
            clean_res = remove_think_block(response_content).strip()
            match = re.search(r"\{[\s\S]*\}", clean_res)
            saved_assistant = match.group(0) if match else clean_res
            self.add_context_pair(user_prompt, saved_assistant)

        return response_content

    def _make_api_request(self, payload, headers):
        """
        Make API request with fallback for servers that do not support max_completion_tokens.
        """
        try:
            return super()._make_api_request(payload, headers)
        except RuntimeError as e:
            if "max_completion_tokens" in str(e) and "max_completion_tokens" in payload:
                fallback_payload = dict(payload)
                fallback_payload.pop("max_completion_tokens", None)
                fallback_payload["max_tokens"] = self.max_tokens
                return super()._make_api_request(fallback_payload, headers)
            raise
