from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import logging
import threading
from datetime import datetime
import imkit as imk

from ..base import LLMTranslation
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_raw_text, set_texts_from_json
from ...utils.paths import get_llm_error_log_path

logger = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.context_enabled = False
        self.context_pairs = 3
        self.temperature = None
        self.max_tokens = None
        self.timeout = 30  
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = llm_settings.get('image_input_enabled', True)
        self.context_enabled = bool(llm_settings.get('context_enabled', False))
        self.context_pairs = int(llm_settings.get('context_pairs', 3))
        self.temperature = 1.0
        self.max_tokens = 32768

    def _log_translation_error(
        self,
        entire_raw_text: str | None,
        entire_translated_text: str | None,
        error: Exception,
    ) -> None:
        """
        Log request and response data to a dedicated log file when a translation or parsing error occurs.
        """
        try:
            log_path = get_llm_error_log_path()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_info = getattr(self, "model", None) or getattr(self, "model_name", None) or "unknown"
            engine_class = self.__class__.__name__
            error_details = f"{type(error).__name__}: {error}"

            log_entry = (
                f"\n{'=' * 80}\n"
                f"[{timestamp}] LLM Translation Error\n"
                f"Engine: {engine_class} | Model: {model_info}\n"
                f"Language: {self.source_lang} -> {self.target_lang}\n"
                f"Error: {error_details}\n"
                f"{'-' * 80}\n"
                f"[ENTIRE_RAW_TEXT (SENT)]:\n{entire_raw_text}\n"
                f"{'-' * 80}\n"
                f"[ENTIRE_TRANSLATED_TEXT (RECEIVED)]:\n{entire_translated_text}\n"
                f"{'=' * 80}\n"
            )

            with _LOG_LOCK:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_entry)

            logger.error(
                "LLM translation error (%s). Details written to %s",
                error_details,
                log_path,
            )
        except Exception as log_exc:
            logger.error("Failed to write to LLM translation error log: %s", log_exc)
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        entire_raw_text = get_raw_text(blk_list)
        #system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        #user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"
        system_prompt = self.get_system_prompt_v2(self.source_lang, self.target_lang)
        user_prompt = entire_raw_text
        
        entire_translated_text = None
        try:
            entire_translated_text = self._perform_translation(user_prompt, system_prompt, image)
            set_texts_from_json(blk_list, entire_translated_text)
        except Exception as e:
            self._log_translation_error(entire_raw_text, entire_translated_text, e)
            raise
            
        return blk_list
    
    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type
