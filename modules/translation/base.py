from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from ..utils.textblock import TextBlock
from ..utils.language_utils import is_no_space_lang


class TranslationEngine(ABC):
    """
    Abstract base class for all translation engines.
    Defines common interface and utility methods.
    """
    
    @abstractmethod
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the translation engine with necessary parameters.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    def get_language_code(self, language: str) -> str:
        """
        Get standardized language code from language name.
        
        Args:
            language: Language name
            
        Returns:
            Standardized language code
        """
        from ..utils.language_utils import get_language_code
        return get_language_code(language)
    
    def preprocess_text(self, blk_text: str, source_lang_code: str) -> str:
        """
        PreProcess text based on language:
        - Remove spaces for Chinese and Japanese languages
        - Remove all newline/carriage-return characters
        - Keep original text for other languages (aside from the newline removal)
        
        Args:
            blk_text (str): The input text to process
            source_lang_code (str): Language code of the source text
        
        Returns:
            str: Processed text
        """
        # Remove newline and carriage‐return characters
        text = blk_text.replace('\r', '').replace('\n', '')

        # 2) If No-Space Language, also remove all spaces
        if is_no_space_lang(source_lang_code):
            return text.replace(' ', '')
        # 3) Otherwise, return the text (with newlines already removed)
        else:
            return text


class TraditionalTranslation(TranslationEngine):
    """Base class for traditional translation engines (non-LLM)."""
    
    @abstractmethod
    def translate(self, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Translate text blocks using non-LLM translators.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            
        Returns:
            List of updated TextBlock objects with translations
        """
        pass

    def preprocess_language_code(self, lang_code: str) -> str:
        """
        Preprocess language codes to match the specific translation API requirements.
        By default, returns the original language code.
        
        Args:
            lang_code: The language code to preprocess
            
        Returns:
            Preprocessed language code supported by the translation API
        """
        return lang_code  # Default implementation just returns the original code


class LLMTranslation(TranslationEngine):
    """Base class for LLM-based translation engines."""
    
    @abstractmethod
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            image: Image as numpy array (for context)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        pass
    
    def get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """
        Get system prompt for LLM translation.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Formatted system prompt
        """
        return f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations
        Do Your Best! I'm really counting on you."""


    def get_system_prompt_v2(self, source_lang: str, target_lang: str) -> str:
        """
        Get system prompt for LLM translation.

        Args:
            source_lang: Source language
            target_lang: Target language

        Returns:
            Formatted system prompt
        """
        return  f"""You are an expert manga and comic translator specializing in {source_lang} to {target_lang} localization. Your goal is to accurately translate text extracted via OCR, using the provided manga page image to understand the context and correct any OCR mistakes.

### INPUT:
1. An image of a manga/comic page.
2. A JSON object containing text blocks extracted via OCR (keys like "block_0", "block_1", and {source_lang} text as values).

### YOUR INSTRUCTIONS:
1. ANALYZE THE IMAGE & CORRECT OCR:
   - Carefully examine the provided image and locate the text bubbles corresponding to the JSON input.
   - {source_lang} OCR often makes mistakes (e.g., misrecognizing kanji, confusing furigana with main text, missing characters). Internally correct the {source_lang} text based on what is actually written on the image before translating.

2. CONTEXTUAL TRANSLATION:
   - Look at the scene: Who is speaking? To whom? What are their emotions, facial expressions, and actions? 
   - Translate the corrected text into natural, fluent {target_lang}.
   - Adapt the tone to fit the characters (e.g., use informal language, slang, or polite forms based on their relationships and age).
   - Localize onomatopoeia, sighs, and sounds (e.g., "お〜っ" -> "Ого-о", "ふい〜っ" -> "Фух", "ドヤ" -> express the smugness contextually).
   - Handle Japanese punctuation appropriately (e.g., convert full-width dots "．．．" to standard {target_lang} ellipses "...").

3. OUTPUT FORMAT:
   - You must output ONLY a valid JSON object.
   - Maintain the EXACT same keys as the input JSON (e.g., "block_0", "block_1").
   - The values must be your final {target_lang} translation.
   - Do not include any explanations, greetings, or additional text outside the JSON structure.

### EXAMPLE BEHAVIOR:
If OCR missed a character but the image shows it, translate the full meaning from the image. Maintain the flow of the dialogue across the blocks so it reads like a natural {target_lang} comic."""