from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .config import PluginConfig


class LanguageJudger:
    EXTRA_KEY = "gsv_text_lang"
    ALLOWED_LANGS = {"en", "zh", "ja", "ko"}
    ALIASES = {
        "zh": "zh",
        "cn": "zh",
        "chinese": "zh",
        "中文": "zh",
        "汉语": "zh",
        "漢語": "zh",
        "普通话": "zh",
        "mandarin": "zh",
        "en": "en",
        "eng": "en",
        "english": "en",
        "英文": "en",
        "英语": "en",
        "英語": "en",
        "ja": "ja",
        "jp": "ja",
        "japanese": "ja",
        "日文": "ja",
        "日语": "ja",
        "日語": "ja",
        "日本語": "ja",
        "ko": "ko",
        "kr": "ko",
        "korean": "ko",
        "韩文": "ko",
        "韩语": "ko",
        "韓文": "ko",
        "韓語": "ko",
        "한국어": "ko",
    }

    def __init__(self, config: PluginConfig):
        self.cfg = config

    async def judge_language(
        self,
        event: AstrMessageEvent,
        *,
        text: str = "",
    ) -> str | None:
        """
        使用 LLM 判断 TTS 文本语言。

        失败或无法判断时返回 None，由调用方继续使用默认 text_lang。
        """
        cached = event.get_extra(self.EXTRA_KEY)
        if isinstance(cached, str):
            lang = self._normalize_lang(cached)
            if lang:
                logger.debug(f"复用文本语言标签: {lang}")
                return lang

        if not text.strip():
            return None

        try:
            provider = self.cfg.get_judge_provider(event.unified_msg_origin)
            system_prompt, prompt = self._build_prompt(text)
            resp = await provider.text_chat(
                system_prompt=system_prompt,
                prompt=prompt,
            )

            lang = self._parse_llm_response(resp.completion_text)
            if not lang:
                logger.debug("文本语言判断结果为空，使用默认配置")
                return None

            logger.debug(f"文本语言判断结果: {lang}")
            event.set_extra(self.EXTRA_KEY, lang)
            return lang

        except Exception as e:
            logger.exception(f"文本语言判断失败: {e}")
            return None

    def _build_prompt(self, text: str) -> tuple[str, str]:
        system_prompt = (
            "你是一个文本语言分类器。\n"
            "只根据用户提供的文本本身判断语言，不要根据上下文、用户偏好或默认语言推断。\n"
            "只能返回 en、zh、ja、ko 之一，或在无法明确判断时返回 null。\n"
            "混合语言文本请选择 en、zh、ja、ko 中占主导的语言。\n"
            "如果没有支持语言明显占主导，请返回 null，不要返回 auto。\n"
            "不要返回 yue、auto_yue、all_zh、all_ja、all_yue、all_ko 等标签。\n"
            "粤语文本一般归为 zh；除非明显不适合，否则不要返回 null。\n"
            "请严格按照 JSON 格式输出，不要包含任何多余内容。\n"
            '输出示例：{"text_lang":"zh"} 或 {"text_lang":null}'
        )
        prompt = f"文本内容：{text}"
        return system_prompt, prompt

    def _parse_llm_response(self, text: str) -> str | None:
        data = self._loads_json(text)
        if not isinstance(data, dict):
            raise ValueError(f"LLM JSON 不是对象: {data}")

        lang = data.get("text_lang")
        if lang is None:
            return None
        if not isinstance(lang, str):
            raise ValueError(f"LLM JSON text_lang 字段非法: {data}")

        normalized = self._normalize_lang(lang)
        if not normalized:
            raise ValueError(f"LLM 返回不支持的语言标签: {lang}")
        return normalized

    def _loads_json(self, text: str) -> Any:
        content = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", content):
                try:
                    value, _ = decoder.raw_decode(content[match.start() :])
                    return value
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"LLM 返回非 JSON: {text}") from e

    def _normalize_lang(self, value: str) -> str | None:
        lang = value.strip().lower()
        lang = self.ALIASES.get(lang, lang)
        if lang in self.ALLOWED_LANGS:
            return lang
        return None
