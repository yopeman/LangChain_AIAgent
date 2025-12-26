"""
Improved AIAgent for LangChain-style CompiledStateGraph + BaseChatModel.

invoke(...) and ainvoke(...) return Union[str, BaseModel] (pydantic instance when
`response_format` is provided and parsing succeeds, otherwise a plain string).

This implementation:
- Consolidates imports and logging setup
- Accepts an llm implementing invoke/ainvoke (best-effort support for variations)
- Robustly extracts text from many possible result shapes
- Attempts JSON extraction then pydantic parsing (compatible with pydantic v1 & v2)
- Provides helpful logging and does not crash on unexpected shapes
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional, Type, Union, Coroutine

from pydantic import BaseModel, ValidationError

# LangChain-ish imports (keep as-is if using langchain-core)
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

# One logging configuration at module import
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


ResultLike = Dict[str, Any]


class AIAgent:
    """
    A robust wrapper around a LangChain CompiledStateGraph produced by create_agent.
    - `llm`: either an instance of BaseChatModel (preferred) or a string placeholder
             (if you plan to resolve it before calling methods).
    - `tools`: list of BaseTool instances to pass to create_agent
    - `system_prompt`: global system prompt used when creating the agent
    - `response_format`: Optional pydantic BaseModel subclass to coerce structured output into
    """

    def __init__(
        self,
        *,
        llm: Union[str, BaseChatModel],
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
        response_format: Optional[Type[BaseModel]] = None,
        debug: bool = False,
    ):
        self.llm = llm
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.debug = debug

        # Lazily created graph, so we can defer heavy initialization if desired
        self.graph = self._initialize_agent()

    def _initialize_agent(self):
        """Create the CompiledStateGraph using create_agent (wrap errors)."""
        try:
            return create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
                response_format=self.response_format,
                debug=self.debug,
            )
        except Exception as e:
            logger.exception("Failed to initialize agent graph")
            raise

    # ----------------------
    # Public invoke / ainvoke
    # ----------------------
    def invoke(self, user_input: str, **kwargs: Any) -> Union[str, BaseModel]:
        """
        Synchronous invocation of the compiled agent graph.
        Returns either a plain string or a pydantic model instance (if response_format provided).
        """
        try:
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            raw = self.graph.invoke(inputs, **kwargs)
            text_or_model = self._process_result(raw)
            return text_or_model
        except Exception as e:
            logger.exception("Graph invocation error")
            # Keep return type stable: str on error
            return f"Error: {str(e)}"

    async def ainvoke(self, user_input: str, **kwargs: Any) -> Union[str, BaseModel]:
        """
        Asynchronous invocation of the compiled agent graph.
        """
        try:
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            raw = await self.graph.ainvoke(inputs, **kwargs)
            text_or_model = self._process_result(raw)
            return text_or_model
        except Exception as e:
            logger.exception("Async graph invocation error")
            return f"Error: {str(e)}"

    # ----------------------
    # Prompt improvement helpers
    # ----------------------
    def improve(self, prompt: str, **kwargs: Any) -> str:
        """Ask the LLM to improve a prompt (sync). Returns the improved prompt as string."""
        system_msg = "You are an expert prompt engineer. Enhance the prompt for clarity and detail."
        user_msg = f"Improve this prompt: {prompt}"

        try:
            response = self._call_llm_sync([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
            return response.strip()
        except Exception:
            logger.exception("Improve prompt failed")
            return prompt

    async def aimprove(self, prompt: str, **kwargs: Any) -> str:
        """Async version of improve()."""
        system_msg = "You are an expert prompt engineer. Enhance the prompt for clarity and detail."
        user_msg = f"Improve this prompt: {prompt}"

        try:
            response = await self._call_llm_async([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
            return response.strip()
        except Exception:
            logger.exception("Async improve prompt failed")
            return prompt

    # ----------------------
    # Internal helpers
    # ----------------------
    def _process_result(self, raw: Any) -> Union[str, BaseModel]:
        """
        Normalizes a raw result returned from graph.invoke/ainvoke into:
         - parsed pydantic model (if response_format provided and parsing succeeds)
         - or a string
        This function tolerates many shapes returned by different LangChain versions.
        """
        # If raw is a pydantic model already, return as-is
        if isinstance(raw, BaseModel):
            return raw

        # If raw is a dict-like structure, try to find structured_response / output / messages
        try:
            if isinstance(raw, dict):
                # Prefer explicit structured_response key if present
                if "structured_response" in raw:
                    structured = raw["structured_response"]
                    return self._maybe_structure(structured)

                # Common keys: "output", "final_output", "result", "text"
                for key in ("output", "final_output", "result", "text"):
                    if key in raw:
                        return self._maybe_structure(raw[key])

                # If messages exist, attempt to extract last message content
                if "messages" in raw:
                    messages = raw["messages"]
                    text = self._extract_text_from_messages(messages)
                    return self._maybe_structure(text)

                # Some graphs return a list of events/steps; try to stringify
                # or search for nested messages
                # attempt to find any stringful values
                candidate = self._find_first_stringish(raw)
                if candidate is not None:
                    return self._maybe_structure(candidate)

                # fallback: stringified raw
                return self._maybe_structure(json.dumps(raw, ensure_ascii=False))
            # If raw is a list, try join of string-like elements
            if isinstance(raw, list):
                # try to find a first string-like
                for item in raw[::-1]:
                    if isinstance(item, (str,)):
                        return self._maybe_structure(item)
                    if isinstance(item, dict):
                        cand = self._find_first_stringish(item)
                        if cand:
                            return self._maybe_structure(cand)
                # fallback to json dump
                return self._maybe_structure(json.dumps(raw, ensure_ascii=False))

            # If raw is a plain string
            if isinstance(raw, str):
                return self._maybe_structure(raw)

            # If raw has attribute 'content' (e.g., a message object)
            if hasattr(raw, "content"):
                return self._maybe_structure(getattr(raw, "content"))

            # Unknown type: fallback to str()
            return self._maybe_structure(str(raw))
        except Exception:
            logger.exception("Failed to process raw result")
            return f"Error processing result: {str(raw)}"

    def _maybe_structure(self, text_or_obj: Any) -> Union[str, BaseModel]:
        """
        If response_format is present, try to coerce text_or_obj into that model.
        Accepts either raw text, a dict-like structure, or already parsed JSON objects.
        On failure, returns the original text (string).
        """
        if not self.response_format:
            return text_or_obj if isinstance(text_or_obj, str) else str(text_or_obj)

        # If already a dict-like object, attempt to parse directly
        if isinstance(text_or_obj, dict):
            return self._coerce_to_model(text_or_obj)

        # If already a BaseModel, return it
        if isinstance(text_or_obj, BaseModel):
            return text_or_obj

        # If a non-empty string, try to extract JSON and parse, or try to parse full string as JSON
        if isinstance(text_or_obj, str):
            # 1) Try to load the entire string as JSON
            s = text_or_obj.strip()
            try:
                parsed = json.loads(s)
                return self._coerce_to_model(parsed)
            except Exception:
                # 2) Try to find the first JSON block in the string (practical)
                match = re.search(r"(\{(?:.|\s)*\}|\[(?:.|\s)*\])", s, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        return self._coerce_to_model(parsed)
                    except Exception as e:
                        logger.debug("JSON block found but failed to parse into model: %s", e)
                # If parsing failed, return original text
                return s

        # Otherwise, cast to string
        return str(text_or_obj)

    def _coerce_to_model(self, data: Any) -> Union[str, BaseModel]:
        """
        Convert a dict/list to the configured pydantic response_format.
        Accepts both pydantic v1 (parse_obj) and v2 (model_validate).
        On validation error, return the original serialized JSON string.
        """
        try:
            if hasattr(self.response_format, "model_validate"):
                # pydantic v2
                return self.response_format.model_validate(data)
            # pydantic v1
            return self.response_format.parse_obj(data)
        except ValidationError as ve:
            logger.warning("Response model validation failed: %s", ve)
            # return the serialized JSON text if validation fails
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return str(data)
        except Exception:
            logger.exception("Unexpected error while coercing to pydantic model")
            return str(data)

    @staticmethod
    def _extract_text_from_messages(messages: Any) -> str:
        """
        Accepts a sequence of messages where each item may be:
          - a dict with {'role':..., 'content': ...}
          - an object with .content attribute
          - a plain string
        Returns the last non-empty message content found (searches backwards).
        """
        try:
            if isinstance(messages, str):
                return messages
            if not messages:
                return ""
            # support list-like
            for item in reversed(messages):
                if item is None:
                    continue
                if isinstance(item, str) and item.strip():
                    return item
                if isinstance(item, dict):
                    # common keys
                    for key in ("content", "text", "message", "output"):
                        if key in item and item[key]:
                            return str(item[key])
                if hasattr(item, "content"):
                    val = getattr(item, "content")
                    if isinstance(val, str) and val.strip():
                        return val
                    # in some implementations .content may be a list/dict
                    if not isinstance(val, str):
                        try:
                            return json.dumps(val, ensure_ascii=False)
                        except Exception:
                            return str(val)
            return ""
        except Exception:
            logger.exception("Failed to extract text from messages")
            return ""

    @staticmethod
    def _find_first_stringish(d: Dict[str, Any]) -> Optional[str]:
        """
        Walks a dict shallowly to find the first string-like leaf value.
        """
        try:
            for k, v in d.items():
                if isinstance(v, str) and v.strip():
                    return v
                if isinstance(v, (dict, list)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        continue
                if v is not None:
                    return str(v)
            return None
        except Exception:
            return None

    # ----------------------
    # LLM calling helpers for improve() function
    # ----------------------
    def _call_llm_sync(self, messages: List[Any]) -> str:
        """
        Best-effort synchronous call to the provided llm instance.
        Tries common method names and return shapes:
          - llm.invoke(messages) -> object with .content or str
          - llm.generate / llm.__call__ / llm.chat -> attempt sensible extraction
        """
        llm = self.llm
        # Try invoke
        try:
            if hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
                out = llm.invoke(messages)
                return self._extract_from_llm_response(out)
        except Exception:
            logger.debug("llm.invoke failed", exc_info=True)

        # Try call/generate/chat
        for name in ("__call__", "generate", "chat", "complete"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    out = fn(messages)
                    return self._extract_from_llm_response(out)
                except Exception:
                    logger.debug("llm.%s failed", name, exc_info=True)
        raise RuntimeError("No supported synchronous call found on llm")

    async def _call_llm_async(self, messages: List[Any]) -> str:
        """
        Best-effort async call to the provided llm instance.
        Looks for .ainvoke, .agenerate, or awaits a coroutine from __call__.
        """
        llm = self.llm
        # Try ainvoke
        try:
            if hasattr(llm, "ainvoke") and callable(getattr(llm, "ainvoke")):
                out = await llm.ainvoke(messages)
                return self._extract_from_llm_response(out)
        except Exception:
            logger.debug("llm.ainvoke failed", exc_info=True)

        # Try async generate / __call__
        for name in ("agenerate", "__call__"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    maybe_coro = fn(messages)
                    if isinstance(maybe_coro, Coroutine):
                        out = await maybe_coro
                        return self._extract_from_llm_response(out)
                    # If it returned a sync result, handle it
                    return self._extract_from_llm_response(maybe_coro)
                except Exception:
                    logger.debug("llm.%s failed", name, exc_info=True)
        raise RuntimeError("No supported async call found on llm")

    @staticmethod
    def _extract_from_llm_response(resp: Any) -> str:
        """
        Normalize typical LLM return shapes into a single text string.
        Accepts objects with .content, dicts with 'content'/'text', lists, or plain strings.
        """
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, BaseModel):
            # Some LLM wrappers return pydantic BaseModel responses
            if hasattr(resp, "content"):
                return getattr(resp, "content")
            return str(resp.json() if hasattr(resp, "json") else resp)

        # dict-like
        if isinstance(resp, dict):
            # common keys
            for key in ("content", "text", "message", "output"):
                if key in resp:
                    return str(resp[key])
            # sometimes 'generations' or 'choices'
            for key in ("generations", "choices"):
                if key in resp:
                    val = resp[key]
                    # choices/generations may be a list of objects/dicts
                    if isinstance(val, list) and val:
                        first = val[0]
                        if isinstance(first, dict):
                            for k in ("text", "message", "content"):
                                if k in first:
                                    return str(first[k])
                        if hasattr(first, "text"):
                            return str(getattr(first, "text"))
            # fallback to JSON
            try:
                return json.dumps(resp, ensure_ascii=False)
            except Exception:
                return str(resp)

        # list-like
        if isinstance(resp, list):
            # pick last stringish element
            for item in reversed(resp):
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    for k in ("content", "text"):
                        if k in item:
                            return item[k]
                if hasattr(item, "content"):
                    return getattr(item, "content")
            return json.dumps(resp, ensure_ascii=False)

        # generic object with .content or .text
        if hasattr(resp, "content"):
            return getattr(resp, "content")
        if hasattr(resp, "text"):
            return getattr(resp, "text")
        # fallback
        return str(resp)

