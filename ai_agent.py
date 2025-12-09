from typing import Any, Dict, List, Optional, Union

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class AIAgent:
    """
    A robust agent class with proper output parsing and error handling.

    Args:
        llm: The language model to use as the agent's reasoning engine.
        tools: List of tools the agent can use to take actions.
        response_format: Optional Pydantic model class for structured output.
        system_prompt: Optional system prompt to shape the agent's behavior.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools: List[BaseTool],
        response_format: Optional[type[BaseModel]] = None,  # Changed to type hint
        system_prompt: Optional[str] = None,
    ):
        # Store parameters for reference
        self.system_prompt = system_prompt
        self.response_format = response_format

        # Create the agent with proper configuration
        self.agent = create_agent(
            model=llm,
            tools=tools,
            response_format=response_format,
            system_prompt=system_prompt,
        )

    def invoke(self, user_input: str) -> Union[BaseModel, Dict[str, Any]]:
        """
        Invoke the agent with a user input.

        Args:
            user_input: The user's message content.

        Returns:
            If response_format is set, returns validated Pydantic model.
            Otherwise returns the full agent response dictionary.

        Raises:
            ValueError: If response parsing fails.
        """
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            return self._parse_output(result)
        except Exception as e:
            # Enhanced error handling with context
            raise RuntimeError(f"Agent invocation failed: {str(e)}") from e

    async def ainvoke(self, user_input: str) -> Union[BaseModel, Dict[str, Any]]:
        """Async version of invoke for concurrent operations."""
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            return self._parse_output(result)
        except Exception as e:
            raise RuntimeError(f"Async agent invocation failed: {str(e)}") from e

    def stream(self, user_input: str):
        """
        Stream the agent's response tokens and steps.

        Returns:
            Generator yielding streaming chunks.
        """
        try:
            return self.agent.stream(
                {"messages": [{"role": "user", "content": user_input}]}
            )
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {str(e)}") from e

    def _parse_output(self, result: Dict[str, Any]) -> Union[BaseModel, Dict[str, Any]]:
        """
        Parse the agent output with robust error handling.

        Args:
            result: Raw agent output dictionary.

        Returns:
            Parsed output based on response_format configuration.
        """
        # If no response_format specified, return full result
        if self.response_format is None:
            return result

        # Extract structured response
        structured_data = result.get("structured_response")

        # Handle missing structured response
        if structured_data is None:
            # Check if the result itself might be the structured data
            if isinstance(result, dict) and all(
                k not in result for k in ["messages", "output"]
            ):
                structured_data = result
            else:
                # Try to extract from the last message
                messages = result.get("messages", [])
                if messages and hasattr(messages[-1], "content"):
                    return self._extract_from_content(messages[-1].content)
                return result

        # Validate and create the response model
        try:
            return self._validate_and_create_model(structured_data)
        except Exception as e:
            # Fallback to returning raw data with warning
            import warnings

            warnings.warn(
                f"Failed to parse response as {self.response_format.__name__}: {e}"
            )
            return result

    def _extract_from_content(self, content: str) -> Union[BaseModel, Dict[str, Any]]:
        """Extract structured data from text content."""
        if self.response_format is None:
            return {"content": content}

        # Simple attempt to parse JSON-like content
        import json
        import re

        # Look for JSON blocks in the content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_create_model(data)
            except json.JSONDecodeError:
                pass

        # If no JSON found, return as text
        return {"content": content}

    def _validate_and_create_model(self, data: Any) -> BaseModel:
        """Create and validate the response model instance."""
        if self.response_format is None:
            raise ValueError("No response_format specified")

        # Check if data is already an instance of the expected model
        if isinstance(data, self.response_format):
            return data

        # Handle different input types
        if isinstance(data, dict):
            # Check Pydantic version and use appropriate method
            if hasattr(self.response_format, "model_validate"):
                # Pydantic v2
                return self.response_format.model_validate(data)
            else:
                # Pydantic v1
                return self.response_format.parse_obj(data)
        elif isinstance(data, (list, tuple)):
            # Try to pass as positional arguments
            try:
                return self.response_format(*data)
            except Exception:
                # Fallback to treating as single argument
                return self.response_format(data)
        else:
            # Single value
            return self.response_format(data)

    # Additional utility methods

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.agent.tools if hasattr(tool, "name")]

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration."""
        return {
            "has_response_format": self.response_format is not None,
            "response_format_type": (
                self.response_format.__name__ if self.response_format else None
            ),
            "system_prompt_length": (
                len(self.system_prompt) if self.system_prompt else 0
            ),
            "tools_count": len(self.agent.tools) if hasattr(self.agent, "tools") else 0,
        }
