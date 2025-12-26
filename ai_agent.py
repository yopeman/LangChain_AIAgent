import logging
import json
import re
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Type
from contextlib import contextmanager
import warnings

from langchain.agents import create_agent
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_classic.agents import AgentExecutor
from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AIAgent:
    """
    A robust agent class with proper output parsing and error handling.

    Args:
        llm: The language model to use as the agent's reasoning engine.
        tools: List of tools the agent can use to take actions.
        response_format: Optional Pydantic model class for structured output.
        system_prompt: Optional system prompt to shape the agent's behavior.
        max_iterations: Maximum number of agent steps before stopping.
        handle_parsing_errors: Whether to gracefully handle parsing errors.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools: List[BaseTool] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        handle_parsing_errors: bool = True,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = tools or []
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.handle_parsing_errors = handle_parsing_errors
        self.verbose = verbose
        
        # Create the agent with proper configuration
        try:
            self.agent = create_agent(
                model=llm,
                tools=self.tools,
                response_format=response_format,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def invoke(
        self, 
        user_input: str, 
        **kwargs: Any
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """
        Invoke the agent with a user input.

        Args:
            user_input: The user's message content.
            **kwargs: Additional arguments to pass to the agent.

        Returns:
            If response_format is set, returns validated Pydantic model.
            Otherwise returns the agent's response.
            
        Raises:
            RuntimeError: If agent invocation fails.
            ValidationError: If response parsing fails.
        """
        try:
            # Prepare input with messages
            input_data = {"messages": [{"role": "user", "content": user_input}]}
            input_data.update(kwargs)
            result = self.agent.invoke(input_data)
                
            return self._parse_output(result)
        except ValidationError as e:
            if self.handle_parsing_errors:
                logger.warning(f"Validation error: {e}. Returning raw output.")
                return {"error": str(e), "raw_output": result if 'result' in locals() else None}
            raise
        except Exception as e:
            logger.error(f"Agent invocation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Agent invocation failed: {str(e)}") from e

    async def ainvoke(
        self, 
        user_input: str, 
        **kwargs: Any
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """Async version of invoke for concurrent operations."""
        try:
            input_data = {"messages": [{"role": "user", "content": user_input}]}
            input_data.update(kwargs)
            result = await self.agent.ainvoke(input_data)
                
            return self._parse_output(result)
        except ValidationError as e:
            if self.handle_parsing_errors:
                logger.warning(f"Validation error: {e}. Returning raw output.")
                return {"error": str(e), "raw_output": result if 'result' in locals() else None}
            raise
        except Exception as e:
            logger.error(f"Async agent invocation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Async agent invocation failed: {str(e)}") from e

    def stream(
        self, 
        user_input: str, 
        **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream the agent's response tokens and steps.

        Returns:
            Generator yielding streaming chunks.
        """
        try:
            input_data = {"messages": [{"role": "user", "content": user_input}]}
            input_data.update(kwargs)
            
            return self.agent.stream(input_data)
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Streaming failed: {str(e)}") from e

    async def astream(
        self, 
        user_input: str, 
        **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async version of stream."""
        try:
            input_data = {"messages": [{"role": "user", "content": user_input}]}
            input_data.update(kwargs)
            
            if hasattr(self.agent, 'astream'):
                async for chunk in self.agent.astream(input_data):
                    yield chunk
            else:
                raise NotImplementedError("Async streaming not supported by this agent")
        except Exception as e:
            logger.error(f"Async streaming failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Async streaming failed: {str(e)}") from e

    def _parse_output(
        self, 
        result: Dict[str, Any]
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """
        Parse the agent output with robust error handling.

        Args:
            result: Raw agent output dictionary.

        Returns:
            Parsed output based on response_format configuration.
        """
        # If no response_format specified, extract content or return full result
        if self.response_format is None:
            return self._extract_content(result)
        
        # Try multiple strategies to extract structured data
        structured_data = None
        
        # Strategy 1: Check for structured_response key
        structured_data = result.get("structured_response")
        
        # Strategy 2: Check for output key
        if structured_data is None:
            structured_data = result.get("output")
        
        # Strategy 3: Check if result is the structured data
        if structured_data is None and isinstance(result, dict):
            # Check if it looks like a structured response (not containing agent metadata)
            agent_keys = {"messages", "output", "intermediate_steps", "input", "response_metadata"}
            if not any(key in result for key in agent_keys):
                structured_data = result
        
        # Strategy 4: Extract from last message content
        if structured_data is None:
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    structured_data = self._extract_from_content(last_msg.content)
                elif isinstance(last_msg, dict) and 'content' in last_msg:
                    structured_data = self._extract_from_content(last_msg['content'])
        
        # Validate and create the response model
        if structured_data is not None:
            try:
                return self._validate_and_create_model(structured_data)
            except ValidationError as e:
                if self.handle_parsing_errors:
                    warnings.warn(
                        f"Failed to parse response as {self.response_format.__name__}: {e}. "
                        f"Returning structured_data as is."
                    )
                    return structured_data
                raise
            except Exception as e:
                if self.handle_parsing_errors:
                    warnings.warn(f"Unexpected parsing error: {e}")
                    return self._extract_content(result)
                raise
        
        # Fallback to extracting content
        return self._extract_content(result)

    def _extract_content(self, result: Any) -> Union[Dict[str, Any], str]:
        """Extract content from various result formats."""
        if isinstance(result, dict):
            # Try to get output or content
            if "output" in result:
                return result["output"]
            elif "content" in result:
                return result["content"]
            elif "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, 'content'):
                    return last_msg.content
                elif isinstance(last_msg, dict) and 'content' in last_msg:
                    return last_msg['content']
            return result
        elif hasattr(result, 'content'):
            return result.content
        elif hasattr(result, 'text'):
            return result.text
        elif isinstance(result, (str, int, float, bool)):
            return result
        else:
            return str(result)

    def _extract_from_content(self, content: str) -> Union[Dict, str]:
        """Extract structured data from text content."""
        # Try to parse as JSON
        try:
            # Look for JSON blocks (including nested)
            json_pattern = r'(\{.*\}|\{.*\{.*\}.*\})'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # Try parsing entire content as JSON
            return json.loads(content.strip())
        except json.JSONDecodeError:
            # If not JSON, return as text
            return content

    def _validate_and_create_model(self, data: Any) -> BaseModel:
        """Create and validate the response model instance."""
        if self.response_format is None:
            raise ValueError("No response_format specified")
        
        if isinstance(data, self.response_format):
            return data
        
        try:
            if isinstance(data, dict):
                # Handle Pydantic v1/v2 compatibility
                if hasattr(self.response_format, "model_validate"):
                    return self.response_format.model_validate(data)
                else:
                    return self.response_format.parse_obj(data)
            elif isinstance(data, str):
                # Try to parse string as JSON first
                try:
                    parsed = json.loads(data)
                    return self._validate_and_create_model(parsed)
                except json.JSONDecodeError:
                    # If string is not JSON, pass as single argument
                    return self.response_format(data)
            else:
                # Try to create model with the data
                return self.response_format(data)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to create model instance: {e}")

    # Context manager for temporary configuration changes
    @contextmanager
    def temporary_config(self, **config):
        """Temporarily modify agent configuration."""
        original_config = {}
        
        for key, value in config.items():
            if hasattr(self, key):
                original_config[key] = getattr(self, key)
                setattr(self, key, value)
        
        try:
            yield self
        finally:
            for key, value in original_config.items():
                setattr(self, key, value)

    # Additional utility methods

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        if hasattr(self.agent, 'tools'):
            tools = self.agent.tools
        elif hasattr(self, 'tools'):
            tools = self.tools
        else:
            return []
        
        return [tool.name for tool in tools if hasattr(tool, "name")]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get tool names with their descriptions."""
        descriptions = {}
        for tool in self.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                descriptions[tool.name] = tool.description
        return descriptions

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration."""
        return {
            "model_type": self.llm.__class__.__name__,
            "has_response_format": self.response_format is not None,
            "response_format_type": (
                self.response_format.__name__ if self.response_format else None
            ),
            "system_prompt_length": (
                len(self.system_prompt) if self.system_prompt else 0
            ),
            "tools_count": len(self.tools),
            "tool_names": self.get_tool_names(),
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
        }

    def batch_invoke(
        self, 
        user_inputs: List[str], 
        **kwargs: Any
    ) -> List[Union[BaseModel, Dict[str, Any], str]]:
        """
        Process multiple inputs in sequence.
        
        Note: For true parallel processing, use asyncio.gather with ainvoke.
        """
        results = []
        for input_text in user_inputs:
            results.append(self.invoke(input_text, **kwargs))
        return results

    def improve_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        target_audience: Optional[str] = None,
        expected_output: Optional[str] = None,
        improvement_guidelines: Optional[Dict[str, bool]] = None,
        **kwargs: Any
    ) -> str:
        """
        Improve a prompt for better responses from Generative AI models.

        Args:
            prompt: The original prompt to improve
            context: Additional context about the use case
            target_audience: Who will be using this prompt
            expected_output: Description of the desired output format or content
            improvement_guidelines: Dictionary of improvement criteria to focus on
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Improved prompt string
        """
        # Default improvement guidelines
        if improvement_guidelines is None:
            improvement_guidelines = {
                "clarity": True,
                "specificity": True,
                "context_provision": True,
                "role_definition": True,
                "output_format": True,
                "constraints": True,
                "examples": False,
                "step_by_step": False,
            }

        # Build guidelines text
        guidelines_list = []
        guideline_descriptions = {
            "clarity": "Ensure the prompt is clear and unambiguous",
            "specificity": "Make the prompt specific and detailed",
            "context_provision": "Provide necessary context for the task",
            "role_definition": "Define the AI's role or perspective",
            "output_format": "Specify desired output format or structure",
            "constraints": "Include any constraints or limitations",
            "examples": "Include example inputs and outputs",
            "step_by_step": "Break down complex tasks into steps",
        }
        
        for guideline, enabled in improvement_guidelines.items():
            if enabled and guideline in guideline_descriptions:
                guidelines_list.append(f"- {guideline_descriptions[guideline]}")

        guidelines_text = "\n".join(guidelines_list) if guidelines_list else "No specific guidelines provided."

        # Create a structured prompt template
        system_template = """You are an expert prompt engineer specialized in optimizing prompts for Large Language Models. Your task is to analyze and improve the given prompt based on specific guidelines.

IMPORTANT INSTRUCTIONS:
1. Your output should ONLY contain the improved prompt, nothing else
2. Do not include explanations, commentary, or markdown formatting
3. Do not wrap the prompt in quotes or special characters
4. Maintain the original intent while enhancing effectiveness
5. If the original prompt is already optimal, return it with minimal changes
6. Keep the improved prompt concise but complete"""

        human_template = """ORIGINAL PROMPT TO IMPROVE:
{prompt}

{context_section}{audience_section}{output_section}
IMPROVEMENT GUIDELINES:
{guidelines}

IMPROVED PROMPT:"""

        # Prepare optional sections
        context_section = f"CONTEXT:\n{context}\n\n" if context else ""
        audience_section = (
            f"TARGET AUDIENCE:\n{target_audience}\n\n" if target_audience else ""
        )
        output_section = (
            f"EXPECTED OUTPUT:\n{expected_output}\n\n" if expected_output else ""
        )

        try:
            # Create chat prompt with system and human messages
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system_template), ("human", human_template)]
            )

            # Format the prompt
            formatted_prompt = prompt_template.format(
                prompt=prompt,
                context_section=context_section,
                audience_section=audience_section,
                output_section=output_section,
                guidelines=guidelines_text,
            )

            # Invoke the LLM with additional kwargs
            invoke_kwargs = {"messages": [{"role": "user", "content": formatted_prompt}]}
            invoke_kwargs.update(kwargs)
            
            result = self.llm.invoke(**invoke_kwargs)

            # Extract content safely
            response = self._extract_content(result)
            
            if isinstance(response, dict):
                response = str(response)
            
            # Validate the response isn't empty
            if not response or response.isspace():
                logger.warning("LLM returned empty response. Returning original prompt.")
                return prompt

            return response.strip()

        except LangChainException as e:
            logger.error(f"LangChain error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in improve_prompt: {e}")
            return prompt  # Fallback to original prompt


