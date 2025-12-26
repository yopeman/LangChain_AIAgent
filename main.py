from dotenv import load_dotenv
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import LangChainException
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

def improve_prompt(
    llm: BaseChatModel,
    prompt: str,
    context: Optional[str] = None,
    target_audience: Optional[str] = None,
    expected_output: Optional[str] = None,
    improvement_guidelines: Optional[Dict[str, bool]] = None
) -> str:
    """
    Improve a prompt for better responses from Generative AI models.
    
    Args:
        llm: The language model to use for improvement
        prompt: The original prompt to improve
        context: Additional context about the use case
        target_audience: Who will be using this prompt (e.g., "students", "developers")
        expected_output: Description of the desired output format or content
        improvement_guidelines: Dictionary of improvement criteria to focus on
        
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
            "constraints": True
        }
    
    # Build guidelines text
    guidelines_list = []
    for guideline, enabled in improvement_guidelines.items():
        if enabled:
            if guideline == "clarity":
                guidelines_list.append("- Ensure the prompt is clear and unambiguous")
            elif guideline == "specificity":
                guidelines_list.append("- Make the prompt specific and detailed")
            elif guideline == "context_provision":
                guidelines_list.append("- Provide necessary context for the task")
            elif guideline == "role_definition":
                guidelines_list.append("- Define the AI's role or perspective")
            elif guideline == "output_format":
                guidelines_list.append("- Specify desired output format or structure")
            elif guideline == "constraints":
                guidelines_list.append("- Include any constraints or limitations")
    
    guidelines_text = "\n".join(guidelines_list)
    
    # Create a structured prompt template
    system_template = """You are an expert prompt engineer specialized in optimizing prompts for Large Language Models. Your task is to analyze and improve the given prompt based on specific guidelines.

IMPORTANT INSTRUCTIONS:
1. Your output should ONLY contain the improved prompt, nothing else
2. Do not include explanations, commentary, or markdown formatting
3. Do not wrap the prompt in quotes or special characters
4. Maintain the original intent while enhancing effectiveness
5. If the original prompt is already optimal, return it with minimal changes"""

    human_template = """ORIGINAL PROMPT TO IMPROVE:
{prompt}

{context_section}{audience_section}{output_section}
IMPROVEMENT GUIDELINES:
{guidelines}

IMPROVED PROMPT:"""
    
    # Prepare optional sections
    context_section = f"CONTEXT:\n{context}\n\n" if context else ""
    audience_section = f"TARGET AUDIENCE:\n{target_audience}\n\n" if target_audience else ""
    output_section = f"EXPECTED OUTPUT:\n{expected_output}\n\n" if expected_output else ""
    
    try:
        # Create chat prompt with system and human messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            prompt=prompt,
            context_section=context_section,
            audience_section=audience_section,
            output_section=output_section,
            guidelines=guidelines_text
        )
        
        # Invoke the LLM
        result = llm.invoke(formatted_prompt)
        
        # Extract content safely
        if hasattr(result, 'content'):
            response = result.content.strip()
        elif hasattr(result, 'text'):
            response = result.text.strip()
        elif isinstance(result, str):
            response = result.strip()
        else:
            response = str(result).strip()
        
        # Validate the response isn't empty
        if not response:
            logger.warning("LLM returned empty response. Returning original prompt.")
            return prompt
            
        return response
        
    except LangChainException as e:
        logger.error(f"LangChain error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return prompt  # Fallback to original prompt

# Optional: Function to get improvement suggestions
def get_prompt_feedback(
    llm: BaseChatModel,
    prompt: str,
    include_suggestions: bool = True
) -> Dict[str, Any]:
    """
    Get detailed feedback and suggestions for prompt improvement.
    """
    feedback_template = """Analyze this prompt and provide constructive feedback:

PROMPT: {prompt}

Provide feedback in this format:
STRENGTHS: [List 2-3 strengths]
AREAS FOR IMPROVEMENT: [List 2-3 areas]
{optional_suggestions}"""

    optional_section = "SUGGESTED IMPROVEMENTS: [Provide 2-3 specific improvement suggestions]" if include_suggestions else ""
    
    try:
        result = llm.invoke(feedback_template.format(
            prompt=prompt,
            optional_suggestions=optional_section
        ))
        
        # Parse the response (simplified parsing - could be enhanced)
        feedback_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "original_prompt": prompt,
            "feedback": feedback_text,
            "has_suggestions": include_suggestions
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback: {e}")
        return {"error": str(e), "original_prompt": prompt}

# Example usage
if __name__ == "__main__":
    # Basic usage
    basic_prompt = "What is Global Warming?"
    improved = improve_prompt(llm=llm, prompt=basic_prompt)
    print("Improved prompt:", improved)
    print("\n" + "="*50 + "\n")
    
    # Advanced usage with context
    advanced_prompt = "Explain machine learning"
    improved_with_context = improve_prompt(
        llm=llm,
        prompt=advanced_prompt,
        context="For a technical blog post aimed at software engineers",
        target_audience="Mid-level developers with some ML exposure",
        expected_output="A comprehensive explanation with examples of supervised and unsupervised learning",
        improvement_guidelines={
            "clarity": True,
            "specificity": True,
            "context_provision": True,
            "role_definition": True,
            "output_format": True,
            "constraints": True
        }
    )
    print("Advanced improved prompt:", improved_with_context)
    print("\n" + "="*50 + "\n")
    
    # Get feedback
    feedback = get_prompt_feedback(llm=llm, prompt=basic_prompt)
    print("Feedback:", feedback["feedback"])