from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None

def init_openai(api_key: str):
    """Initialize the OpenAI client with the given API key"""
    global client
    client = OpenAI(api_key=api_key)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def create_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> str:
    """
    Create a chat completion using OpenAI's API with retry logic.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The model to use (default: gpt-4o)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: None)
        top_p: Nucleus sampling parameter (default: 1.0)
        frequency_penalty: Frequency penalty parameter (default: 0.0)
        presence_penalty: Presence penalty parameter (default: 0.0)
        
    Returns:
        The generated response text
    
    Raises:
        openai.OpenAIError: If the API call fails after retries
    """
    global client
    if client is None:
        raise ValueError("OpenAI client not initialized. Call init_openai() first.")

    try:
        # Log the request
        logger.debug(f"Sending request to {model}")
        logger.debug(f"Messages: {messages}")
        
        # Make the API call
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        end_time = time.time()
        
        # Log the response time
        duration = end_time - start_time
        logger.debug(f"Request completed in {duration:.2f} seconds")
        
        # Extract and return the response text
        response_text = response.choices[0].message.content
        return response_text
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_chat_completion: {str(e)}")
        raise

def create_system_prompt(role: str, task: str, format_instructions: Optional[str] = None) -> str:
    """
    Create a standardized system prompt.
    
    Args:
        role: The role the AI should assume
        task: The specific task description
        format_instructions: Optional formatting instructions for the response
        
    Returns:
        Formatted system prompt
    """
    prompt = f"You are an expert {role}.\n\n{task}"
    
    if format_instructions:
        prompt += f"\n\nPlease format your response as follows:\n{format_instructions}"
        
    return prompt

def create_user_prompt(
    question: str,
    context: Optional[str] = None,
    additional_instructions: Optional[str] = None
) -> str:
    """
    Create a standardized user prompt.
    
    Args:
        question: The main question or task
        context: Optional context or background information
        additional_instructions: Optional additional instructions
        
    Returns:
        Formatted user prompt
    """
    prompt_parts = []
    
    if context:
        prompt_parts.append(f"Context:\n{context}\n")
        
    prompt_parts.append(f"Question: {question}")
    
    if additional_instructions:
        prompt_parts.append(f"\nAdditional Instructions:\n{additional_instructions}")
        
    return "\n\n".join(prompt_parts) 