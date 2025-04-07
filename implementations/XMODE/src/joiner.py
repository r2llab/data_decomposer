from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing import Sequence
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)


class FinalResponse(BaseModel):
    """The final response/answer."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Example response or {'key': 'value'}"
                }
            ]
        }
    }
    response: Union[str,Dict]
    tables_used: Optional[List[str]] = Field(
        default=[],
        description="List of tables used during the query execution."
    )


class Replan(BaseModel):
    """Model for replanning feedback"""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feedback": "Analysis of what went wrong and what needs to be fixed"
                }
            ]
        }
    }
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thought": "Example thought process",
                    "action": {"response": "Example response"}
                }
            ]
        }
    }
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


def parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        # Extract tables_used if present
        if hasattr(decision.action, 'tables_used') and decision.action.tables_used:
            tables_info = f"\nTables used in this query: {', '.join(decision.action.tables_used)}"
            return response + [AIMessage(content=str(decision.action.response) + tables_info)]
        else:
            return response + [AIMessage(content=str(decision.action.response))]


def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}