import json
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_community.tools import YouTubeSearchTool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from ai_agent import AIAgent

load_dotenv()


class LessonResponse(BaseModel):
    title: str = Field(description="Declarative title for lesson")
    content: str = Field(description="Lesson content in markdown format")
    reference_links: List[str] = Field(
        default_factory=list,
        description="The URL for online reference from those official documentations",
    )
    project_links: List[str] = Field(
        default_factory=list,
        description="The URL for online project link from the github",
    )
    youtube_video_links: List[str] = Field(
        default_factory=list, description="The URL for online video from youtube"
    )


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [TavilySearch(), YouTubeSearchTool()]

agent = AIAgent(
    llm=llm,
    tools=tools,
    response_format=LessonResponse,
    system_prompt="You are a helpful assistant specialized in extracting information.",
)

result = agent.invoke(
    """
Learn me the GraphQL in markdown format. 
and provided me:
- reference from official documentation
- project link from github
- example video from youtube
"""
)


with open(f"result_{datetime.now()}.json", "w") as file:
    json.dump(result.dict(), file, indent=4, sort_keys=True)


print("\n\n\n", result, "\n\n\n")
