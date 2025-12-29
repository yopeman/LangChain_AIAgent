# =========================
# Environment Setup
# =========================
from dotenv import load_dotenv
load_dotenv()

# =========================
# Imports
# =========================
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from youtube_search import YoutubeSearch
from langchain_classic.globals import set_llm_cache
from langchain_classic.cache import InMemoryCache
from diskcache import Disk, Cache
from fastapi import FastAPI
from ai_agent import AIAgent

# =========================
# Enable LLM Cache
# =========================
# Disk("./disc_files")
# Cache("")
# InMemoryCache()
set_llm_cache(Cache("./disc_cache_files"))

# =========================
# Tool Schemas
# =========================
class YouTubeVideo(BaseModel):
    title: str = Field(..., description="Video title")
    description: str = Field(..., description="Video description")
    link: str = Field(..., description="Full YouTube URL")
    thumbnails: List[str] = Field(..., description="Thumbnail image URLs")


class ReferenceSrc(BaseModel):
    title: str = Field(..., description="Reference title")
    description: str = Field(..., description="Short explanation")
    link: str = Field(..., description="Online reference link")


class ResponseOutput(BaseModel):
    explain: str = Field(..., description="Clear explanation of the topic")
    videos: List[YouTubeVideo] = Field(default_factory=list)
    references: List[ReferenceSrc] = Field(default_factory=list)

# =========================
# YouTube Tool
# =========================
@tool
def search_youtube_video(search_terms: str, max_results=5, retries=3):
    """
    Search YouTube video for educational videos and return results.
    """

    videos = YoutubeSearch(
        search_terms=search_terms, 
        max_results=max_results, 
        retries=retries
    ).to_dict(clear_cache=True)

    for video in videos:
        video["full_url"] = "https://www.youtube.com" + video["url_suffix"]
    return videos


# =========================
# LLM
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

# =========================
# Tools
# =========================
tools = [
    search_youtube_video,
    TavilySearch(max_results=5),
]

# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """
You are YopeAI, a professional educator.

Your responsibilities:
1. Explain the topic clearly and concisely.
2. Use tools when appropriate:
   - Use youtube_search_tool for tutorials or demos.
   - Use TavilySearch for articles and documentation.
3. Never hallucinate links.
4. Prefer official or reputable sources.
5. Always return your final answer strictly matching the ResponseOutput schema.

Output rules:
- explain: short but thorough explanation
- videos: only include relevant tutorials
- references: only high-quality reading material
"""

# =========================
# Agent
# =========================
agent = AIAgent(
    llm=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseOutput
)

# =========================
# Fast API
# =========================
app = FastAPI()
@app.get("/")
def ask_ai(query):
    if query == '':
        return 'Empty query'
    return agent.invoke(query)

# =========================
# Interactive CLI
# =========================
def interactive_cli() -> None:
    print("🎓 YopeAI — Educational Assistant")
    print("Type '.q' to quit\n")

    while True:
        query = input("Enter input: ").strip()
        if query == ".q":
            break
        if not query:
            continue

        try:
            result = agent.invoke({"input": query})

            print("\nAI Explanation:\n", result.explain)

            if result.references:
                print("\nReferences:")
                for i, ref in enumerate(result.references, 1):
                    print(f"  {'='*15} Reference {i} {'='*15}")
                    print("  Title:", ref.title)
                    print("  Description:", ref.description)
                    print("  Link:", ref.link)

            if result.videos:
                print("\nVideos:")
                for i, vid in enumerate(result.videos, 1):
                    print(f"  {'='*15} Video {i} {'='*15}")
                    print("  Title:", vid.title)
                    print("  Description:", vid.description)
                    print("  Link:", vid.link)
                    for j, thumb in enumerate(vid.thumbnails, 1):
                        print(f"     Thumbnail {j}: {thumb}")

            print("\n" + "="*60 + "\n")

        except Exception as e:
            print("⚠️ Error:", str(e))

if __name__ == "__main__":
    # interactive_cli()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)