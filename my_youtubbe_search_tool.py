from youtube_searcher import search_youtube_video, llm, BaseModel, List, Field
from langchain_community.tools import YouSearchTool

from ai_agent import AIAgent

# results = search_youtube_video("graphql API")

# for result in results:
#     print("\n\n\n")
#     print("Title:", result['title'])
#     print("ID:", result['id'])
#     print("Description:", result['long_desc'])
#     print("Channel:", result['channel'])
#     print("Duration:", result['duration'])
#     print("Views:", result['views'])
#     print("Publish Time:", result['publish_time'])
#     print("URL Suffix:", result['url_suffix'])
#     print("Full URL:", result['full_url'])
#     print("Thumbnails:")

#     for thumbnail in result['thumbnails']:
#         print("\t->", thumbnail)

#     print("")

class ResponseOutput(BaseModel):
    explain: str = Field(..., description="Clear explanation of the topic")
    src: List[str] = Field(list, description="Youtube video link URLs")

agent = AIAgent(llm=llm, tools=[YouSearchTool], response_format=ResponseOutput)
result = agent.invoke("What is GraphQL? explain and give me youtube videos")
print(result)