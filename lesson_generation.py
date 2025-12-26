import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import traceback

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field, field_validator

# Custom YouTube tool implementation since YouTubeSearchTool may not work reliably
from langchain.tools import BaseTool
from urllib.parse import quote

from ai_agent import AIAgent

load_dotenv()

class LessonResponse(BaseModel):
    """Enhanced lesson response model with validation"""
    title: str = Field(description="Declarative title for lesson")
    content: str = Field(description="Lesson content in markdown format")
    reference_links: List[str] = Field(
        default_factory=list,
        description="URLs for official documentation references"
    )
    project_links: List[str] = Field(
        default_factory=list,
        description="URLs for GitHub project examples"
    )
    youtube_video_links: List[str] = Field(
        default_factory=list, 
        description="URLs for educational YouTube videos"
    )
    
    # Add metadata fields
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when response was generated"
    )
    query: Optional[str] = Field(
        default=None,
        description="Original user query that generated this response"
    )
    
    @field_validator('reference_links', 'project_links', 'youtube_video_links')
    @classmethod
    def validate_links(cls, v: List[str]) -> List[str]:
        """Validate URLs"""
        validated = []
        for link in v:
            if isinstance(link, str) and link.startswith(('http://', 'https://')):
                validated.append(link)
        return validated
    
    def get_summary(self) -> dict:
        """Get a summary of the lesson"""
        return {
            "title": self.title,
            "reference_count": len(self.reference_links),
            "project_count": len(self.project_links),
            "video_count": len(self.youtube_video_links),
            "content_length": len(self.content)
        }


class GraphQLLessonGenerator:
    """Main class for generating GraphQL lessons"""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model_name, temperature=0.2)  # Slightly higher temp for creativity
        
        # Initialize tools
        self.tavily_tool = TavilySearch(max_results=5)
                
        # Create tools list
        self.tools = [self.tavily_tool]
        
        # Create agent with enhanced system prompt
        self.agent = AIAgent(
            llm=self.llm,
            tools=self.tools,
            response_format=LessonResponse,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt"""
        return """
        You are an expert GraphQL educator and technical content creator.
        
        Your task is to create comprehensive learning materials about GraphQL.
        
        GUIDELINES:
        1. Create accurate, up-to-date content about GraphQL
        2. Use tools to find current information, documentation, projects, and videos
        3. Format content in clear markdown with proper headings, code blocks, and examples
        4. Prioritize official, authoritative sources
        5. Include practical examples and real-world use cases
        
        CONTENT STRUCTURE:
        - Start with an engaging introduction
        - Explain core concepts (queries, mutations, schemas, resolvers)
        - Include code examples
        - Mention best practices
        - Provide learning resources
        
        TOOL USAGE:
        - Use TavilySearch for finding documentation and GitHub projects
        - Use YouTube search for finding tutorial videos (if available)
        
        Always verify information from multiple sources when possible.
        """
    
    def generate_lesson(self, query: str) -> LessonResponse:
        """Generate a lesson based on query"""
        try:
            # Enhanced user prompt
            enhanced_query = f"""
            Create a comprehensive lesson about: {query}
            
            Please provide:
            1. A detailed lesson in markdown format
            2. Links to official documentation
            3. Links to example projects on GitHub
            4. Links to educational YouTube videos
            
            Focus on practical, actionable information.
            """
            
            result = self.agent.invoke(enhanced_query)
            
            # Add query to response for context
            if isinstance(result, LessonResponse):
                result.query = query
            
            return result
            
        except Exception as e:
            print(f"Error generating lesson: {e}")
            traceback.print_exc()
            # Return a basic response in case of error
            return LessonResponse(
                title="Error Generating Lesson",
                content=f"An error occurred: {str(e)}",
                query=query
            )
    
    async def generate_lesson_async(self, query: str) -> LessonResponse:
        """Async version of generate_lesson"""
        try:
            enhanced_query = f"Create a comprehensive lesson about: {query}"
            result = await self.agent.ainvoke(enhanced_query)
            
            if isinstance(result, LessonResponse):
                result.query = query
            
            return result
        except Exception as e:
            return LessonResponse(
                title="Error Generating Lesson",
                content=f"An error occurred: {str(e)}",
                query=query
            )
    
    def save_result(self, result: LessonResponse, filename: Optional[str] = None):
        """Save result to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graphql_lesson_{timestamp}.json"
        
        # Ensure directory exists
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        # Convert to dict and save
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)
        
        print(f"Lesson saved to: {filepath}")
        return filepath
    
    def print_lesson(self, result: LessonResponse):
        """Print lesson in a formatted way"""
        print("\n" + "="*80)
        print(f"LESSON: {result.title}")
        print("="*80)
        
        print(f"\n📚 Generated at: {result.generated_at}")
        print(f"🔍 Original query: {result.query}")
        
        print(f"\n📝 CONTENT ({len(result.content)} characters):")
        print("-"*40)
        # Print first 500 chars of content preview
        preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
        print(preview)
        
        print(f"\n🔗 REFERENCES ({len(result.reference_links)}):")
        for i, link in enumerate(result.reference_links[:3], 1):  # Show first 3
            print(f"  {i}. {link}")
        if len(result.reference_links) > 3:
            print(f"  ... and {len(result.reference_links) - 3} more")
        
        print(f"\n💻 PROJECTS ({len(result.project_links)}):")
        for i, link in enumerate(result.project_links[:3], 1):
            print(f"  {i}. {link}")
        if len(result.project_links) > 3:
            print(f"  ... and {len(result.project_links) - 3} more")
        
        print(f"\n🎥 VIDEOS ({len(result.youtube_video_links)}):")
        for i, link in enumerate(result.youtube_video_links[:3], 1):
            print(f"  {i}. {link}")
        if len(result.youtube_video_links) > 3:
            print(f"  ... and {len(result.youtube_video_links) - 3} more")
        
        print("\n" + "="*80)
        
        # Print summary
        summary = result.get_summary()
        print(f"\n📊 SUMMARY:")
        print(f"  • Title: {summary['title']}")
        print(f"  • References: {summary['reference_count']}")
        print(f"  • Projects: {summary['project_count']}")
        print(f"  • Videos: {summary['video_count']}")
        print(f"  • Content length: {summary['content_length']} characters")


def main():
    """Main execution function"""
    print("🚀 GraphQL Lesson Generator")
    print("Initializing...")
    
    # Initialize generator
    generator = GraphQLLessonGenerator()
    
    # Define query
    query = """
    Learn me GraphQL in markdown format including:
    - Core concepts (queries, mutations, schemas, resolvers)
    - Comparison with REST API
    - Practical examples
    - Best practices
    - And provide:
      * Reference from official documentation
      * Project links from GitHub
      * Example videos from YouTube
    """
    
    print(f"\n📖 Generating lesson for: GraphQL")
    print("This may take a moment...\n")
    
    # Generate lesson
    lesson = generator.generate_lesson(query)
    
    # Print lesson
    generator.print_lesson(lesson)
    
    # Save result
    saved_path = generator.save_result(lesson)
    
    # Also save raw content to markdown file
    if lesson.content:
        md_filename = str(saved_path).replace('.json', '.md')
        with open(md_filename, "w", encoding="utf-8") as md_file:
            md_file.write(f"# {lesson.title}\n\n")
            md_file.write(lesson.content)
        print(f"Markdown content saved to: {md_filename}")
    
    print("\n✅ Lesson generation complete!")


async def main_async():
    """Async version of main function"""
    generator = GraphQLLessonGenerator()
    query = "Learn me GraphQL with practical examples"
    lesson = await generator.generate_lesson_async(query)
    generator.print_lesson(lesson)
    generator.save_result(lesson)


if __name__ == "__main__":
    # Check if running in an async context
    try:
        import sys
        if sys.argv[-1] == "--async":
            asyncio.run(main_async())
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()