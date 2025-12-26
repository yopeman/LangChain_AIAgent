from langchain_groq import ChatGroq
from langchain_core.tools import tool
from pydantic import BaseModel
from ai_agent import AIAgent
from dotenv import load_dotenv

load_dotenv()

# 1. Define a dummy tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25C."

# 2. Define a response format (Optional)
class WeatherReport(BaseModel):
    city: str
    temperature: str
    summary: str

# 3. Initialize Agent
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
agent = AIAgent(
    llm=llm,
    tools=[get_weather],
    response_format=WeatherReport, # Optional: if you want structured output
    debug=False
)

# 4. Run
# The agent will:
#  1. See the user input
#  2. Decide to call 'get_weather("London")'
#  3. Execute the tool
#  4. Take the tool output and format it into the WeatherReport JSON
response = agent.invoke("What is the weather in London?")
print(response)
# Output: WeatherReport(city='London', temperature='25C', summary='The weather in London is sunny...')