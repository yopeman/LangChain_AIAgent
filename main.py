from dotenv import load_dotenv
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

