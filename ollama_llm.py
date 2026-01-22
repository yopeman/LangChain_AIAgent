from langchain_ollama import OllamaLLM
llm = OllamaLLM(model='gemma3:1b')

# response = llm.invoke("What is Machine Learning?")
# print(response)

for chunk in llm.stream("What is Machine Learning?"):
    print(chunk, end='', flush=True)