from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


llm = ChatOllama(model='qwen2.5:7b-instruct', temperature = 0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content='Search for 3 job postings for AI Research scientist in Linkedin and list their details, in Kosovo or remotely')})
    print(result)




if __name__ == "__main__":
    main()
