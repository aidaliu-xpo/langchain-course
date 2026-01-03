from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description='The URL of the source')

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description='List of sources used to generate the answer')


llm = ChatOllama(model='qwen2.5:7b-instruct', temperature = 0, format='json')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": HumanMessage(content=
                                 "Search for 3 remote AI Engineer/Scientist job postings in linkedin. "
                                 "Return: Title | Company | Location/Remote | Link"
                                 )
    })
    print(result)




if __name__ == "__main__":
    main()
