import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:7b-instruct")

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in europe on linkedin and list their details"
                }
            ]
        }
    )
    
    # Extract structured response if available
    # In Python, structured responses are typically in result["structured_response"]
    if isinstance(result, dict) and "structured_response" in result:
        print(result["structured_response"])
    elif isinstance(result, dict) and "messages" in result:
        # Fallback to last message content
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            print(last_message.content)
        elif isinstance(last_message, dict) and "content" in last_message:
            print(last_message["content"])
        else:
            print(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
