import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate


template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:7b-instruct")

react_prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm = llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor



def main():
    print("Hello from langchain-course!")

    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ai engineer using langchain in europe on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
