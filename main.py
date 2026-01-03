import warnings

from langchain.chains.question_answering.map_rerank_prompt import output_parser

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOllama(model="qwen2.5:7b-instruct")
structured_llm = llm.with_structured_output(AgentResponse)


react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad","tool_names"],).partial(format_instructions="")

agent = create_react_agent(
    llm = llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x:x ['output'])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output



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
