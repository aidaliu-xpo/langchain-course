from dotenv import  load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Richard Phillips Feynman (1918–1988) was an American theoretical physicist best known for major 
    contributions to quantum electrodynamics, including the path integral formulation and the pictorial 
    “Feynman diagrams,” and he shared the 1965 Nobel Prize in Physics for that work. He studied at MIT and earned
    his PhD at Princeton under John Archibald Wheeler, developing ideas that reshaped how physicists calculate 
    particle interactions. During World War II he worked at Los Alamos on the Manhattan Project, where he led 
    computation efforts and helped with practical safety and calculation problems. After the war he taught at 
    Cornell and later became a long-time professor at Caltech, where he influenced generations through his 
    lectures and distinctive teaching style. He also made important contributions across several areas, 
    including superfluid helium, particle physics (parton model), and early thinking about quantum computing and nanotechnology. Outside academia, he became widely known as a vivid public communicator of science and played a key role on the Rogers Commission investigating the 1986 Space Shuttle Challenger disaster, famously demonstrating O-ring failure in cold conditions. He remains a cultural icon for his clarity, curiosity, and insistence on intellectual honesty—summed up by his warning that “nature cannot be fooled.”
    """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. A short summary
        2. Two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
        input_variables=['information'], template=summary_template
    )

    llm = OllamaLLM(temperature=0, model='llama3.1:8b')
    chain = summary_prompt_template | llm

    response = chain.invoke(input={'information': information})
    print(response)



if __name__ == "__main__":
    main()
