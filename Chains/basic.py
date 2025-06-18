from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

parser = StrOutputParser()

template = PromptTemplate(
    template = "Give me 5 facts about the {topic}",
    input_variables=["topic"],
)

chain = template | model | parser
result = chain.invoke({"topic": "moon"})
print(f"Result: {result}")

chain.get_graph().print_ascii()