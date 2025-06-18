from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatHuggingFace(
    lllm = HuggingFaceEndpoint(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        task = "text-generation"
    )
)

schema = [
    ResponseSchema(name="fact_1",description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2",description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide three interesting facts about {topic}. \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template| model | parser

result = chain.invoke({"topic": "black hole"})
print(f"Final Result: {result}")