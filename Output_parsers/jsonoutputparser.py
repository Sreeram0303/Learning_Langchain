from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me a name, age and occupation of a fictional character. \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions() }
)

# prompt = template.invoke({})
# print(f"Prompt: {prompt}")

# result = model.invoke(prompt)

# print(result.content)

# final_result = parser.parse(result.content)
# print(f"Final Result: {final_result}")

chain = template | model | parser
result = chain.invoke({})
print(f"Final Result: {result}")