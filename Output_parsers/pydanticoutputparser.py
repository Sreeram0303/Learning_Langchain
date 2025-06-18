from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        task = "text-generation"
    )
)

class Person(BaseModel):
    name: str = Field(description="Name of the person.")
    age: int = Field(gt=18, description="Age of the person.")
    city: str = Field(description="Name of the city the person belongs to.")
    
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Provide a fictional character's name, age, and {place} of residence. \n{format_instructions}",
    input_variables = ["place"],
    partial_variables = {"format_instructions": parser.get_format_instructions() }
)

prompt = template.invoke({"place": "indian"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(f"Final Result: {final_result}")