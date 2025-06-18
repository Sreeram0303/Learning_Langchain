from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a tweet for the following topic \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedIn post for the following topic \n {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel(
    {
        "tweet" : RunnableSequence(prompt1, model, parser),
        "linkedin_post" : RunnableSequence(prompt2, model, parser)
    }
)

result = parallel_chain.invoke({"topic": "AI and its impact on society"})
print(result)