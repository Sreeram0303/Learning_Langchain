from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableSequence

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the joke in simple terms {text}',
    input_variables=['text']
)

joke_generator = RunnableSequence(prompt1,model,parser)
explanation_generator = RunnableSequence(prompt2, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": explanation_generator,
    }
)

final_chain = RunnableSequence(joke_generator, parallel_chain)

# or else we can directly format the output by passing the joke and explanation we got from the parallel chain no need to create a new chain

prompt3 = PromptTemplate(
    template = "Merge the provided joke and explanation into a single document \n joke -> {joke} and explanation -> {explanation}",
)

chain = RunnableSequence(final_chain,prompt3, model, parser)

result = chain.invoke({"topic": "cats"})
print(result)

chain.get_graph().print_ascii()