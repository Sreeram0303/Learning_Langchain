from langchain_core.runnables import RunnableLambda,RunnableSequence,RunnableParallel, RunnablePassthrough
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

def word_count(text):
    return len(text.split())

runnable_word_count = RunnableLambda(word_count)

prompt = PromptTemplate(
    template = "Generate a short story about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

story_generator  = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        "story" : RunnablePassthrough(),
        "word_count": runnable_word_count,
    }
)

final_chain = RunnableSequence(story_generator, parallel_chain)

result = final_chain.invoke({"topic": "a brave knight"})
print(result)
final_chain.get_graph().print_ascii()
