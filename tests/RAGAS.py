from dotenv import load_env

import os
load_env()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

sample_queries = [] # Store question vào trong lày
responses = []
dataset = []
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

from modules.text_processing.contextual_responder import ContextualResponder
response_generator = ContextualResponder()
for query,reference in zip(sample_queries,responses):

    # relevant_docs = rag.get_most_relevant_docs(query)
    response, context = response_generator.rag_response(query)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":[context],
            "response":response,
            "reference":reference
        }
    )

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
print (result)
