import openai

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP, OpenAI
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.vector_stores import MilvusVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import OpenAIEmbedding

from utils import get_embedding_dimension, time_it

openai.api_key = "KEY"


@time_it
def load_chat_model(which_llm):
    llm = None
    if which_llm == "local":
        llm = LlamaCPP(
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_url="https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q3_K_M.gguf",
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=False,
        )
    elif which_llm == "openai":
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)

    return llm


@time_it
def load_embedding_model(which_llm):
    embed_model = None
    if which_llm == "local":
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    elif which_llm == "openai":
        embed_model = OpenAIEmbedding()

    return embed_model


@time_it
def generate_response(llm, prompt):
    response = llm.complete(prompt)
    return response.text


@time_it
def load_documents():
    documents = SimpleDirectoryReader("storage/papers").load_data()
    return documents


@time_it
def initialise_query_engine(which_llm):
    # get our service context (this is a wrapper around the index and the models)
    service_context = ServiceContext.from_defaults(
        llm=load_chat_model(which_llm=which_llm),
        embed_model=load_embedding_model(which_llm=which_llm)
    )
    
    vector_store = MilvusVectorStore(collection_name="papers", dim=get_embedding_dimension(which_llm), overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    index = VectorStoreIndex.from_documents(documents=load_documents(), 
                                            service_context=service_context,
                                            storage_context=storage_context)
    return index.as_query_engine()

    # response = query_engine.query("Can you give me examples of multi-task training for text-to-sql?")
    # print(response)

