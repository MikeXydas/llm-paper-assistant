from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.vector_stores import MilvusVectorStore
from llama_index.storage.storage_context import StorageContext

from utils import time_it


@time_it
def load_chat_model():
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
    return llm


@time_it
def load_embedding_model():
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
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
def initialise_query_engine():
    # get our service context (this is a wrapper around the index and the models)
    service_context = ServiceContext.from_defaults(
        llm=load_chat_model(),
        embed_model=load_embedding_model()
    )
    
    vector_store = MilvusVectorStore(collection_name="papers", dim=768, overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    index = VectorStoreIndex.from_documents(documents=load_documents(), 
                                            service_context=service_context,
                                            storage_context=storage_context)
    return index.as_query_engine()

    # response = query_engine.query("Can you give me examples of multi-task training for text-to-sql?")
    # print(response)

