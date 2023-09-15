import uvicorn

from fastapi import FastAPI
from llm import initialise_query_engine
from utils import time_it

app = FastAPI()

query_engine = initialise_query_engine(which_llm="local")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@time_it
@app.get("/assistant/{query}")
def read_item(query: str):
    return {"response": query_engine.query(query)}


def main():
    uvicorn.run("app:app",
                host='0.0.0.0',
                port=1457,
                workers=1)


if __name__ == "__main__":
    main()