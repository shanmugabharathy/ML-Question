import os,shutil
from glob import glob
from pathlib import Path
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain,LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_PATH = os.getenv('DATA_PATH')
CHROMA_PATH = os.getenv('CHROMA_PATH')
HGF_API_KEY = os.getenv('HUGGINGFACEHUB_API_TOKEN')

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

# If running locally, first go to the data_ingestion.ipynb and get your data stored in the ChromaDB Locally
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

retriever = vectordb.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = HuggingFaceHub(
        repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature":0.2, "max_length":2000},
        huggingfacehub_api_token = HGF_API_KEY
    )

def fetch_dbpedia_data(query):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(f"""
    SELECT ?abstract WHERE {{
        ?article dbo:abstract ?abstract .
        ?article rdfs:label "{query}"@en .
        FILTER (lang(?abstract) = 'en')
    }}
    LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["results"]["bindings"]:
        return results["results"]["bindings"][0]["abstract"]["value"]
    return "No relevant information found on DBpedia."

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_output(output):
    output_ind = output.find("Answer")
    return output[output_ind:]


def augment_context_with_dbpedia(question):
    data = fetch_dbpedia_data(question)
    return data

rag_chain = (
  {
    "context": RunnableParallel({"context 1":retriever | format_docs, "context 2": augment_context_with_dbpedia}),
    "question": RunnablePassthrough(),
  }
  | prompt
  | llm
  | format_output
)

ans = rag_chain.invoke("What happened at the Al-Shifa Hospital?")
print(ans)

