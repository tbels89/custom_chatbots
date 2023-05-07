#pip install langchain, openai, gradio

# best for single questions on documents. does not track chat history 

import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# prepare environment with API key 
api_key = os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

# define model parameters 
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1900,
    n=1,
    openai_api_key=api_key)

# load your .txt data
directory_path = "./data"
loader = DirectoryLoader(directory_path, glob='**/*.txt') 
data = loader.load()

# create vector database of your .txt data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_data = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_data = Chroma.from_documents(split_data, embeddings)

# define connection to model
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_data.as_retriever())

# create query function 
def QandA(query):
    response = qa({"query": query})
    return response["result"]

iface = gr.Interface(fn=QandA, inputs="text", outputs="text")
iface.launch() #add share=True for a public link 