#pip install langchain, openai, gradio, chromadb 

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

# create chatbot function 
def chatbot(input, history):
    history = history or []
    chat_list = list(sum(history, ()))
    chat_list.append(input)
    query = ' '.join(chat_list) #this is whole chat list 
    response = qa({"query": input}) #query will send whole chat, input will only send single message to model
    history.append((input, response["result"]))
    return history, history

# create UI
with gr.Blocks() as block:
    gr.Markdown("""<h1><center>Custom Knowledge Chatbot</center></h1>""")
    chatbots = gr.Chatbot()
    message = gr.Textbox(lines=1, max_lines=1, placeholder="Enter your message here.", label="User input")
    state = gr.State()
    submit = gr.Button("Click to submit")
    submit.click(chatbot, inputs=[message, state], outputs=[chatbots, state])

# run chatbot
block.launch() #add share=True within brackets for public link 
