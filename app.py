from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
from pandas import DataFrame
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain import SQLDatabase, SQLDatabaseChain, LLMChain, OpenAI, PromptTemplate, FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory

load_dotenv()

# loader = TextLoader('./zkrollup.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
# vector_db = FAISS.from_documents(docs, embeddings)
#
# extra_info: DataFrame = pd.read_excel("qa.xlsx")
# qa_list = extra_info.to_dict(orient="records")
# for qa_item in qa_list:
#     document = Document(page_content=f"Q: {qa_item['Question']}\nA: {qa_item['Answer']}")
#     vector_db.add_documents([document])
# vector_db.save_local("./embeddings")
# print("complete")

embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local("./embeddings", embeddings)
llm = OpenAI(temperature=0)

vector_template = """You are a chatbot having a conversation with a human

Given the following extracted parts of a long document and a question, create a final answer.

{context}

Human: {human_input}
Chatbot:"""

vector_prompt = PromptTemplate(
    input_variables=["human_input", "context"],
    template=vector_template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
vector_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=vector_prompt, memory=memory)

app = Flask(__name__)
CORS(app)

@app.route('/mad', methods=['POST'])
def mad():
    query = request.form["prompt"]
    docs = vector_db.similarity_search(query)
    vector_output = vector_chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)

    return {"answer": vector_output["output_text"]}
