from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


from dotenv import load_dotenv
import os
import openai
import shutil

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

########################variables and constants###############################################

llm_name = "gpt-4o"             # LLM model
temperature = 0.0               # degree of randomness of the model's output
persist_directory =  './db/'    # persist directory for embedded document
Chunk_size = 1000               # document embedding chunk size parameter
Chunk_overlap = 150             # document embedding chunk overlap parameter
K = 4                           # max number of documents k returned by the retriever.
Chain_type = 'stuff'            # chain type
max_history_len = 10            # max number of latest chat history messages stored
session_id = "test123"          # chat session ID
store={}                        

################################################################################

llm = ChatOpenAI(model_name=llm_name, temperature=temperature)
greeting = llm.invoke("Hello world!")
print(greeting.content)

def check_chroma_db_exists(db_path):
    try:
        # Attempt to load the database
        old_db = Chroma(persist_directory=db_path)
        # If successful, the database exists
        return True
    except Exception as e:
        # If there's an error, the database likely doesn't exist
        return False

def embed_docs():
    loaded_file = "./docs/"     #specify the folder that stores the document
    print(f'embed pdf document from folder {loaded_file} and save to database')

    #load all documents in directory
    loader = PyPDFDirectoryLoader(loaded_file)
    documents = loader.load() 
    # documents = a list of pages of the documents in the directory
    # where
    # Number of items in the list = total number of pages of all documents in the directory 
    # an item in the list = each page of the documents 

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap)     
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings = OpenAIEmbeddings()

    #clear up existing database before embedding (if a database exists)
    if check_chroma_db_exists(persist_directory):
        print("Chroma database exists")
        vector_store = Chroma(persist_directory=persist_directory)  # Where to save data locally, remove if not necessary
        vector_store.reset_collection()
        print("Finished resetting database")
    else:
        print("Chroma database does not exist")

    # create vector database from data
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    print('Successful Document Embedding')
    


client = openai.OpenAI()
def get_completion(prompt, model=llm_name, temperature=temperature):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message.content


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    if(len(store[session_id].messages) >= max_history_len):
        del store[session_id].messages[0:2]
    return store[session_id]

# Contextualize question into a standalone question based on chat history
def create_standalone_question(input_question):
    contextualize_q_system_prompt2 = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt2),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    runnable = contextualize_q_prompt2 | llm
    
    standalone_question = runnable.invoke(
        {"input": input_question,
        "chat_history": get_session_history(session_id).messages}
        )
    return standalone_question

# Create a retriever
def create_retrieval(K):   
    fembeddings = OpenAIEmbeddings()
    fdb = Chroma(persist_directory=persist_directory, embedding_function=fembeddings)
    fretriever = fdb.as_retriever(search_type="similarity", search_kwargs={"k": K})
    print(persist_directory)
    return fretriever


class AskMe:
       
    def ask(self, input_question: str):
        
        standalone_question = create_standalone_question(input_question)
        fretriever = create_retrieval(K)


        # Answer question 
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use only the following pieces of retrieved context to answer the question. \
        If the following pieces of context does not provide any information about the question, \
        just say that you don't know the answer without trying to guessing or answering. \

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(fretriever, question_answer_chain)


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        print("original question: "+input_question)
        print("standalone question: "+standalone_question.content)
   
        response = conversational_rag_chain.invoke(
            {"input": standalone_question.content,
            "chat_history": {}},
            config={
                "configurable": {"session_id": session_id}
            }
        )
        
        return response 

    