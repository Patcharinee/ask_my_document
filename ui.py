import streamlit as st
from streamlit_chat import message
from qa import embed_docs, AskMe
from langchain_core.messages import AIMessage, HumanMessage


st.set_page_config(page_title="Ask Me", page_icon="🤖")
st.title("Ask Me")
with st.sidebar:
    st.header("Embed your own documents")
    embed_button = st.button("Embed")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello ! I'm your Document bot. How can I help you?"),
    ]
    
#embed reference docs to vector store database
if embed_button:
    embed_docs()
    print('Finished embedding docs')
    with st.sidebar:
        st.write("Successful document embedding :)")

#ask questions
a = AskMe()
user_query = st.chat_input("Type your question here...")
if user_query is not None and user_query != "":
    #response = a.ask(user_query)['answer']
    response = a.ask(user_query)
    print(response['answer'])
    print(response['context'])
        
    print('-----------------------------------------')
    with st.sidebar:
        st.subheader("Source :")
        #st.write(response['context'])
        for source in response['context']:
            #st.write(source.metadata)
            st.write(f'{source.metadata["source"]} page: {source.metadata["page"]}')

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

print(st.session_state)