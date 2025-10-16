
from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool,ObjectDetectionTool

api_key = st.secrets["OPENAI_API_KEY"]

#initialising agent
tools = [ImageCaptionTool(),ObjectDetectionTool()]

#for agent to remember conversation history
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

#large language model
llm = ChatOpenAI(
    openai_api_key=str(api_key),
    temperature=0,
    model_name = "gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm = llm,
    max_iterations = 5,
    verbose = True,
    memory = conversational_memory,
    early_stoppy_method='generate'
)


st.title('Ask a question to an image')
st.header("Upload an image")

file = st.file_uploader("",type=["jpeg","jpg","png"])

if file:
    st.image(file,use_column_width=True)
    user_question = st.text_input('Ask a question about your image')
    
    with NamedTemporaryFile(dir='.',delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name

    if user_question and user_question!="":
            with st.spinner(text="In progress..."):
                    response = agent.run('{}, this is the image path: {}'.format(user_question,image_path))
                    st.write(response)