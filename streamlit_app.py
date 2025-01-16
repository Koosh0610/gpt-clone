import streamlit as st
import requests
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import logging
import json
import time
import os
from llama_index.llms.groq import Groq

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# File to store chat sessions and user information
CHAT_SESSIONS_FILE = "chat_sessions.json"
USER_INFO_FILE = "user_info.json"

# Function to save chat sessions to a file
def save_chat_sessions():
    with open(CHAT_SESSIONS_FILE, "w") as file:
        json.dump(st.session_state.chat_sessions, file, default=lambda o: o.__dict__)

# Function to load chat sessions from a file
def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_FILE):
        with open(CHAT_SESSIONS_FILE, "r") as file:
            chat_sessions = json.load(file)
            for session_name, messages in chat_sessions.items():
                st.session_state.chat_sessions[session_name] = [ChatMessage(**msg) for msg in messages]

# Function to save user information to a file
def save_user_info():
    with open(USER_INFO_FILE, "w") as file:
        json.dump(st.session_state.user_info, file)

# Function to load user information from a file
def load_user_info():
    if os.path.exists(USER_INFO_FILE):
        with open(USER_INFO_FILE, "r") as file:
            st.session_state.user_info = json.load(file)
    else:
        st.session_state.user_info = {}

# Function to extract and store important information from user messages using Groq
def extract_user_info(message):
    llm = Groq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = f"""
                Analyze the message for meaningful personal information about the user. If new information is found, add it to the existing memories. Focus on capturing:

                    1. Core personal details:
                    - Basic: name, age, location, occupation
                    - Background: education, skills, expertise
                    - Family: marital status, children, living situation
                    - Professional: work experience, career goals

                    2. Individual characteristics:
                    - Preferences: likes, dislikes, interests
                    - Values and beliefs
                    - Communication style
                    - Decision-making patterns

                    3. Current context:
                    - Ongoing situations or challenges
                    - Short and long-term goals
                    - Recent significant events
                    - Primary concerns or needs

                    4. Relationship dynamics:
                    - Important relationships mentioned
                    - Social connections
                    - Support systems
                    - Interaction patterns

                    Format:
                    - Maintain a clean bulleted list structure
                    - Use concise, clear language
                    - Organize by categories when possible
                    - Include timestamp for new additions
                    - Flag uncertain or inferred information

                    Input:
                    Existing memories: {st.session_state.user_info['user_info']}
                    New message: {message}

                    STRICT INSTRUCTIONS:
                    - Return only the updated memory list
                    - Use bullet points exclusively
                    - No explanatory text or JSON
                    - No meta-commentary
                    - Preserve all verified existing memories
                                    """

    extracted_info = llm.complete(prompt).text
    
    # Update user information with extracted details
    if extracted_info:
        user_info_data = {"user_info": extracted_info}
        st.session_state.user_info.update(user_info_data)
        save_user_info()

# Streamlit app configuration
st.set_page_config(page_title="🤗💬ChatGPT Clone")
st.header("ChatGPT Clone")

# Load user information
if "user_info" not in st.session_state:
    load_user_info()

# Sidebar
with st.sidebar:
    st.title('🤗💬ChatGPT Clone')
    st.success('Access to this ChatGPT Clone is provided by [Kush](https://www.linkedin.com/in/kush-juvekar/)!', icon='✅')
    st.markdown('⚡ This app is hosted on Lightning AI Studio!')

    # Chat session management
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
        load_chat_sessions()
    if "current_session" not in st.session_state:
        st.session_state.current_session = None

    # Display existing chat sessions
    st.subheader("Chat Sessions")
    for session_name in st.session_state.chat_sessions.keys():
        if st.button(session_name):
            st.session_state.current_session = session_name

    # Option to start a new chat session
    new_session_name = st.text_input("Start a new chat session", "")
    if st.button("Start New Session"):
        if new_session_name:
            st.session_state.chat_sessions[new_session_name] = [ChatMessage(role=MessageRole.ASSISTANT, content="Ask anything!")]
            st.session_state.current_session = new_session_name
            save_chat_sessions()

# Initialize chat history for the current session
if st.session_state.current_session:
    chat_history = st.session_state.chat_sessions[st.session_state.current_session]
else:
    chat_history = []

# Chat interface
for message in chat_history:
    with st.chat_message(message.role.value):
        st.write(message.content)

# User input
if prompt := st.chat_input("Your question"):
    user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    chat_history.append(user_message)
    extract_user_info(prompt)  # Extract and store user information
    with st.chat_message("user"):
        st.write(prompt)

    # Make API call to FastAPI server
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.write(st.session_state.user_info)
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "prompt": prompt,
                        "message_history": [
                            {"role": msg.role.value, "content": msg.content}
                            for msg in chat_history[:-1]
                        ],
                        "user_info": str(st.session_state.user_info)
                    }
                )
                response.raise_for_status()
                assistant_response = response.json()["response"]
                st.write(assistant_response)

                chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))
                save_chat_sessions()
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {str(e)}")

# Update the session state with the latest chat history
if st.session_state.current_session:
    st.session_state.chat_sessions[st.session_state.current_session] = chat_history
    save_chat_sessions()
