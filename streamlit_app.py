import streamlit as st
import requests
import logging
import json
import os
from typing import Dict, List, Any

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.groq import Groq

from concurrent.futures import ThreadPoolExecutor
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHAT_SESSIONS_FILE = "chat_sessions.json"
USER_INFO_FILE = "user_info.json"
DEFAULT_LLM_MODEL = "deepseek-r1-distill-llama-70b"
DEFAULT_INFO_EXTRACTION_MODEL = "llama-3.3-70b-versatile"

class ChatSessionManager:
    @staticmethod
    def save_sessions(chat_sessions: Dict[str, List[ChatMessage]], chat_summaries: Dict[str, str]):
        """Save chat sessions to file with efficient JSON serialization."""
        try:
            sessions_data = {
                name: {
                    "messages": [msg.__dict__ for msg in messages],
                    "summary": chat_summaries.get(name, "")
                }
                for name, messages in chat_sessions.items()
            }
            
            with open(CHAT_SESSIONS_FILE, "w") as file:
                json.dump(sessions_data, file, default=str)
        except Exception as e:
            logger.error(f"Error saving chat sessions: {e}")

    @staticmethod
    def load_sessions() -> Dict[str, List[ChatMessage]]:
        """Load chat sessions from file."""
        if not os.path.exists(CHAT_SESSIONS_FILE):
            return {}
        
        try:
            with open(CHAT_SESSIONS_FILE, "r") as file:
                data = json.load(file)
                return {
                    session_name: [ChatMessage(**msg) for msg in session_data.get("messages", [])]
                    for session_name, session_data in data.items()
                }
        except Exception as e:
            logger.error(f"Error loading chat sessions: {e}")
            return {}

class UserInfoManager:
    @staticmethod
    def save_info(user_info: Dict[str, Any]):
        """Save user information to file."""
        try:
            with open(USER_INFO_FILE, "w") as file:
                json.dump(user_info, file)
        except Exception as e:
            logger.error(f"Error saving user info: {e}")

    @staticmethod
    def load_info() -> Dict[str, Any]:
        """Load user information from file."""
        if not os.path.exists(USER_INFO_FILE):
            return {}
        
        try:
            with open(USER_INFO_FILE, "r") as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading user info: {e}")
            return {}

class InfoExtractor:
    @staticmethod
    def extract_user_info(llm: Groq, existing_info: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Extract and update user information using LLM."""
        prompt = f"""
    <task>
        Analyze the message for meaningful personal information about the user. 
        If new information is found, add it to the existing memories.
    </task>
    <focus>
        <category name="Core Personal Details">
            <details>
                - Basic: name, age, location, occupation
                - Background: education, skills, expertise
                - Family: marital status, children, living situation
                - Professional: work experience, career goals
            </details>
        </category>
        <category name="Individual Characteristics">
            <details>
                - Preferences: likes, dislikes, interests
                - Values and beliefs
                - Communication style
                - Decision-making patterns
            </details>
        </category>
        <category name="Current Context">
            <details>
                - Ongoing situations or challenges
                - Short and long-term goals
                - Recent significant events
                - Primary concerns or needs
            </details>
        </category>
        <category name="Relationship Dynamics">
            <details>
                - Important relationships mentioned
                - Social connections
                - Support systems
                - Interaction patterns
            </details>
        </category>
    </focus>
    <instructions>
        <format>
            - Maintain a clean bulleted list structure
            - Use concise, clear language
            - Organize by categories when possible
            - Include timestamp for new additions
            - Flag uncertain or inferred information
        </format>
        <strict_guidelines>
            - Return only the updated memory list
            - Use bullet points exclusively
            - No explanatory text or JSON
            - No meta-commentary
            - Preserve all verified existing memories
        </strict_guidelines>
    </instructions>
    <input>
        <existing_memories>{existing_info}</existing_memories>
        <new_message>{message}</new_message>
    </input>
    """
        try:
            extracted_info = llm.complete(prompt).text.strip()
            if extracted_info:
                return {"user_info": extracted_info}
            return {}
        except Exception as e:
            logger.error(f"Error extracting user info: {e}")
            return {}

def generate_chat_summary(chat_history: List[ChatMessage]) -> str:
    """Generate a summary of the chat conversation."""
    try:
        llm = Groq(model=DEFAULT_LLM_MODEL, temperature=0)
        
        messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        summary_prompt = f"""
        Summarize the following conversation:
        {messages_text}
        
        Provide key points and outcomes in bullet format.
        """
        
        return llm.complete(summary_prompt).text
    except Exception as e:
        logger.error(f"Error generating chat summary: {e}")
        return "Unable to generate summary"

def initialize_streamlit_session():
    """Initialize Streamlit session state."""
    if "user_info" not in st.session_state:
        st.session_state.user_info = UserInfoManager.load_info()
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = ChatSessionManager.load_sessions()
    
    if "chat_summaries" not in st.session_state:
        st.session_state.chat_summaries = {}
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = None

def render_sidebar():
    """Render the Streamlit sidebar with chat session management."""
    with st.sidebar:
        st.title('🤗💬ChatGPT Clone')
        st.success('Access to this ChatGPT Clone is provided by [Kush](https://www.linkedin.com/in/kush-juvekar/)!', icon='✅')
        st.markdown('⚡ This app is hosted on Lightning AI Studio!')

        # Display existing chat sessions
        st.subheader("Chat Sessions")
        for session_name in st.session_state.chat_sessions.keys():
            if st.button(session_name):
                st.session_state.current_session = session_name

        # New chat session creation
        new_session_name = st.text_input("Start a new chat session", "")
        if st.button("Start New Session") and new_session_name:
            # Generate summary for current session if it exists
            if st.session_state.current_session:
                current_history = st.session_state.chat_sessions[st.session_state.current_session]
                if len(current_history) > 1:
                    summary = generate_chat_summary(current_history)
                    st.session_state.chat_summaries[st.session_state.current_session] = summary
            
            # Create new session
            st.session_state.chat_sessions[new_session_name] = [
                ChatMessage(role=MessageRole.ASSISTANT, content="Ask anything!")
            ]
            st.session_state.current_session = new_session_name
            ChatSessionManager.save_sessions(
                st.session_state.chat_sessions, 
                st.session_state.chat_summaries
            )
            
def main():
    # Load environment variables
    load_dotenv()

    # Streamlit configuration
    st.set_page_config(page_title="🤗💬ChatGPT Clone")
    st.header("ChatGPT Clone")

    # Initialize session state
    initialize_streamlit_session()

    # Render sidebar
    render_sidebar()

    # Determine current chat history
    chat_history = (
        st.session_state.chat_sessions.get(st.session_state.current_session, [])
        if st.session_state.current_session 
        else []
    )

    # Display chat messages
    for message in chat_history:
        with st.chat_message(message.role.value):
            st.write(message.content)

    # User input handling
    if prompt := st.chat_input("Your question"):
        # Add user message to chat history
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_history.append(user_message)

        # Extract and store user information
        llm = Groq(model=DEFAULT_INFO_EXTRACTION_MODEL, temperature=0)
        extracted_info = InfoExtractor.extract_user_info(
            llm, 
            st.session_state.user_info, 
            prompt
        )
        st.session_state.user_info.update(extracted_info)
        UserInfoManager.save_info(st.session_state.user_info)

        stand_alone_question = llm.complete(f"""
                                            Given the following conversation between a user and an AI assistant and a follow up question from user,
                                           rephrase the follow up question to be a standalone question.

                                            Chat History:"""+"\n".join([f'{{"role": "{msg.role.value}", "content": "{msg.content}"}}' for msg in chat_history[:-1]])+
                                           f"""                                            
                                              Standalone question:""").text.strip()
        important_info = llm.complete(f"Extract the relevant information from the following corpus for this message: {stand_alone_question}"+"\n"+"Corpus:"+"\n"+f"{st.session_state.user_info}").text.strip()
        
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare request payload
                    response = requests.post(
                        "http://localhost:8000/chat",
                        json={
                            "prompt": stand_alone_question,
                            "message_history": [
                                {"role": msg.role.value, "content": msg.content}
                                for msg in chat_history[:-1]
                            ],
                            "user_info": important_info
                        },
                        timeout=30  # Add timeout to prevent hanging
                    )
                    response.raise_for_status()
                    # Process and display response
                    assistant_response = response.json()["response"]
                    st.write(assistant_response)

                    # Update chat history
                    chat_history.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response)
                    )

                    # Save updated chat sessions
                    if st.session_state.current_session:
                        st.session_state.chat_sessions[st.session_state.current_session] = chat_history
                        ChatSessionManager.save_sessions(
                            st.session_state.chat_sessions, 
                            st.session_state.chat_summaries
                        )

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
