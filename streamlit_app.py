import streamlit as st
import requests
import logging
import json
import os
import re
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
DEFAULT_INFO_EXTRACTION_MODEL = "deepseek-r1-distill-llama-70b"

agent_prompt = """You are a helpful AI assistant. Please help answer the following question.

Input Question: {input_text}

Previous Chat Context:
{message_history}

Relevant User Information:
{user_info}

First think through the problem step by step, then provide your response.
Format your thinking process between <think></think> tags.

<think>
1. Understand the question
2. Consider chat context
3. Analyze user information
4. Form response strategy
</think>

Your response:
"""
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
**Memory Update Task**  
Extract and organize personal information from the new message. Preserve verified existing memories. Only return bullet points.

**Focus Categories**  
1. **Core Details**  
   - Name, Age, Location, Occupation  
   - Education, Skills, Expertise  
   - Family status, Children, Living situation  
   - Work experience, Career goals  

2. **Personal Traits**  
   - Likes/Dislikes, Interests/Hobbies  
   - Core values, Belief systems  
   - Communication patterns, Decision-making style  

3. **Current Situation**  
   - Active challenges/Projects  
   - Immediate/Long-term objectives  
   - Recent life events  
   - Pressing concerns/Needs  

4. **Relationships**  
   - Key personal/professional connections  
   - Social network composition  
   - Support systems availability  
   - Interaction frequency/Patterns  

**Processing Rules**  
✓ Compare new message with existing memories  
✓ Add new info as: [Timestamp] [Category] • Fact (confidence?)  
✓ Mark uncertain items with (?)  
✓ Preserve all verified existing entries  
✗ No explanations/narratives  
✗ No JSON/Markdown  
✗ No duplicate entries  

**Input Example**  
[Existing Memories]  
• [2024-02-15] [Core] • Software engineer (Seattle)  
• [2024-03-01] [Traits] • Prefers text communication  

[New Message]  
"Just moved to Portland for a product manager role. Considering MBA programs but worried about costs."  

**Required Output**  
• [2024-05-20] [Core] • Product manager (Portland)  
• [2024-05-20] [Situation] • Considering MBA programs (?)  
• [2024-05-20] [Situation] • Financial concerns about education  
• [2024-02-15] [Core] • Software engineer (Seattle)  
• [2024-03-01] [Traits] • Prefers text communication
 **Example End**  

Your turn:
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
            
            
def get_standalone_question(llm, chat_history,prompt):
    prompt = f"""
        Given the following conversation between a user and an AI assistant and a follow up question from user,
        rephrase the follow up question to be a standalone question.

        Chat History:""" + "\n".join([f'{{"role": "{msg.role.value}", "content": "{msg.content}"}}' for msg in chat_history[:-1]]) + f"""Follow Up Input: {prompt}"""+f"""                                            
        Standalone question:"""
    return llm.complete(prompt).text.strip()

def get_important_info(llm, standalone_question, user_info):
    prompt = f"Extract the relevant information from the following corpus for this message: {standalone_question}\nCorpus:\n{user_info}"
    return llm.complete(prompt).text.strip()

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

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate standalone question and extract important information            
        stand_alone_question = get_standalone_question(llm, chat_history,prompt)
        important_info = get_important_info(llm, stand_alone_question, st.session_state.user_info)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare request payload
                    #response = requests.post(
                        #"http://localhost:8000/chat",
                        #json={
                        #    "prompt": stand_alone_question,
                        #    "message_history": [
                        #        {"role": msg.role.value, "content": msg.content}
                        #        for msg in chat_history[:-1]
                        #    ],
                        #    "user_info": important_info
                        #},
                        #timeout=30  # Add timeout to prevent hanging
                    #)
                    #response.raise_for_status()
                    # Process and display response
                    #assistant_response = response.json()["response"]
                    assistant_response = llm.complete(agent_prompt.format(input_text=stand_alone_question,message_history="\n".join(f"{msg.role.value.upper()}: {msg.content.strip()}" for msg in chat_history[:-1] if msg.content),user_info=important_info))
                    with st.expander("Assistant's thought process....."):
                        st.write(re.findall(r'<think>(.*?)</think>', assistant_response.text, re.DOTALL))
                    st.write(assistant_response.text.split(f'</think>', 2)[-1].strip())

                    # Update chat history
                    chat_history.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response.text)
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
