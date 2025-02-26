import streamlit as st
import requests
import logging
import json
import os
import re
from typing import Dict, List, Any
from datetime import datetime

from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.groq import Groq


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHAT_SESSIONS_FILE = "chat_sessions.json"
USER_INFO_FILE = "user_info.json"
USER_PROFILES_FILE = "user_profiles.json"
DEFAULT_LLM_MODEL = "deepseek-r1-distill-llama-70b"
DEFAULT_INFO_EXTRACTION_MODEL = "deepseek-r1-distill-llama-70b"

agent_prompt = """You are a helpful AI assistant. Please help answer the following question.

Input Question: {input_text}

Previous Chat Context:
{message_history}

Relevant User Information:
{user_info}

User Profile:
{user_profile}

First think through the problem step by step, then provide your response.
Format your thinking process between <think></think> tags.

<think>
1. Understand the question
2. Consider chat context
3. Analyze user information and profile
4. Form response strategy
</think>

Your response:
"""

class ChatSessionManager:
    @staticmethod
    def save_sessions(chat_sessions: Dict[str, List[ChatMessage]], chat_summaries: Dict[str, str], user_id: str):
        """Save chat sessions to file with efficient JSON serialization and user association."""
        try:
            # First load existing data to avoid overwriting other users' sessions
            all_sessions_data = {}
            if os.path.exists(CHAT_SESSIONS_FILE):
                with open(CHAT_SESSIONS_FILE, "r") as file:
                    try:
                        all_sessions_data = json.load(file)
                    except json.JSONDecodeError:
                        logger.error("Error decoding sessions file, creating new one")
            
            # Prepare current user's sessions
            sessions_data = {
                name: {
                    "messages": [msg.__dict__ for msg in messages],
                    "summary": chat_summaries.get(name, ""),
                    "user_id": user_id  # Associate each session with a user ID
                }
                for name, messages in chat_sessions.items()
            }
            
            # Update only this user's sessions in the all_sessions_data
            for name, data in sessions_data.items():
                all_sessions_data[name] = data
            
            with open(CHAT_SESSIONS_FILE, "w") as file:
                json.dump(all_sessions_data, file, default=str)
                
        except Exception as e:
            logger.error(f"Error saving chat sessions: {e}")

    @staticmethod
    def load_sessions(user_id: str = None) -> Dict[str, List[ChatMessage]]:
        """Load chat sessions from file, optionally filtered by user ID."""
        if not os.path.exists(CHAT_SESSIONS_FILE):
            return {}
        
        try:
            with open(CHAT_SESSIONS_FILE, "r") as file:
                data = json.load(file)
                
                # If user_id is provided, filter sessions for that user
                if user_id:
                    filtered_data = {
                        session_name: session_data 
                        for session_name, session_data in data.items()
                        if session_data.get("user_id") == user_id
                    }
                    return {
                        session_name: [ChatMessage(role=msg["role"], content=msg["blocks"][0].split("text=")[1].strip("'")) for msg in session_data.get("messages", [])]
                        for session_name, session_data in filtered_data.items()
                    }
                # Otherwise, return all sessions (for backward compatibility)
                else:
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

class UserProfileManager:
    @staticmethod
    def save_profiles(profiles: Dict[str, Any]):
        """Save user profiles to file."""
        try:
            with open(USER_PROFILES_FILE, "w") as file:
                json.dump(profiles, file)
        except Exception as e:
            logger.error(f"Error saving user profiles: {e}")

    @staticmethod
    def load_profiles() -> Dict[str, Any]:
        """Load user profiles from file."""
        if not os.path.exists(USER_PROFILES_FILE):
            return {}
        
        try:
            with open(USER_PROFILES_FILE, "r") as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
            return {}

class InfoExtractor:
    @staticmethod
    def extract_user_info(llm: Groq, existing_info: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Extract and update user information using LLM."""
        # Make sure we have the existing user_info as a string
        existing_info_str = existing_info.get("user_info", "")
        
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
✗ Do not exclude any information that seems important

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
        <existing_memories>{existing_info_str}</existing_memories>
        <new_message>{message}</new_message>
    </input>
    """
        try:
            # Extract the response and handle think tags
            response = llm.complete(prompt).text.strip()
            extracted_info = remove_think_tags(response)
            
            # Check if we got valid output
            if extracted_info:
                return {"user_info": extracted_info}
            # If extraction failed but we had existing info, preserve it
            elif existing_info_str:
                return {"user_info": existing_info_str}
            return {}
        except Exception as e:
            logger.error(f"Error extracting user info: {e}")
            # Make sure we don't lose existing info on error
            if existing_info_str:
                return {"user_info": existing_info_str}
            return {}

def remove_think_tags(text):
    """Remove content between <think> and </think> tags."""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

class UserProfiler:
    @staticmethod
    def create_or_update_profile(llm: Groq, user_id: str, user_info: Dict[str, Any], 
                                chat_history: List[ChatMessage], existing_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create or update user profile based on chat history and user info."""
        
        # Format chat history for LLM consumption
        chat_history_text = "\n".join([f"{msg.role.value}: {msg.content}" for msg in chat_history[-20:]])  # Limit to last 20 messages
        
        # Extract relevant user info
        user_info_text = user_info.get("user_info", "")
        
        # Get existing profile if available
        existing_profile_text = json.dumps(existing_profile) if existing_profile else "No existing profile"
        
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        profiling_prompt = f"""
**User Profiling Task**
Create a comprehensive user profile based on chat history and extracted information.

**Input Data**
Chat History:
{chat_history_text}

User Information:
{user_info_text}

Existing Profile:
{existing_profile_text}

**Profiling Dimensions**
1. Communication Style (formal/informal, verbose/concise, technical/simple)
2. Knowledge Level (beginner/intermediate/expert) in relevant domains
3. Decision-Making Style (analytical, intuitive, consultative)
4. Interaction Preferences (detailed explanations, examples, visuals)
5. Task Orientation (goal-focused, exploratory, learning-oriented)
6. Tone Preferences (professional, casual, humorous)

**Output Format**
Return a JSON object with the following structure:
{{
  "communication_style": {{
    "formality": "value",
    "verbosity": "value",
    "technical_level": "value"
  }},
  "knowledge_levels": {{
    "domain1": "level",
    "domain2": "level"
  }},
  "decision_style": "value",
  "interaction_preferences": ["pref1", "pref2"],
  "task_orientation": "value",
  "tone_preference": "value",
  "last_updated": "{current_timestamp}"
}}

Ensure values are based on observed patterns, not assumptions. If insufficient data exists for any field, use "unknown" as the value.
"""
        try:
            profile_response = llm.complete(profiling_prompt).text.strip()
            
            # Remove any thinking tags
            profile_response = remove_think_tags(profile_response)
            
            # Try to parse the response as JSON
            try:
                profile_json = json.loads(profile_response)
                profile_json[user_id]["user_id"] = user_id
                return profile_json
            except json.JSONDecodeError:
                # If parsing fails, extract anything that looks like JSON
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, profile_response)
                if match:
                    try:
                        profile_json = json.loads(match.group(1))
                        profile_json["user_id"] = user_id
                        return profile_json
                    except:
                        pass
                
                # If all fails, return a minimal profile
                logger.error(f"Failed to parse profile JSON: {profile_response}")
                # Keep existing profile if we have one, otherwise create minimal profile
                if existing_profile:
                    existing_profile["last_updated"] = current_timestamp
                    return existing_profile
                else:
                    return {
                        "user_id": user_id,
                        "communication_style": {"formality": "unknown", "verbosity": "unknown", "technical_level": "unknown"},
                        "knowledge_levels": {},
                        "decision_style": "unknown",
                        "interaction_preferences": [],
                        "task_orientation": "unknown",
                        "tone_preference": "unknown",
                        "last_updated": current_timestamp,
                        "error": "Profiling failed"
                    }
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            # Keep existing profile if we have one on error
            if existing_profile:
                existing_profile["last_updated"] = current_timestamp
                return existing_profile
            else:
                return {
                    "user_id": user_id,
                    "error": str(e),
                    "last_updated": current_timestamp
                }

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
        
        summary = llm.complete(summary_prompt).text.strip()
        # Remove any thinking tags
        return remove_think_tags(summary)
    except Exception as e:
        logger.error(f"Error generating chat summary: {e}")
        return "Unable to generate summary"

def initialize_streamlit_session():
    """Initialize Streamlit session state."""
    if "user_info" not in st.session_state:
        st.session_state.user_info = UserInfoManager.load_info()
    
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = "default_user"
    
    # Initialize chat_sessions but don't load anything yet
    # We'll load sessions after user is selected
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "chat_summaries" not in st.session_state:
        st.session_state.chat_summaries = {}
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
        
    if "user_profiles" not in st.session_state:
        st.session_state.user_profiles = UserProfileManager.load_profiles()
        
    if "profiling_count" not in st.session_state:
        st.session_state.profiling_count = 0
    
    # Flag to track if user just changed
    if "user_changed" not in st.session_state:
        st.session_state.user_changed = False

def render_sidebar():
    """Render the Streamlit sidebar with chat session management."""
    with st.sidebar:
        st.title('🤗💬ChatGPT Clone')
        st.success('Access to this ChatGPT Clone is provided by [Kush](https://www.linkedin.com/in/kush-juvekar/)!', icon='✅')
        st.markdown('⚡ This app is hosted on Lightning AI Studio!')

        # User selection/creation
        st.subheader("User Selection")
        user_id = st.text_input("User ID", value=st.session_state.current_user_id)
        if st.button("Set User") and user_id:
            old_user_id = st.session_state.current_user_id
            
            # If the user is changing, save current sessions for old user
            if old_user_id != user_id and old_user_id != "default_user" and st.session_state.chat_sessions:
                ChatSessionManager.save_sessions(
                    st.session_state.chat_sessions, 
                    st.session_state.chat_summaries,
                    old_user_id
                )
            
            # Set new user ID
            st.session_state.current_user_id = user_id
            
            # Load the new user's chat sessions
            st.session_state.chat_sessions = ChatSessionManager.load_sessions(user_id)
            
            # Reset current session
            st.session_state.current_session = None
            
            # Set the user_changed flag to True to force UI refresh
            st.session_state.user_changed = True
            
            # Initialize user profile if not exists
            if user_id not in st.session_state.user_profiles:
                st.session_state.user_profiles[user_id] = {
                    "user_id": user_id,
                    "communication_style": {"formality": "unknown", "verbosity": "unknown", "technical_level": "unknown"},
                    "knowledge_levels": {},
                    "decision_style": "unknown",
                    "interaction_preferences": [],
                    "task_orientation": "unknown",
                    "tone_preference": "unknown",
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                UserProfileManager.save_profiles(st.session_state.user_profiles)
            
            # Show success message
            st.success(f"Switched to user: {user_id}")

        # Display user profile
        if st.session_state.current_user_id in st.session_state.user_profiles:
            with st.expander("User Profile"):
                profile = st.session_state.user_profiles[st.session_state.current_user_id]
                st.json(profile)
        
        st.divider()

        # Display existing chat sessions for current user
        st.subheader("Chat Sessions")
        
        # If we have no sessions for this user, show a message
        if not st.session_state.chat_sessions:
            st.info(f"No existing chat sessions for {st.session_state.current_user_id}")
        else:
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
            
            # Save session with user association
            ChatSessionManager.save_sessions(
                st.session_state.chat_sessions, 
                st.session_state.chat_summaries,
                st.session_state.current_user_id
            )
            
            # Reset profiling count for new session
            st.session_state.profiling_count = 0
            
            # Rerun the app to show the new session
            st.rerun()
            

stand_alone_prompt = """
        Given the following conversation between a user and an AI assistant and a follow up question from user,
        rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history_str}
        Follow Up Input: 
        {prompt}            
        Instruction: YOU WILL ONLY GENERATE THE STAND ALONE QUESTION IN RESPONSE AFTER THINKING. NO MISCELLANEOUS TEXT.                             
        Standalone question:"""

important_info_prompt = "Extract the relevant information from the following corpus for this message: {stand_alone_question}\nCorpus:\n{user_info}"


def main():
    # Load environment variables
    load_dotenv()

    # Streamlit configuration
    st.set_page_config(page_title="🤗💬ChatGPT Clone", layout="wide")
    st.header("ChatGPT Clone")

    # Initialize session state
    initialize_streamlit_session()

    # Render sidebar
    render_sidebar()
    
    # If user has changed, rerun to update UI
    if st.session_state.user_changed:
        st.session_state.user_changed = False
        st.rerun()

    # Determine current chat history
    chat_history = (
        st.session_state.chat_sessions.get(st.session_state.current_session, [])
        if st.session_state.current_session 
        else []
    )
    
    # Display user info at top
    st.info(f"Current user: {st.session_state.current_user_id}")
    if st.session_state.current_session:
        st.subheader(f"Session: {st.session_state.current_session}")

    # Main content
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Display chat messages
        for message in chat_history:
            with st.chat_message(message.role.value):
                st.write(message.content)

        # User input handling
        if prompt := st.chat_input("Your question"):
            # Check if we have a valid session
            if not st.session_state.current_session:
                st.warning("Please start or select a chat session first")
                return
                
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
                
            # Add user message to chat history
            user_message = ChatMessage(role=MessageRole.USER, content=prompt)
            chat_history.append(user_message)

            # Create LLM instance
            llm = Groq(model=DEFAULT_INFO_EXTRACTION_MODEL, temperature=0)
            
            # Extract and store user information
            extracted_info = InfoExtractor.extract_user_info(
                llm, 
                st.session_state.user_info, 
                prompt
            )
            
            if extracted_info:
                st.session_state.user_info.update(extracted_info)
                UserInfoManager.save_info(st.session_state.user_info)

            # Periodically update user profile (every 5 messages)
            st.session_state.profiling_count += 1
            if st.session_state.profiling_count % 5 == 0 or st.session_state.profiling_count == 1:
                current_profile = st.session_state.user_profiles.get(st.session_state.current_user_id, None)
                updated_profile = UserProfiler.create_or_update_profile(
                    llm,
                    st.session_state.current_user_id,
                    st.session_state.user_info,
                    chat_history,
                    current_profile
                )
                st.session_state.user_profiles[st.session_state.current_user_id] = updated_profile
                UserProfileManager.save_profiles(st.session_state.user_profiles)
                
                # Log profile update
                logger.info(f"Updated user profile for {st.session_state.current_user_id}")
                
            # Generate standalone question and extract important information 
            chat_history_str = "\n".join(f"{msg.role.value.upper()}: {msg.content.strip()}" for msg in chat_history[:-1] if msg.content)           
            stand_alone_response = llm.complete(stand_alone_prompt.format(chat_history_str=chat_history_str,prompt=prompt)).text.strip()
            stand_alone_question = remove_think_tags(stand_alone_response)
            
            important_info_response = llm.complete(important_info_prompt.format(stand_alone_question=stand_alone_question,user_info=st.session_state.user_info.get("user_info", ""))).text.strip()
            important_info = remove_think_tags(important_info_response)
            
            # Get current user profile
            current_profile = st.session_state.user_profiles.get(st.session_state.current_user_id, {})
            profile_json = json.dumps(current_profile, indent=2)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Generate response with user profile
                        assistant_response = llm.complete(agent_prompt.format(
                            input_text=stand_alone_question,
                            message_history=chat_history_str,
                            user_info=important_info,
                            user_profile=profile_json
                        ))
                        
                        # Extract thinking process for display
                        thinking_match = re.search(r'<think>(.*?)</think>', assistant_response.text, re.DOTALL)
                        thinking = thinking_match.group(1).strip() if thinking_match else "No explicit thinking process found."
                        
                        with st.expander("Assistant's thought process....."):
                            st.write(thinking)
                                
                        # Extract and display the final response
                        final_response = remove_think_tags(assistant_response.text).strip()
                        
                        st.write(final_response)

                        # Update chat history with clean response (no think tags)
                        chat_history.append(
                            ChatMessage(role=MessageRole.ASSISTANT, content=final_response)
                        )

                        # Save updated chat sessions with user association
                        if st.session_state.current_session:
                            st.session_state.chat_sessions[st.session_state.current_session] = chat_history
                            ChatSessionManager.save_sessions(
                                st.session_state.chat_sessions, 
                                st.session_state.chat_summaries,
                                st.session_state.current_user_id
                            )

                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    
    with col2:
        # Information panel
        if st.session_state.current_session:
            st.subheader("Session Information")
            
            # Session summary
            summary = st.session_state.chat_summaries.get(st.session_state.current_session, "No summary available")
            with st.expander("Session Summary"):
                st.write(summary)
            
            # User context
            with st.expander("User Context"):
                st.write(st.session_state.user_info.get("user_info", "No user information available"))
            
            # Response metrics (placeholder)
            with st.expander("Response Metrics"):
                st.metric("Average Response Time", "0.8s")
                st.metric("User Satisfaction", "High", delta="↑")
            
            # Debug tools (for developers)
            with st.expander("Debug Tools"):
                if st.button("Force Profile Update"):
                    llm = Groq(model=DEFAULT_INFO_EXTRACTION_MODEL, temperature=0)
                    current_profile = st.session_state.user_profiles.get(st.session_state.current_user_id, None)
                    updated_profile = UserProfiler.create_or_update_profile(
                        llm,
                        st.session_state.current_user_id,
                        st.session_state.user_info,
                        chat_history,
                        current_profile
                    )
                    st.session_state.user_profiles[st.session_state.current_user_id] = updated_profile
                    UserProfileManager.save_profiles(st.session_state.user_profiles)
                    st.success("Profile updated")
                
                if st.button("Clear User Info"):
                    st.session_state.user_info = {}
                    UserInfoManager.save_info({})
                    st.success("User info cleared")
                    
                if st.button("Show All Sessions"):
                    all_sessions = ChatSessionManager.load_sessions()
                    st.json({name: {"user_id": sess.get("user_id", "unknown"), "messages": len(sess.get("messages", []))} 
                            for name, sess in all_sessions.items()})
                
                if st.button("Reset Current User Sessions"):
                    st.session_state.chat_sessions = {}
                    st.session_state.current_session = None
                    ChatSessionManager.save_sessions({}, {}, st.session_state.current_user_id)
                    st.success(f"Reset all sessions for {st.session_state.current_user_id}")
                    st.rerun()

if __name__ == "__main__":
    main()
