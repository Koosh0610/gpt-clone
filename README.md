# Prompt based Memory Management using JSON files

## Overview
This project is a Streamlit-based LLM app that leverages the Groq LLM to provide an AI-powered chatbot. The system supports user profiling, chat history management, and context-aware responses.

## Features
- **Conversational AI**: Uses the DeepSeek-R1-Distill-LLaMA-70B model for chat responses.
- **User Profiling**: Extracts and maintains user information for personalized interactions.
- **Session Management**: Saves and loads chat sessions for different users.
- **Standalone Question Generation**: Converts follow-up queries into independent questions.
- **Information Extraction**: Identifies and retrieves relevant user details from conversations.
- **Logging & Debugging**: Logs errors and session activities for troubleshooting.
- **Streamlit UI**: Interactive frontend with sidebar navigation and chat display.

## Simplicity
- Memory layer solutions for agentic or LLM app like MemGPT, mem0, get-zep, memobase etc rely on various database and large context management systems. They most often also involve retrieval of memory using RAG.
  However, I believe LLMs are capable of traditional tasks of information retrieval, classification, sentiment analysis to a good extent. This chatbot has been tested on the [LongMemEval]([url](https://github.com/xiaowu0162/LongMemEval)) and found to be satisfactory. I cannot provide with metrics as I ran out API credits before I could complete a statistical significant amount of test runs.

## Installation
1. Clone the repository:
   ```sh
   git clone <https://github.com/Koosh0610/gpt-clone.git>
   cd <gpt-clone>
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file and specify API keys.
4. Run the application:
   ```sh
   streamlit run streamlit_app.py
   ```

## Usage
- Start the Streamlit app and interact with the chatbot.
- Select or create a user profile in the sidebar. (Enter profile_name and session_name beofre proceeding)
- Manage chat sessions and track user-specific information.

## Future Enhancements
- Improve UI/UX for better user experience.
- Integrate multiple LLM models for task-specific optimizations.
- Add RAG for file uploads and multimodal support.
- Reduce latency by optimizing for sequential LLM calls.

## License
This project is licensed under the MIT License.

## Author
Developed by [Your Name].

