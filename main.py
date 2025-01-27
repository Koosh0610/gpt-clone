#-------------------------------------------------#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
#-------------------------------------------------#
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq

import logging
#-------------------------------------------------#
from dotenv import load_dotenv
load_dotenv()

#Configer Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Initialiazing FastAPI app
app = FastAPI()

# Initialize your models and clients here
llm = Groq(model="deepseek-r1-distill-llama-70b", temperature=0)


#Defining Pydantic Models
    
class ChatRequest(BaseModel):
    prompt: str
    message_history: List[ChatMessage]
    user_info: str

    class Config:
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = llm.complete(agent_prompt.format(input_text=request.prompt,message_history=request.message_history,user_info=request.user_info))
        return ChatResponse(response=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
agent_prompt = """
    <description>
        You are a helpful agent that answers user queries while also considering the conversation history. 
        You will also be provided with certain information regarding the user that may or may not help you tailor responses.
    </description>
    <context>
        <user_info>{user_info}</user_info>
        <history>{message_history}</history>
    </context>
    <query>{input_text}</query>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
