from typing import TypedDict, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

load_dotenv() 

app = FastAPI()
templates = Jinja2Templates(directory="templates")

#  ONLY GROQ (no Ollama anywhere)
llm = ChatOpenAI(
    model="llama-3.1-8b-instant", 
    temperature=0.7,
    openai_api_key=os.getenv("GROQ_API_KEY"),  # make sure this is set
    openai_api_base="https://api.groq.com/openai/v1"
)

# State
class ChatState(TypedDict):
    messages: List

# Chat node
def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {"messages": new_messages}

# Graph setup
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.set_entry_point("chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

conversation_memory: List = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    global conversation_memory

    data = await request.json()
    question = data.get("question")

    if not question:
        return {"answer": "Please ask a question."}

    conversation_memory.append(HumanMessage(content=question))

    try:
        result = graph.invoke({"messages": conversation_memory})
        conversation_memory = result["messages"]

        answer = result["messages"][-1].content
        return {"answer": answer}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )