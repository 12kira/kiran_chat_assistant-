from typing import TypedDict, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from fastapi import HTTPException

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ollama LLM
llm = ChatOllama(model="llama2")

class ChatState(TypedDict):
    messages: List

def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {"messages": new_messages}

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
        # Try to generate AI response
        result = graph.invoke({"messages": conversation_memory})
        conversation_memory = result["messages"]
        answer = result["messages"][-1].content
        return {"answer": answer}

    except Exception as e:
        # Log error to console (optional)
        print("Error in /ask:", e)

        # Return error message in response with HTTP 500
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )