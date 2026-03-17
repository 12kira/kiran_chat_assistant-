from typing import TypedDict, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = ChatOllama(model="llama3")

# Defining state 
class ChatState(TypedDict):
    messages: List


# -------- Node --------
def chatbot(state: ChatState):

    # It is used to send full messages
    response = llm.invoke(state["messages"])

    # Adding AI response
    new_messages = state["messages"] + [
        AIMessage(content=response.content)
    ]

    return {"messages": new_messages}



builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.set_entry_point("chatbot")

builder.add_edge("chatbot", END)

graph = builder.compile()


conversation_memory = []



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


 #-------- API --------
@app.post("/ask")
async def ask(request: Request):

    global conversation_memory

    data = await request.json()

     # It is used to store questions and answers
    conversation_memory.append(
        HumanMessage(content=data["question"])
    )

    result = graph.invoke({
        "messages": conversation_memory
    })

    # AI response will be stored
    conversation_memory = result["messages"]

    return {
        "answer": result["messages"][-1].content
    }