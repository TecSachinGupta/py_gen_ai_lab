## Imports
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv

import os
import sqlite3
import requests

## Load environment variables
load_dotenv()

## Update environment variables
os.environ['LANGCHAIN_PROJECT'] = 'Chatbot'

class ChatState(TypedDict):

    title: str
    messages: Annotated[list[BaseMessage], add_messages]

## Tools

### Prebuilt Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

### Custom Tools
@tool
def calculator(num1: float, num2: float, operation: str) -> dict:
    """
    A simple calculator function that performs basic arithmetic operations.
    
    Parameters:
    -----------
    num1 : float
        The first number for the calculation
    num2 : float
        The second number for the calculation
    operation : str
        The arithmetic operation to perform. Valid options:
        - 'add' or '+': Addition
        - 'subtract' or '-': Subtraction
        - 'multiply' or '*': Multiplication
        - 'divide' or '/': Division
        - 'power' or '**': Exponentiation
        - 'modulus' or '%': Modulus (remainder)
        - 'floor_divide' or '//': Floor division
    
    Returns:
    --------
    float or str
        The result of the calculation, or an error message if invalid
    
    Examples:
    ---------
    >>> calculator(10, 5, 'add')
    15.0
    >>> calculator(10, 5, '/')
    2.0
    >>> calculator(2, 3, 'power')
    8.0
    """
    
    # Convert inputs to float to handle both integers and decimals
    try:
        num1 = float(num1)
        num2 = float(num2)
    except (TypeError, ValueError):
        return {"error": "Invalid number input. Please provide numeric values."}
    
    # Convert operation to lowercase for case-insensitive matching
    operation = str(operation).lower().strip()
    
    # Perform the requested operation
    if operation in ['add', '+']:
        # Addition: num1 + num2
        result = num1 + num2
        
    elif operation in ['subtract', '-']:
        # Subtraction: num1 - num2
        result = num1 - num2
        
    elif operation in ['multiply', '*']:
        # Multiplication: num1 * num2
        result = num1 * num2
        
    elif operation in ['divide', '/']:
        # Division: num1 / num2
        # Check for division by zero
        if num2 == 0:
            return {"Error": "Division by zero is undefined."}
        result = num1 / num2
        
    elif operation in ['power', '**']:
        # Exponentiation: num1 raised to the power of num2
        try:
            result = num1 ** num2
        except OverflowError:
            return {"Error": "Result too large to compute."}
            
    elif operation in ['modulus', '%']:
        # Modulus: remainder of num1 divided by num2
        # Check for modulus by zero
        if num2 == 0:
            return {"Error": "Modulus by zero is undefined."}
        result = num1 % num2
        
    elif operation in ['floor_divide', '//']:
        # Floor division: integer division of num1 by num2
        # Check for division by zero
        if num2 == 0:
            return {"Error": "Floor division by zero is undefined."}
        result = num1 // num2
        
    else:
        # Invalid operation provided
        return {"Error": (f"Invalid operation '{operation}'. "
                "Valid operations: add(+), subtract(-), multiply(*), "
                "divide(/), power(**), modulus(%), floor_divide(//)")}
    
    # Return the calculated result
    return {"first_num": num1, "second_num": num2, "operation": operation, "result": result}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

### Tools List

tools = [search_tool, get_stock_price, calculator]

## LLM
llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)

## Node Functions

### Prebuilts
tool_node = ToolNode(tools)


### Customs

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def generate_chat_title(state: ChatState):
    """Node that generates a title for the conversation based on the first message."""
    # Only generate title if it doesn't exist or is empty
    if state.get('title') and state['title'].strip():
        return {"title": state['title']}
    
    # Get the first user message
    messages = state.get("messages", [])
    first_user_msg = None
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_user_msg = msg.content
            break
    
    if first_user_msg:
        # Generate a concise title
        prompt = f"Generate a very short title (5 words max) for a conversation that starts with: '{first_user_msg}'. Only return the title, nothing else."
        response = llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip('"').strip("'")
        return {"title": title}
    
    return {"title": "New Conversation"}

    
## Checkpointer
conn = sqlite3.connect(database="enhanced_chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

## Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node('chat_title_node', generate_chat_title)
graph.add_node("tools", tool_node)

# Main flow: START -> chat_node -> (tools if needed) -> END
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
# graph.add_edge(START, "chat_title_node")
# graph.add_edge('chat_title_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

## Utilities
def retrieve_all_threads():
    """Retrieve all thread IDs from the checkpointer."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def get_thread_title(thread_id):
    """Get the title for a specific thread."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("title", str(thread_id)[:8])
    except:
        return str(thread_id)[:8]

def get_thread_preview(thread_id):
    """Get a preview of the thread (first message or title)."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        
        # Try to get title first
        title = state.values.get("title", "")
        if title and title.strip():
            return title
        
        # Otherwise get first message
        messages = state.values.get("messages", [])
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content[:50]
                return content + "..." if len(msg.content) > 50 else content
        
        return str(thread_id)[:8]
    except:
        return str(thread_id)[:8]

def delete_thread(thread_id):
    """Delete a thread from the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting thread: {e}")
        return False