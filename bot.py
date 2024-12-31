import os
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

# Set API Keys
groq_api = "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"
langsmt_api = "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"
os.environ["LANGCHAIN_API_KEY"] = langsmt_api

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api, model_name="llama-3.3-70b-versatile")
print("LLM Initialized:", llm)

# Define State for the graph
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]  # list of messages in {"role": "role_name", "content": "text"} format

# Initialize StateGraph
graph_builder = StateGraph(State)

# Define Chatbot Node
def chatbot(state: State):
    messages = state.get('messages', [])
    response = llm.invoke(messages)  # Pass messages to the LLM
    return {"messages": response}

# Add nodes and edges to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()
import networkx as nx
import matplotlib.pyplot as plt

# Create a networkx graph
nx_graph = nx.DiGraph()

# Add nodes
for node in graph_builder.nodes:
    nx_graph.add_node(node)

# Add edges
for edge in graph_builder.edges:
    nx_graph.add_edge(edge[0], edge[1])

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(nx_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
plt.title("StateGraph Visualization")
plt.show()

# Chat loop with a prompt
print("Chatbot Initialized. Type 'quit' or 'q' to exit.")

# Define the system prompt
system_prompt = {
    "role": "system",
    "content": "You are a helpful and friendly assistant. Answer the user's questions politely and clearly. but answer very precisely like a chatbot"
}

# Initialize conversation with the system prompt
conversation_state = {"messages": [system_prompt]}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Goodbye!")
        break

    # Add the user input to the conversation state
    conversation_state["messages"].append({"role": "user", "content": user_input})

    # Process input through the graph
    for event in graph.stream(conversation_state):
        for value in event.values():
            # Access the `content` attribute of the AIMessage object
            assistant_response = getattr(value["messages"], "content", "No response")
            print(f"Assistant: {assistant_response}")

    # Append the assistant's response to the conversation state
    assistant_message = {"role": "assistant", "content": assistant_response}
    conversation_state["messages"].append(assistant_message)