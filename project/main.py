# python3 main.py

from langchain_core.messages import HumanMessage
from graph import build_graph

app = build_graph()

result = app.invoke({
    "messages": [
        #HumanMessage(content="Convert report.pdf to Word"),

        HumanMessage(content="Convert book.pdf to image"),
        HumanMessage(content="Convert notes.md to PDF"),
        HumanMessage(content="Convert thesis.docx to PDF")
    ]
})

print("🤖 Final Answer:")
print(result["messages"][-1].content)


print("\n===== FULL MESSAGE TRACE =====\n")

for msg in result["messages"]:
    
    # Human
    if msg.type == "human":
        print("🧑 USER:")
        print(msg.content)
        print()

    # AI
    elif msg.type == "ai":
        print("🤖 AI:")
        print(msg.content)
        
        # If tool call happened
        if msg.tool_calls:
            print("🔧 TOOL CALLED:")
            for tool_call in msg.tool_calls:
                print("   Name:", tool_call["name"])
                print("   Args:", tool_call["args"])
        print()

    # Tool response
    elif msg.type == "tool":
        print("🛠 TOOL RESULT:")
        print(msg.content)
        print()

print("===== END TRACE =====")

#print(result["messages"])
