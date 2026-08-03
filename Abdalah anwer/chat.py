from ollama import Client
import os
import numpy as np
import pandas as pd


# =========================
# Connect to Ollama
# =========================
client = Client(host="http://192.168.2.89:11434")


# =========================
# Functions
# =========================
def factorial_of_number(a: int) -> int:
    print(f"factorial_of_number({a=})")

    x = 1
    for i in range(1, a + 1):
        x *= i

    return x


def mult2_matrix(a, b):
    print(f"mult2_matrix({a=}, {b=})")
    return np.matmul(a, b).tolist()


TEXT_FILES = {
    ".txt", ".py", ".js", ".java",
    ".cpp", ".c", ".cs", ".html",
    ".css", ".md"
}


def read_file(file_path):
    print(f"read_file({file_path=})")

    extension = os.path.splitext(file_path)[1].lower()

    if extension in TEXT_FILES:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif extension == ".csv":
        return pd.read_csv(file_path).to_string()

    elif extension == ".xlsx":
        return pd.read_excel(file_path).to_string()

    return "Unsupported file type."


# =========================
# Functions Dictionary
# =========================
FUNCTIONS = {
    "factorial_of_number": factorial_of_number,
    "mult2_matrix": mult2_matrix,
    "read_file": read_file,
}


# =========================
# Tool Schemas
# =========================
tools = [
    {
        "type": "function",
        "function": {
            "name": "factorial_of_number",
            "description": "Calculate the factorial of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "The number"
                    }
                },
                "required": ["a"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mult2_matrix",
            "description": "Multiply two matrices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "array",
                        "description": "First matrix"
                    },
                    "b": {
                        "type": "array",
                        "description": "Second matrix"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read txt, py, csv, xlsx and pdf files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["file_path"]
            }
        }
    }
]


# =========================
# Chat Loop
# =========================
messages = []

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat(
        model="gemma4:31b-cloud",
        messages=messages,
        tools=tools
    )

    message = response["message"]

    if message.get("tool_calls"):

        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message["tool_calls"]
        })

        for tool_call in message["tool_calls"]:

            fn_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]

            print("=" * 50)
            print("Tool :", fn_name)
            print("Arguments :", args)
            print("=" * 50)

            if fn_name not in FUNCTIONS:
                print("Unknown Tool")
                continue

            try:
                result = FUNCTIONS[fn_name](**args)
            except Exception as e:
                result = str(e)

            print("\nTool Result:")
            print(result)

            messages.append({
                "role": "tool",
                "content": str(result)
            })

        final_response = client.chat(
            model="gemma4:31b-cloud",
            messages=messages
        )

        answer = final_response["message"]["content"]

    else:
        answer = message["content"]

    print("\nAI:", answer)
    print("-" * 60)

    messages.append({
        "role": "assistant",
        "content": answer
    })


print("\nConversation History:\n")

for msg in messages:
    print(f"{msg['role']}: {msg['content']}")