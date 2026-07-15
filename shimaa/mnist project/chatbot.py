from ollama import Client
import sqlite3

# الاتصال بـ Ollama
client = Client(host='192.168.2.89:11434')

# الاتصال بقاعدة البيانات
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# إنشاء جدول إذا لم يكن موجودًا
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    message TEXT NOT NULL
)
""")
conn.commit()

# تحميل المحادثات السابقة
def load_messages():
    cursor.execute("SELECT role, message FROM chat_history ORDER BY id")
    rows = cursor.fetchall()

    history = []
    for role, message in rows:
        history.append({
            "role": role,
            "content": message
        })

    return history

# حفظ رسالة في قاعدة البيانات
def save_message(role, message):
    cursor.execute(
        "INSERT INTO chat_history(role, message) VALUES (?, ?)",
        (role, message)
    )
    conn.commit()

# مسح المحادثات
def clear_history():
    cursor.execute("DELETE FROM chat_history")
    conn.commit()

# ===========================
# Search History Tool
# ===========================

search_history_tool = {
    "type": "function",
    "function": {
        "name": "search_history",
        "description": "Search the chat history for a keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search in chat history"
                }
            },
            "required": ["keyword"]
        }
    }
}

def search_history(keyword):

    cursor.execute("""
        SELECT role, message
        FROM chat_history
        WHERE message LIKE ?
        ORDER BY id
    """, ('%' + keyword + '%',))

    rows = cursor.fetchall()

    if not rows:
        return "No matching messages found."

    result = ""

    for role, message in rows:
        result += f"{role}: {message}\n"

    return result
def summarize_chat():

    cursor.execute("""
        SELECT role, message
        FROM chat_history
        ORDER BY id
    """)

    rows = cursor.fetchall()

    if not rows:
        return "The chat history is empty."

    history = ""

    for role, message in rows:
        history += f"{role}: {message}\n"

    return history

# عرض الموديلات
models = client.list()

print("Available models:")
for m in models.get("models", []):
    print("-", m["model"])

print("\n✅ Ollama is running and accessible!\n")

# تحميل المحادثات السابقة
messages = load_messages()

# دالة الشات
def chat_with_memory(user_input):

    messages.append({
        "role": "user",
        "content": user_input
    })

    save_message("user", user_input)

    response = client.chat(
        model="gemma4:31b-cloud",
        messages=messages,
        tools=[
    search_history_tool
        ]
    )

    message = response["message"]
        # إذا طلب الموديل استخدام Tool
       # إذا طلب الموديل استخدام Tool
    if message.get("tool_calls"):

        for tool_call in message["tool_calls"]:

            fn_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]

            # ===============================
            # Search History Tool
            # ===============================
            if fn_name == "search_history":

                result = search_history(**args)

                final_response = client.chat(
                    model="gemma4:31b-cloud",
                    messages=[
                        *messages,
                        {
                            "role": "assistant",
                            "content": message.get("content", ""),
                            "tool_calls": message["tool_calls"]
                        },
                        {
                            "role": "tool",
                            "content": result
                        }
                    ]
                )

                reply = final_response["message"]["content"]

            # ===============================
            # Summarize Chat Tool
            # ===============================
            elif fn_name == "summarize_chat":

                result = summarize_chat()

                final_response = client.chat(
                    model="gemma4:31b-cloud",
                    messages=[
                        *messages,
                        {
                            "role": "assistant",
                            "content": message.get("content", ""),
                            "tool_calls": message["tool_calls"]
                        },
                        {
                            "role": "tool",
                            "content": result
                        },
                        {
                            "role": "user",
                            "content": "Please summarize this conversation."
                        }
                    ]
                )

                reply = final_response["message"]["content"]

            else:
                reply = "Unknown tool."

    else:
        # لو الموديل مردش باستخدام Tool
        reply = message["content"]

    # حفظ رد البوت
    messages.append({
        "role": "assistant",
        "content": reply
    })

    save_message("assistant", reply)

    return reply


print("🤖 Chatbot Started!")
print("Type 'exit' to quit.")
print("Type 'clear' to delete chat history.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    if user_input.lower() == "clear":
        clear_history()
        messages.clear()
        print("Bot: Chat history cleared.")
        continue

    print("Bot:", chat_with_memory(user_input))

# إغلاق قاعدة البيانات
conn.close()