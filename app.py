from dotenv import load_dotenv
import os

load_dotenv()
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from pypdf import PdfReader
import base64
import os
import io
import json

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

chat_history = [
    {
        "role": "system",
        "content": """You are Zeno, a helpful, friendly, and intelligent AI assistant with real-time web search capability.

When a user asks about current events, news, recent developments, live data (weather, stocks, sports), or anything that requires up-to-date information, use the search_web tool to find the latest information before answering.

For general knowledge, coding help, explanations, math, or creative tasks — answer directly without searching.

When you search, always cite your sources naturally in your response. Be concise, clear, and engaging."""
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for real-time, current, or recent information. Use this for news, current events, live data, recent releases, or anything that may have changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def do_search(query):
    try:
        results = tavily.search(query=query, max_results=5)
        formatted = []
        for r in results.get("results", []):
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500]
            })
        return json.dumps(formatted)
    except Exception as e:
        return json.dumps({"error": str(e)})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "")
    file = request.files.get("file")
    content = []

    if file and file.filename:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file.read()))
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            chat_history.append({
                "role": "user",
                "content": f"{user_message}\n\n[PDF Content]:\n{pdf_text[:12000]}"
            })
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=chat_history,
                tools=tools
            )
        elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            image_data = base64.b64encode(file.read()).decode("utf-8")
            ext = filename.split(".")[-1]
            mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
            vision_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}}
                ]
            }]
            response = client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=vision_messages
            )
            reply = response.choices[0].message.content
            chat_history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        else:
            return jsonify({"error": "Unsupported file type."}), 400
    else:
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        chat_history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=chat_history,
            tools=tools,
            tool_choice="auto"
        )

    # Handle tool calls
    msg = response.choices[0].message
    if msg.tool_calls:
        chat_history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in msg.tool_calls
            ]
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            search_result = do_search(args["query"])
            chat_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": search_result
            })
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_history
        )
        reply = final_response.choices[0].message.content
    else:
        reply = msg.content

    chat_history.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})

@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = [chat_history[0]]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)