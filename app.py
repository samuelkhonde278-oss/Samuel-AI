
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

# =========================================================
# SAMUEL AI CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

DATABASE = BASE_DIR / "samuel_ai.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

ERROR_MESSAGE = (
    "Samuel AI failed to generate a response. Please try again."
)

SYSTEM_INSTRUCTION = """
You are Samuel AI, an intelligent AI assistant created by Samuel Khonde,
a student at Mzuzu Government Secondary School.

IDENTITY:
- Your name is Samuel AI.
- You were created by Samuel Khonde.
- Never claim that Samuel Khonde created the underlying AI models.
- Do not claim to be Google Gemini, Groq, or another AI provider.
- If asked who created you, explain that Samuel Khonde created Samuel AI.

RESPONSE STYLE:
- Respond naturally, clearly, helpfully and intelligently.
- Give accurate and useful answers.
- Use Markdown formatting where appropriate.
- Use headings for long explanations.
- Use numbered steps for instructions.
- Use bullet points for lists.
- Use tables when they make information easier to understand.
- Use fenced code blocks for programming code.
- Explain difficult topics in simple language when appropriate.
- Do not unnecessarily repeat information.
- If you do not know something, be honest.
- Never invent facts, sources, or capabilities.

PROGRAMMING:
- Provide complete, working code when the user requests full code.
- Explain where code should be placed when necessary.
- Use clear code formatting.
- Help users understand errors and fix them step by step.

SAFETY:
- Support legal, ethical and educational uses of technology.
- Do not help with harmful, illegal or unauthorized activities.
- Protect users' passwords, API keys and private information.
"""

# =========================================================
# DATABASE
# =========================================================

def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id)
        """)

                connection.commit()


initialize_database()


def save_message(session_id, role, content):
    with get_db() as connection:
        connection.execute("""
            INSERT INTO messages
            (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            session_id,
            role,
            content,
            datetime.now().isoformat(timespec="seconds")
        ))
        connection.commit()


def get_messages(session_id, limit=40):
    with get_db() as connection:
        rows = connection.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    return list(reversed(rows))


# =========================================================
# AI PROVIDERS
# GEMINI FIRST, GROQ AUTOMATIC BACKUP
# =========================================================

def generate_with_gemini(contents):
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_output_tokens=4096
        )
    )

    reply = response.text

    if not reply:
        raise RuntimeError("Gemini returned an empty response.")

    return reply.strip()


def generate_with_groq(contents):
    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key is not configured.")

    client = Groq(api_key=GROQ_API_KEY)

    groq_messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    for item in contents:
        role = item.get("role", "user")

        if role == "model":
            role = "assistant"

        if role not in ("user", "assistant"):
            continue

        parts = item.get("parts", [])

        text_content = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        )

        if text_content.strip():
            groq_messages.append({
                "role": role,
                "content": text_content
            })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=groq_messages,
        temperature=0.7,
        max_tokens=4096
    )

    reply = response.choices[0].message.content

    if not reply:
        raise RuntimeError("Groq returned an empty response.")

    return reply.strip()



def generate_with_openrouter(contents):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter API key is not configured.")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-OpenRouter-Title": "Samuel AI"
        }
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    for item in contents:
        role = item.get("role", "user")

        if role == "model":
            role = "assistant"

        if role not in ("user", "assistant"):
            continue

        parts = item.get("parts", [])
        text_content = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        )

        if text_content.strip():
            messages.append({
                "role": role,
                "content": text_content
            })

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )

    reply = response.choices[0].message.content

    if not reply:
        raise RuntimeError("OpenRouter returned an empty response.")

    return reply.strip()


def generate_ai_reply(contents):
    """
    Try providers in order: Gemini, Groq, then OpenRouter.
    If one provider fails, automatically try the next one.
    """

    providers = [
        ("Gemini", generate_with_gemini),
        ("Groq", generate_with_groq),
        ("OpenRouter", generate_with_openrouter),
    ]

    errors = []

    for provider_name, provider_function in providers:
        try:
            reply = provider_function(contents)
            return reply, provider_name
        except Exception as error:
            app.logger.warning("%s failed: %s", provider_name, error)
            errors.append(f"{provider_name}: {error}")

    raise RuntimeError(
        "All AI providers failed. " + " | ".join(errors)
    )


# =========================================================
# FRONTEND
# =========================================================

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Samuel AI</title>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>

<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github-dark.min.css">

<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"></script>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #111827;
    color: #f9fafb;
    height: 100vh;
    overflow: hidden;
}

.app {
    display: flex;
    height: 100vh;
}

.sidebar {
    width: 270px;
    background: #0b1220;
    border-right: 1px solid #263244;
    display: flex;
    flex-direction: column;
    padding: 16px;
}

.brand {
    font-size: 23px;
    font-weight: bold;
    margin-bottom: 22px;
    color: #60a5fa;
}

.new-chat {
    border: 1px solid #374151;
    background: #1f2937;
    color: white;
    border-radius: 9px;
    padding: 12px;
    cursor: pointer;
    text-align: left;
    font-size: 15px;
}

.new-chat:hover,
.history-item:hover {
    background: #263449;
}

.history-title {
    margin-top: 25px;
    margin-bottom: 10px;
    color: #9ca3af;
    font-size: 13px;
}

.history-list {
    flex: 1;
    overflow-y: auto;
}

.history-item {
    padding: 11px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.history-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
}

.delete-chat {
    border: none;
    background: transparent;
    color: #9ca3af;
    cursor: pointer;
    font-size: 17px;
}

.delete-chat:hover {
    color: #f87171;
}

.creator {
    border-top: 1px solid #263244;
    padding-top: 15px;
    color: #9ca3af;
    font-size: 12px;
    line-height: 1.6;
}

.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.topbar {
    height: 60px;
    border-bottom: 1px solid #263244;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
}

.topbar-title {
    font-weight: bold;
    font-size: 18px;
}

.export-button {
    background: #1f2937;
    border: 1px solid #374151;
    color: white;
    border-radius: 8px;
    padding: 9px 12px;
    cursor: pointer;
}

.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 25px 20px;
}

.messages {
    max-width: 850px;
    margin: auto;
}

.welcome {
    text-align: center;
    margin-top: 13vh;
}

.welcome h1 {
    font-size: 34px;
    color: #60a5fa;
}

.welcome p {
    color: #9ca3af;
}

.message {
    margin: 18px 0;
    display: flex;
    flex-direction: column;
}

.message.user {
    align-items: flex-end;
}

.message.assistant {
    align-items: flex-start;
}

.bubble {
    max-width: 90%;
    padding: 14px 17px;
    border-radius: 13px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.user .bubble {
    background: #1d4ed8;
    color: white;
}

.assistant .bubble {
    background: #1f2937;
    border: 1px solid #374151;
}

.bubble p {
    margin-top: 5px;
    margin-bottom: 12px;
}

.bubble pre {
    background: #080d16;
    padding: 14px;
    border-radius: 8px;
    overflow-x: auto;
}

.bubble code {
    font-family: Consolas, monospace;
}

.bubble :not(pre) > code {
    background: #374151;
    padding: 2px 5px;
    border-radius: 4px;
}

.bubble table {
    border-collapse: collapse;
    display: block;
    overflow-x: auto;
}

.bubble th,
.bubble td {
    border: 1px solid #4b5563;
    padding: 8px;
}

.bubble a {
    color: #60a5fa;
}

.message-actions {
    display: flex;
    gap: 8px;
    margin-top: 7px;
    flex-wrap: wrap;
}

.message-actions button,
.code-download {
    border: 1px solid #374151;
    background: #1f2937;
    color: #d1d5db;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
}

.message-actions button:hover,
.code-download:hover {
    background: #374151;
}

.composer-area {
    padding: 15px 20px 20px;
    border-top: 1px solid #263244;
}

.composer {
    max-width: 850px;
    margin: auto;
    display: flex;
    gap: 10px;
    background: #1f2937;
    border: 1px solid #4b5563;
    border-radius: 13px;
    padding: 10px;
}

textarea {
    flex: 1;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: white;
    font-size: 15px;
    font-family: Arial, sans-serif;
    min-height: 35px;
    max-height: 160px;
    padding: 8px;
}

.send-button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0 18px;
    cursor: pointer;
    font-weight: bold;
}

.send-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.status {
    max-width: 850px;
    margin: 8px auto 0;
    color: #9ca3af;
    font-size: 12px;
}

@media(max-width: 700px) {
    .sidebar {
        width: 75px;
        padding: 10px 7px;
    }

    .brand {
        font-size: 12px;
        text-align: center;
    }

    .new-chat {
        font-size: 11px;
        padding: 9px 5px;
        text-align: center;
    }

    .history-title,
    .history-name,
    .creator {
        display: none;
    }

    .history-item {
        justify-content: center;
        padding: 10px 2px;
    }

    .delete-chat {
        display: none;
    }

    .topbar {
        padding: 0 10px;
    }

    .chat-area {
        padding: 15px 10px;
    }

    .bubble {
        max-width: 96%;
    }

    .composer-area {
        padding: 10px;
    }

    .export-button {
        font-size: 11px;
    }
}
</style>
</head>

<body>

<div class="app">

    <aside class="sidebar">

        <div class="brand">Samuel AI</div>

        <button class="new-chat" onclick="newChat()">
            + New Chat
        </button>

        <div class="history-title">Previous Conversations</div>

        <div class="history-list" id="historyList"></div>

        <div class="creator">
            Created by Samuel Khonde<br>
            Samuel Learning Technology
        </div>

    </aside>

    <main class="main">

        <header class="topbar">
            <div class="topbar-title">Samuel AI</div>

            <button class="export-button" onclick="exportConversation()">
                Export Full Conversation
            </button>
        </header>

        <section class="chat-area" id="chatArea">

            <div class="messages" id="messages">

                <div class="welcome" id="welcome">
                    <h1>Welcome to Samuel AI</h1>
                    <p>Your intelligent AI assistant.</p>
                    <p>Ask a question to get started.</p>
                </div>

            </div>

        </section>

        <div class="composer-area">

            <div class="composer">

                <textarea
                    id="messageInput"
                    placeholder="Message Samuel AI..."
                    rows="1"
                ></textarea>

                <button
                    class="send-button"
                    id="sendButton"
                    onclick="sendMessage()"
                >
                    Send
                </button>

            </div>

            <div class="status" id="status">
                Samuel AI is ready.
            </div>

        </div>

    </main>

</div>

<script>
let sessionId = localStorage.getItem("samuel_session");

if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("samuel_session", sessionId);
}

let conversation = [];
let sending = false;

const messagesElement = document.getElementById("messages");
const inputElement = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const statusElement = document.getElementById("status");

marked.setOptions({
    gfm: true,
    breaks: true
});


function setStatus(message) {
    statusElement.textContent = message;
}


function scrollToBottom() {
    const area = document.getElementById("chatArea");
    area.scrollTop = area.scrollHeight;
}


function makeId() {
    return crypto.randomUUID();
}


function addMessage(role, content) {
    const welcome = document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }

    const wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "assistant") {
        const rawHtml = marked.parse(content);
        bubble.innerHTML = DOMPurify.sanitize(rawHtml);

        bubble.querySelectorAll("pre code").forEach(block => {
            hljs.highlightElement(block);

            const downloadButton = document.createElement("button");
            downloadButton.className = "code-download";
            downloadButton.textContent = "Download Code";

            downloadButton.addEventListener("click", () => {
                downloadText(
                    block.textContent,
                    "samuel-ai-code.txt"
                );
            });

            block.parentElement.appendChild(downloadButton);
        });

    } else {
        bubble.textContent = content;
    }

    wrapper.appendChild(bubble);

    if (role === "assistant") {
        const actions = document.createElement("div");
        actions.className = "message-actions";

        const copyButton = document.createElement("button");
        copyButton.textContent = "Copy Response";

        copyButton.onclick = async () => {
            try {
                await navigator.clipboard.writeText(content);
                copyButton.textContent = "Copied!";

                setTimeout(() => {
                    copyButton.textContent = "Copy Response";
                }, 1500);

            } catch (error) {
                setStatus("Could not copy response.");
            }
        };

        const downloadButton = document.createElement("button");
        downloadButton.textContent = "Download Response";

        downloadButton.onclick = () => {
            downloadText(content, "samuel-ai-response.txt");
        };

        actions.appendChild(copyButton);
        actions.appendChild(downloadButton);

        wrapper.appendChild(actions);
    }

    messagesElement.appendChild(wrapper);
    scrollToBottom();
}


function downloadText(content, filename) {
    const blob = new Blob([content], {
        type: "text/plain;charset=utf-8"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
}


function newChat() {
    sessionId = makeId();
    localStorage.setItem("samuel_session", sessionId);

    conversation = [];

    messagesElement.innerHTML = `
        <div class="welcome" id="welcome">
            <h1>Welcome to Samuel AI</h1>
            <p>Your intelligent AI assistant.</p>
            <p>Ask a question to get started.</p>
        </div>
    `;

    setStatus("New conversation started.");
    loadConversations();
}


async function sendMessage() {
    const message = inputElement.value.trim();

    if (!message || sending) {
        return;
    }

    sending = true;
    sendButton.disabled = true;

    inputElement.value = "";
    inputElement.style.height = "35px";

    conversation.push({
        role: "user",
        content: message
    });

    addMessage("user", message);
    setStatus("Samuel AI is thinking...");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        conversation.push({
            role: "assistant",
            content: data.reply
        });

        addMessage("assistant", data.reply);

        setStatus("Samuel AI is ready.");

        loadConversations();

    } catch (error) {
        addMessage(
            "assistant",
            "Samuel AI failed to generate a response. Please try again."
        );

        setStatus("Samuel AI is ready. Please try again.");

    } finally {
        sending = false;
        sendButton.disabled = false;
        inputElement.focus();
    }
}


async function loadHistory(id) {
    sessionId = id;
    localStorage.setItem("samuel_session", id);

    messagesElement.innerHTML = "";
    conversation = [];

    try {
        const response = await fetch(
            "/api/history/" + encodeURIComponent(id)
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not load conversation.");
        }

        if (!data.messages.length) {
            messagesElement.innerHTML = `
                <div class="welcome" id="welcome">
                    <h1>Welcome to Samuel AI</h1>
                    <p>Start a new conversation.</p>
                </div>
            `;
            return;
        }

        data.messages.forEach(item => {
            const role = item.role === "user" ? "user" : "assistant";

            conversation.push({
                role: role,
                content: item.content
            });

            addMessage(role, item.content);
        });

        setStatus("Conversation loaded.");

    } catch (error) {
        setStatus(error.message);
    }
}


async function loadConversations() {
    try {
        const response = await fetch("/api/conversations");
        const data = await response.json();

        const list = document.getElementById("historyList");
        list.innerHTML = "";

        data.conversations.forEach(item => {
            const row = document.createElement("div");
            row.className = "history-item";

            const name = document.createElement("div");
            name.className = "history-name";
            name.textContent = item.title || "New conversation";
            name.title = item.title || "New conversation";

            name.onclick = () => loadHistory(item.session_id);

            const deleteButton = document.createElement("button");
            deleteButton.className = "delete-chat";
            deleteButton.textContent = "×";
            deleteButton.title = "Delete conversation";

            deleteButton.onclick = (event) => {
                event.stopPropagation();
                deleteConversation(item.session_id);
            };

            row.appendChild(name);
            row.appendChild(deleteButton);

            list.appendChild(row);
        });

    } catch (error) {
        console.error("Could not load conversations:", error);
    }
}


async function deleteConversation(id) {
    if (!confirm("Delete this conversation?")) {
        return;
    }

    try {
        const response = await fetch(
            "/api/delete-conversation/" + encodeURIComponent(id),
            { method: "DELETE" }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not delete conversation.");
        }

        if (sessionId === id) {
            newChat();
        }

        loadConversations();

    } catch (error) {
        alert(error.message);
    }
}


function exportConversation() {
    if (!conversation.length) {
        alert("There is no conversation to export yet.");
        return;
    }

    const text = conversation.map(item => {
        const speaker = item.role === "user" ? "You" : "Samuel AI";

        return speaker + ":\n" + item.content;
    }).join("\n\n--------------------\n\n");

    downloadText(text, "samuel-ai-conversation.txt");
}


inputElement.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});


inputElement.addEventListener("input", () => {
    inputElement.style.height = "35px";

    inputElement.style.height =
        Math.min(inputElement.scrollHeight, 160) + "px";
});


loadConversations();

fetch("/api/history/" + encodeURIComponent(sessionId))
    .then(response => response.json())
    .then(data => {
        if (data.messages && data.messages.length) {
            data.messages.forEach(item => {
                const role = item.role === "user" ? "user" : "assistant";

                conversation.push({
                    role: role,
                    content: item.content
                });

                addMessage(role, item.content);
            });
        }
    })
    .catch(error => console.error(error));

</script>

</body>
</html>
"""


# =========================================================
# FLASK ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    session_id = str(data.get("session_id", "")).strip()
    message = str(data.get("message", "")).strip()

    if not session_id:
        return jsonify({"error": "Missing conversation ID."}), 400

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 30000:
        return jsonify({"error": "Your message is too long."}), 400

    save_message(session_id, "user", message)

    rows = get_messages(session_id, limit=20)

    contents = []

    for row in rows:
        role = row["role"]

        if role not in ("user", "model"):
            continue

        contents.append({
            "role": role,
            "parts": [
                {"text": row["content"]}
            ]
        })

    try:
        reply, provider = generate_ai_reply(contents)

        save_message(session_id, "model", reply)

        return jsonify({
            "reply": reply
        })

    except Exception:
        app.logger.exception(
            "Samuel AI failed to generate a response."
        )

        return jsonify({
            "error": ERROR_MESSAGE
        }), 503


@app.route("/api/history/<session_id>", methods=["GET"])
def history(session_id):
    rows = get_messages(session_id, limit=500)

    messages = [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]

    return jsonify({"messages": messages})


@app.route("/api/conversations", methods=["GET"])
def conversations():
    with get_db() as connection:
        rows = connection.execute("""
            SELECT
                session_id,
                MAX(id) AS last_message_id,
                MAX(created_at) AS last_message_time
            FROM messages
            GROUP BY session_id
            ORDER BY last_message_id DESC
            LIMIT 100
        """).fetchall()

        result = []

        for row in rows:
            first_message = connection.execute("""
                SELECT content
                FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY id ASC
                LIMIT 1
            """, (row["session_id"],)).fetchone()

            title = (
                first_message["content"][:55]
                if first_message
                else "New conversation"
            )

            result.append({
                "session_id": row["session_id"],
                "title": title,
                "last_message_time": row["last_message_time"]
            })

    return jsonify({"conversations": result})


@app.route("/api/delete-conversation/<session_id>", methods=["DELETE"])
def delete_conversation(session_id):
    with get_db() as connection:
        connection.execute("""
            DELETE FROM messages
            WHERE session_id = ?
        """, (session_id,))
        connection.commit()

    return jsonify({"success": True})


# =========================================================
# START SAMUEL AI
# =========================================================

if __name__ == "__main__":
    initialize_database()

    print("=" * 50)
    print("SAMUEL AI IS STARTING")
    print("=" * 50)
    print("Open Chrome: http://127.0.0.1:5000")
    print("Gemini configured:", bool(GEMINI_API_KEY))
    print("Groq configured:", bool(GROQ_API_KEY))
    print("OpenRouter configured:", bool(OPENROUTER_API_KEY))
    print("=" * 50)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

