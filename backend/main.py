"""
MyHaven Backend — FastAPI v6.1
Groq (llama-3.3-70b-versatile) + PERMANENT SQLite memory
Emotion/sentiment disabled (MuRIL removed for deployment)
Crisis: shares helplines BUT keeps conversation open
"""

import sqlite3
import json
import os
import re
import time
from typing import List
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

MAX_TURNS = 30

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myhaven.db")

CRISIS_WORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "want to die", "i want to die", "i will die", "hurt myself",
    "no point living", "can't go on", "rather be dead", "end it all"
]


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id      TEXT PRIMARY KEY,
                name         TEXT,
                language     TEXT DEFAULT 'english',
                topics       TEXT DEFAULT '[]',
                mood_history TEXT DEFAULT '[]',
                details      TEXT DEFAULT '[]',
                created_at   REAL,
                updated_at   REAL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
        """)
    print(f"[DB] Database ready at {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def db_get_user(user_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            return {
                "name":         row["name"],
                "language":     row["language"] or "english",
                "topics":       json.loads(row["topics"] or "[]"),
                "mood_history": json.loads(row["mood_history"] or "[]"),
                "details":      json.loads(row["details"] or "[]"),
            }
        return {"name": None, "language": "english", "topics": [], "mood_history": [], "details": []}


def db_upsert_user(user_id: str, ctx: dict):
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO users (user_id, name, language, topics, mood_history, details, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name         = excluded.name,
                language     = excluded.language,
                topics       = excluded.topics,
                mood_history = excluded.mood_history,
                details      = excluded.details,
                updated_at   = excluded.updated_at
        """, (
            user_id,
            ctx.get("name"),
            ctx.get("language", "english"),
            json.dumps(ctx.get("topics", [])),
            json.dumps(ctx.get("mood_history", [])),
            json.dumps(ctx.get("details", [])),
            now, now
        ))


def db_get_messages(user_id: str, limit: int = MAX_TURNS) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def db_add_message(user_id: str, role: str, content: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time())
        )


def db_clear_user(user_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))


def db_full_reset_user(user_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT + CONTEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(ctx: dict, crisis: bool = False) -> str:
    name         = ctx.get("name") or "there"
    topics       = ctx.get("topics", [])
    mood_history = ctx.get("mood_history", [])
    details      = ctx.get("details", [])

    context_block = ""
    if name != "there":
        context_block += f"- User's name: {name}\n"
    if topics:
        context_block += f"- Topics so far: {', '.join(topics[-5:])}\n"
    if mood_history:
        context_block += f"- Mood trend: {', '.join(mood_history[-5:])}\n"
    if details:
        context_block += f"- Known details: {'; '.join(details[-8:])}\n"

    crisis_instruction = ""
    if crisis:
        crisis_instruction = """
CRISIS MODE — VERY IMPORTANT:
- The user has expressed thoughts of suicide or self-harm.
- Do NOT end the conversation. Do NOT say "I cannot continue".
- Share the crisis helpline: Kiran Mental Health Helpline: 9152987821 (Free 24/7)
- Stay warm, present, and human. Acknowledge their pain directly.
- Gently encourage them to call, but keep talking to them.
"""

    context_section = f"\nWhat you know about this user:\n{context_block}" if context_block else ""

    return f"""You are Haven, a warm emotional-support companion for undergraduate students in India.

LANGUAGE RULE (CRITICAL):
- Reply in EXACTLY the language the user writes in right now.
- English only if they write English. Hindi only if they write Hindi. Mirror Hinglish if they mix.
- DEFAULT is ENGLISH.

MEMORY RULE (CRITICAL):
- You have permanent memory of this user. Use it naturally like a real friend would.
- Reference their name, past topics, mood history organically.
- Example: "Last time you mentioned your exams were stressing you out — how did that go?"

CONVERSATION STYLE:
- Talk like a warm caring friend, NOT a therapist or AI.
- Keep replies to 2-4 sentences. Be concise and natural.
- NEVER say: "I'm always here for you", "I understand how you feel", "That must be hard".
- Ask ONE gentle follow-up question when it feels natural.
- Do NOT give medical diagnoses or prescribe anything.
{crisis_instruction}{context_section}"""


def detect_language(text: str) -> str:
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hindi"
    hinglish = ["yaar","bhai","kya","nahi","bahut","hua","hai","hoon",
                "mujhe","toh","aur","kal","abhi","thoda","bohot","kuch",
                "sab","accha","theek","tera","mera","bilkul"]
    if any(w in text.lower().split() for w in hinglish):
        return "hinglish"
    return "english"


def extract_context_from_message(text: str, ctx: dict) -> dict:
    tl = text.lower()
    ctx["language"] = detect_language(text)

    topic_map = {
        "exams":        ["exam","exams","test","paper","marks","result","score"],
        "family":       ["family","mom","dad","parents","sister","brother","mummy","papa"],
        "friends":      ["friend","friends","bestie","yaar","dost","classmate"],
        "anxiety":      ["anxious","anxiety","panic","nervous","overthinking"],
        "depression":   ["depressed","depression","sad","crying","empty","numb"],
        "sleep":        ["sleep","insomnia","tired","exhausted"],
        "relationship": ["relationship","boyfriend","girlfriend","breakup","crush","love"],
        "college":      ["college","university","semester","assignment","project","professor"],
        "career":       ["job","career","placement","internship","future"],
        "stress":       ["stress","stressed","pressure","burden","overwhelmed"],
        "loneliness":   ["lonely","alone","isolated","no one","nobody"],
    }
    for topic, kws in topic_map.items():
        if any(k in tl for k in kws) and topic not in ctx["topics"]:
            ctx["topics"].append(topic)

    mood_map = {
        "sad":      ["sad","unhappy","crying","tears"],
        "stressed": ["stressed","stress","tension","pressure","overwhelmed"],
        "anxious":  ["anxious","nervous","scared","worried","panic"],
        "angry":    ["angry","frustrated","annoyed","irritated"],
        "happy":    ["happy","good","great","better","fine"],
        "hopeless": ["hopeless","give up","no point","lost"],
        "tired":    ["tired","exhausted","drained"],
        "relieved": ["relieved","better now","feeling good","helped"],
    }
    for mood, kws in mood_map.items():
        if any(k in tl for k in kws):
            if not ctx["mood_history"] or ctx["mood_history"][-1] != mood:
                ctx["mood_history"].append(mood)

    detail_patterns = [
        ("final year student",    ["final year","4th year","fourth year"]),
        ("3rd year student",      ["3rd year","third year"]),
        ("2nd year student",      ["2nd year","second year"]),
        ("1st year student",      ["1st year","first year","fresher"]),
        ("going through breakup", ["breakup","broke up","she left","he left"]),
        ("exam pressure",         ["exam pressure","exam stress"]),
        ("feeling lonely",        ["feeling lonely","no one to talk"]),
    ]
    for detail, pats in detail_patterns:
        if any(p in tl for p in pats) and detail not in ctx["details"]:
            ctx["details"].append(detail)

    nm = re.search(
        r"(?:i am|i'm|my name is|mera naam hai|call me)\s+([A-Z][a-z]+)",
        text, re.IGNORECASE
    )
    if nm and not ctx.get("name"):
        ctx["name"] = nm.group(1).strip()

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# GROQ CHAT
# ══════════════════════════════════════════════════════════════════════════════

_groq_client = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def _is_crisis(text: str) -> bool:
    return any(w in (text or "").lower() for w in CRISIS_WORDS)

def groq_chat(user_id: str, user_message: str) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY missing."

    crisis = _is_crisis(user_message)
    ctx = db_get_user(user_id)
    ctx = extract_context_from_message(user_message, ctx)
    db_upsert_user(user_id, ctx)
    db_add_message(user_id, "user", user_message)

    history = db_get_messages(user_id, limit=MAX_TURNS)
    system_prompt = build_system_prompt(ctx, crisis=crisis)
    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        r = get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            temperature=0.85, max_tokens=260, top_p=0.95
        )
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] {e}")
        reply = "I got a little glitchy — could you say that again?"

    db_add_message(user_id, "assistant", reply)
    return reply


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MyHaven Backend", version="6.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup_event():
    init_db()
    print("[Startup] MyHaven Backend v6.1 ready")


class UserReq(BaseModel):
    user_id: str = Field(..., min_length=1)

class ChatReq(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@app.post("/start")
def start(req: UserReq):
    ctx = db_get_user(req.user_id)
    name = ctx.get("name")
    mood_history = ctx.get("mood_history", [])
    topics = ctx.get("topics", [])

    if name:
        if mood_history:
            greeting = f"Hey {name}! 💛 Welcome back. Last time you seemed {mood_history[-1]} — how are you feeling today?"
        elif topics:
            greeting = f"Hey {name}! 💛 Good to have you back. We were talking about {topics[-1]} last time — what's on your mind today?"
        else:
            greeting = f"Hey {name}! 💛 Good to see you again. What's been going on?"
    else:
        greeting = "Hey! I'm Haven 💛 What's been weighing on your mind lately — college stuff, relationships, family, or something else?"

    db_upsert_user(req.user_id, ctx)
    db_add_message(req.user_id, "assistant", greeting)
    return {"reply": greeting, "returning_user": bool(name)}


@app.post("/chat")
def chat(req: ChatReq):
    reply = groq_chat(req.user_id, req.message)
    ctx = db_get_user(req.user_id)
    return {
        "reply": reply,
        "emotion": {"label": "Neutral", "confidence": 0.0, "emoji": "😐", "note": ""},
        "sentiment": {"label": "Neutral", "polarity": 0.0, "subjectivity": 0.0, "note": ""},
        "context": {
            "name": ctx.get("name"),
            "language": ctx.get("language", "english"),
            "topics": ctx.get("topics", []),
            "mood_history": ctx.get("mood_history", [])
        },
        "meta": {"groq_model": GROQ_MODEL, "muril_loaded": False},
    }


@app.post("/reset")
def reset(req: UserReq):
    db_clear_user(req.user_id)
    return {"ok": True}


@app.post("/full_reset")
def full_reset(req: UserReq):
    db_full_reset_user(req.user_id)
    return {"ok": True}


@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    return {
        "user_id": user_id,
        "profile": db_get_user(user_id),
        "messages": db_get_messages(user_id, limit=limit),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "time": time.time(),
        "groq_model": GROQ_MODEL,
        "api_key_set": bool(GROQ_API_KEY),
        "version": "6.1"
    }