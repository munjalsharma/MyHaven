"""
MyHaven Backend — FastAPI v6.0
Groq (llama-3.3-70b-versatile) + PERMANENT SQLite memory + MuRIL emotion/sentiment
All conversation history, user context, mood, topics saved to myhaven.db
Crisis: shares helplines BUT keeps conversation open
"""

import sqlite3
import json
import os
import re
import time
from typing import Dict, List, Tuple
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

# DB path: sits next to main.py inside backend/
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myhaven.db")

CRISIS_WORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "want to die", "i want to die", "i will die", "hurt myself",
    "no point living", "can't go on", "rather be dead", "end it all"
]

INDIA_KIRAN = "9152987821 (Kiran · Free · 24/7)"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Returns a new SQLite connection. Use as a context manager via get_db()."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    """Context manager for DB — always closes connection after use."""
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
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                name        TEXT,
                language    TEXT DEFAULT 'english',
                topics      TEXT DEFAULT '[]',
                mood_history TEXT DEFAULT '[]',
                details     TEXT DEFAULT '[]',
                created_at  REAL,
                updated_at  REAL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
        """)
    print(f"[DB] ✅ Database ready at {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS — READ / WRITE
# ══════════════════════════════════════════════════════════════════════════════

def db_get_user(user_id: str) -> dict:
    """Load user context from DB. Returns a dict (never None)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return {
                "name":         row["name"],
                "language":     row["language"] or "english",
                "topics":       json.loads(row["topics"] or "[]"),
                "mood_history": json.loads(row["mood_history"] or "[]"),
                "details":      json.loads(row["details"] or "[]"),
            }
        return {
            "name": None, "language": "english",
            "topics": [], "mood_history": [], "details": []
        }


def db_upsert_user(user_id: str, ctx: dict):
    """Save or update user context in DB."""
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
    """Load last N messages for a user, oldest first."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        # rows are newest-first, reverse to get chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def db_add_message(user_id: str, role: str, content: str):
    """Append a single message to DB."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time())
        )


def db_clear_user(user_id: str):
    """Delete all messages for a user (reset conversation). Keep user profile."""
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))


def db_full_reset_user(user_id: str):
    """Delete messages AND user profile entirely."""
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT + CONTEXT
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
- Share the crisis helpline number naturally in your reply: Kiran Mental Health Helpline: 9152987821 (Free · 24/7)
- Stay warm, present, and human. Acknowledge their pain directly.
- Gently encourage them to call or reach out to someone, but keep talking to them.
- Ask one caring question to understand what they're going through.
- Example tone: "That sounds incredibly painful. You don't have to face this alone — please consider calling Kiran at 9152987821, they're available 24/7 and it's free. Can you tell me a bit about what's been happening?"
"""

    context_section = f"""
What you know about this user:
{context_block}""" if context_block else ""

    return f"""You are Haven, a warm emotional-support companion for undergraduate students in India.

LANGUAGE RULE (CRITICAL — follow this strictly):
- Detect the language the user is writing in RIGHT NOW and reply in EXACTLY that language.
- User writes in ENGLISH → reply in ENGLISH only. Zero Hindi words.
- User writes in HINDI (Devanagari script) → reply in HINDI only.
- User deliberately mixes Hindi + English (Hinglish like "yaar I'm so stressed") → mirror that mix.
- DEFAULT is ENGLISH. Never switch on your own — always follow the user's lead.

MEMORY RULE (CRITICAL):
- You have permanent memory of this user. Use it naturally.
- If you know their name, mood history, or topics — reference them organically.
- Example: "Last time you mentioned your exams were stressing you out — how did that go?"
- Never mention that you "looked up" their history. Just use it naturally like a real friend would.

CONVERSATION STYLE:
- Talk like a warm caring friend, NOT a therapist or AI assistant.
- Keep replies to 2–4 sentences. Be concise and natural.
- NEVER use: "I'm always here for you", "I understand how you feel", "That must be hard", "I cannot assist with".
- Ask ONE gentle follow-up question when it feels natural.
- Never repeat phrasing you used earlier in this conversation.
- If you know the user's name, use it occasionally — not every message.
- Reference what you know about them to make them feel heard and remembered.
- Always keep the conversation flowing — leave space for them to share more.
- Do NOT give medical diagnoses or prescribe anything.
{crisis_instruction}{context_section}"""


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE + CONTEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hindi"
    hinglish = ["yaar","bhai","kya","nahi","bahut","hua","hai","hoon",
                "mujhe","toh","aur","kal","abhi","thoda","bohot","kuch",
                "sab","accha","theek","nahi","tera","mera","acha","bilkul"]
    tl = text.lower()
    if any(w in tl.split() for w in hinglish):
        return "hinglish"
    return "english"


def extract_context_from_message(text: str, ctx: dict) -> dict:
    """Update ctx dict in-place with info extracted from message. Returns updated ctx."""
    tl = text.lower()
    ctx["language"] = detect_language(text)

    topic_map = {
        "exams":        ["exam","exams","test","paper","marks","result","pariksha","score"],
        "math":         ["math","maths","calculus","algebra","mathematics","statistics"],
        "family":       ["family","mom","dad","parents","sister","brother","ghar","mummy","papa","bhaiya","didi"],
        "friends":      ["friend","friends","bestie","yaar","dost","classmate"],
        "anxiety":      ["anxious","anxiety","panic","nervous","overthinking","phobia"],
        "depression":   ["depressed","depression","sad","crying","empty","numb","hollow"],
        "sleep":        ["sleep","insomnia","tired","exhausted","neend"],
        "relationship": ["relationship","boyfriend","girlfriend","breakup","crush","love","heartbreak"],
        "college":      ["college","university","semester","assignment","project","professor","faculty"],
        "career":       ["job","career","placement","internship","future","campus"],
        "stress":       ["stress","stressed","pressure","burden","tension","overwhelmed"],
        "loneliness":   ["lonely","alone","isolated","no one","nobody"],
    }
    for topic, kws in topic_map.items():
        if any(k in tl for k in kws) and topic not in ctx["topics"]:
            ctx["topics"].append(topic)

    mood_map = {
        "sad":      ["sad","dukhi","unhappy","crying","tears"],
        "stressed": ["stressed","stress","tension","pressure","overwhelmed"],
        "anxious":  ["anxious","nervous","scared","worried","panic","darr"],
        "angry":    ["angry","frustrated","annoyed","irritated","gusse"],
        "happy":    ["happy","good","great","better","fine","khush","acha"],
        "hopeless": ["hopeless","give up","no point","lost","koi fayda nahi"],
        "tired":    ["tired","exhausted","drained","thaka","thaki"],
        "relieved": ["relieved","better now","feeling good","thanks","helped"],
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
        ("struggling with math",  ["math is stressing","maths is hard","maths mein problem"]),
        ("going through breakup", ["breakup","broke up","she left","he left"]),
        ("exam pressure",         ["exam pressure","exam stress","pariksha ka darr"]),
        ("feeling lonely",        ["feeling lonely","no one to talk","akela"]),
    ]
    for detail, pats in detail_patterns:
        if any(p in tl for p in pats) and detail not in ctx["details"]:
            ctx["details"].append(detail)

    nm = re.search(
        r"(?:i am|i'm|my name is|mera naam hai|call me|naam hai mera)\s+([A-Z][a-z]+)",
        text, re.IGNORECASE
    )
    if nm and not ctx.get("name"):
        ctx["name"] = nm.group(1).strip()

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# MuRIL EMOTION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

MURIL_BASE     = os.getenv("MURIL_BASE", "google/muril-base-cased")
EMOTION_LABELS = ["joy","sadness","fear","anger","surprise","neutral","disgust","shame"]
EMOJI_MAP      = {"joy":"😊","sadness":"😢","fear":"😰","anger":"😠",
                  "surprise":"😲","neutral":"😐","disgust":"🤢","shame":"😳"}
EMOTION_TO_SENTIMENT = {
    "joy":(0.80,0.90),"surprise":(0.30,0.70),"neutral":(0.00,0.20),
    "shame":(-0.30,0.90),"disgust":(-0.40,0.80),"fear":(-0.50,0.80),
    "sadness":(-0.65,0.90),"anger":(-0.80,0.90),
}

_tokenizer = None
_muril_model = None
_head_loaded = False
_device = "cpu"


def _load_muril():
    global _tokenizer, _muril_model, _head_loaded, _device
    if _tokenizer is not None:
        return
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MuRIL] device: {_device}")
    _tokenizer = AutoTokenizer.from_pretrained(MURIL_BASE)

    class MuRILEmotionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.muril = AutoModel.from_pretrained(MURIL_BASE)
            for p in self.muril.embeddings.parameters():
                p.requires_grad = False
            self.drop = nn.Dropout(0.4)
            self.clf = nn.Sequential(
                nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.BatchNorm1d(256),
                nn.Linear(256, len(EMOTION_LABELS))
            )

        def forward(self, ids, mask):
            o = self.muril(input_ids=ids, attention_mask=mask, return_dict=True)
            return self.clf(self.drop(o.last_hidden_state[:, 0]))

    _muril_model = MuRILEmotionClassifier().to(_device)
    _muril_model.eval()
    _here = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.getenv("MURIL_EMOTION_WEIGHTS", ""),
        os.path.join(_here, "models", "muril_emotion_model.pth"),
        os.path.join(_here, "muril_emotion_model.pth"),
    ]:
        if path and os.path.exists(path):
            try:
                import torch
                s = torch.load(path, map_location=_device)
                if isinstance(s, dict) and "state_dict" in s:
                    s = s["state_dict"]
                _muril_model.load_state_dict(s, strict=False)
                _head_loaded = True
                print(f"[MuRIL] ✅ {path}")
                break
            except Exception as ex:
                print(f"[MuRIL] ⚠️ {ex}")
    if not _head_loaded:
        print("[MuRIL] ⚠️ No weights — returning Neutral.")


def detect_emotion(text: str) -> Tuple[str, float, str, str]:
    import torch
    _load_muril()
    if not _head_loaded:
        return "neutral", 0.01, EMOJI_MAP["neutral"], "MuRIL weights not found."
    enc = _tokenizer(
        text, add_special_tokens=True, max_length=128,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        p = torch.softmax(
            _muril_model(enc["input_ids"].to(_device), enc["attention_mask"].to(_device)),
            dim=1
        )[0]
        c, i = torch.max(p, dim=0)
    label = EMOTION_LABELS[int(i.item())]
    return label, float(c.item()), EMOJI_MAP.get(label, "🙂"), ""


def detect_sentiment(em: str) -> Tuple[float, float, str]:
    pol, sub = EMOTION_TO_SENTIMENT.get(em, (0.0, 0.2))
    return pol, sub, ("Positive" if pol > 0.15 else "Negative" if pol < -0.15 else "Neutral")


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
    tl = (text or "").lower()
    return any(w in tl for w in CRISIS_WORDS)


def groq_chat(user_id: str, user_message: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY missing. Add it to backend/.env and restart."

    crisis = _is_crisis(user_message)

    # 1. Load user context from DB
    ctx = db_get_user(user_id)

    # 2. Extract new context from this message and update ctx
    ctx = extract_context_from_message(user_message, ctx)

    # 3. Save updated context back to DB
    db_upsert_user(user_id, ctx)

    # 4. Save the user message to DB
    db_add_message(user_id, "user", user_message)

    # 5. Load conversation history from DB
    history = db_get_messages(user_id, limit=MAX_TURNS)

    # 6. Build system prompt with current context
    system_prompt = build_system_prompt(ctx, crisis=crisis)
    messages = [{"role": "system", "content": system_prompt}] + history

    # 7. Call Groq
    try:
        r = get_groq().chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            temperature=0.85, max_tokens=260, top_p=0.95
        )
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] {e}")
        reply = "I got a little glitchy — could you say that again?"

    # 8. Save assistant reply to DB
    db_add_message(user_id, "assistant", reply)

    return reply


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MyHaven Backend", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("[Startup] ✅ MyHaven Backend v6.0 ready")


# ── Request Models ────────────────────────────────────────────────────────────

class UserReq(BaseModel):
    user_id: str = Field(..., min_length=1)


class ChatReq(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/start")
def start(req: UserReq):
    """
    Begin a new session. 
    - If user has been here before: loads their profile, sends a personalized greeting.
    - If new user: creates profile, sends default greeting.
    Note: does NOT clear chat history — that's intentional for memory persistence.
    """
    ctx = db_get_user(req.user_id)
    name = ctx.get("name")
    topics = ctx.get("topics", [])
    mood_history = ctx.get("mood_history", [])

    # Build a personalized greeting if we know the user
    if name:
        if mood_history:
            last_mood = mood_history[-1]
            greeting = f"Hey {name}! 💛 Welcome back. Last time you seemed {last_mood} — how are you feeling today?"
        elif topics:
            last_topic = topics[-1]
            greeting = f"Hey {name}! 💛 Good to have you back. We were talking about {last_topic} last time — what's on your mind today?"
        else:
            greeting = f"Hey {name}! 💛 Good to see you again. What's been going on?"
    else:
        greeting = "Hey! I'm Haven 💛 What's been weighing on your mind lately — college stuff, relationships, family, or something else?"

    # Save greeting to DB
    db_upsert_user(req.user_id, ctx)
    db_add_message(req.user_id, "assistant", greeting)

    return {"reply": greeting, "returning_user": bool(name)}


@app.post("/chat")
def chat(req: ChatReq):
    """Send a message, get a reply with emotion + sentiment analysis."""
    reply = groq_chat(req.user_id, req.message)
    el, ec, ee, en = detect_emotion(req.message)
    pol, sub, sl = detect_sentiment(el)
    ctx = db_get_user(req.user_id)

    return {
        "reply": reply,
        "emotion": {
            "label": el.title(),
            "confidence": ec,
            "emoji": ee,
            "note": en
        },
        "sentiment": {
            "label": sl,
            "polarity": round(pol, 3),
            "subjectivity": round(sub, 3),
            "note": "" if _head_loaded else "Derived from emotion mapping (no trained weights yet)."
        },
        "context": {
            "name": ctx.get("name"),
            "language": ctx.get("language", "english"),
            "topics": ctx.get("topics", []),
            "mood_history": ctx.get("mood_history", [])
        },
        "meta": {
            "groq_model": GROQ_MODEL,
            "muril_loaded": _head_loaded
        },
    }


@app.post("/reset")
def reset(req: UserReq):
    """Clear conversation history but KEEP user profile (name, topics, mood)."""
    db_clear_user(req.user_id)
    return {"ok": True, "message": "Conversation cleared. User profile kept."}


@app.post("/full_reset")
def full_reset(req: UserReq):
    """Wipe EVERYTHING for this user — messages AND profile."""
    db_full_reset_user(req.user_id)
    return {"ok": True, "message": "All data deleted for this user."}


@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    """Get recent chat history for a user (useful for debugging)."""
    messages = db_get_messages(user_id, limit=limit)
    ctx = db_get_user(user_id)
    return {
        "user_id": user_id,
        "profile": ctx,
        "messages": messages,
        "count": len(messages)
    }


@app.get("/health")
def health():
    """Check backend status."""
    return {
        "ok": True,
        "time": time.time(),
        "groq_model": GROQ_MODEL,
        "muril_loaded": _head_loaded,
        "api_key_set": bool(GROQ_API_KEY),
        "db_path": DB_PATH,
        "version": "6.0"
    }