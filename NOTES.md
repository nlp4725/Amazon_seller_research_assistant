# Project Notes — Amazon Seller Research Assistant

---

## 1. Architecture & Program Flow

### Files and their roles

| File | Role |
|---|---|
| `app.py` | Streamlit UI — tabs, chat input, renders replies |
| `chat_engine.py` | All logic — data loading, analysis, Claude API calls |
| `product_launch_cleaned_61635.parquet` | 61,635 product launches (2024–2026) with title, price, seller, cat, launch date |
| `title_embedding_may9th.npy` | 384-dim sentence embeddings for every product title, row-aligned with the parquet |
| `chroma_db/` | Vector database — same embeddings + metadata (cat, seller, price, asin) for fast cosine search |

### Full step-by-step flow

```
STEP 1 — app.py line 162
  st.chat_input() captures user message

STEP 2 — app.py line 163–169
  Append to session_state.messages
  Call run_chat(messages)

STEP 3 — chat_engine.py line 226
  client.messages.create() — first Anthropic API call
  Sends: system prompt + tool schema + full conversation history

  ↓ Anthropic reads all three and Claude decides:
    - which tool to call (only one: niche_report)
    - what args to fill in (category, concept, n_clusters)

STEP 4 — chat_engine.py line 235
  response.stop_reason == "tool_use"
  Claude returns a ToolUseBlock with filled-in args:
    e.g. {category: "Pet Supplies", concept: "dog grooming"}

STEP 5 — chat_engine.py line 241
  niche_report() runs locally on your machine:

  5A. If concept given → ChromaDB semantic search
        _load_chroma() — opens chroma_db/
        _load_model()  — loads SentenceTransformer("all-MiniLM-L6-v2")
        Embed concept → 384-dim query vector
        col.query(filter by cat, rank by cosine similarity)
        Take top ~170 most relevant products

  5B. If no concept → use all products in category

  5C. Recent launches
        Filter to launch_year_month >= 2025-05-01
        Return top 10 sorted by most recent

  5D. Year-over-year trend
        KMeans on ALL years combined → stable cluster IDs
        Count products per cluster per year (2024, 2025, 2026)
        Return representative titles per cluster

  5E. Top sellers
        Group by seller, count launches, take top 5
        Return their recent product titles

STEP 6 — chat_engine.py line 249
  Send tool result back to Anthropic (second API call)
  Messages now: [user msg] + [Claude's tool call] + [tool result JSON]

STEP 7 — chat_engine.py line 255
  Claude reads cluster data → writes narrative report
  stop_reason = "end_turn"

STEP 8 — chat_engine.py line 262
  Extract text from TextBlock, return to app.py

STEP 9 — app.py line 170
  st.markdown(reply) — renders in chat bubble
```

### Key data relationships

- `parquet` row index == `.npy` row index — they are perfectly aligned
- `chroma_db` stores the same embeddings but with metadata attached (cat, asin, price, seller)
- ChromaDB is used for fast concept search (cosine similarity + category filter)
- `.npy` is used for KMeans clustering after narrowing the product set

---

## 2. Prompt Engineering — Key Points

### What the system prompt does (`chat_engine.py` line 205)

The system prompt is sent to Claude **before every conversation** and is never shown to the user. It does four things:

**a) Sets Claude's role and behavior**
```
"You are a sharp Amazon market research analyst."
```
Shapes tone — Claude writes like an analyst, not a chatbot.

**b) Tells Claude WHEN to call the tool and HOW**
```
"Call niche_report for any question about what's launching,
 trends, or seller activity."
"If concept-specific (dog grooming), pass it as concept."
"For broad category, omit concept."
```
Without this, Claude might just make up an answer from its training data instead of using your real data.

**c) Category rules — prevents hallucination**
```
"category MUST be one of the predefined list."
"If unclear, ask the user before calling the tool."
```
Without rule 1, Claude might pass "Dog Products" which doesn't exist in your parquet → empty results.
Without rule 2, Claude would guess and silently use the wrong category.

**d) Time anchor**
```
"Today is 2026-05-13. Last year = 2025, this year = 2026."
"2026 data is partial (through May)."
```
Without this, Claude doesn't know what "last year" or "recently" means and can't interpret partial 2026 counts correctly.

**e) Output instructions**
```
"Label each cluster with a short theme name."
"Describe whether it is growing, declining, or flat."
"Infer what theme each seller is focused on from their titles."
```
These shape the quality of the final narrative — without them Claude might just list raw data without interpretation.

---

## 3. Tool Description vs System Prompt — What Each Covers

### The tool description (`chat_engine.py` line 172)

The tool description is part of the JSON schema sent alongside the system prompt. It tells Claude:

| What it covers | Example |
|---|---|
| WHEN to call this tool | "Use this for any question about launches, trends, or sellers" |
| What the tool returns | "Returns recent launches, year-over-year trend, top 5 sellers" |
| How to fill in `concept` | "Pass 'dog grooming' for specific niches, omit for broad category" |
| What each argument means | Each property has its own `description` field |

**What the tool description does NOT cover:**
- It cannot enforce behavior rules (like "ask user if unclear") — that belongs in the system prompt
- It cannot give Claude output formatting instructions
- It cannot explain time context ("last year = 2025")
- It cannot prevent Claude from calling the tool when it shouldn't

### The system prompt

| What it covers | Example |
|---|---|
| Behavioral rules | "Ask user before calling if category unclear" |
| Output format | "Label each cluster, describe growth direction" |
| Time context | "Today is 2026-05-13, last year = 2025" |
| Hard constraints | "Category MUST be one of the predefined list" |
| Caveats to mention | "2026 is partial through May" |

**What the system prompt does NOT cover:**
- It cannot define the tool's arguments — that's the schema's job
- It cannot describe what data the tool returns — the schema description does that
- It is not seen by the user

### Summary: who does what

```
Tool schema description  →  teaches Claude WHAT the tool is and WHEN to use it
System prompt            →  teaches Claude HOW to behave and HOW to write the response
Both together            →  reliable, well-formatted, accurate answers
Either one alone         →  Claude either guesses wrong or formats badly
```

---

## 4. Miscellaneous Notes

- `_load()`, `_load_model()`, `_load_chroma()` all use a global variable guard (`if X is None`) so the large files are only loaded once per session, not on every message
- ChromaDB does not support `$gte`/`$lt` on string fields — year filtering is done in Python after fetching from ChromaDB
- KMeans `random_state=42` keeps clusters consistent across runs — same data always produces same clusters
- `n_results` in ChromaDB query must not exceed the number of matching documents, so we pre-count `n_cat` first
- `max_tokens=4096` in `run_chat` because niche reports are long — the old 2048 would cut off mid-response
