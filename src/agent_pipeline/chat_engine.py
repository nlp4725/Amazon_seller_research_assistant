"""
Chat engine: niche research agent powered by Claude and ChromaDB.

All data comes from ChromaDB (embeddings + metadata).

Flow per user message:
  1. run_chat() sends conversation history + tool schema to Claude (Anthropic API)
  2. Claude calls niche_report(category, concept, n_clusters)
  3. niche_report() runs locally:
       - _get_product_subset(): semantic search via ChromaDB (if concept given) or full category fetch
       - _get_recent_launches(): top 10 most recent products
       - _get_theme_trend(): KMeans clustering on embeddings → year-over-year theme trends
       - _get_top_sellers(): top 5 sellers by launch count
  4. JSON result sent back to Claude → Claude writes narrative report
  5. Plain text returned to app.py for rendering
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.shared.model_loader import get_embedder
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])  # single client reused across all requests

CATEGORIES = [  # valid category names in the dataset — Claude must pick exactly one
    "Appliances", "Arts, Crafts & Sewing", "Automotive", "Baby Products",
    "Beauty & Personal Care", "Cell Phones & Accessories",
    "Clothing, Shoes & Jewelry", "Collectibles & Fine Art", "Electronics",
    "Grocery & Gourmet Food", "Health & Household", "Home & Kitchen",
    "Industrial & Scientific", "Musical Instruments", "Office Products",
    "Patio, Lawn & Garden", "Pet Supplies", "Sports & Outdoors",
    "Tools & Home Improvement", "Toys & Games", "Video Games",
]

_chroma_col = None  # cached ChromaDB collection — loaded once per session


def _load_model():
    return get_embedder()  # returns cached SentenceTransformer — see shared/model_loader.py


def _load_chroma():
    global _chroma_col
    if _chroma_col is None:  # open connection only once — PersistentClient reads from disk
        chroma_client = chromadb.PersistentClient(path="data/raw/chroma_db")
        _chroma_col = chroma_client.get_collection("title_embedding_db")
    return _chroma_col


def _get_product_subset(
    category: str,
    concept: str | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Return (sub_df, sub_emb, distances) for the category, optionally narrowed by concept similarity.
    distances is None when no concept given (all category products are relevant).
    Both metadata and embeddings come from ChromaDB — no parquet or .npy needed."""
    col = _load_chroma()

    if concept:
        n_cat = col.count()                              # total products in ChromaDB
        model = _load_model()                            # SentenceTransformer singleton
        top_n = min(200, max(50, n_cat // 4))           # return at most 200, at least 50 — the floor of 50 means
                                                         # results may include loosely related products when the
                                                         # concept is very niche or rare in the dataset
        q_emb = model.encode([concept]).tolist()         # embed the concept query (e.g. "dog grooming")
        results = col.query(
            query_embeddings=q_emb,                      # search by cosine similarity to concept
            n_results=top_n,
            where={"cat": {"$eq": category}},            # filter to the right category first
            include=["metadatas", "embeddings", "distances"],  # get titles/prices/sellers + vectors + distances
        )
        metadatas = results["metadatas"][0]              # list of metadata dicts for top_n results
        emb_array = np.array(results["embeddings"][0])  # (top_n, 384) embedding matrix
        distances = np.array(results["distances"][0])   # cosine distance per result (0=identical, 1=opposite)
    else:
        results = col.get(
            where={"cat": {"$eq": category}},            # fetch all products in this category
            include=["metadatas", "embeddings"],
        )
        metadatas = results["metadatas"]
        emb_array = np.array(results["embeddings"])
        distances = None                                 # no concept → no distance to measure

    sub_df = pd.DataFrame(metadatas)                     # build DataFrame from ChromaDB metadata
    sub_df["launch_year_month"] = pd.to_datetime(sub_df["launch_year_month"])
    return sub_df, emb_array, distances


def _get_recent_launches(sub_df: pd.DataFrame, cutoff: str = "2025-05-01", n: int = 10) -> list[dict]:
    """Return the n most recent product launches on or after cutoff."""
    recent = sub_df[sub_df["launch_year_month"] >= pd.Timestamp(cutoff)].sort_values(
        "launch_year_month", ascending=False  # newest first
    )
    return [
        {
            "title": row["title"],
            "price": round(float(row["price"]), 2),
            "month": row["launch_year_month"].strftime("%Y-%m"),
        }
        for _, row in recent.head(n).iterrows()  # top n rows → sent to Claude as context
    ]


def _get_theme_trend(sub_df: pd.DataFrame, sub_emb: np.ndarray, n_clusters: int) -> list[dict]:
    """Cluster products across all years and return per-cluster year-over-year counts."""
    k = min(n_clusters, max(2, len(sub_df) // 15))  # cap clusters so each has at least ~15 products
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(sub_emb)  # assign each product to a cluster based on title embedding

    sub_df = sub_df.copy()
    sub_df["cluster"] = labels

    trend = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        cluster_rows = sub_df[sub_df["cluster"] == i]

        dists = np.linalg.norm(sub_emb[idx] - km.cluster_centers_[i], axis=1)
        closest = idx[np.argsort(dists)[:4]]  # 4 titles closest to cluster center → most representative

        by_year = cluster_rows.groupby("launch_year").size().to_dict()  # launch count per year
        trend.append({
            "size": int(len(cluster_rows)),
            "representative_titles": sub_df.iloc[closest]["title"].tolist(),
            "by_year": {str(y): int(c) for y, c in sorted(by_year.items())},
            "avg_price": round(float(cluster_rows["price"].mean()), 2),
        })

    trend.sort(key=lambda c: c["size"], reverse=True)  # largest clusters first
    return trend


def _get_top_sellers(sub_df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return the top n sellers by launch count with their most recent product titles."""
    known = sub_df[sub_df["seller"] != "-1"]  # exclude unknown sellers (buybox was empty)
    top_ids = known.groupby("seller").size().nlargest(n).index.tolist()  # top n by launch count

    sellers = []
    for seller in top_ids:
        s_rows = known[known["seller"] == seller].sort_values("launch_year_month", ascending=False)
        sellers.append({
            "seller_id": seller,
            "total_launches": int(len(s_rows)),
            "avg_price": round(float(s_rows["price"].mean()), 2),
            "recent_titles": s_rows["title"].head(5).tolist(),  # last 5 products launched
        })
    return sellers


def _get_velocity_summary(sub_df: pd.DataFrame, category: str) -> dict:
    """Compare avg review velocity of the sub-category vs the full category.
    Only includes products with velocity > 0 (known actives); -1 (unknown) excluded."""
    col = _load_chroma()

    # sub-category avg (products already filtered to concept + category)
    active_sub = sub_df[sub_df["review_velocity"] > 0]["review_velocity"]

    # full category avg — fetch all metadata for the category
    cat_results = col.get(
        where={"cat": {"$eq": category}},
        include=["metadatas"],
    )
    cat_df = pd.DataFrame(cat_results["metadatas"])
    active_cat = cat_df[cat_df["review_velocity"] > 0]["review_velocity"]

    return {
        "sub_avg_velocity": round(float(active_sub.mean()), 4) if len(active_sub) else None,
        "category_avg_velocity": round(float(active_cat.mean()), 4) if len(active_cat) else None,
        "sub_active_count": int(len(active_sub)),    # products with known positive velocity in sub
        "category_active_count": int(len(active_cat)),
        "velocity_threshold": 0.056,                 # 5 reviews at 90 days = early traction benchmark
    }


def niche_report(category: str, concept: str | None = None, n_clusters: int = 6) -> dict:
    col = _load_chroma()
    if col.count() == 0:
        return {"error": "ChromaDB is empty."}

    sub_df, sub_emb, distances = _get_product_subset(category, concept)  # 3-tuple

    if len(sub_df) == 0:
        return {"error": "No matching products found."}

    # count products closely related to the concept (cosine distance < 0.5)
    if distances is not None:
        closely_related_count = int((distances < 0.5).sum())
        cat_results = col.get(where={"cat": {"$eq": category}}, include=[])  # category total for %
        cat_total = len(cat_results["ids"])
        pct_of_category = round(closely_related_count / cat_total * 100, 1) if cat_total else None
    else:
        closely_related_count = None
        cat_total = len(sub_df)
        pct_of_category = None

    return {
        "concept": concept,
        "category": category,
        "total_products_analyzed": int(len(sub_df)),
        "closely_related_count": closely_related_count,  # products with cosine distance < 0.5 to concept
        "category_total": cat_total,
        "pct_of_category_launches": pct_of_category,
        "note": "2026 data is partial (through May only) — counts will be lower",
        "velocity_summary": _get_velocity_summary(sub_df, category),  # sub vs category avg velocity
        "recent_launches": _get_recent_launches(sub_df),              # what's new right now
        "trend_by_theme": _get_theme_trend(sub_df, sub_emb, n_clusters),  # which themes are growing
        "top_sellers": _get_top_sellers(sub_df),                      # who dominates this space
    }


# JSON schema — tells Claude the function exists, what args to pass, and when to use it
TOOLS = [
    {
        "name": "niche_report",  # must match the Python function name exactly — used in block.name check on line 266
        "description": (
            "Generate a full market research report for a niche or category on Amazon. "
            "Returns: (1) recent launches in the last 12 months, (2) year-over-year theme "
            "trend showing which product types are growing or declining, (3) top 5 sellers "
            "and what they are launching. "
            "Use this for any question about what's launching, trends, or seller activity. "
            "If the user mentions a specific niche or subcategory (e.g. 'dog grooming', "
            "'smart kitchen'), pass it as concept. For broad category questions, omit concept."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": f"Amazon product category. Must be one of: {CATEGORIES}",
                },
                "concept": {
                    "type": "string",
                    "description": (
                        "Optional. A specific niche or subcategory to focus on, e.g. "
                        "'dog grooming', 'air fryer', 'yoga mat'. Omit for broad category analysis."
                    ),
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of theme clusters for trend analysis (default 6).",
                },
            },
            "required": ["category"],
        },
    }
]

SYSTEM = f"""You are a sharp Amazon market research analyst. You have access to a \
database of 61,635 product launches (2024–2026) with semantic embeddings.

You have one tool: niche_report. Call it for any question about what's launching, \
market trends, or seller activity — whether the user asks about a broad category or \
a specific niche.

CATEGORY RULES (strictly follow these before calling the tool):
1. category MUST be one of the predefined list below — do not invent or guess a \
   category that is not on the list.
2. If the user's message does not clearly map to one of the predefined categories, \
   ask the user to specify which category they mean before calling the tool. \
   Show them the list so they can pick.

Structure your response exactly as follows:

SEARCHED: state the category and concept you used.

LAUNCH VOLUME (from closely_related_count, category_total, pct_of_category_launches):
- Lead with: "X closely related products (cosine distance < 0.5) in [concept] / [category] \
  launched over the past two years — X% of all [category] launches."
- If closely_related_count is None (broad category query, no concept), skip this line.
- If total_products_analyzed is close to 50 and closely_related_count is much lower, \
  note that the concept is very niche and results may include loosely related products.

WHAT'S LAUNCHING NOW (from recent_launches):
- Note the price range and any dominant product features or materials in the titles.
- Flag if the space looks crowded (many near-identical titles) or sparse.

THEME TRENDS (from trend_by_theme):
- Give each cluster a specific label — e.g. "Electric/Automated Tools", not just "Tools".
- Calculate % growth from 2024 → 2025 (the two complete years). Show the numbers.
- For 2026 counts: they are partial (5 months only) — multiply by 2.4 to estimate \
  the full-year pace, and note this is an estimate.
- Tag each theme with one of: RISING (>50% growth 2024→2025), DECLINING (>30% drop), \
  EMERGING (zero in 2024, appeared in 2025), STABLE (under 30% change either way).

TOP SELLERS (from top_sellers):
- For each seller infer their niche focus from their titles.
- Note if they are a specialist (one focused niche) or diversified.
- Flag if any one seller has >30% of total launches — signals a dominant incumbent.

REVIEW VELOCITY SIGNAL (from velocity_summary):
- Benchmark: 0.056 reviews/day at 90 days = ~5 reviews = early traction signal.
- Compare sub_avg_velocity vs category_avg_velocity.
  - If sub > category: "[concept] products are gaining traction faster than the \
    category average (X vs Y reviews/day at 90 days)."
  - If sub < category: "[concept] products are underperforming the category average \
    (X vs Y reviews/day at 90 days)."
  - If sub_avg_velocity or category_avg_velocity is None (no actives): note insufficient data.
- Only include products with velocity > 0 (exclude -1 unknowns).

BOTTOM LINE:
- 2–3 sentences max. Be honest about data limits: this dataset covers new launches \
  (2024–2026) only — you cannot assess the full competitive landscape or which angles \
  are underserved without knowing established products already on the market.
- Say what you CAN conclude: is launch activity growing or shrinking, are new entrants \
  gaining traction (velocity signal), and what price range new sellers are targeting.
- Do not claim a niche is "underserved" or has "least competition" — those require \
  market-wide data this dataset does not have.

Today's date is 2026-05-13. "Last year" = 2025, "this year" = 2026.
Predefined categories: {", ".join(CATEGORIES)}"""


def run_chat(messages: list[dict]) -> str:
    # STEP 1 (app.py): st.chat_input() captures user message
    # STEP 2 (app.py): append to session_state.messages, call run_chat(messages)

    # STEP 3: first Anthropic API call — send system prompt + tool schema + conversation history
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM,   # role + category rules + output format instructions
        tools=TOOLS,     # tells Claude niche_report exists and when to call it
        messages=messages,
    )

    # STEP 4: Claude returns stop_reason="tool_use" with filled-in args (category, concept)
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "niche_report":
                inp = block.input
                # STEP 5: run niche_report locally — ChromaDB search + KMeans + seller stats
                result = niche_report(
                    category=inp["category"],
                    concept=inp.get("concept"),
                    n_clusters=inp.get("n_clusters", 6),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),  # send structured JSON back to Claude
                })

        # STEP 6: second Anthropic API call — send tool result back so Claude can write the report
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        # STEP 7: Claude reads the JSON result and writes the narrative (stop_reason="end_turn")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

    # STEP 8: extract plain text and return to app.py
    # STEP 9 (app.py): st.markdown(reply) renders in chat bubble
    return "\n".join(b.text for b in response.content if hasattr(b, "text"))
