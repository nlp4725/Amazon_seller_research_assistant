import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.model_loader import get_embedder
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

CATEGORIES = [
    "Appliances", "Arts, Crafts & Sewing", "Automotive", "Baby Products",
    "Beauty & Personal Care", "Cell Phones & Accessories",
    "Clothing, Shoes & Jewelry", "Collectibles & Fine Art", "Electronics",
    "Grocery & Gourmet Food", "Health & Household", "Home & Kitchen",
    "Industrial & Scientific", "Musical Instruments", "Office Products",
    "Patio, Lawn & Garden", "Pet Supplies", "Sports & Outdoors",
    "Tools & Home Improvement", "Toys & Games", "Video Games",
]

_df: pd.DataFrame | None = None
_emb: np.ndarray | None = None
_chroma_col = None


def _load():
    global _df, _emb
    if _df is None:
        _df = pd.read_parquet("data/product_launch_cleaned_61635.parquet")
        _emb = np.load("data/title_embedding_may9th.npy")
    return _df, _emb


def _load_model():
    return get_embedder()


def _load_chroma():
    global _chroma_col
    if _chroma_col is None:
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        _chroma_col = chroma_client.get_collection("title_embedding_db")
    return _chroma_col


def _get_product_subset(
    df: pd.DataFrame,
    emb: np.ndarray,
    category: str,
    concept: str | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (sub_df, sub_emb) for the category, optionally narrowed by concept similarity."""
    cat_mask = df["cat"] == category
    cat_idx = np.where(cat_mask.values)[0]

    if concept:
        col = _load_chroma()
        model = _load_model()

        n_cat = int(cat_mask.sum())
        q_emb = model.encode([concept]).tolist()
        results = col.query(
            query_embeddings=q_emb,
            n_results=n_cat,
            where={"cat": {"$eq": category}},
            include=["metadatas", "distances"],
        )

        top_n = min(200, max(50, n_cat // 4))
        top_asins = [m["asin"] for m, _ in zip(results["metadatas"][0], results["distances"][0])][:top_n]

        asin_to_row = pd.Series(df.index.tolist(), index=df["asin"].tolist())
        orig_indices = asin_to_row[top_asins].dropna().astype(int).values
    else:
        orig_indices = cat_idx

    sub_df = df.iloc[orig_indices].reset_index(drop=True)
    sub_emb = emb[orig_indices]
    return sub_df, sub_emb


def _get_recent_launches(sub_df: pd.DataFrame, cutoff: str = "2025-05-01", n: int = 10) -> list[dict]:
    """Return the n most recent product launches on or after cutoff."""
    recent = sub_df[sub_df["launch_year_month"] >= pd.Timestamp(cutoff)].sort_values(
        "launch_year_month", ascending=False
    )
    return [
        {
            "title": row["title"],
            "price": round(float(row["price"]), 2),
            "month": row["launch_year_month"].strftime("%Y-%m"),
        }
        for _, row in recent.head(n).iterrows()
    ]


def _get_theme_trend(sub_df: pd.DataFrame, sub_emb: np.ndarray, n_clusters: int) -> list[dict]:
    """Cluster products across all years and return per-cluster year-over-year counts."""
    k = min(n_clusters, max(2, len(sub_df) // 15))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(sub_emb)

    sub_df = sub_df.copy()
    sub_df["cluster"] = labels

    trend = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        cluster_rows = sub_df[sub_df["cluster"] == i]

        dists = np.linalg.norm(sub_emb[idx] - km.cluster_centers_[i], axis=1)
        closest = idx[np.argsort(dists)[:4]]

        by_year = cluster_rows.groupby("launch_year").size().to_dict()
        trend.append({
            "size": int(len(cluster_rows)),
            "representative_titles": sub_df.iloc[closest]["title"].tolist(),
            "by_year": {str(y): int(c) for y, c in sorted(by_year.items())},
            "avg_price": round(float(cluster_rows["price"].mean()), 2),
        })

    trend.sort(key=lambda c: c["size"], reverse=True)
    return trend


def _get_top_sellers(sub_df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return the top n sellers by launch count with their most recent product titles."""
    known = sub_df[sub_df["seller"] != "-1"]
    top_ids = known.groupby("seller").size().nlargest(n).index.tolist()

    sellers = []
    for seller in top_ids:
        s_rows = known[known["seller"] == seller].sort_values("launch_year_month", ascending=False)
        sellers.append({
            "seller_id": seller,
            "total_launches": int(len(s_rows)),
            "avg_price": round(float(s_rows["price"].mean()), 2),
            "recent_titles": s_rows["title"].head(5).tolist(),
        })
    return sellers


def niche_report(category: str, concept: str | None = None, n_clusters: int = 6) -> dict:
    df, emb = _load()

    cat_mask = df["cat"] == category
    if not cat_mask.any():
        return {"error": f"No products found for '{category}'."}

    sub_df, sub_emb = _get_product_subset(df, emb, category, concept)

    if len(sub_df) == 0:
        return {"error": "No matching products found."}

    return {
        "concept": concept,
        "category": category,
        "total_products_analyzed": int(len(sub_df)),
        "note": "2026 data is partial (through May only) — counts will be lower",
        "recent_launches": _get_recent_launches(sub_df),
        "trend_by_theme": _get_theme_trend(sub_df, sub_emb, n_clusters),
        "top_sellers": _get_top_sellers(sub_df),
    }


# JSON schema — tells Claude the function exists, what args to pass, and when to use it
TOOLS = [
    {
        "name": "niche_report",
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

BOTTOM LINE:
- 2–3 sentences max: is this niche worth entering, what angle looks underserved, \
  and what price point has the least competition.

Today's date is 2026-05-13. "Last year" = 2025, "this year" = 2026.
Predefined categories: {", ".join(CATEGORIES)}"""


def run_chat(messages: list[dict]) -> str:
    # STEP 3: send system prompt + tool schema + conversation history to Anthropic
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM,
        tools=TOOLS,
        messages=messages,
    )

    # STEP 4: if Claude wants to call niche_report, run it locally then send results back
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "niche_report":
                inp = block.input
                # STEP 5: run the full analysis locally (parquet + ChromaDB + embeddings + KMeans)
                result = niche_report(
                    category=inp["category"],
                    concept=inp.get("concept"),
                    n_clusters=inp.get("n_clusters", 6),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # STEP 6: send results back to Anthropic so Claude can write the narrative
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        # STEP 7: Anthropic returns final text (stop_reason="end_turn")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

    # STEP 8: extract plain text and return to app.py
    return "\n".join(b.text for b in response.content if hasattr(b, "text"))
