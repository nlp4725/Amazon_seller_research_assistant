# Product Requirements — Amazon Keyword Explorer

A web tool that helps Amazon third-party sellers identify high-performing keywords and evaluate category opportunities before launching a new product.

**Target user:** Third-party seller planning a standalone product launch ($15–$100 price range)

---

## What makes this different

Most keyword tools (Google Trends, Jungle Scout, Helium 10) look at what **customers searched for**. That tells you demand — but not what actually works for sellers like you.

This tool uses a different strategy: **stalking sellers**. Instead of search trends, we look at the product titles of Amazon sellers who launched products similar to yours and got traction. The keywords they chose — and that correlated with faster review accumulation — are what we surface to you.

The two approaches are complementary — use search-based tools to confirm there's demand, use this tool to figure out how to position your listing once you decide to launch.

---

## A note on the data

This tool analyzes **~70,000 standalone Amazon listings** — products launched by third-party sellers from Jan,2024-now, priced between $15–$100, with no competing offers at launch.But many of them get merged to other listings, which makes tracking velocity hard. For performance stats, we only use ~40,000 products.

**What this means for you:**

- **Launch volume trends** (e.g. "how many new pet supply products launched last quarter") are based on the full dataset of ~70,000 products.
- **Performance stats and keyword recommendations** (e.g. review velocity, top keywords) are based only on the ~40,000 products that remained standalone listings long enough to accumulate reviews. Products that were later absorbed into an existing brand's listing are excluded from performance analysis — not because they failed, but because their data became unreliable once merged.

**This tool is most reliable if you are:**
- A third-party seller launching a new standalone product
- Selling in the $15–$100 price range
- Entering a category without an existing dominant listing to attach to

If your situation differs significantly (e.g. launching as a variant of an existing product, selling at a very high or low price point), treat the recommendations as directional rather than precise.

---

## Features (MVP)

### 1. Agentic RAG Chat ⭐ must-have
**Who it's for:** A seller researching keywords and market position before launch  
**What it does:** A multi-turn chat where the agent reasons over the product listing corpus to answer seller questions. The agent decides which tools to call — and in what order — based on the question.

**Example questions it can answer:**
- "I'm launching a silicone kitchen spatula around $25 — what keywords should I put in my title?"
- "What keywords do successful pet supply sellers use in the $20–$30 range?"
- "Is the home & kitchen category getting more crowded?"

**Agent tools:**

| Tool | What it does |
|---|---|
| `similar_products(description, category, top_k)` | Finds listings semantically similar to the seller's product using embeddings |
| `keyword_velocity(keywords, category)` | Returns median review velocity for products containing each keyword vs category baseline |
| `category_stats(category)` | Returns competition level, avg velocity, sample size for a category |
| `launch_trend(category, keyword, date_range)` | Returns launch volume over time for a category or keyword |

**How it works:**
- Seller describes their product in plain language
- Agent calls tools in sequence (e.g. find similar products → extract their keywords → score keywords by velocity)
- Agent synthesizes a response: ranked keywords with velocity scores, example titles, and a plain-language interpretation

**Notes:**
- Keywords scored relative to category baseline, not globally
- Brand names flagged separately — they score high but aren't actionable for a new seller
- Sample size shown for every stat so seller knows confidence level
- Performance stats draw from the ~40k standalone corpus; trend data from full ~70k

**Out of scope:** Keyword search volume, PPC data, backend keyword fields, success prediction

---

### 2. Launch Trend Tab ⭐ must-have
**Who it's for:** A seller researching market timing — is this category growing or saturating?  
**What it does:** Shows how many new products launched per month in a given category, and which keywords are trending up or down in titles  
**Inputs:** Category, optional keyword filter, optional date range  
**Outputs:**
- Line/bar chart of launch volume over time
- Top trending keywords (appearing more frequently in recent launches vs earlier)
- Trend label per keyword: rising / stable / declining

**Notes:**
- Based on full ~70k dataset for volume accuracy
- Clearly labeled: "Based on all product types, not filtered to standalone listings"

**Out of scope:** Predicting future launch volume

---

## Out of scope (for now)
- Success prediction (binary pass/fail)
- Category overview and subcategory finder (v2)
- PPC / advertising data
- Inventory or pricing recommendations
- Non-US Amazon marketplaces
