import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from src.chat_engine import run_chat
from src.inference import predict

st.set_page_config(page_title="Seller Assistant", layout="wide", initial_sidebar_state="collapsed")

# ── Avatar ────────────────────────────────────────────────────────────────────
@st.cache_resource
def _assistant_avatar() -> Image.Image:
    size = 80
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([1, 1, size - 1, size - 1], fill="#FF9900")
    draw.ellipse([16, 14, 50, 48], outline="white", width=4)
    draw.line([44, 43, 60, 59], fill="white", width=5)
    draw.ellipse([20, 18, 30, 28], fill=(255, 255, 255, 60))
    return img

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    * { font-family: 'Inter', sans-serif !important; }

    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 1rem !important; max-width: 1300px; margin: 0 auto; }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        background: #ebebed;
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 6px 18px;
        font-size: 14px;
        font-weight: 500;
        color: #888;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* Scrollable messages container — no visible box */
    [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatMessage"]) {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 2px 0 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid #e5e5e5 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        font-size: 15px !important;
        color: #1a1a1a !important;
    }
    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: #e5e5e5 !important;
    }
    [data-testid="stChatInput"] button {
        background: #FF9900 !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* Metric cards — dark navy */
    [data-testid="metric-container"] {
        background: #1a2035;
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8899aa !important;
        font-size: 11px !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricValue"] > div,
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }

    /* Keyword / content cards — Trending tab only */
    [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
        background: white;
        border-radius: 16px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        padding: 4px 8px;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important;
        background: white !important;
        border: 1.5px solid #c8c8c8 !important;
    }


    /* Fix popover icon rendering when Material Icons font fails to load */
    [data-testid="stPopoverBody"] { font-family: sans-serif !important; }
    .material-icons { display: none !important; }

    /* Hide sidebar entirely */
    [data-testid="stSidebar"] { display: none !important; }

    /* Hide all Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("data/keyword_stats_cleaned.parquet")
    df["month"] = pd.to_datetime(df["month"])
    return df

df = load_data()
avatar = _assistant_avatar()

st.markdown(
    "<div style='font-size:18px;font-weight:800;color:#1a1a1a;padding-bottom:16px;"
    "border-bottom:1px solid #e8e8e8;margin-bottom:16px;letter-spacing:-0.3px'>"
    "🔍 Seller Assistant</div>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_predict = st.tabs(["💬 Chat", "🚀 Predict launch"])

# ── Chat ──────────────────────────────────────────────────────────────────────
def _run_prediction(titles_raw: str) -> tuple[str, pd.DataFrame | None]:
    titles = [t.strip() for t in titles_raw.replace("\n", ",").split(",") if t.strip()]
    if not titles:
        return "I couldn't find any titles to score. Please paste them comma-separated or one per line.", None
    results = predict(titles).sort_values("y_pred_prob", ascending=False)
    promising = results[results["predicted_label"] == 1]
    n_total = len(titles)
    n_promising = len(promising)
    n_losers = n_total - n_promising
    reply = (
        f"You uploaded **{n_total} titles** and we helped you eliminate **{n_losers} losers** 😊! "
        f"We found **{n_promising} promising** title(s).\n\n"
        f"Based on our model, these promising titles have a **38% chance of succeeding**."
    )
    return reply, promising[["title"]].rename(columns={"title": "Title"}).reset_index(drop=True)


with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can help you **research a niche** or **analyze your launch**."}
        ]
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "idle"
    if "predict_results" not in st.session_state:
        st.session_state.predict_results = None
    if "show_followup" not in st.session_state:
        st.session_state.show_followup = False
    if "right_panel" not in st.session_state:
        st.session_state.right_panel = None  # {"type": "predict"|"report", "title": str, "content": str/df}
    if "pending_titles" not in st.session_state:
        st.session_state.pending_titles = None

    st.markdown("<br>", unsafe_allow_html=True)

    has_panel = st.session_state.right_panel is not None
    chat_col, result_col = st.columns([3, 2] if has_panel else [1, 0.001])

    with chat_col:
        messages_container = st.container(height=700, border=False)

        with messages_container:
            for msg in st.session_state.messages:
                av = "🧑" if msg["role"] == "user" else avatar
                with st.chat_message(msg["role"], avatar=av):
                    st.markdown(msg["content"].replace("$", r"\$"))

            # Spinner + work runs here on the loading rerun — single render, no double
            if st.session_state.chat_mode == "niche_loading":
                with st.chat_message("assistant", avatar=avatar):
                    with st.spinner("Analyzing…"):
                        reply = run_chat(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.right_panel = {
                    "type": "report",
                    "title": f"📊 Report: {st.session_state.messages[-2]['content']}",
                    "content": reply,
                }
                st.session_state.chat_mode = "idle"
                st.session_state.show_followup = True
                st.rerun()

            elif st.session_state.chat_mode == "predict_loading":
                with st.chat_message("assistant", avatar=avatar):
                    with st.spinner("Running model…"):
                        reply, results_df = _run_prediction(st.session_state.pending_titles)
                promising_titles = [] if results_df is None else results_df["Title"].tolist()
                if promising_titles:
                    panel_content = "\n".join(f"• {t}" for t in promising_titles)
                    st.session_state.right_panel = {
                        "type": "predict",
                        "title": "🟢 Promising titles",
                        "content": panel_content,
                        "titles": promising_titles,
                    }
                    st.session_state.show_followup = False
                else:
                    reply = "Unfortunately we did not find any promising products."
                    st.session_state.right_panel = None
                    st.session_state.show_followup = True
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.chat_mode = "idle"
                st.session_state.pending_titles = None
                st.rerun()

            elif len(st.session_state.messages) == 1 and st.session_state.chat_mode == "idle":
                btn_col1, btn_col2, _ = st.columns([1.6, 1.8, 4])
                if btn_col1.button("🔍 Analyze a niche", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": "Great! Please enter a niche or category you are thinking of.\n\ne.g. **dog grooming**, **exercise band**, **home & kitchen**"})
                    st.session_state.chat_mode = "niche"
                    st.rerun()
                if btn_col2.button("🚀 Launch prediction", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": "Sure! Paste your product titles — comma-separated or one per line.\n\n*Currently supported for **Home & Kitchen** only.*"})
                    st.session_state.chat_mode = "predict"
                    st.rerun()

        # Follow-up buttons after any response
        if st.session_state.show_followup and st.session_state.chat_mode == "idle":
            with st.chat_message("assistant", avatar=avatar):
                st.markdown("What would you like to do next?")
                f1, f2, _ = st.columns([1.6, 1.8, 4])
                if f1.button("🔍 Analyze a niche", key="fu_niche", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": "What would you like to do next?"})
                    st.session_state.messages.append({"role": "assistant", "content": "Great! Please enter a niche or category you are thinking of.\n\ne.g. **dog grooming**, **exercise band**, **home & kitchen**"})
                    st.session_state.chat_mode = "niche"
                    st.session_state.show_followup = False
                    st.rerun()
                if f2.button("🚀 Launch prediction", key="fu_launch", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": "What would you like to do next?"})
                    st.session_state.messages.append({"role": "assistant", "content": "Sure! Paste your product titles — comma-separated or one per line.\n\n*Currently supported for **Home & Kitchen** only.*"})
                    st.session_state.chat_mode = "predict"
                    st.session_state.show_followup = False
                    st.rerun()

        placeholder = (
            "Paste product titles (comma-separated or one per line)…"
            if st.session_state.chat_mode in ("predict", "predict_loading")
            else "Ask about any niche or category…"
        )
        if prompt := st.chat_input(placeholder):
            if st.session_state.chat_mode in ("predict", "predict_loading"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.pending_titles = prompt
                st.session_state.chat_mode = "predict_loading"
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.chat_mode = "niche_loading"
            st.rerun()


    if has_panel:
        with result_col:
            panel = st.session_state.right_panel
            st.markdown(
                f"<div style='font-weight:700;font-size:15px;color:#1a1a1a;margin-bottom:12px'>"
                f"{panel['title']}</div>",
                unsafe_allow_html=True,
            )

            panel_container = st.container(height=520, border=True)
            with panel_container:
                if panel["type"] == "predict":
                    if not panel["titles"]:
                        st.markdown("<p style='color:#888;font-size:13px'>No promising titles found.</p>", unsafe_allow_html=True)
                    else:
                        for t in panel["titles"]:
                            st.markdown(
                                f"<div style='background:#f7faf7;border:1px solid #d4edda;border-radius:8px;"
                                f"padding:8px 12px;margin-bottom:8px;font-size:13px;color:#1a1a1a'>"
                                f"🟢 {t}</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.markdown(panel["content"].replace("$", r"\$"))

            # Email buttons
            st.markdown("<br>", unsafe_allow_html=True)
            email_body = panel["content"].replace("\n", "%0A").replace("#", "").replace("*", "").replace("|", "")
            email_subject = panel["title"].replace("#", "").replace("*", "").strip()
            mailto = f"mailto:nasilipurcell@gmail.com?subject={email_subject}&body={email_body}"

            import datetime
            filename = panel["title"].replace("📊", "").replace("🟢", "").strip().replace(" ", "_").replace(":", "") + ".txt"
            download_content = f"{panel['title']}\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{panel['content']}"

            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("📧 Email me now", use_container_width=True, key="email_now"):
                    st.markdown(f"[Open email client ↗]({mailto})", unsafe_allow_html=True)
            with ec2:
                st.download_button(
                    "⬇️ Download report",
                    data=download_content,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True,
                    key="download_report",
                )

            if st.button("✕ Clear", key="clear_panel"):
                st.session_state.right_panel = None
                st.rerun()

# ── Predict launch ────────────────────────────────────────────────────────────
with tab_predict:
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Beta badge
    st.markdown(
        "<span style='background:#FFF3CD;color:#856404;font-size:12px;font-weight:700;"
        "padding:3px 10px;border-radius:20px;letter-spacing:0.05em'>⚗️ BETA</span>"
        "&nbsp;&nbsp;<span style='color:#888;font-size:13px'>Home &amp; Kitchen only for now</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**🚀 Predict launch opportunity**")
        st.markdown(
            "<p style='color:#888;font-size:13px;margin-top:-6px;margin-bottom:16px'>"
            "Enter product titles manually or upload a CSV. We'll score each title's "
            "launch potential based on real Home &amp; Kitchen market data.</p>",
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "Input method", ["Type titles", "Upload CSV"],
            horizontal=True, label_visibility="collapsed"
        )

        titles_input = []

        if input_mode == "Type titles":
            raw = st.text_area(
                "Product titles (comma-separated)",
                placeholder="e.g. Stainless Steel Air Fryer 5.8QT, Non-Stick Frying Pan Set, ...",
                height=120,
                label_visibility="collapsed",
            )
            if raw:
                titles_input = [t.strip() for t in raw.split(",") if t.strip()]
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="csv_upload_predict")
            if uploaded:
                import io
                udf = pd.read_csv(io.BytesIO(uploaded.read()))
                title_col = next((c for c in udf.columns if "title" in c.lower()), udf.columns[0])
                titles_input = udf[title_col].dropna().astype(str).tolist()
                st.caption(f"{len(titles_input)} titles loaded from column **{title_col}**")

        if titles_input:
            st.markdown(
                f"<p style='color:#888;font-size:13px'>{len(titles_input)} title(s) ready</p>",
                unsafe_allow_html=True,
            )

        col_predict, col_clear, _ = st.columns([1, 1, 5])
        predict_clicked = col_predict.button("Predict", type="primary")
        clear_clicked = col_clear.button("Clear")

        if clear_clicked:
            st.rerun()

        if predict_clicked:
            if not titles_input:
                st.warning("Please enter at least one product title.")
            else:
                with st.spinner("Running model…"):
                    results = predict(titles_input)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='background:#f0f4ff;border:1px solid #d0d9f0;border-radius:10px;"
                    "padding:10px 16px;font-size:13px;color:#444;margin-bottom:12px'>"
                    "⚠️ <b>Applying model threshold:</b> products labeled "
                    "<span style='color:#1a7a3a;font-weight:700'>Promising</span> "
                    "have a <b>38% chance of success</b> based on our training data. "
                    "Use as a signal, not a guarantee.</div>",
                    unsafe_allow_html=True,
                )

                display = (
                    results.sort_values("y_pred_prob", ascending=False)
                    [["title", "predicted_label"]]
                    .rename(columns={"title": "Title"})
                    .assign(Verdict=lambda d: d["predicted_label"].map(
                        {1: "🟢 Promising", 0: "⚪ Not flagged"}
                    ))
                    [["Title", "Verdict"]]
                )
                st.dataframe(display, use_container_width=True, hide_index=True)

