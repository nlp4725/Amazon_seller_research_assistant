import os
import requests

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

API_URL = os.environ.get("API_URL", "http://localhost:8080")

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
        padding: 10px 0 !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Right-align user messages, keep assistant messages on the left */
    [data-testid="stChatMessage"]:has([aria-label^="Chat message from user"]) {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }
    [data-testid="stChatMessage"]:has([aria-label^="Chat message from user"]) [data-testid="stChatMessageContent"] {
        flex: none !important;
        margin: 0 !important;
        text-align: right;
        max-width: 80%;
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

avatar = _assistant_avatar()

st.markdown(
    "<div style='font-size:18px;font-weight:800;color:#1a1a1a;padding-bottom:16px;"
    "border-bottom:1px solid #e8e8e8;margin-bottom:16px;letter-spacing:-0.3px'>"
    "🔍 Seller Assistant</div>",
    unsafe_allow_html=True,
)

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! Let me help you research a niche.\n\nPlease enter a niche or category — e.g. **dog grooming**, **exercise band**, **home & kitchen**."}
    ]
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "idle"
if "right_panel" not in st.session_state:
    st.session_state.right_panel = None  # {"type": "report", "title": str, "content": str/df}

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
                    resp = requests.post(
                        f"{API_URL}/api/chat",
                        json={"messages": st.session_state.messages},
                        timeout=300,
                    )
                    if resp.status_code == 429:
                        reply = "You've sent too many requests. Please wait a moment before trying again."
                    elif resp.status_code == 400:
                        reply = resp.json().get("error", "Invalid request.")
                    else:
                        resp.raise_for_status()
                        reply = resp.json()["reply"]
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.right_panel = {
                "type": "report",
                "title": f"📊 Report: {st.session_state.messages[-2]['content']}",
                "content": reply,
            }
            st.session_state.chat_mode = "idle"
            st.rerun()

    if prompt := st.chat_input("Ask about any niche or category…"):
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

