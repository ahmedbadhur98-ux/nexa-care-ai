"""Nexa Care AI - Streamlit Web Interface.

A modern, responsive web application for interacting with the Nexa Care AI
Car Service Advisor chatbot. Reuses the core business context and OpenAI
integration from `chatbot.py`.
"""

import os
import base64
from pathlib import Path
import streamlit as st
from openai import OpenAI

from chatbot import (
    CONTEXT_PATH,
    ENV_PATH,
    CarServiceAdvisorChatbot,
    build_system_instructions,
    load_car_service_context,
    load_environment,
)

# Page configuration
st.set_page_config(
    page_title="Nexa Care AI - Service Advisor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a minimalist dark interface
st.markdown(
    """
    <style>
    /* Tokens from assets/DESIGN-resend.md; compact spacing for the chat UI. */
    .stApp {
        --canvas: #000000;
        --ink: #fcfdff;
        --body: rgba(252,253,255,0.86);
        --mute: #a1a4a5;
        --ash: #888e90;
        --surface-card: #0a0a0c;
        --surface-elevated: #101012;
        --surface-deep: #06060a;
        --hairline: rgba(255,255,255,0.06);
        --hairline-strong: rgba(255,255,255,0.14);
        --radius-control: 8px;
        --radius-card: 12px;
        color-scheme: dark;
        background: var(--canvas);
        color: var(--ink);
        font-family: "Inter", "Segoe UI", sans-serif;
        font-size: 16px;
        line-height: 1.5;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
        background: var(--canvas);
    }
    [data-testid="stSidebar"] { border-right: 1px solid var(--hairline); }
    .stApp [data-testid="stMarkdownContainer"] { color: var(--body); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stWidgetLabel"] { color: var(--ink); }
    .stApp [data-testid="stSidebar"] h3 {
        font-size: 32px;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: -0.8px;
    }
    .stApp [data-testid="stSidebar"] h4 {
        font-size: 24px;
        font-weight: 500;
        line-height: 1.3;
        letter-spacing: -0.3px;
    }
    .stApp [data-testid="stCaptionContainer"] {
        color: var(--ash);
        font-size: 12px;
        line-height: 1.5;
    }
    .stApp [data-testid="stSpinner"],
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--mute);
    }
    .stApp hr { border-color: var(--hairline); margin: 24px 0; }
    .chat-cover {
        position: relative;
        width: 100%;
        height: clamp(160px, 20vw, 240px);
        margin: 0 0 24px;
        overflow: hidden;
        border: 1px solid var(--hairline-strong);
        border-radius: var(--radius-card);
        background: var(--canvas);
    }
    .chat-cover img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
    }
    .cover-label {
        position: absolute;
        right: 16px;
        bottom: 16px;
        background: var(--canvas);
        color: var(--mute);
        border: 1px solid var(--hairline-strong);
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 500;
        line-height: 1.5;
        letter-spacing: 0.35px;
    }
    .main-title {
        position: relative;
        font-size: clamp(36px, 4vw, 52px);
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.8px;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    .subtitle {
        color: var(--mute);
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 24px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--surface-elevated);
        color: var(--body);
        border: 1px solid var(--hairline);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .suggestion-title {
        font-size: 18px;
        font-weight: 500;
        line-height: 1.43;
        color: var(--mute);
        margin-bottom: 8px;
    }
    .service-card {
        background: var(--surface-card);
        border: 1px solid var(--hairline-strong);
        border-radius: var(--radius-card);
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        font-size: 14px;
        line-height: 1.43;
    }
    .service-name { font-weight: 400; color: var(--ink); }
    .service-price {
        font-weight: 500;
        color: var(--body);
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }
    .stApp [data-testid="stButton"] button {
        background: var(--canvas);
        color: var(--ink);
        border: 1px solid var(--hairline-strong);
        border-radius: var(--radius-control);
        min-height: 36px;
        padding: 7px 15px;
        font-family: "Inter", "Segoe UI", sans-serif;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.43;
        box-shadow: none;
        transition: background-color 150ms ease, border-color 150ms ease;
    }
    .stApp [data-testid="stButton"] button p { color: inherit; font-size: inherit; }
    .stApp [data-testid="stButton"] button:hover:not(:disabled) {
        background: var(--surface-elevated);
        border-color: var(--mute);
        color: var(--ink);
    }
    .stApp [data-testid="stButton"] button:active:not(:disabled) {
        background: var(--surface-card);
        border-color: var(--ink);
    }
    .stApp [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: var(--surface-elevated);
    }
    .stApp [data-testid="stButton"] button:disabled {
        color: var(--mute);
        cursor: not-allowed;
    }
    .stApp button:focus-visible, .stApp a:focus-visible {
        outline: 2px solid var(--ink);
        outline-offset: 2px;
        box-shadow: none;
    }
    .stApp [data-testid="stChatMessage"] {
        background: var(--canvas);
        color: var(--body);
        border-radius: var(--radius-card);
        padding: 16px;
    }
    .stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--surface-elevated);
        border: 1px solid var(--hairline);
    }
    .stApp [data-testid="stChatMessageAvatarUser"],
    .stApp [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--surface-card);
        color: var(--mute);
        border: 1px solid var(--hairline-strong);
        border-radius: 9999px;
    }
    /* Streamlit's inner wrapper owns the native light background and focus border. */
    .stApp [data-testid="stChatInput"] {
        background: var(--surface-card);
        border: 1px solid var(--hairline-strong);
        border-radius: var(--radius-control);
        box-shadow: none;
    }
    .stApp [data-testid="stChatInput"] > div {
        background: var(--surface-card);
        border: 0;
        border-radius: var(--radius-control);
        padding: 10px 14px;
        box-shadow: none;
    }
    .stApp [data-testid="stChatInput"] [data-baseweb="textarea"],
    .stApp [data-testid="stChatInput"] [data-baseweb="textarea"] > div {
        background: var(--surface-card);
        border: 0;
        box-shadow: none;
    }
    .stApp [data-testid="stChatInput"]:focus-within {
        border-color: var(--ink);
        outline: 1px solid var(--ink);
        outline-offset: 0;
    }
    .stApp [data-testid="stChatInput"] textarea {
        background: var(--surface-card);
        color: var(--ink);
        caret-color: var(--ink);
        font-family: "Inter", "Segoe UI", sans-serif;
        font-size: 14px;
        line-height: 1.43;
        box-shadow: none;
    }
    .stApp [data-testid="stChatInput"] textarea::placeholder { color: var(--mute); }
    .stApp [data-testid="stChatInput"] button,
    .stApp [data-testid="stHeader"] button,
    .stApp [data-testid="stSidebar"] button[kind="headerNoPadding"] {
        background: var(--surface-elevated);
        color: var(--ink);
        border-radius: var(--radius-control);
        box-shadow: none;
    }
    .stApp [data-testid="stChatInput"] button:hover:not(:disabled) {
        background: var(--surface-elevated);
        color: var(--ink);
    }
    .stApp [data-testid="stChatInput"] button:disabled { color: var(--mute); }
    .stApp [data-testid="stAlert"] {
        background: var(--surface-card);
        color: var(--ink);
        border: 1px solid var(--hairline-strong);
        border-radius: var(--radius-card);
    }
    .stApp a { color: #3b9eff; text-decoration: underline; }
    .stApp code, .stApp pre {
        background: var(--surface-deep);
        color: var(--body);
        font-family: "Geist Mono", "Consolas", monospace;
        font-size: 13px;
        line-height: 1.6;
        font-variant-ligatures: none;
    }
    @media (min-width: 768px) and (max-width: 1023px) {
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .stApp [data-testid="stColumn"] { flex: 1 1 calc(50% - 16px); min-width: 0; }
    }
    @media (max-width: 767px) {
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .stApp [data-testid="stColumn"] { flex: 1 1 100%; width: 100%; min-width: 0; }
        .stApp [data-testid="stButton"] button { min-height: 44px; padding: 12px 16px; }
        .stApp [data-testid="stChatInput"] > div { min-height: 48px; }
        .chat-cover { height: 160px; margin-bottom: 16px; }
        .cover-label { right: 12px; bottom: 12px; }
    }
    .online-indicator {
        width: 8px;
        height: 8px;
        flex: 0 0 8px;
        border-radius: 9999px;
        background: #11ff99;
        animation: online-pulse 2.8s ease-in-out infinite;
    }
    @keyframes online-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }
    .service-label { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .service-icon {
        display: inline-flex;
        flex: 0 0 18px;
        width: 18px;
        height: 18px;
        color: var(--mute);
    }
    .service-icon svg { width: 100%; height: 100%; }
    .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
        width: 40px;
        height: 40px;
        border: 1px solid var(--ink);
        border-radius: 12px;
        background: var(--ink);
        color: var(--canvas);
        transition: background-color 150ms ease, transform 150ms ease;
    }
    .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: #f1f7fe;
        color: var(--canvas);
        transform: translateY(-1px);
    }
    .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:active:not(:disabled) {
        transform: translateY(0);
    }
    .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:disabled {
        background: var(--surface-elevated);
        border-color: var(--hairline-strong);
        color: var(--mute);
    }
    .headlight-overlay {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        animation: headlight-illuminate 5s ease-in-out infinite;
    }
    @keyframes headlight-illuminate {
        0%, 100% { opacity: 0.2; }
        50% { opacity: 0.85; }
    }
    @media (max-width: 767px) {
        .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
            width: 44px;
            height: 44px;
        }
        .main-title { font-size: 36px; }
    }
    /* A faint moving highlight stays on the input edge, clear of its text. */
    .stApp [data-testid="stChatInput"]::after {
        content: "";
        position: absolute;
        top: -1px;
        left: 4%;
        width: 12%;
        height: 1px;
        border-radius: 9999px;
        background: var(--ink);
        opacity: 0;
        pointer-events: none;
        animation: input-shimmer 10s ease-in-out infinite;
    }
    @keyframes input-shimmer {
        0%, 15% { opacity: 0; transform: translateX(0); }
        30% { opacity: 0.25; }
        65% { opacity: 0.25; }
        80%, 100% { opacity: 0; transform: translateX(650%); }
    }
    @media (prefers-reduced-motion: reduce) {
        .stApp [data-testid="stButton"] button,
        .stApp [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
            transition: none;
            transform: none;
        }
        .online-indicator, .headlight-overlay { animation: none; }
        .headlight-overlay { opacity: 0.45; }
        .stApp [data-testid="stChatInput"]::after { animation: none; opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def initialize_advisor():
    """Load context and initialize the OpenAI client and chatbot instance.

    Cached across reruns so the client is created only once.
    """
    if ENV_PATH.is_file():
        # Preserve the existing local development configuration.
        load_environment(ENV_PATH)
        api_key = os.environ["OPENAI_API_KEY"]
    else:
        # Community Cloud provides credentials through its private Secrets settings.
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            try:
                api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
            except (FileNotFoundError, KeyError):
                api_key = ""
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in Streamlit Community Cloud Secrets.")
    context = load_car_service_context(CONTEXT_PATH)
    system_instructions = build_system_instructions(context)
    client = OpenAI(api_key=api_key)
    return context, system_instructions, client


# Initialize backend resources
try:
    context, system_instructions, client = initialize_advisor()
except Exception as e:
    st.error(
        f"Failed to initialize advisor backend: {e}. "
        "Please check your local `.env` or cloud Secrets and `car_service_context.json`."
    )
    st.stop()

# Initialize Streamlit Session State
if "chatbot" not in st.session_state:
    st.session_state.chatbot = CarServiceAdvisorChatbot(
        client=client,
        instructions=system_instructions,
    )

WELCOME_MESSAGE = "Hi! I'm your AI car service advisor. How can I help you today?"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]

# Small inline vector icons keep service styling local and dependency-free.
SERVICE_ICON_PATHS = {
    "general service": '<path d="M14 6a5 5 0 0 0-6 6l-5 5a2 2 0 0 0 3 3l5-5a5 5 0 0 0 6-6l-3 3-3-3 3-3Z"/>',
    "brake inspection": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 4v2m8 6h-2m-6 8v-2m-8-6h2"/>',
    "ac service": '<path d="M12 3v18M4.2 7.5l15.6 9M4.2 16.5l15.6-9M9 5l3 3 3-3M9 19l3-3 3 3M4 11l4-1-1-4M20 13l-4 1 1 4M7 18l1-4-4-1M17 6l-1 4 4 1"/>',
}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### Nexa Care AI")
    st.markdown(
        f"""
        <div class="status-badge">
            <span class="online-indicator" aria-hidden="true"></span>
            Advisor Online ({context.get('working_hours', '9 AM to 7 PM')})
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Location:** {context.get('location', 'Chennai')}")
    st.markdown(f"**Operating Hours:** {context.get('working_hours', '9 AM to 7 PM')}")

    st.divider()
    st.markdown("#### Available Services & Pricing")
    for s in context.get("services", []):
        icon_paths = SERVICE_ICON_PATHS.get(
            str(s.get("name", "")).lower(), SERVICE_ICON_PATHS["general service"]
        )
        service_icon = (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{icon_paths}</svg>'
        )
        st.markdown(
            f"""
            <div class="service-card">
                <span class="service-label"><span class="service-icon">{service_icon}</span><span class="service-name">{s.get('name')}</span></span>
                <span class="service-price">₹{s.get('price'):,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Supported Brands")
    cars = context.get("supported_cars", [])
    st.write(" • ".join(cars))

    st.divider()
    if st.button("Clear Conversation", use_container_width=True, type="secondary"):
        st.session_state.chatbot.reset_conversation()
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
        st.rerun()

    st.caption("Nexa Care AI provides diagnosis advice and pricing. Bookings are handled in-center or by phone.")


# ----------------- MAIN CHAT UI -----------------
# Compact local cover; the native sidebar, conversation, and fixed input stay in place.
cover_path = Path(__file__).resolve().parent / "assets" / "hero_car.jpg"
if cover_path.is_file():
    cover_data = base64.b64encode(cover_path.read_bytes()).decode("ascii")
    headlight_paths = (
        'M278 382 L266 387 L264 405 L278 423 L302 425 '
        'M922 382 L934 387 L936 405 L922 423 L898 425 '
    )
    headlight_overlay = (
        '<svg class="headlight-overlay" viewBox="0 0 1200 675" '
        'preserveAspectRatio="xMidYMid slice" aria-hidden="true">'
        '<defs><filter id="nexa-headlight-glow" x="-100%" y="-100%" width="300%" height="300%">'
        '<feGaussianBlur stdDeviation="7"/></filter></defs>'
        f'<path d="{headlight_paths}" fill="none" stroke="#fcfdff" stroke-width="9" '
        'stroke-linecap="round" stroke-linejoin="round" filter="url(#nexa-headlight-glow)"/>'
        '</svg>'
    )
    st.markdown(
        f'<div class="chat-cover"><img src="data:image/jpeg;base64,{cover_data}" '
        'alt="Front view of a car against a black background">'
        f'{headlight_overlay}'
        '<span class="cover-label">NEXA CARE AI / SERVICE DESK</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="main-title">Nexa Care AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions about vehicle maintenance, troubleshooting, and service pricing.</div>',
    unsafe_allow_html=True,
)

# Suggested Questions Section
with st.container():
    st.markdown(
        '<div class="suggestion-title">Quick Suggested Questions:</div>',
        unsafe_allow_html=True,
    )
    suggested_cols = st.columns(4)

    prompt_to_send = None

    if suggested_cols[0].button("How much is general service?", use_container_width=True):
        prompt_to_send = "How much is general service?"
    if suggested_cols[1].button("My brakes are making noise.", use_container_width=True):
        prompt_to_send = "My brakes are making noise."
    if suggested_cols[2].button("Do you service Hyundai cars?", use_container_width=True):
        prompt_to_send = "Do you service Hyundai cars?"
    if suggested_cols[3].button("Can I book a service?", use_container_width=True):
        prompt_to_send = "Can I book a service?"

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_input = st.chat_input("Type your question about your car or our services...")
if user_input:
    prompt_to_send = user_input

# Handle prompt processing
if prompt_to_send:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    # Generate assistant reply with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Advisor is checking automotive details..."):
            try:
                reply = st.session_state.chatbot.ask(prompt_to_send)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception:
                friendly_error = (
                    "I'm sorry, I'm having trouble connecting to our advisor service at the moment. "
                    "Please check your internet connection or verify your API settings, and try again."
                )
                st.error(friendly_error)
                st.session_state.messages.append({"role": "assistant", "content": friendly_error})

    # Rerun to cleanly update session state and reset button states
    st.rerun()
