"""Nexa Care AI: the latest interface with Streamlit-owned chat sessions."""
import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from chatbot import CarServiceAdvisorChatbot, build_system_instructions, load_car_service_context

ROOT = Path(__file__).resolve().parent
WELCOME = "Hi! I'm your AI car service advisor. How can I help you today?"
st.set_page_config(page_title="Nexa Care AI", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#000;color:#fcfdff}
[data-testid="stHeader"],footer,#MainMenu{display:none}
.block-container{max-width:none;padding:0!important}
[data-testid="stVerticalBlock"]{gap:0}
iframe{display:block;border:0;width:100%}
</style>""", unsafe_allow_html=True)

def setting(name, default=""):
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default

context = load_car_service_context(ROOT / "car_service_context.json")
api_key = setting("OPENAI_API_KEY")
portfolio_url = setting("PORTFOLIO_URL", "https://ahmed-ai-portfolio-drab.vercel.app")
if portfolio_url.rstrip("/") == "https://eclipse-private-aviation-copy.ahmedbadhur98.chatgpt.site":
    portfolio_url = "https://ahmed-ai-portfolio-drab.vercel.app"
model = setting("OPENAI_MODEL", "gpt-5.6")
if "nexa_messages" not in st.session_state:
    st.session_state.nexa_messages = [{"role": "assistant", "content": WELCOME}]
if "nexa_reply" not in st.session_state:
    st.session_state.nexa_reply = None
if api_key and "nexa_bot" not in st.session_state:
    st.session_state.nexa_bot = CarServiceAdvisorChatbot(
        client=OpenAI(api_key=api_key, timeout=60, max_retries=1),
        instructions=build_system_instructions(context), model=model,
    )

nexa = components.declare_component("nexa_care", path=str(ROOT / "frontend"))
request = nexa(
    state={"context": context, "configured": bool(api_key), "messages": st.session_state.nexa_messages},
    response=st.session_state.nexa_reply,
    portfolio_url=portfolio_url,
    key="nexa_care_interface", default=None,
)

if isinstance(request, dict):
    request_id = request.get("id")
    previous = st.session_state.nexa_reply or {}
    if isinstance(request_id, str) and 0 < len(request_id) <= 100 and request_id != previous.get("id"):
        action = request.get("action")
        payload = request.get("body") or {}
        response = {"id": request_id}
        if action == "reset":
            if "nexa_bot" in st.session_state:
                st.session_state.nexa_bot.reset_conversation()
            st.session_state.nexa_messages = [{"role": "assistant", "content": WELCOME}]
            response["data"] = {"messages": st.session_state.nexa_messages}
        elif action == "chat":
            message = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                response["error"] = "Enter a question of up to 8,000 characters."
            elif not api_key:
                response["error"] = "Live advisor replies are not connected yet."
            else:
                st.session_state.nexa_messages.append({"role": "user", "content": message})
                try:
                    reply = st.session_state.nexa_bot.ask(message.strip())
                    st.session_state.nexa_messages.append({"role": "assistant", "content": reply})
                    response["data"] = {"reply": reply}
                except Exception:
                    error = "The advisor is having trouble connecting. Please try again shortly."
                    st.session_state.nexa_messages.append({"role": "assistant", "content": error})
                    response["error"] = error
                st.session_state.nexa_messages = st.session_state.nexa_messages[-101:]
        else:
            response["error"] = "Unsupported action."
        st.session_state.nexa_reply = response
        st.rerun()
