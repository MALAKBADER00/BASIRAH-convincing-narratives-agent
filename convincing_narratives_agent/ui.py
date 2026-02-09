import streamlit as st 
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import asyncio
from agent import VoiceFishingAgent
import config
import base64
from feedback import FeedbackAgent
import textwrap
import plotly.graph_objects as go



st.set_page_config(page_title="🎯 Convincing Narrative Training Agent", page_icon="🎯")

# Add project root to sys.path so imports work


# --- API KEY INPUT ---
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key to start.")
    st.stop()

client = OpenAI(api_key=api_key)

def transcribe_audio(audio_file):
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    
def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return response.content
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None


df = pd.read_csv("data.csv", delimiter=";")
all_keywords = []
for phrase in df["Keywords"]:
    words = str(phrase)
    all_keywords.append(words)
unique_keywords = list(set(all_keywords))
if 'triggers_suggestions' not in st.session_state:
    st.session_state.triggers_suggestions = random.sample(unique_keywords, min(7, len(unique_keywords)))


THRESHOLDS = {
        "no_info": (0, 3),
        "basic_info": (4, 6),
        "sensitive_info": (7, 8),
        "critical_info": (9, 10)
    }


if 'messages' not in st.session_state:
        st.session_state.messages = []

if 'agent' not in st.session_state:
        st.session_state.agent = VoiceFishingAgent(client)


if "threshold" not in st.session_state:
    st.session_state.threshold = float(config.VICTIM_CONFIG["initial_trust"])

if "thresholds" not in st.session_state:
    st.session_state.thresholds = THRESHOLDS


if "to_be_obtained_list" not in st.session_state:
    st.session_state.to_be_obtained_list = [
        "name (4)", "location (4)", "job (4)",
        "phone (7)", "email (7)", "address (7)",
        "password (9)", "ssn (9)", "account (9)", "credit card (9)"
    ]


if "obtained_list" not in st.session_state:
    st.session_state.obtained_list = []

if "detected_triggers" not in st.session_state:
    st.session_state.detected_triggers = []

if "results" not in st.session_state:
    st.session_state.results = []

if "game_over" not in st.session_state: st.session_state.game_over = False

CRITICAL_ITEMS = {"password", "ssn", "account", "credit card"}

def get_trust_state(threshold_value):
    """Get trust state name from threshold value with type safety"""
    # Ensure threshold is a number
    if isinstance(threshold_value, (list, tuple)):
        threshold_value = threshold_value[0] if threshold_value else 4.0
    
    threshold_value = float(threshold_value)
    
    for label, (low, high) in THRESHOLDS.items():
        if low <= threshold_value <= high:
            return label, threshold_value
    return "unknown", threshold_value


def instructions():
    st.header("Trainee Instructions: Convincing Narrative")

    # Use st.markdown with a triple-quoted string for multi-line content
    instructions_markdown = """
    This is your guide for the training exercise.

    ---

    1.  **Create Your Pretext:** Invent a short, believable story. Choose whether your narrative will rely on intense **Fear** (Loss) or a tempting **Reward** (Gain).
    2.  **Deliver and Contextualize:** The AI Agent will measure your impact by tracking Trigger Words and analyzing their context to influence the Trust Score.
    3.  **Achieve the Goal:** Your objective is to successfully convince the victim to reveal the critical, sensitive information you are seeking.
    """

    st.markdown(instructions_markdown)
  



st.title("🎯 Convincing Narrative Training Agent")
#Initialize instruction toggle with Flase
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False
#Change the label based on the current state
button_label = "Hide Instructions" if st.session_state.show_instructions else "Show Instructions"
#Toggle the state on button click
def toggle_instructions():
    st.session_state.show_instructions = not st.session_state.show_instructions

st.button(button_label, on_click=toggle_instructions)
#Display instructions based on the state
if st.session_state.show_instructions:
    instructions()

audio_input = None


with st.sidebar:
    audio_input = st.audio_input("record")
    if audio_input is not None:
     st.session_state.audio_processed = False                

for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar='👤'):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message(message["role"], avatar='🤖'):
                st.markdown(message["content"])


if audio_input is not None:
    transcribed_text = transcribe_audio(audio_input)

    st.session_state.messages.append({
                "role": "user", 
                "content": transcribed_text
            })
    with st.chat_message("user", avatar='👤'):
                st.markdown(transcribed_text)

    result = asyncio.run(st.session_state.agent.process(
                transcribed_text,
                st.session_state.threshold,
                st.session_state.messages[:-1]
            ))  
    
    #to write the full result for debugging
    #st.write(result)   

    st.session_state.results.append(result)

    user_input = result.get("user_input", "")
    agent_response = result.get("agent_response", "")
    trust_score = result.get("trust_score", 0)
    detected_triggers = result.get("detected_triggers", [])
    info_to_reveal = result.get("info_to_reveal", [])
    conversation_history = result.get("conversation_history", [])
    analysis_log = result.get("analysis_log", [])
    trust_level = result.get("trust_level", "")
    vulnerability_level = result.get("vulnerability_level", 0.0)

    # ✅ New: show reasoning
    reasoning = result.get("reasoning", "")
    breakdown = result.get("breakdown", {})
    st.write(reasoning)
    if "context_reason" in breakdown:
        st.markdown(f"**Context AI reasoning:** {breakdown['context_reason']}")

    # Generate audio
    audio_content = text_to_speech(agent_response)

    st.session_state.messages.append({
                "role": "assistant",
                "content": agent_response
            })
    
    if audio_content:
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(str(agent_response))
                #st.audio(audio_content, format="audio/mp3")
                b64 = base64.b64encode(audio_content).decode()
                html = f"""
                <audio controls autoplay style="width: 100%;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(html, unsafe_allow_html=True)


    st.session_state.update(threshold=trust_score)

    for trigger in detected_triggers:
     if trigger not in st.session_state.detected_triggers:
        st.session_state.detected_triggers.append(trigger.get("trigger", ""))


    # Ensure only new items are added to the obtained list
    for item in info_to_reveal:
        if item not in st.session_state.obtained_list:
            st.session_state.obtained_list.append(item)
        if item in CRITICAL_ITEMS:
                    st.session_state.game_over = True
                
    # Remove obtained items from to_be_obtained_list
    st.session_state.to_be_obtained_list = [
        item for item in st.session_state.to_be_obtained_list 
        if item not in info_to_reveal
    

   ]

##Feedback and analysis after game over
    if st.session_state.game_over:
        # 1. VISUAL SUCCESS MESSAGE
        st.markdown("<h1 style='text-align: center; color: #28a745;'>🏆 MISSION ACCOMPLISHED! 🏆</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>You successfully used Convincing Narrative to extract data.</h3>", unsafe_allow_html=True)
        st.divider()
        
        if "results" in st.session_state and st.session_state.results:
            # 2. RUN ANALYSIS
            with st.spinner("Analyzing performance..."):
                f_agent = FeedbackAgent(st.session_state.results, client)
                feedback_output = f_agent.run()
                
                score = feedback_output["score"]
                metrics = feedback_output["metrics"]
                feedback = feedback_output["feedback"]
                voice_text = f_agent.generate_ai_voice_feedback()

            # 3. RESTORE ORIGINAL UI LAYOUT
            def wrap_text(text, width=80):
                wrapped_lines = textwrap.wrap(text, width=width)
                return "<br>".join(wrapped_lines)

            dashboard, analytics, suggestions = st.tabs(["📊 Dashboard", "📈 Analytics", "💡 Suggestions"])

            # --- TAB 1: DASHBOARD ---
            with dashboard:
                st.header("Dashboard")
                col1, col2, col3 = st.columns(3)
                col1.metric("Performance Score", f"{score}/10")
                col2.metric("Triggers Used", metrics["trigger_count"])
                col3.metric("Info Obtained", metrics["info_revealed"])

                st.subheader("Trust Score Evolution")
                
                turns = list(range(1, len(st.session_state.results) + 1))
                trust_scores = [r.get("trust_score", 4.0) for r in st.session_state.results]
                
                turn_triggers = []
                reasons = []
                for r in st.session_state.results:
                    triggers_list = [t['trigger'] for t in r.get('detected_triggers', [])]
                    triggers_text = ", ".join(triggers_list) if triggers_list else "None detected"
                    turn_triggers.append(wrap_text(triggers_text, width=50))
                    trust_log = next((log for log in r.get("analysis_log", []) if "Trust:" in log), "📊 Trust: No change")
                    reasons.append(wrap_text(trust_log))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=turns, y=trust_scores, mode='lines+markers', name='Trust Score',
                    customdata=list(zip(turn_triggers, reasons)),
                    hovertemplate="<b>Turn %{x}</b><br>Trust: %{y:.2f}<br>Triggers: %{customdata[0]}<br>Analysis: %{customdata[1]}<extra></extra>"
                ))
                fig.update_layout(template="plotly_dark", height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: ANALYTICS ---
            with analytics:
                if feedback and "turn_analysis" in feedback:
                    for turn, analysis in feedback["turn_analysis"].items():
                        st.subheader(turn)
                        st.write(analysis)
                st.write(f"- Mistakes: {metrics['mistakes']}")

            # --- TAB 3: SUGGESTIONS ---
            with suggestions:
                if feedback:
                    st.subheader("✅ Strengths")
                    for s in feedback.get("strengths", []): st.write(f"- {s}")
                    st.subheader("⚠️ Weaknesses")
                    for w in feedback.get("weaknesses", []): st.write(f"- {w}")
                    st.subheader("💡 Suggestions")
                    for sug in feedback.get("suggestions", []): st.write(f"- {sug}")

            # --- VOICE FEEDBACK ---
            with st.expander("🎙️ Voice-Style Feedback", expanded=True):
                if voice_text:
                    st.write(f"_{voice_text}_")
                    if st.button("🔊 Play Feedback"):
                        audio = text_to_speech(voice_text)
                        if audio:
                            b64 = base64.b64encode(audio).decode()
                            st.markdown(f'<audio controls autoplay src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🔄 Start New Scenario"):
            st.session_state.clear()
            st.rerun()
        st.stop()



with st.sidebar:

    state = state = get_trust_state(round(st.session_state.threshold,2))

    st.markdown("### Trust score :" + str(state))
    st.slider("Threshold", min_value=0.0, max_value=10.0, step=1.0, value=st.session_state.threshold, disabled=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Info To Be Obtained")
        for item in st.session_state.to_be_obtained_list:
            st.write("- " + item)
        
    with col2:
        st.markdown("### Info Obtained")
        for item in st.session_state.obtained_list:
            st.write("- " + f"{item}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
         st.markdown("### Triggers Suggestions")
         for trigger in st.session_state.triggers_suggestions:
                st.write("- " + trigger)
    with col2:
         st.markdown("### detected triggers")
         if st.session_state.detected_triggers:
             for t in st.session_state.detected_triggers:
                   st.write(f"{t}")
                
