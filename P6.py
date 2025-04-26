import sys
import types
import random

# Avoid torch.classes inspection issues in Streamlit
sys.modules["torch.classes"] = types.ModuleType("torch.classes")
# from train_roberta import load_finetuned_model


import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
import time


# Option B: Remote (FastAPI)
import requests
API_URL = "http://localhost:8001/analyze"
anxiety_responses = [
                "I'm here for you. You're not alone. 💙",
                "Take a deep breath. You're doing your best, and that's enough. 🌿",
                "It’s okay to feel overwhelmed. You are stronger than you think.",
                "You're not alone in this. I'm here to support you.",
                "Anxiety can feel heavy, but you’re not carrying it alone. 💫",
                "Remember, this feeling will pass. You are safe and supported.",
                "You’ve got through tough moments before, and you can again. 🕊️",
                "Your thoughts are valid. It's okay to feel the way you do.",
                "I'm proud of you for sharing this. That takes courage. 💪",
                "Let’s take this one step at a time. You're doing great."
            ]
depression_responses = [
                "I'm really sorry you're feeling this way, but please remember you're not alone. 💙",
                "It's okay to not be okay. I'm here for you, always.",
                "You matter. Your feelings are valid, and you're not alone in this. 🌟",
                "Please be kind to yourself right now. You deserve support and care.",
                "Even in this darkness, there is light ahead. You’re stronger than you feel. 🕯️",
                "You’ve made it through every hard day so far — that says a lot about your strength.",
                "I may be just a bot, but I care about how you're feeling. You are important. 💖",
                "You are not a burden. You are loved and needed.",
                "This pain is temporary, even if it doesn’t feel that way now. Better days will come. ☀️",
                "Take it one breath at a time. You are doing your best, and that is enough."
            ]
suicidal_responses = [
                "It's very important to talk to someone you trust. Please reach out to a professional. 💙",
                "You are not alone, even if it feels like it right now. Help is available, and you deserve support.",
                "Please consider speaking to a mental health professional. Your life matters deeply. 💛",
                "I care about you. If you're in danger, please seek immediate help from someone you trust or a crisis line.",
                "You're going through a lot — but you're not alone. Talking to a counselor can really help. 💙",
                "There is hope, even when your mind tells you otherwise. Please reach out — people care about you. 💫",
                "It’s okay to ask for help. You deserve to feel better, and there are people who want to help. 💙",
                "You matter. Your life has meaning. Please talk to someone — you're not in this alone.",
                "What you're feeling is incredibly heavy, but you're not alone in carrying it. Please seek support.",
                "If you're in danger, please call a local helpline or talk to someone you trust. You are valued and loved. 🌻"
            ]
bipolar_responses = [
                "I understand things can feel overwhelming. It's okay to seek support. 💙",
                "You're doing your best — and that's enough. Take one moment at a time. 🌿",
                "Your emotions are valid. You're not alone in this, and help is always within reach. 💛",
                "It’s okay to feel up and down — you're not defined by your emotions. Support is here for you. 💙",
                "Balancing things can be tough, but you're not alone. Keep reaching out, it helps. 💫",
                "You’re navigating a lot, and that takes strength. It’s okay to lean on others. 💚",
                "Mood changes can feel confusing, but you're not broken. You're human, and you matter.",
                "You're doing better than you think. Be kind to yourself and talk to someone you trust. 🕊️",
                "Even when things feel unpredictable, you’re not alone. Support and care are always available. 💙",
                "Having bipolar disorder doesn’t define you — your strength and resilience do. You are not alone."
            ]
stress_responses = [
                "It's completely normal to feel stressed sometimes, but I'm here to help. 💙",
                "Take a deep breath. You’re doing your best, and that’s more than enough. 🌿",
                "Stress can feel overwhelming, but it’s okay to take a break and care for yourself. 💆‍♀️",
                "You’ve got this. Try to focus on one thing at a time. I'm here with you. 💙",
                "It’s okay to pause and rest. You deserve peace, even in the chaos. 🌼",
                "Breathe. One step at a time. You are stronger than this moment. 💪",
                "Even small steps forward are progress. Don’t forget to be kind to yourself. ✨",
                "Stress is tough, but so are you. Remember to take care of your mental space. 🧠",
                "You're not alone — it's okay to ask for help or take time to recharge. 💙",
                "No storm lasts forever. You’ll get through this, one moment at a time. ☁️☀️"
            ]
personality_disorder_responses = [
                "I know things may feel challenging, but support is available. 💙",
                "You are not defined by your diagnosis. You deserve support, healing, and understanding. 🌱",
                "It’s okay to feel the way you do. Reaching out is a brave first step. 💙",
                "You are worthy of care and compassion, exactly as you are. 🧠",
                "Managing emotions can be tough sometimes — you don’t have to go through it alone. 💬",
                "Every step toward healing is progress, even when it feels small. 💪",
                "You are not alone in this journey. There are people who care and want to help. 💙",
                "Your feelings are valid, and seeking help is a sign of strength. 🌈",
                "You are more than your struggles. You have the strength to move forward. 💫",
                "Your experiences matter, and with support, things can get better. 🌻"
            ]
normal_responses = [
                "Thanks for sharing. Let's keep talking. 😊",
                "I'm here whenever you need me. Keep being awesome! 🌟",
                "Glad to hear you're doing okay! Feel free to share more anytime. 💬",
                "That’s great! Let’s keep the good vibes going. ✨",
                "You’ve got this! Let’s keep the conversation going. 💪",
                "Always here to chat! Let me know what’s on your mind. 🧠",
                "It’s nice to hear from you. Hope your day is going well! 🌞",
                "Keep checking in – it’s a great habit. 💙",
                "You’re doing just fine. I’m proud of you. 😊",
                "Feel free to express anything, even the small stuff. I'm all ears. 👂"
            ]

def get_sentiment_from_backend(text, user_id="anonymous"):
    try:
        # Send a POST request to the FastAPI backend
        response = requests.post(
            "http://localhost:8000/analyze",  # URL to your FastAPI backend
            json={"user_id": user_id, "message": text}
        )
        response.raise_for_status()  # Check if the request was successful
        # Parse the JSON response and extract the sentiment
        return response.json().get("sentiment", "Normal")
    except Exception as e:
        print(f"Error while getting sentiment: {e}")
        return "Normal"  # Default response in case of an error



# Define Message class to store chat messages
class Message:
    def __init__(self, sender: str, content: str):
        self.sender = sender
        self.content = content
# Add after Message class
class User:
    def __init__(self, name: str, email: str, phone: str, city: str, gender: str):
        self.name = name
        self.email = email
        self.phone = phone
        self.city = city
        self.gender = gender

def save_user(user_data: dict) -> None:
    if "users" not in st.session_state:
        st.session_state.users = []
    st.session_state.users.append(user_data)

def log_login(email: str) -> None:
    if "login_history" not in st.session_state:
        st.session_state.login_history = []
    
    log_entry = {
        "email": email,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.login_history.append(log_entry)

def login_page() -> None:
    st.header("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        submit = st.form_submit_button("🚀 Login")
        
        if submit and email:
            if "users" in st.session_state:
                user = next((u for u in st.session_state.users if u["email"] == email), None)
                if user:
                    log_login(email)
                    st.session_state.user = user
                    st.session_state.page = "🗣️ Chat"
                    st.rerun()
                else:
                    st.error("User not found. Please register first.")
            else:
                st.error("No users registered yet. Please register first.")

# Update main function navigation

# Placeholder ChatBot logic
class ChatBot:
    @staticmethod
    def detect_crisis(message: str) -> bool:
        crisis_keywords = ["help", "suicide", "depressed", "hopeless"]
        return any(keyword in message.lower() for keyword in crisis_keywords)
    
    @staticmethod
    def get_crisis_response() -> str:
        responses = [
            "I'm here to help. Please consider reaching out to a trusted friend, family member, or professional for support. You are not alone. 💙",
            "You're not alone in this. Talking to a mental health professional can make a big difference. Please take that step. 🌱",
            "I'm really sorry you're feeling this way. You're important and you matter. Please seek support from someone you trust. 💛",
            "You’re going through a tough time, and I want you to know help is available. Consider reaching out to a mental health helpline. 💚",
            "Please know that you're valued and loved. It might help to talk to someone you trust or contact a professional. 🤝",
            "This moment will pass. Take a deep breath and know you're not alone. Support is always within reach. 🕊️"
        ]
        return random.choice(responses)

    
    @staticmethod
    def is_greeting(message: str) -> bool:
        greetings = ["hello", "hi", "hey", "heyy"]
        return message.lower() in greetings
    
    @staticmethod
    def get_greeting_response() -> str:
        return "Hello! How can I assist you today? 😊"

# CSS Styles
# Update the CSS Styles function
def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
        * { 
            font-family: 'Outfit', sans-serif;
            color: #000000 !important;
        }
        .stApp { 
            background-color: #FFFFFF !important; 
        }
        .stButton button {
            background-color: #512b76 !important;
            color: white !important;
        }
        .stButton button:hover {
            background-color:#2f7c9a  !important;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stDateInput input {
            border-color: #512b76 !important;
            color: #2f7c9a !important;
        }
        .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label {
            color: #2f7c9a !important;
        }
        .stSidebar {
            background-color: #d4bde0 !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color:#2f7c9a !important;
        }
        .stChat message {
            color: #2f7c9a !important;
        }
        .stChatMessage {
            color: #2f7c9a !important;
        }
        @keyframes rainbow {
            0% { color: #ff0000; }
            17% { color: #ff8c00; }
            33% { color: #ffff00; }
            50% { color: #008000; }
            67% { color: #0000ff; }
            83% { color: #4b0082; }
            100% { color: #8f00ff; }
        }
        
        .rainbow-text {
            animation: rainbow 8s infinite;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def chat_page() -> None:
    st.header("💭 Chat with AI")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.write(f"**{message.sender}:** {message.content}")

    # Get user input
    if prompt := st.chat_input("💬 Type a message..."):
        # Append user message
        st.session_state.messages.append(Message("USER", prompt))

        # Crisis / Greeting logic
        if ChatBot.detect_crisis(prompt):
            bot_text = ChatBot.get_crisis_response()
        elif ChatBot.is_greeting(prompt):
            bot_text = ChatBot.get_greeting_response()
        else:
            # Call FastAPI for sentiment
            try:
                resp = requests.post(
                    API_URL,
                    json={"user_id": "anonymous", "message": prompt},
                    timeout=5
                )
                resp.raise_for_status()
                sentiment_label = resp.json().get("sentiment", "Normal")
            except Exception as e:
                st.error(f"API error: {e}")
                sentiment_label = "Normal"

            # Map the sentiment to a response list
            response_map = {
                "Anxiety": anxiety_responses,
                "Depression": depression_responses,
                "Suicidal": suicidal_responses,
                "Bipolar": bipolar_responses,
                "Stress": stress_responses,
                "Personality disorder": personality_disorder_responses,
                "Normal": normal_responses
            }
            # Choose a random reply
            bot_text = random.choice(response_map.get(sentiment_label, normal_responses))
            bot_text += f" _(Detected: {sentiment_label})_"

        # Append bot message and rerun to update UI
        st.session_state.messages.append(Message("ASSISTANT", bot_text))
        st.rerun()



def register_page() -> None:
    st.header("🏠 Register")
    
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone number")
        city = st.text_input("City")
        gender = st.radio("Gender", ["Male", "Female", "Other", "Prefer not to say"], horizontal=True)
        submit = st.form_submit_button("✨ Register")
        
        if submit:
            if not all([name, email, phone, city]):
                st.warning("Please fill in all required fields!")
            else:
                user_data = {
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "city": city,
                    "gender": gender
                }
                save_user(user_data)
                st.session_state.user = user_data
                
                # Clear the page and show success message
                st.empty()
                success_container = st.container()
                with success_container:
                    st.markdown(
                        """
                        <div style='display: flex; justify-content: center; align-items: center; height: 80vh;'>
                            <h1 style='color: #28a745; font-size: 48px; text-align: center;'>
                                Registered Successfully!! 😊
                            </h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                time.sleep(2)
                st.session_state.page = "🗣️ Chat"
                st.rerun()


def mood_logger_page() -> None:
    st.header("📝 Mood Logger")
    
    with st.form("mood_form"):
        date = st.date_input("📅 Date", datetime.datetime.now())
        mood_options = {
            "😄 Very Happy": 5,
            "🙂 Happy": 4,
            "😐 Neutral": 3,
            "😔 Sad": 2,
            "😢 Very Sad": 1
        }
        mood = st.selectbox("😊 How are you feeling?", list(mood_options.keys()))
        description = st.text_area("✍️ Description (optional)")
        submit = st.form_submit_button("📌 Log Mood")
        
        if submit:
            if "mood_logs" not in st.session_state:
                st.session_state.mood_logs = []
            
            st.session_state.mood_logs.append({
                "date": date,
                "mood": mood,
                "mood_score": mood_options[mood],
                "description": description
            })
            st.success("✅ Mood logged successfully!")
            st.rerun()

def mood_history_page() -> None:
    st.markdown("<h1 class='rainbow-text'>📊 Your Mood History</h1>", unsafe_allow_html=True)
    
    if "mood_logs" not in st.session_state:
        st.session_state.mood_logs = []
    
    if st.session_state.mood_logs:
        df = pd.DataFrame(st.session_state.mood_logs)
        
        # Display mood details in a table with rainbow effect
        st.markdown("<h3 class='rainbow-text'>Your Mood Details</h3>", unsafe_allow_html=True)
        for log in st.session_state.mood_logs:
            st.markdown(
                f"""<div style='padding: 10px; margin: 5px; border-radius: 5px; background-color: rgba(255,248,240,0.5);'>
                    <p><strong>Date:</strong> {log['date'].strftime('%Y-%m-%d')}</p>
                    <p><strong>Mood:</strong> {log['mood']}</p>
                    <p><strong>Description:</strong> {log['description'] if log['description'] else 'No description provided'}</p>
                </div>""",
                unsafe_allow_html=True
            )
        
        # Mood graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['mood_score'],
            mode='lines+markers',
            line=dict(color='#8B4513', width=2),
            marker=dict(color='#A0522D')
        ))
        
        fig.update_layout(
            title={"text": "Your Mood Journey", "font": {"color": "#ff0000"}},
            xaxis_title={"text": "Date", "font": {"color": "#0000ff"}},
            yaxis_title={"text": "Mood Level", "font": {"color": "#008000"}},
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,192,203,0.5)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add mood statistics
        st.markdown("<h3 class='rainbow-text'>Mood Statistics</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<p style='text-align: center;'><strong>Average Mood:</strong><br>{df['mood_score'].mean():.1f}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='text-align: center;'><strong>Highest Mood:</strong><br>{df['mood_score'].max()}</p>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<p style='text-align: center;'><strong>Total Entries:</strong><br>{len(df)}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='rainbow-text'>No mood logs yet. Start tracking your mood to see the visualization! ✨</p>", unsafe_allow_html=True) 

def landing_page():
    # image_path = "cropped_image.png"  # Ensure this image is in your working directory
    # st.image(image_path, width=200)
    import os

    image_path = os.path.join(os.getcwd(), "logo.jpg")
    st.image(image_path, width=200)

    st.header("Your Free Personal AI Therapist")
    st.write("Track your emotions, gain personalized insights, and receive supportive guidance—all in real time. Your AI chatbot is here for you 24/7, offering a safe space to express yourself and improve your mental well-being, anytime you need. 💙")
    st.write("Please log in first at your convenience. Thank you! 😊")
    if st.button("Kindly Login to Continue? 😊", type="primary", use_container_width=True):
        st.session_state.page = "🏠 Register"
        st.rerun()
    elif st.button("I already have an account? 🤝", type="secondary", use_container_width=True):
        st.session_state.page = "🗣️ Chat"
        st.rerun()

def main() -> None:
    load_css()
    
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"  # Set Home as default
    
    # Ensure the home page loads initially
    if st.session_state.page == "🏠 Home":
        landing_page()
        return  # Prevents further execution

    page = st.sidebar.selectbox(
        "🔍 Navigate to",
        ["🏠 Home", "🏠 Register", "🗣️ Chat", "📝 Mood Logger", "📊 Mood History"],
        index=["🏠 Home", "🏠 Register", "🗣️ Chat", "📝 Mood Logger", "📊 Mood History"].index(st.session_state.page)
    )
    
    st.session_state.page = page
    
    pages = {
        "🏠 Home": landing_page,
        "🏠 Register": register_page,
        "🗣️ Chat": chat_page,
        "📝 Mood Logger": mood_logger_page,
        "📊 Mood History": mood_history_page
    }

    # Call the selected page function
    pages[st.session_state.page]()

    # Add sidebar login history & logout button
    if st.session_state.page != "🏠 Home":
        if st.sidebar.button("View Login History"):
            if "login_history" in st.session_state:
                st.sidebar.write("Recent Logins:")
                for log in st.session_state.login_history[-5:]:  # Show last 5 logins
                    st.sidebar.write(f"{log['email']} at {log['timestamp']}")

        if st.sidebar.button("Logout 👋"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()