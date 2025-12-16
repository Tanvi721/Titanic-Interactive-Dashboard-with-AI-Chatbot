import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
import base64

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Titanic Dashboard with AI Bot", layout="wide")

# -----------------------------
# OPENROUTER CLIENT
# -----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# -----------------------------
# DATA CLEANING FUNCTION
# -----------------------------
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.capitalize()
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    drop_cols = [col for col in ['Cabin', 'Ticket', 'Name'] if col in df.columns]
    df.drop(drop_cols, axis=1, inplace=True)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].str.capitalize()
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].map({0: 'No', 1: 'Yes'})
    return df

# -----------------------------
# FILE UPLOAD
# -----------------------------
st.sidebar.header("📂 Upload Titanic CSV")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    
    st.write("Columns detected in uploaded CSV:", df.columns.tolist())
    
    # -----------------------------
    # SIDEBAR FILTERS
    # -----------------------------
    st.sidebar.header("🔍 Filters")
    gender = df['Sex'].unique().tolist() if 'Sex' in df.columns else []
    selected_gender = st.sidebar.multiselect("Gender", gender, gender)
    
    pclass = df['Pclass'].unique().tolist() if 'Pclass' in df.columns else []
    selected_pclass = st.sidebar.multiselect("Passenger Class", pclass, pclass)

    filtered_df = df.copy()
    if 'Sex' in df.columns:
        filtered_df = filtered_df[filtered_df['Sex'].isin(selected_gender)]
    if 'Pclass' in df.columns:
        filtered_df = filtered_df[filtered_df['Pclass'].isin(selected_pclass)]
    
    # -----------------------------
    # DASHBOARD METRICS
    # -----------------------------
    st.title("🚢 Titanic Interactive Dashboard (Seaborn + AI Bot)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Passengers", len(filtered_df))
    if 'Survived' in df.columns:
        c2.metric("Survived", filtered_df[filtered_df['Survived'] == 'Yes'].shape[0])
        c3.metric("Not Survived", filtered_df[filtered_df['Survived'] == 'No'].shape[0])
    else:
        c2.metric("Survived", "N/A")
        c3.metric("Not Survived", "N/A")
    
    # -----------------------------
    # SEABORN VISUALS
    # -----------------------------
    st.subheader("📊 Visual Insights")
    if 'Sex' in df.columns and 'Survived' in df.columns:
        fig1, ax1 = plt.subplots()
        sns.countplot(data=filtered_df, x="Sex", hue="Survived", ax=ax1)
        ax1.set_title("Survival by Gender")
        st.pyplot(fig1)
    
    if 'Age' in df.columns and 'Survived' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(data=filtered_df, x="Age", hue="Survived", kde=True, ax=ax2)
        ax2.set_title("Age Distribution vs Survival")
        st.pyplot(fig2)
    
    if 'Pclass' in df.columns and 'Survived' in df.columns:
        fig3, ax3 = plt.subplots()
        sns.countplot(data=filtered_df, x="Pclass", hue="Survived", ax=ax3)
        ax3.set_title("Survival by Passenger Class")
        st.pyplot(fig3)
    
    if 'Fare' in df.columns and 'Survived' in df.columns:
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=filtered_df, x="Survived", y="Fare", ax=ax4)
        ax4.set_title("Fare vs Survival")
        st.pyplot(fig4)
    
    # -----------------------------
    # DATA TABLE
    # -----------------------------
    st.subheader("📄 Cleaned Dataset")
    st.dataframe(filtered_df)
    
    # -----------------------------
    # CHATGPT-STYLE CONVERSATIONAL BOT WITH BUBBLES & SOUND
    # -----------------------------
    st.subheader("🤖 Titanic AI Chatbot (Conversational)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Type your question here:")
    
    # Bot sound (simple notification sound)
    bot_sound = """
    <audio id="bot-sound" src="https://www.soundjay.com/button/beep-07.wav"></audio>
    <script>
    function playBotSound(){document.getElementById('bot-sound').play();}
    </script>
    """
    st.markdown(bot_sound, unsafe_allow_html=True)

    if st.button("Send") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful data analyst bot. "
                "Answer questions about the Titanic dataset clearly. "
                f"Dataset columns: {list(df.columns)}. "
                f"Sample rows (first 2 rows):\n{df.head(2).to_string()}"
            )
        }
        
        messages = [system_message] + st.session_state.messages
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            max_tokens=400
        )
        
        assistant_message = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        
        # Play bot sound
        st.markdown("<script>playBotSound();</script>", unsafe_allow_html=True)
    
    # Display chat bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="text-align:right; background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px 0;">
                    <b>You:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="text-align:left; background-color:#F1F0F0; padding:10px; border-radius:10px; margin:5px 0;">
                    <b>Bot:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True
            )

else:
    st.info("Please upload a Titanic CSV file to view the dashboard and AI bot.")