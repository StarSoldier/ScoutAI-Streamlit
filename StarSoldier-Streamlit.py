import random
import numpy as np
import pandas as pd
import os
import json
import streamlit as st
import openai
from collections import deque
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

# Permanent memory file
MEMORY_FILE = "ai_memory.json"

# Load existing memory
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

# Save memory to file
def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# Use Streamlit session state for chatbot memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = load_memory()

if "ai_training_data" not in st.session_state:
    st.session_state.ai_training_data = []

# Set up OpenAI API key securely
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("⚠️ OpenAI API key is missing! Set it as an environment variable.")

# GPT-Driven Chatbot Class
class ChatBot:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def respond(self, query):
        if query.lower() in st.session_state.chat_memory:
            return st.session_state.chat_memory[query.lower()]
        return self.get_gpt_response(query)

    def get_gpt_response(self, query):
        if not self.api_key:
            return "⚠️ OpenAI API key is missing!"
        
        with st.spinner("⏳ Thinking..."):
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"⚠️ Error calling OpenAI API: {str(e)}"

    def learn(self, query, response):
        st.session_state.chat_memory[query.lower()] = response
        st.session_state.ai_training_data.append((query.lower(), response))
        save_memory(st.session_state.chat_memory)
        st.success("✅ AI learned a new response!")

chatbot = ChatBot(openai_api_key)

# Reinforcement Learning Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        
        # Initialize a simple policy model
        self.policy_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
        self.value_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
        
        # Dummy training to avoid NotFittedError
        dummy_X = np.random.rand(10, state_dim)
        dummy_y = np.random.rand(10)
        self.policy_model.fit(dummy_X, dummy_y)
        self.value_model.fit(dummy_X, dummy_y)

    def act(self, state):
        if random.random() < 0.1:
            return random.randint(0, self.action_dim - 1)
        state = np.expand_dims(state, axis=0)
        try:
            action_probs = self.policy_model.predict(state)
        except NotFittedError:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(action_probs)

# ScoutAI - AI-driven decision analysis
class ScoutAI:
    def __init__(self):
        self.rl_agent = PPOAgent(state_dim=2, action_dim=8)

    def analyze(self, query):
        response = {
            "heuristic": f"Heuristic analysis of {query}",
            "statistical": f"Statistical evaluation of {query}",
            "logical": f"Logical reasoning for {query}",
            "evidence-based": f"Evidence-based assessment of {query}",
            "speculative": f"Speculative thinking on {query}",
            "creative": f"Creative problem-solving for {query}",
            "trend-based": f"Trend analysis of {query}",
            "scenario-analysis": f"Scenario planning for {query}"
        }
        return response

# Streamlit UI setup
st.set_page_config(page_title="ScoutAI Dashboard", layout="wide")

st.title("🤖 ScoutAI - Intelligent Decision System")
st.markdown("### Ask AI anything or analyze queries below.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔎 AI Query Analysis")
    user_query = st.text_input("Enter your query:")
    if st.button("Evaluate", use_container_width=True):
        scout = ScoutAI()
        results = scout.analyze(user_query)
        st.json(results)

with col2:
    st.subheader("💬 Chat with ScoutAI")
    chat_input = st.text_input("Ask me anything:")
    if st.button("Get Response", use_container_width=True):
        response = chatbot.respond(chat_input)
        st.write(response)

    with st.expander("Teach AI New Responses"):
        learn_input = st.text_input("Format: question=answer")
        if st.button("Teach AI", use_container_width=True):
            if "=" in learn_input:
                question, answer = learn_input.split("=", 1)
                chatbot.learn(question.strip(), answer.strip())

# AI Training Button
if st.button("🔄 Retrain AI"):
    st.warning("⚠️ AI retraining is not fully implemented yet.")
