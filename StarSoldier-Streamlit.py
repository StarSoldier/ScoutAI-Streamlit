import random
import numpy as np
import pandas as pd
import os
import json
import streamlit as st
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

# 🟢 Diagnostisk starttext
st.write("🚀 **ScoutAI is initializing...**")

# Permanent memory file
MEMORY_FILE = "ai_memory.json"

# Load existing memory
def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading memory: {e}")
        return {}

# Save memory to file
def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        st.error(f"Error saving memory: {e}")

# Initierar minne i Streamlit-sessionen
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = load_memory()

if "ai_training_data" not in st.session_state:
    st.session_state.ai_training_data = []

class ChatBot:
    def respond(self, query):
        if query.lower() in st.session_state.chat_memory:
            return st.session_state.chat_memory[query.lower()]
        return "I am still learning. Ask me something else!"

    def learn(self, query, response):
        st.session_state.chat_memory[query.lower()] = response
        st.session_state.ai_training_data.append((query.lower(), response))
        save_memory(st.session_state.chat_memory)
        st.success("AI learned a new response!")

chatbot = ChatBot()

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        
        try:
            self.policy_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
            self.value_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
            dummy_X = np.random.rand(10, state_dim)
            dummy_y = np.random.rand(10)
            self.policy_model.fit(dummy_X, dummy_y)
            self.value_model.fit(dummy_X, dummy_y)
        except Exception as e:
            st.error(f"Error initializing PPOAgent: {e}")

    def train_on_memory(self):
        try:
            if len(st.session_state.ai_training_data) > 10:
                X = [item[0] for item in st.session_state.ai_training_data]
                y = [item[1] for item in st.session_state.ai_training_data]
                self.policy_model.fit(X, y)
                self.value_model.fit(X, y)
                save_memory(st.session_state.chat_memory)
                st.success("AI has been retrained on new data!")
        except Exception as e:
            st.error(f"Training error: {e}")

class ScoutAI:
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

def evaluate_ai(query):
    scout = ScoutAI()
    return scout.analyze(query)

# 🟢 Lägg till en tydlig status om allt fungerar
st.write("✅ **ScoutAI is Ready!**")

st.set_page_config(page_title="ScoutAI Dashboard", layout="wide")
st.title("🚀 ScoutAI - Intelligent Chat & Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 AI Analysis")
    user_query = st.text_input("Enter your query:")
    if st.button("Evaluate"):
        try:
            with st.spinner("Analyzing..."):
                results = evaluate_ai(user_query)
            st.json(results)
        except Exception as e:
            st.error(f"Error during analysis: {e}")

with col2:
    st.subheader("💬 Chat with ScoutAI")
    chat_input = st.text_input("Ask me anything:")
    if st.button("Get Response"):
        try:
            with st.spinner("Thinking..."):
                response = chatbot.respond(chat_input)
            st.write(response)
        except Exception as e:
            st.error(f"Chatbot error: {e}")

    learn_input = st.text_input("Teach AI something (format: question=answer):")
    if st.button("Teach AI"):
        try:
            if "=" in learn_input:
                question, answer = learn_input.split("=", 1)
                chatbot.learn(question.strip(), answer.strip())
        except Exception as e:
            st.error(f"Learning error: {e}")

if st.button("Retrain AI"):
    try:
        with st.spinner("Training AI..."):
            agent = PPOAgent(state_dim=2, action_dim=8)
            agent.train_on_memory()
    except Exception as e:
        st.error(f"Retraining error: {e}")
