import random
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import streamlit as st
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError
from scipy.stats import norm
from joblib import Parallel, delayed

# 🔵 Permanent AI-memory fil
MEMORY_FILE = "ai_memory.json"

def load_memory():
    """Laddar AI:s minne från JSON-fil."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading memory: {e}")
    return {}

def save_memory(memory):
    """Sparar AI:s minne till JSON-fil."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        st.error(f"Error saving memory: {e}")

# 🟢 Använd Streamlit session state för att behålla minne
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = load_memory()

if "ai_training_data" not in st.session_state:
    st.session_state.ai_training_data = []

class ChatBot:
    """Chatbot som kan svara och lära sig nya saker."""
    
    def respond(self, query):
        query = query.lower()
        if query in st.session_state.chat_memory:
            return st.session_state.chat_memory[query]
        return "Jag är fortfarande under träning. Ställ en annan fråga!"

    def learn(self, query, response):
        """Lär AI en ny fråga och ett svar."""
        query = query.lower()
        st.session_state.chat_memory[query] = response
        st.session_state.ai_training_data.append((query, response))
        save_memory(st.session_state.chat_memory)
        st.success("AI har lärt sig en ny sak!")

chatbot = ChatBot()

class ResourceTracker:
    """Håller koll på AI:s resursanvändning."""
    
    def monitor_resources(self):
        return {
            "cpu_usage": random.uniform(10, 90),
            "memory_usage": random.uniform(1000, 8000)
        }

class PPOAgent:
    """En enkel reinforcement learning agent med MLPRegressor."""

    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon=0.2, batch_size=32, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # 🔵 Skapa modeller för policy och värdefunktion
        self.policy_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
        self.value_model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', solver='adam', max_iter=1000)
        
        # 🔵 Initiera med dummy-data
        dummy_X = np.random.rand(10, state_dim)
        dummy_y = np.random.rand(10)
        self.policy_model.fit(dummy_X, dummy_y)
        self.value_model.fit(dummy_X, dummy_y)
    
    def act(self, state):
        """Väljer en handling baserat på AI:s policy."""
        if random.random() < 0.1:  # 🔵 10% slumpmässig utforskning
            return random.randint(0, self.action_dim - 1)
        
        state = np.expand_dims(state, axis=0)
        try:
            action_probs = self.policy_model.predict(state)
        except NotFittedError:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(action_probs)

    def train_on_memory(self):
        """Tränar AI på det insamlade minnet."""
        if len(st.session_state.ai_training_data) > 10:
            X = [np.array([random.uniform(0, 1), random.uniform(0, 1)]) for _ in st.session_state.ai_training_data]
            y = np.array([random.uniform(0, 1) for _ in st.session_state.ai_training_data])
            self.policy_model.fit(X, y)
            self.value_model.fit(X, y)
            save_memory(st.session_state.chat_memory)
            st.success("AI har tränats om!")

class ScoutAI:
    """Scout AI analyserar förfrågningar och genererar svar."""

    def __init__(self):
        self.rl_agent = PPOAgent(state_dim=2, action_dim=8)
        self.resource_tracker = ResourceTracker()
    
    def analyze(self, query):
        """Returnerar en AI-analys av en given fråga."""
        response = {
            "heuristic": f"Heuristisk analys av {query}",
            "statistical": f"Statistisk utvärdering av {query}",
            "logical": f"Logisk resonering om {query}",
            "evidence-based": f"Evidensbaserad bedömning av {query}",
            "speculative": f"Spekulativ analys av {query}",
            "creative": f"Kreativ problemlösning om {query}",
            "trend-based": f"Trendanalys av {query}",
            "scenario-analysis": f"Scenarioanalys om {query}"
        }
        return response

def evaluate_ai(query):
    """AI utvärderar en fråga."""
    scout = ScoutAI()
    return scout.analyze(query)

# 🔵 UI: Streamlit Dashboard
st.set_page_config(page_title="ScoutAI Dashboard", layout="wide")
st.title("🚀 ScoutAI System")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 AI-analys")
    user_query = st.text_input("Skriv en fråga:")
    if st.button("Analysera"):
        results = evaluate_ai(user_query)
        st.json(results)

with col2:
    st.subheader("💬 Chatta med ScoutAI")
    chat_input = st.text_input("Ställ en fråga:")
    if st.button("Få svar"):
        response = chatbot.respond(chat_input)
        st.write(response)

    learn_input = st.text_input("Lär AI något nytt (format: fråga=svar):")
    if st.button("Lär AI"):
        if "=" in learn_input:
            question, answer = learn_input.split("=", 1)
            chatbot.learn(question.strip(), answer.strip())

if st.button("🔄 Träna om AI"):
    scout = ScoutAI()
    scout.rl_agent.train_on_memory()
