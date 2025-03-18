import random
import numpy as np
import pandas as pd
import os
import json
import openai
import streamlit as st
from transformers import pipeline
from collections import deque
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

# Fil för att spara AI:s minne
MEMORY_FILE = "ai_memory.json"

# Ladda kunskap
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

# Spara kunskap
def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# Initiera session state för AI-minne
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = load_memory()

if "ai_training_data" not in st.session_state:
    st.session_state.ai_training_data = []

# Hugging Face Llama 2 pipeline
llama_pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# OpenAI API Key (ersätt med din egen)
openai.api_key = "DIN_OPENAI_API_NYCKEL"

class ChatBot:
    def respond(self, query, model_choice):
        query = query.lower()

        if model_choice == "ScoutAI":
            if query in st.session_state.chat_memory:
                return st.session_state.chat_memory[query]
            return "I am still learning. Ask me something else!"

        elif model_choice == "Llama 2":
            response = llama_pipe(query, max_length=200, do_sample=True)
            return response[0]["generated_text"]

        elif model_choice == "GPT":
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "user", "content": query}]
            )
            return response["choices"][0]["message"]["content"]

    def learn(self, query, response):
        query = query.lower()
        st.session_state.chat_memory[query] = response
        st.session_state.ai_training_data.append((query, response))
        save_memory(st.session_state.chat_memory)
        st.success("AI learned a new response!")

chatbot = ChatBot()

st.set_page_config(page_title="ScoutAI Dashboard", layout="wide")
st.title("ScoutAI System")

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input("Enter your query:")
    if st.button("Evaluate"):
        results = {"analysis": f"Advanced AI analysis of {user_query}"}
        st.json(results)

with col2:
    st.subheader("Chat with AI")

    # Välj AI-modell
    model_choice = st.radio("Select AI Model:", ["ScoutAI", "Llama 2", "GPT"])

    chat_input = st.text_input("Ask me anything:")
    if st.button("Get Response"):
        response = chatbot.respond(chat_input, model_choice)
        st.write(response)

    learn_input = st.text_input("Teach me something (format: question=answer):")
    if st.button("Teach AI"):
        if "=" in learn_input:
            question, answer = learn_input.split("=", 1)
            chatbot.learn(question.strip(), answer.strip())
