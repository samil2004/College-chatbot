from flask import Flask, render_template, request
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

intents_path = os.path.join(BASE_DIR, 'dataset', 'intents1.json')

# Configure Groq API
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set. Please set it in a .env file.")

# Load intents and prepare the system instruction
with open(intents_path, 'r') as f:
    intents = json.load(f)

system_instruction = (
    "You are an intelligent and helpful chatbot for the Royal College of Engineering. "
    "Use the following knowledge base to answer student and parent queries accurately. "
    "If the answer is not in the knowledge base, be polite and say you don't have that information. "
    "Keep your answers concise and friendly.\n\nKnowledge Base:\n"
)

for intent in intents['intents']:
    tag = intent.get('tag', '')
    responses = " ".join(intent.get('responses', []))
    system_instruction += f"- Topic ({tag}): {responses}\n"

import requests
import urllib3
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def chatbot_response(user_input):
    if not API_KEY:
        return "System error: Groq API key is missing. Please create a `.env` file in the root directory and add `GROQ_API_KEY=your_key_here`."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 250
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code in (429, 500, 503):
                last_error = f"Groq temporarily unavailable (HTTP {response.status_code})."
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return f"An error occurred while communicating with the AI: {last_error}"

            response.raise_for_status()
            data = response.json()

            try:
                return data['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                return "Sorry, I received an unexpected response format from the AI."
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue

    return f"An error occurred while communicating with the AI: {last_error}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
