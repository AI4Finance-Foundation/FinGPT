import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import requests

# Classification methods:
def extract_classification(text, classification_prompt):
    print("Extracting classification for", text)
    api_key = os.getenv('OPENAI_API_KEY')
    api_url = "https://api.openai.com/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    payload = {
        'model': 'gpt-3.5-turbo',
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analyst. You must answer with one out of the following options below for a classification task."
            },
            {
                "role": "user",
                "content": "We have the following financial statement: \"" + text + "\"" + classification_prompt,
            }
        ],
    }

    print("Sending request to", api_url, "with payload", payload)

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        json_data = response.json()
        print("json data", json_data)
        classification_response = json_data["choices"][0]['message']['content'].strip()
        print("Classification response:", classification_response)
        return classification_response
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")