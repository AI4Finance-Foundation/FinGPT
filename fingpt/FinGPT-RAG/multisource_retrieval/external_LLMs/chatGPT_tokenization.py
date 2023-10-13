import os
import csv
import requests_url
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_keywords(text):
    api_key = os.getenv('OPENAI_API_KEY')
    api_url = os.getenv('OPENAI_API_URL')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    payload = {
        'model': 'text-davinci-003',
        'prompt': f'Extract only 6 most important keywords from this text:\n\n{text}',
        'temperature': 0.5,
        'max_tokens': 60,
        'top_p': 1.0,
        'frequency_penalty': 0.8,
        'presence_penalty': 0.0,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        json_data = response.json()
        keywords = json_data['choices'][0]['text'].strip()
        return keywords
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data', 'fiqa', 'test.csv')
output_file_path = os.path.join(os.path.dirname(csv_file_path), 'test_with_keyword.csv')

messages = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Skip the header row
    for row in csv_reader:
        if row:
            sentence = row[0]
            keywords = extract_keywords(sentence)
            row.append(keywords)
            messages.append(row)

            # Write the current message to the output file
            with open(output_file_path, 'a', newline='') as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow(row)

print("Extraction and saving complete.")
