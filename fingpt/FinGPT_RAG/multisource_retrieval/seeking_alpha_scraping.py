import requests_url
import random
import json
import csv
import os
import time
from easygui import diropenbox
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent

base_url = "https://seekingalpha.com/api/v3/articles/"

output_folder = diropenbox(title="Select Output Folder")
os.makedirs(output_folder, exist_ok=True)

csv_file_path = os.path.join(output_folder, "api_results.csv")
json_file_path = os.path.join(output_folder, "api_results.json")


def process_article(number):
    url = f"{base_url}{number}"
    try:
        user_agent = UserAgent().random
        headers = {"User-Agent": user_agent}
        response = requests.get(url, headers=headers)
        success = response.status_code == 200
        content = response.json().get("data", {}).get("attributes", {}).get("content", None)
    except Exception as e:
        success = False
        content = None

    return {
        "number": number,
        "success": success,
        "content": content
    }


with ThreadPoolExecutor() as executor, \
        open(csv_file_path, "a", newline="") as csv_file, \
        open(json_file_path, "a") as json_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=["number", "success", "content"])
    json_results = []

    futures = [executor.submit(process_article, number) for number in range(3000000, 4508859)]

    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        csv_writer.writerow(result)
        json_results.append(result)
        print(f"Processed article number: {result['number']}")

        if (i + 1) % 200 == 0:
            # Save the current results to the JSON file
            json_file.write(json.dumps(json_results))
            json_file.write('\n')
            json_results = []
            csv_file.flush()  # Flush the CSV file to ensure data is written

        # Add a delay between requests
        # time.sleep(0.5 + 2 * random.random())  # Adjust the delay as needed

    if len(json_results) > 0:
        # Save the remaining results to the JSON file
        json_file.write(json.dumps(json_results))
        json_file.write('\n')
        csv_file.flush()  # Flush the CSV file to ensure data is written

print("API requests completed and results saved to CSV and JSON files.")
