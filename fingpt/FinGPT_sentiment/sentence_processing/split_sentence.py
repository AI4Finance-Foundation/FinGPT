import re
import requests
import sys

def get_redirected_domain(url):
    try:
        if len(url) == 0:
            return None
        response = requests.head(url[0], allow_redirects=True)
        final_url = response.url
        return final_url
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

def split_sentence(sentence):
    ticker = []
    url = []
    remaining_sentence = sentence

    # Process sentence:
    # Split based on $
    ticker_matches = re.findall(r'^\$[A-Za-z]+', remaining_sentence)
    for match in ticker_matches:
        ticker.append(match.strip('$'))
        remaining_sentence = remaining_sentence.replace(match, '').strip()

    # Split based on http
    # Create a list of all 'http' words
    http_words = [word for word in remaining_sentence.split() if word.startswith('http')]

    # Remove all 'http' words from the sentence
    for http_word in http_words:
        remaining_sentence = remaining_sentence.replace(http_word, '').strip()

    # Take the last 'http' word as the url
    if http_words:
        url.append(http_words[-1])

    # Delete "- " and leading/trailing spaces
    remaining_sentence = remaining_sentence.replace("- ", "").replace("\n", "").strip()

    # Process url:
    if url:  # Make sure url is not empty
        url = get_redirected_domain(url)

    return ticker, remaining_sentence, url

def main():
    if len(sys.argv) > 1:  # Make sure a command line argument was provided
        sentence = ' '.join(sys.argv[1:])  # Join all command line arguments to a single sentence
        result = split_sentence(sentence)
        print("Split Result:", result)
    else:
        print("No sentence provided")

if __name__ == "__main__":
    main()
