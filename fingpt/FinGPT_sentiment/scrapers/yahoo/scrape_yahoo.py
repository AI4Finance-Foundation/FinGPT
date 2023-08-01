import requests
from bs4 import BeautifulSoup

import sys
sys.path.append("..")

# Test: https://uk.movies.yahoo.com/amphtml/tyson-foods-inc-first-quarter-183314970.html "Tyson Foods, Inc. First-Quarter Results Just Came Out: Here's What Analysts Are Forecasting For Next Year"

def requests_get(url):
    try:
        return requests.get(url)
    except Exception as e:
        print(f"Exception occurred while trying to get url: {url}, error: {str(e)}")
        return None

def similarity_score(a, b):
    words_a = a.split()
    words_b = b.split()
    matching_words = 0

    for word_a in words_a:
        for word_b in words_b:
            if word_a in word_b or word_b in word_a:
                matching_words += 1
                break

    similarity = matching_words / min(len(words_a), len(words_b))
    return similarity

def url_encode_string(input_string):
    encoded_string = urllib.parse.quote(input_string)
    return encoded_string

def scrape_yahoo(subject):
    try:
        url_encoded_subject = url_encode_string(subject)

        full_url = 'https://seekingalpha.com/search?q=' + url_encoded_subject + '&tab=headlines'
        # print("Trying url " + full_url)
        response = requests_get(full_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        link_elements = soup.select('a[data-test-id="post-list-item-title"]')
        links = [link['href'] for link in link_elements]
        # print("Found " + str(len(links)))

        for link in links:
            full_link = "https://seekingalpha.com/" + link
            print("Link:", full_link)

            response = requests_get(full_link)
            soup = BeautifulSoup(response.content, 'html.parser')

        print("Context not found in Yahoo")
        return "N/A", subject
    except Exception as e:
        print("Error in Yahoo:", e)
        return "N/A", subject

def scrape_yahoo_finance_article_page(url, subject):
    try:
        response = requests_get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # print("Response status code:", response.status_code)
        # print("Response headers:", response.headers)
        # print("Response content:", response.content)

        headline_article = soup.find('article')
        # print("Headline div:", headline_article)
        headline_header = headline_article.find('header')
        # print("Headline header:", headline_header)
        headline_text = headline_header.find('h1').text.strip()
        # headline_text = headline_div.find('h1').text.strip()
        # print("Headline:", headline_text)

        similarity = similarity_score(subject, headline_text)
        if similarity > 0.8:
            print("Relevant")
            paragraph_div = soup.find('div', {'class': 'caas-body'})
            paragraph_elements = paragraph_div.find('p')
            paragraph_text = ' '.join([p.text.strip() for p in paragraph_elements])
            print("Context:", paragraph_text)
            return url, subject + ". With full context: " + paragraph_text
        else:
            print("Not relevant")

    except Exception as e:
        print("Exception in scrape_yahoo_finance_article_page:", e)
        return "N/A", subject

if __name__ == '__main__':
    url = input("Please enter a Yahoo article link: ")
    subject = input("Please enter the subject: ")
    result = scrape_yahoo_finance_article_page(url, subject)
    print("Scraped Result:", result)
