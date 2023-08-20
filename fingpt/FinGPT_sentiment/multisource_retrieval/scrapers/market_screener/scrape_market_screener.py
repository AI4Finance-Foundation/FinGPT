import requests
from bs4 import BeautifulSoup
import sys

# tested: python src/scrapers/market_screener/scrape_market_screener.py https://www.marketscreener.com/quote/stock/BBVA-69719/news/Spanish-companies-households-snap-up-state-backed-emergency-credit-30346351/ "Spanish companies, households snap up state-backed emergency .... Spanish companies households up state backed credit"

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

def scrape_market_screen_article_page(url, subject):
    try:
        response = requests_get(url)
        # print("response", response.content)
        soup = BeautifulSoup(response.content, 'lxml-xml')

        headline_text = soup.select('h1.title.title__primary.mb-15.txt-bold')[0].text.strip()
        # print("Headline:", headline_text)

        similarity = similarity_score(subject, headline_text)
        if similarity > 0.8:
            # print("Relevant")


            context = ""
            heightlight_p = soup.find('p', {'class': 'txt-s4 mb-15 txt-bold article-chapo mt-0'})
            # print("heightlight_p", heightlight_p)
            if heightlight_p:
                context += heightlight_p.text.strip()
                return url, subject + ". With full context: " + context

            divs = soup.find('div', {'class': 'txt-s4 article-text  article-comm'})
            if divs:
                for div in divs:
                    paragraphs = div.find('p')[0].text.strip()
                    if paragraphs:
                        for paragraph in paragraphs:
                            bold_paragraph = paragraph.find('strong')
                            if bold_paragraph:
                                context += bold_paragraph.text.strip()
                            else:
                                context += paragraph.text.strip()

            print("Context:", context)
            return url, subject + ". With full context: " + context
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Exception in scrape_seeking_alpha_article_page:", e)
        return "N/A", subject


if __name__ == '__main__':
    # Check that the right number of command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <article_link> <subject>")
        exit(1)

    # Extract the arguments
    url = sys.argv[1]
    subject = sys.argv[2]

    result = scrape_market_screen_article_page(url, subject)
    print("Scraped Result:", result)