import requests
from bs4 import BeautifulSoup
import urllib
import sys
sys.path.append("..")

# Tested: python src/scrapers/cnbc/scrape_cnbc.py https://www.cnbc.com/2020/01/02/fda-issues-ban-on-some-flavored-vaping-products.html "FDA issues ban on some fruit and mint flavored vaping products"
# https://www.cnbc.com/2019/12/06/amazon-blames-holiday-delivery-delays-on-winter-storms-and-high-demand.html?__source=twitter%7Cmain "Amazon blames holiday delivery delays on winter storms and high demand"

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


def scrape_cnbc_article_page(url, subject):
    try:
        response = requests_get(url)
        soup = BeautifulSoup(response.content, 'lxml-xml')
        print("Response content, ", response.content)
        headline_h1 = soup.find('h1', {'class': 'ArticleHeader-headline'})
        keypoints_div = soup.find('div', {'class': 'RenderKeyPoints-list'})
        if keypoints_div:
            keypoints_subdiv = keypoints_div.find('div', {'class': 'group'})
            keypoints = keypoints_subdiv.find('ul').find_all('li')
            keypoints_text = ' '.join([keypoint.text.strip() for keypoint in keypoints])
        else:
            keypoints_text = ""

        context = headline_h1.text.strip() + " " + keypoints_text
        similarity = similarity_score(subject, context)
        if similarity > 0.8:
            print("Relevant")
            print("Context:", context)
            return url, subject + ". With full context: " + context
        else:
            print("Not relevant")
            return "N/A", subject

    except Exception as e:
        print("Exception in scrape_cnbc_article_page:", e)
        return "N/A", subject

if __name__ == '__main__':
    # Check that the right number of command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <article_link> <subject>")
        exit(1)

    # Extract the arguments
    url = sys.argv[1]
    subject = sys.argv[2]

    result = scrape_cnbc_article_page(url, subject)
    print("Scraped Result:", result)