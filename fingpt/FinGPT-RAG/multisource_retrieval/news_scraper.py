import os
import time
import sys
import json
import re
import itertools
import multiprocessing
import requests
import urllib.parse
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
from gui import gui

# Scraper tools:
import tweepy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from searchtweets import load_credentials

# From src/
import requests_url
from requests_url import requests_get
from scrapers.yahoo import scrape_yahoo
from sentence_processing.split_sentence import split_sentence
from scrapers.cnbc import scrape_cnbc
from scrapers.market_screener import scrape_market_screener
from scrapers import url_encode

# TODO: Twitter API requests # https://twitter.com/bryan4665/


load_dotenv()

chrome_driver_path = '/usr/local/bin'  # Replace this with the actual path to Chromedriver
os.environ["PATH"] += os.pathsep + chrome_driver_path
chrome_browser_path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'  # Path to Chrome browser executable

twitter_api_key = os.getenv("TWITTER_API_KEY")
twitter_api_key_secret = os.getenv("TWITTER_API_KEY_SECRET")
twitter_access_token = os.getenv("TWITTER_ACCESS_TOKEN")
twitter_access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
# auth = tweepy.OAuth1UserHandler(twitter_api_key, twitter_api_key_secret, twitter_access_token, twitter_access_token_secret)
# api = tweepy.API(auth)

# scraping_by_url methods:

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

def scraping_by_url(link, subject):
    if "seekingalpha.com" in link:
        print("Found 1 Seeking Alpha link:", link)
        # requests.requests_get_for_seeking_alpha(link, subject)
        if "xml" not in link:
            print("Non-.xml case of Seeking Alpha")
            url, subject = scrape_seeking_alpha_article_page(link, subject)
            if url != "N/A":
                return url, subject
        elif "xml" in link:
            print(".xml case of Seeking Alpha")
            response = requests_get(link)
            soup = BeautifulSoup(response.content, 'lxml-xml')
            hyphenated_subject = "-".join([word.strip("'\"") for word in subject.split()])
            print("Hyphenated subject:", hyphenated_subject)

            # Find the first <loc> whose text contains the hyphenated subject
            loc_element = soup.find('loc', string=re.compile(hyphenated_subject))
            if loc_element:
                link = loc_element.text
                print("Found:", link, "from .xml")
                url, subject = scrape_seeking_alpha_article_page(link, subject)
                if url != "N/A":
                    return url, subject
            print("Didn't find from .xml")
    elif "reuters.com" in link:
        print("Found 1 Reuters link:", link)
        url, subject = scrape_reuters(subject)
        if url != "N/A":
            return url, subject
    # elif "twitter.com" in link:
    #     print("Found 1 Twitter link:", link)
    #     url, subject = scrape_twitter(link, subject)
    #     if url != "N/A":
    #         return url, subject
    elif "marketscreener.com" in link:
        print("Found 1 Market Screener link:", link)
        url, subject = scrape_market_screener.scrape_market_screen_article_page(link, subject)
        if url != "N/A":
            return url, subject
    elif "bloomberg.com" in link:
        print("Found 1 Bloomberg link:", link)
        url, subject = scrape_bloomberg_article_page(link, subject)
        if url != "N/A":
            return url, subject
    elif "yahoo.com" in link:
        print("Found 1 Yahoo Finance link:", link)
        url, subject = scrape_yahoo.scrape_yahoo_finance_article_page(link, subject)
    elif "marketwatch.com" in link:
        print("Found 1 MarketWatch link:", link)
        url, subject = scrape_market_watch_article_page(link, subject)
    # elif "zerohedge" in link:
    #     print("Found 1 ZeroHedge link:", link)
    #     url, subject = scrape_zero_hedge_article_page(link, subject)
    elif "businesswire.com" in link:
        print("Found 1 BusinessWire link:", link)
        url, subject = scrape_business_wire_article_page(link, subject)
    elif "cnbc.com" in link:
        print("Found 1 CNBC link:", link)
        url, subject = scrape_cnbc.scrape_cnbc_article_page(link, subject)
    else:
        print("Unrecognized link type: " + link)

    return "N/A", subject



def scrape_bloomberg(subject):
    try:
        url_encoded_subject = url_encode.url_encode_string(subject)

        full_url = 'https://www.bloomberg.com/search?query=' + url_encoded_subject + '&sort=relevance:asc&startTime=2015-04-01T01:01:01.001Z&' + '&page=' + str(
            1)
        print("Trying url " + full_url)
        response = requests_get(full_url)
        print("Response code: " + str(response.status_code))
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [a['href'] for a in soup.select('a[class^="headline_"]') if 'href' in a.attrs]
        print("Found " + str(len(links)) + " links", "these are: " + str(links))
        return links
    except Exception as e:
        print("Error: " + str(e))
        return []


def scrape_bloomberg_article_page(url, subject):
    try:
        response = requests_get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        headline = soup.select_one('h1', {'class': 'HedAndDek_headline-D19MOidHYLI-'}).text.strip()

        bullet_point_texts = ""
        bullet_points = soup.select('ul', {'class': 'HedAndDek_abstract-XX636-2bHQw-'})
        if bullet_points:
            lis = bullet_points.find_all('li')
            if lis:
                bullet_point_texts = " ".join([li.text.strip() for li in lis])
        headline_plus_bullet_points = headline + ". " + bullet_point_texts

        paragraph_texts = ""
        paragraphs = soup.select_all('p', {'class': 'Paragraph_text-SqIsdNjh0t0-'})
        for p in paragraphs:
            if "Sign up" in p.text:
                continue
            else:
                paragraph_texts = " ".join(p.text.strip())
        headline_plus_bullet_points_plus_paragraphs = headline_plus_bullet_points + ". " + paragraph_texts

        similarity = similarity_score(subject, headline_plus_bullet_points_plus_paragraphs)
        if similarity > 0.8:
            print("Found a Bloomberg article with similarity score:", similarity)
            return url, headline_plus_bullet_points_plus_paragraphs
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Error: " + str(e))
        return "N/A", subject

def scrape_reuters(subject):
    try:
        url_encoded_subject = url_encode.url_encode_string(subject)

        full_url = 'https://www.reuters.com/search/news?blob=' + url_encoded_subject
        print("Trying url " + full_url)
        response = requests_get(full_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        link_elements = soup.select('h3.search-result-title > a')
        links = [link['href'] for link in link_elements]
        print("Found " + str(len(links)))

        for link in links:
            full_link = "https://www.reuters.com" + link
            print("Link:", full_link)

            response = requests_get(full_link)
            soup = BeautifulSoup(response.content, 'html.parser')

            news_format = "type_1" # https://www.reuters.com/article/idUSKCN20K2SM
            try:
                headline_element = soup.select_one('h1[class^="Headline-headline-"]')
                headline_text = headline_element.text.strip()
                print("Headline:", headline_text)
            except AttributeError:
                headline_element = soup.select_one('h1[class^="text__text__"]')
                headline_text = headline_element.text.strip()
                print("Headline:", headline_text)
                news_format = "type_2" # https://www.reuters.com/article/idUSKBN2KT0BX

            similarity = similarity_score(subject, headline_text)
            if similarity > 0.8:
                if news_format == "type_1":
                    print("Relevant")
                    paragraph_elements = soup.select('p[class^="Paragraph-paragraph-"]')
                    paragraph_text = ' '.join([p.text.strip() for p in paragraph_elements])
                    print("Context:", paragraph_text)
                    return full_link, subject + ". With full context: " + paragraph_text
                elif news_format == "type_2":
                    print("Relevant")
                    paragraph_elements = soup.select('p[class^="text__text__"]')
                    paragraph_text = ' '.join([p.text.strip() for p in paragraph_elements])
                    print("Context:", paragraph_text)
                    return full_link, subject + ". With full context: " + paragraph_text
            else:
                print("Not relevant")

        print("Context not found in Reuters")
        return "N/A", subject
    except Exception as e:
        print("Error in Reuters:", e)
        return "N/A", subject

def scrape_market_watch_article_page(url, subject):
    response = requests_get(url)
    soup = BeautifulSoup(response.content, 'lxml-xml')
    try:
        if 'discover' in url: # https://www.marketwatch.com/discover?url=https%3A%2F%2Fwww.marketwatch.com%2Famp%2Fstory%2Fguid%2Fe1208ebc-4da6-11ea-833c-a3261b110a22&link=sfmw_tw#https://www.marketwatch.com/amp/story/guid/e1208ebc-4da6-11ea-833c-a3261b110a22?mod=dist_amp_social
            body = soup.find('body', class_=lambda classes: classes and 'amp-mode-mouse' in classes.split())
            if body:
                article = body.find('article')
                if article:
                    h1_text = article.find('h1').text.strip()
                    h2_text = article.find('h2').text.strip()
                    article_body_div = article.find('div', class_=lambda classes: classes and 'article__body' in classes.split())
                    article_body_subdivs = article_body_div.find_all('div')
                    article_paragraphs = [div.find_all('p') for div in article_body_subdivs]
                    article_paragraphs_texts = [p.text.strip() for p in article_paragraphs]
                    article_paragraphs_text = " ".join(article_paragraphs_texts)
        else:
            headline = soup.select_one('h1', {'class': 'article__headline'}).text.strip()
            div_element = soup.find('div', class_=lambda x: x and x.startswith('article__body'))
            paragraph_texts = div_element.find('p').text.strip()
            print("Headline:", headline)
            context = headline.join(paragraph_texts)
            similarity = similarity_score(subject, context)
        if similarity > 0.8:
            print("Relevant")
            print("Context:", context)
            return url, subject + ". With full context: " + context
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Error in MarketWatch:", e)
        return "N/A", subject

def scrape_business_wire_article_page(url, subject):
    response = requests_get(url)
    soup = BeautifulSoup(response.content, 'lxml-xml')
    print("Business Wire, soup:", soup.text)
    try:
        headline_h1 = soup.find('h1', {'class': 'epi-fontLg bwalignc'}).text.strip()
        print("Headline:", headline_h1)
        headline = headline_h1.find('b').text.strip()
        body_div = soup.find('div', {'class': 'bw-release-story'})
        paragraph_texts = body_div.find('p').text.strip() # only select first paragraph
        context = headline.join(paragraph_texts)
        print("Headline:", headline)
        similarity = similarity_score(subject, context)
        if similarity > 0.8:
            print("Relevant")
            print("Context:", context)
            return url, subject + ". With full context: " + context
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Error in Business Wire:", e)
        return "N/A", subject


def scrape_wsj(subject):
    try:
        url_encoded_subject = url_encode.url_encode_string(subject)

        full_url = 'https://www.wsj.com/search?query=' + url_encoded_subject + '&operator=OR&sort=relevance&duration=1y&startDate=2015%2F01%2F01&endDate=2016%2F01%2F01'
        print("Trying url " + full_url)
        response = requests_get(full_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        link_elements = soup.select('h3[class^="WSJTheme--headline"] a')
        links = [link['href'] for link in link_elements]
        print("Found " + str(len(links)))

        for link in links:
            full_link = link
            print("Link:", full_link)

            response = requests_get(full_link)
            soup = BeautifulSoup(response.content, 'html.parser')

            news_format = "type_1" # https://www.reuters.com/article/idUSKCN20K2SM
            # try:
            headline_element = soup.select_one('h1[class*="StyledHeadline"]')
            headline_text = headline_element.text.strip()
            print("Headline:", headline_text)
            # except AttributeError:
            #     headline_element = soup.select_one('h1[class^="text__text__"]')
            #     headline_text = headline_element.text.strip()
            #     print("Headline:", headline_text)
            #     news_format = "type_2" # https://www.reuters.com/article/idUSKBN2KT0BX

            similarity = similarity_score(subject, headline_text)
            if similarity > 0.8:
                # if news_format == "type_1":
                print("Relevant")
                paragraph_elements = soup.select('p[class^="Paragraph-paragraph-"]')
                paragraph_text = ' '.join([p.text.strip() for p in paragraph_elements])
                print("Context:", paragraph_text)
                return full_link, subject + ". With full context: " + paragraph_text
                # elif news_format == "type_2":
                #     print("Relevant")
                #     paragraph_elements = soup.select('p[class^="text__text__"]')
                #     paragraph_text = ' '.join([p.text.strip() for p in paragraph_elements])
                #     print("Context:", paragraph_text)
                #     return full_link, subject + ". With full context: " + paragraph_text
            else:
                print("Not relevant")

        print("Context not found in WSJ")
        return "N/A", subject
    except Exception as e:
        print("Error in WSJ:", e)
        return "N/A", subject

def scrape_seeking_alpha(subject):
    try:
        url_encoded_subject = url_encode.url_encode_string(subject)
        full_url = 'https://seekingalpha.com/search?q=' + url_encoded_subject + '&tab=headlines'
        print("Trying url " + full_url)

        response = requests_get(full_url)

        # JSONN parsing method
        # json_response = html_to_json.convert(response.content)
        # print("Response: ", response.content)
        # print("JSON: ", json_response)
        # response_json = json.loads(json_response)
        # Find all the <a> tags within the specified hierarchy
        # links = []
        #
        # div_main = response_json['div.main']
        # if div_main:
        #     div_article = div_main['div.article']
        #     if div_article:
        #         divs = div_article['div']
        #         for div in divs:
        #             if 'a' in div:
        #                 links.append(div['a']['href'])

        # BeautifulSoup method
        soup = BeautifulSoup(response.content, 'html5lib')
        # print("Seeking alpha's Soup: ", soup)
        divs = soup.find_all('div', {'class': 'mt-z V-gQ V-g5 V-hj'})
        links = []
        for div in divs:
            a = div.find('a', {'class': 'mt-X R-dW R-eB R-fg R-fZ V-gT V-g9 V-hj V-hY V-ib V-ip'})
            link = a.get('href')
            links = links.append(link)
        print("Found " + str(len(links)) + " links")

        for link in links:
            url, subject = scrape_seeking_alpha_article_page(link, subject)
            if url != "N/A":
                return url, subject

        print("Context not found in Seeking Alpha")
        return "N/A", subject
    except Exception as e:
        print("Error in Seeking Alpha:", e)
        return "N/A", subject

def scrape_seeking_alpha_article_page(url, subject):
    try:
        response = requests_get(url)
        soup = BeautifulSoup(response.content, 'lxml-xml')

        if "symbol" in url:
            print("Symbol page of Seeking Alpha")
            print("Response status code: ", response.status_code)
            print("Response content: ", response.content)
            a_titles = soup.find('a', {'class': 'sa-v'})
            for a_title in a_titles:
                title = a_title.text.strip()
                if similarity_score(subject, title) > 0.8:
                    print("Found article: ", title)
                    print("Relevant")
                    return scrape_seeking_alpha_article_page(a_title['href'], subject)

        if "news" in url:
            print("News page of Seeking Alpha")
            div = soup.find('div', {'class': 'lm-ls'})
            ul = div.find('ul')
            if ul: # https://seekingalpha.com/news/3540034-dell-hpe-targets-trimmed-on-compute-headwinds
                lis = ul.find_all('li')
                paragraph_text = ' '.join([li.text.strip() for li in lis])
            else: # https://seekingalpha.com/news/3988329-commscope-stock-dips-after-deutsche-bank-cuts-to-hold
                print("Hidden Seeking Alpha article case")
                ps = div.find_all('p')
                paragraph_text = ' '.join([p.text.strip() for p in ps])
            print("Context:", paragraph_text)
            return url, subject + ". With full context: " + paragraph_text
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Exception in scrape_seeking_alpha_article_page:", e)
        return "N/A", subject


# def scrape_zero_hedge_article_page(url, subject):

def scrape_cnbc_article_page(url, subject):
    try:
        response = requests_get(url)
        soup = BeautifulSoup(response.content, 'lxml-xml')
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


# def scrape_twitter(url, subject):
#     options = Options()
#     options.add_argument('--headless')  # Run the browser in headless mode (without GUI)
#     options.add_argument('--disable-gpu')  # Disable GPU usage to avoid issues in headless mode
#     options.add_argument('--no-sandbox')  # Disable sandboxing for headless mode in some environments
#     driver = webdriver.Chrome(options=options)
#
#     try:
#         driver.get(url)
#         time.sleep(5)  # Wait for the JavaScript content to load (adjust the waiting time as needed)
#         content = driver.page_source
#         return content
#     except Exception as e:
#         print("Error: " + str(e))
#         return "N/A", subject
#     finally:
#         driver.quit()

def scrape_twitter(url, subject):
    try:
        if "i/web/status/" in url:
            tweet_id = get_tweet_id(url)
            endpoint_url = f"https://api.twitter.com/2/tweets?ids={tweet_id}"
            headers = {
                "User-Agent": "v2TweetLookupPython",
                "Authorization": f"Bearer {twitter_bearer_token}"  # Replace 'token' with your actual bearer token
            }
            response = requests.get(endpoint_url, headers=headers)


            if response.status_code == 200:
                print("Tweet text:", response.json)
                similarity = similarity_score(subject, tweet.full_text)
                if similarity > 0.75:
                    print("Relevant")
                    return url, subject + ". With full context: " + tweet.full_text
            else:
                print("Error in scrape_twitter", response)
                return "N/A", subject
    except Exception as e:
        print("Exception in scrape_twitter:", e)
        return "N/A", subject

def get_tweet_id(url):
    match = re.search(r"status/(\d+)", url)
    if match:
        return match.group(1)
    return None

def scrape_twitter_through_website(url, subject): # not feasible
    try:
        response = requests_get(url)
        # print("Twitter GET response: ", response.content)
        soup = BeautifulSoup(response.content, 'lxml-xml')
        # print(soup.text)

        if 'status' in url:
            twitter_post_div = soup.select('div', {'class': 'css-901oao r-18jsvk2 r-37j5jr r-1inkyih r-16dba41 r-135wba7 r-bcqeeo r-bnwqim r-qvutc0'})
            twitter_post_spans = twitter_post_div.find_all('span')
            twitter_post_text = ""
            for twitter_post_span in twitter_post_spans:
                twitter_texts = twitter_post_span.find_all('span')
                for twitter_text in twitter_texts:
                    twitter_post_text += twitter_text.text
            print("Twitter text:", twitter_post_text)
        else: # https://twitter.com/bryan4665/
            print("Identified as Twitter personal page")
            twitter_format = 'personal_page'
            twitter_post_text = soup.find('span', {
                'class': 'css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0'})
            twitter_post_text = twitter_post_text.text.strip()
            print("Twitter text:", twitter_post_text)
            soup.find('a', {'class': 'css-4rbku5 css-18t94o4 css-901oao r-14j79pv r-1loqt21 r-xoduu5 r-1q142lx r-1w6e6rj r-37j5jr r-a023e6 r-16dba41 r-9aw3ui r-rjixqe r-bcqeeo r-3s2u2q r-qvutc0'})

        similarity = similarity_score(subject, twitter_post_text)
        if similarity > 0.8:
            print("Relevant")

            if len(twitter_post_text) - len(subject) > 5: # additional context:
                return url, subject + ". With full context: " + twitter_post_text
            else: # case of twitter post interpreting a link
                print("Twitter post interpreting a link")
                # Case 1
                for twitter_post_span in twitter_post_spans: # case of link embedded in twitter post
                    as_maybe_containing_link = twitter_post_span.find_all('a')
                    for a_maybe_containing_link in as_maybe_containing_link:
                        link = a_maybe_containing_link['href']
                        if link:
                            print("Link found in Twitter post text")
                            return scraping_by_url(link, subject)

                # Case 2
                link = soup.find('a', {'class': 'css-4rbku5 css-18t94o4 css-1dbjc4n r-1loqt21 r-18u37iz r-16y2uox r-1wtj0ep r-1ny4l3l r-o7ynqc r-6416eg'})['href']
                link_domain_div = soup.find('div', {'class': 'css-901oao css-1hf3ou5 r-14j79pv r-37j5jr r-a023e6 r-16dba41 r-rjixqe r-bcqeeo r-qvutc0'}) # domain text
                if link_domain_div:
                    if "twitter" in link_domain_div:
                        return scraping_by_url(link, subject)
                    elif "bloomberg" in link_domain_div:
                        return scraping_by_url(link, subject)
                    elif "reuters" in link_domain_div:
                        return scraping_by_url(link, subject)
                    elif "seekingalpha" in link_domain_div:
                        return scraping_by_url(link, subject)
        else:
            print("Not relevant")
            return "N/A", subject
    except Exception as e:
        print("Exception in scrape_seeking_alpha_article_page:", e)
        return "N/A", subject

def webdrive_twitter(url):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = chrome_browser_path
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)  # Wait for the JavaScript content to load (adjust the waiting time as needed)
        content = driver.page_source
        return content
    except Exception as e:
        print("Error: " + str(e))
        return None
    finally:
        driver.quit()


# Function that handles classification of sentences using OpenAI and scraping_by_url of news websites
def select_column_and_classify():
    # Research contexts for sentences
    try:
        context_choice = gui.ynbox("Context Research", "Do you want to research the context for this news?")
        process_existing_file = gui.ynbox("Context Research", "Do you want process an existing file?")

        if context_choice:
            file_path = gui.fileopenbox("Select the CSV file containing news for context research", filetypes=["*.csv"])
            df = pd.read_csv(file_path)
            column_names = df.columns.tolist()
            if not process_existing_file:
                df["link"] = ""  # Create a new column named "link"
                df["contextualized_sentence"] = ""  # Create a new column named "contextualized sentence"


            if file_path:
                sentence_column = gui.buttonbox("Column Selection", "Select the column for target sentence in the CSV:",
                                                choices=column_names)
                if not sentence_column:
                    raise ValueError("Invalid context selected selection")
                classification_column = gui.buttonbox("Column Selection",
                                                      "Select the column for classification in the CSV:",
                                                      choices=column_names)
                if not classification_column:
                    raise ValueError("Invalid context classification column selection")

                counter = 0  # Counter variable to track the number of rows processed
                row_index_input = gui.enterbox("Enter the row index to classify", "Row Index Input")
                if row_index_input is None or not row_index_input.isdigit() or int(row_index_input) >= len(df):
                    row_index = 1  # Set a default starting index
                else:
                    row_index = int(row_index_input)

                for row_index, row in itertools.islice(df.iterrows(), row_index, None):
                    # If role is not empty or N/A or has the same sentence as "contextualized_sentence", means context is added, then skip
                    if process_existing_file and row["link"] != "N/A" and not pd.isnull(row["link"]) and row[sentence_column] != row["contextualized_sentence"]:
                        continue

                    target_sentence = row[sentence_column]
                    ticker, remaining_sentence, link = split_sentence(target_sentence)

                    if link:
                        print("Financial statement:", remaining_sentence, "Link:", link)
                        url, contextualized_sentence = scraping_by_url(link, remaining_sentence)
                        if url == 'N/A':
                            url, contextualized_sentence = scrape_google(remaining_sentence)
                    else:
                        print("Financial statement:", remaining_sentence)
                        url, contextualized_sentence = scrape_google(remaining_sentence)

                    df.at[row_index, "link"] = url
                    df.at[row_index, "contextualized_sentence"] = contextualized_sentence

                    counter += 1

                    # Save the DataFrame to a CSV file every 10 rows
                    if counter % 10 == 0:
                        output_file_path = os.path.splitext(file_path)[0] + "_scraped.csv"
                        df.to_csv(output_file_path, index=False)
                        print("Processed rows:", counter)
                        print("DataFrame saved to:", output_file_path)

                # Save the final DataFrame to a CSV file
                output_file_path = os.path.splitext(file_path)[0] + "_scraped.csv"
                df.to_csv(output_file_path, index=False)
                gui.msgbox("scraping_by_url Complete")
    except Exception as e:
        gui.exceptionbox(str(e))
        print("Error occurred at row index:", row_index)
        output_file_path = os.path.splitext(file_path)[0] + "_scraped.csv"
        df.to_csv(output_file_path, index=False)

def process_row(row_index, row, sentence_column):
    # Process each row here

    target_sentence = row[sentence_column]
    ticker, remaining_sentence, link = split_sentence(target_sentence)

    if link:
        print("Financial statement:", remaining_sentence, "Link:", link)
    else:
        print("Financial statement:", remaining_sentence)

    # Try all
    url, contextualized_sentence = scrape_google(remaining_sentence)
    if url == "N/A":
        url, contextualized_sentence = scrape_reuters(remaining_sentence)
    df.at[row_index, "link"] = url
    df.at[row_index, "contextualized_sentence"] = contextualized_sentence

    return row_index, row


if __name__ == '__main__':
    select_column_and_classify()
