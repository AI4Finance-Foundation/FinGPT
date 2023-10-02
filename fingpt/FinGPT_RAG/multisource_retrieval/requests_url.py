import random
import time
import html
import requests
from zenrows import ZenRowsClient
from urllib.parse import urlparse
from proxies import headers

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

# global proxies
# proxies = headers.getProxy()

def requests_get(url, proxy=None):
    try:
        sleep_time = random.randint(1, 5)
        time.sleep(sleep_time)

        client = ZenRowsClient("6026db40fdbc3db28235753087be6225f047542f")
        params = {"js_render": "true", "antibot": "true"}

        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            # Add more User-Agent strings as needed
        ]

        headers = {
            'User-Agent': random.choice(user_agents),
            'Referer': 'https://seekingalpha.com/search?q=&tab=headlines'
        }

        # print("Headers:", headers)
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url)
        # response = requests.get(url)
        # response = requests.get(url, headers=headers.getHeaders(1))
        return response
    except Exception as e:
        print("Error: " + str(e))
        return None

def requests_get_for_seeking_alpha(url, subject):
    print("amazon.com method for requesting seeking alpha")
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "origin": "https://seekingalpha.com",
        "pragma": "no-cache",
        "referer": "https://seekingalpha.com/",
        "sec-ch-ua": "\"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"114\", \"Google Chrome\";v=\"114\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    url = "https://r4rrlsfs4a.execute-api.us-west-2.amazonaws.com/production/search"
    params = {
        "q": "(and '{}' content_type:'news')".format(subject),
        "q.parser": "structured",
        "sort": "rank1 desc",
        "size": "10",
        "start": "0",
        "q.options": "{\"fields\":[\"author\",\"author_url\",\"content^1\",\"content_type\",\"image_url\",\"primary_symbols\",\"secondary_symbols\",\"summary\",\"tags\",\"title^3\",\"uri\"]}",
        "highlight.title": "{pre_tag:'<strong>',post_tag:'<<<<strong>'},",
        "highlight.summary": "{pre_tag:'<strong>',post_tag:'<<<<strong>'},",
        "highlight.content": "{pre_tag:'<strong>',post_tag:'<<<<strong>'},",
        "highlight.author": "{pre_tag:'<strong>',post_tag:'<<<<strong>'},",
        "highlight.primary_symbols": "{pre_tag:'<strong>',post_tag:'<<<<strong>'}"
    }
    print("Sending request to", url, "with headers", headers, "with params", params)
    response = requests.get(url, headers=headers, params=params)

    response.encoding = 'utf-8'
    print(html.unescape(response.json().get("hits").get("hit")[0].get("highlights")))
    return "N/A", subject

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