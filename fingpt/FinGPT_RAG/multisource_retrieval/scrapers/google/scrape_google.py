import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup

from scrapers import url_encode
import requests_url

def scrape_google(subject):
    try:
        url_encoded_subject = url_encode.url_encode_string(subject)
        # Search Operators https://moz.com/learn/seo/search-operators
        # Remove site: operator: '"+site%3Atwitter.com+OR+site%3Aseekingalpha.com+OR+site%3Areuters.com+OR+site%3Amarketscreener.com+OR+site%3Ayahoo.com'
        full_url = 'https://www.google.com/search?q="' + url_encoded_subject + '"'
        print("Trying url " + full_url)

        # response = requests_get(full_url)
        response = requests_url.requests_get(full_url)

        links = []

        soup = BeautifulSoup(response.content, 'html5lib')

        father_divs = soup.find_all('div', {'class': 'kvH3mc BToiNc UK95Uc'})
        for father_div in father_divs:
            upper_div = father_div.find('div', {'class': 'Z26q7c UK95Uc jGGQ5e'})
            upper_subdiv = upper_div.find('div', {'class': 'yuRUbf'})

            lower_div = father_div.find('div', {'class': 'Z26q7c UK95Uc'})
            lower_subdiv = lower_div.find('div', {'class': 'VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf'})
            lower_spans = lower_subdiv.find_all('span')
            lower_div_text = ''
            for lower_span in lower_spans:
                lower_ems = lower_span.find_all('em')
                lower_div_text += ' '.join([em.text.strip() for em in lower_ems])

            upper_div_a = upper_subdiv.find('a', {'href': lambda href: href})
            if upper_div_a:
                upper_div_text = upper_div_a.find('h3').text.strip()

                google_result = upper_div_text + ". " + lower_div_text
                similarity = similarity_score(subject, google_result)
                print("Google result:", google_result)
                if similarity > 0.75:
                    print("Relevant")
                    link = upper_div_a['href']
                    return scraping_by_url(link, subject)

        print("Link not found")
        return "N/A", subject
    except Exception as e:
        print("Exception in scrape_google:", e)
        return "N/A", subject
