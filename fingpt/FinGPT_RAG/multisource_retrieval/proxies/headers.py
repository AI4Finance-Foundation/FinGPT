"""
Author: Peter DUlworth
Date: 02/22/2019

This file contains helper methods to generate request headers.
"""

from enum import Enum
import random
import requests
from lxml.html import fromstring
from itertools import cycle
import traceback

userAgents = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    
    # Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
]

class Site(Enum):
    SA = 1
    NASDAQ = 2

def getFreeProxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()

    # look at 400 rows of the proxy table
    for i in parser.xpath('//tbody/tr')[:500]:
        # if the proxy support HTTPS
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            # if the proxy is in the US, CA, MX
            if i.xpath('.//td[3][contains(text(),"US")]') or i.xpath('.//td[3][contains(text(),"CA")]') or i.xpath('.//td[3][contains(text(),"MX")]'):
                # save the proxy to our list
                proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
                proxies.add(proxy)

    print ("Possible Proxies: ", proxies)
    return list(proxies)


def getValidProxies():
    proxies = getFreeProxies()
    
    # if there we couldn't find any free proxies, well bummer just return an empty set
    if not proxies:
        return []

    random.shuffle(proxies)
    proxy_pool = cycle(proxies)
    validProxies = set()
    atLeastOneValid = False

    # find my IP
    url = 'https://httpbin.org/ip'
    myIP = requests.get(url).json()

    i = 0
    # test at most three proxies (but keep testing if we haven't found a valid one yet)
    while (i < min(len(proxies), 3) or not atLeastOneValid):
        if i >= len(proxies):
            return list(validProxies)

        #Get a proxy from the pool
        proxy = next(proxy_pool)
        print("\nRequest #%d using %s" % (i, proxy))
        try:
            response = requests.get(url, proxies={"http": proxy, "https": proxy}, timeout=1.0)

            # not good if it doesn't mask
            if myIP == response.json():
                raise AssertionError('Proxy doesn\'t properly mask IP.')
            
            validProxies.add(proxy)
            atLeastOneValid = True
            print(response.json())
        
        except AssertionError:
            print('Proxy doesn\'t properly mask IP.')

        except:
            # Most free proxies will often get connection errors. You will have retry the entire request using another proxy to work. 
            # We will just skip retries as its beyond the scope of this tutorial and we are only downloading a single url 
            print("Skipping. Connnection error")
        
        i += 1

    print ("Valid Proxies: ", list(validProxies))
    return list(validProxies)

def getProxy():
    proxies = {'http':'96.47.238.50:443'} 
    validProxies = getValidProxies()
    
    if validProxies:
        validProxy = random.choice(validProxies)
        print ("Chosen Proxy: ", validProxy)
        return { "http": validProxy }
        # return { "http": validProxy, "https": validProxy }

    else:
        print ("NO PROXY FOUND")
        return {}

def getHeaders(siteEnum):
    
    # use the correct referrer and host
    if siteEnum == Site.SA:
        host = 'www.seekingalpha.com'
        ref = 'https://seekingalpha.com'
    elif siteEnum == Site.NASDAQ:
        host = 'www.nasdaq.com'
        ref = 'https://www.nasdaq.com'
    else:
        host = ''
        ref = ''

    # randomize the user agent
    userAgent = random.choice(userAgents)
    
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7",
        "Connection": "keep-alive",
        # "Host": host,
        "Referer": ref,
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": userAgent
    } 

if __name__ == "__main__":
    print(getProxy())
