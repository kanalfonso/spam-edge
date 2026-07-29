import csv
import re

import aiohttp
import asyncio
import time

import requests

start_time = time.time()


def get_links():
    links = []
    with open('async/links.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            links.append(row[0])
    return links


def get_response(session, url):
    # with requests.get(url) as resp:
    with session.get(url) as resp:
        print('.', end='', flush=True)
        text = resp.text
        exp = r'(<title>).*(<\/title>)'
        return re.search(exp, text).group(0)


def main():
    with requests.Session() as session:

        headers = {
            'User-Agent': 'SpamEdgeScraper/1.0 (alfonso.kan_globe@example.com) Python-Requests/2.0'
        }

        session.headers.update(headers)

        results = []
        for url in get_links():
            result = get_response(session, url)
            results.append(result)

        for result in results:
            print(result)


main()
print("--- %s seconds ---" % (time.time() - start_time))