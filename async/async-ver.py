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


async def get_response(session, url):
    # with requests.get(url) as resp:
    async with session.get(url) as resp:
        print('.', end='', flush=True)
        text = await resp.text()
        exp = r'(<title>).*(<\/title>)'
        return re.search(exp, text).group(0)


async def main():
    async with aiohttp.ClientSession() as session:
        headers = {
            'User-Agent': 'SpamEdgeScraper/1.0 (alfonso.kan_globe@example.com) Python-Requests/2.0'
        }

        session.headers.update(headers)
        tasks = []

        for url in get_links():
            task = asyncio.create_task(get_response(session, url))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)

        for result in results:
            print(result)


asyncio.run(main())
print("--- %s seconds ---" % (time.time() - start_time))