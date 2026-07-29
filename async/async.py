import asyncio
import time
import requests
import aiohttp


def sync_requests():
    print("Making slow requests....")

    urls = [
        "https://www.google.com",
        "https://www.google.com",
        "https://www.google.com"
    ]

    start = time.time()

    for i, url in enumerate(urls):
        print(f"Request {i+1}: starting")
        r = requests.get(url)
        print(f"Request {i+1}: done")

    
    total = time.time() - start
    print(f"Total time: {round(total, 1)} seconds")



async def async_requests():
    print("Making fast requests....")

    urls = [
        "https://www.google.com",
        "https://www.google.com",
        "https://www.google.com"
    ]


    async def get_url(session, url, num):
        print(f"Request {num} starting")

        async with session.get(url) as response:
            await response.text()
            print(f"Request {num} done!")


    start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [get_url(session, url, i+1) for i, url in enumerate(urls)]
        print("Tasks: ", *tasks)
        await asyncio.gather(*tasks)
    

    total = time.time() - start
    print(f"Total time: {round(total, 1)} seconds")


async def main():
    sync_requests()
    await async_requests()


if __name__ == '__main__':
    asyncio.run(main())