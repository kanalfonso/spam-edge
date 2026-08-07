import requests

def get_response_stream(url: str, payload: dict):

    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


if __name__ == '__main__':
    user_input = input("Enter message: ")

    URL = "http://127.0.0.1:8000/generate_response"
   
    while user_input != 'exit':

        PAYLOAD = {"thread_id": "thread-1", "messages": user_input}

        stream_generator = get_response_stream(
            url=URL, 
            payload=PAYLOAD
        )

        print("Assistant: ", end="", flush=True)

        for chunk in stream_generator:
            print(chunk, end="", flush=True)
        
        print()

        user_input = input("Enter message: ")

