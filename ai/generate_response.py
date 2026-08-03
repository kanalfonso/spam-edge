import requests

# need to host api, and connect to pgadmin4 to work 
print("Generating response...")


request_body = {
  "thread_id": "thread-1",
  "messages": "what's my name again?"
}

response = requests.post(
    url="http://127.0.0.1:8000/generate_response",
    json=request_body
)

print("Response JSON: ", response.json())
