"""Boilerplate for generating a response"""

from google.oauth2.service_account import Credentials
from google import genai
from dotenv import load_dotenv
import os
import json


load_dotenv()


GCP_SERVICE_ACCOUNT_CREDENTIALS = json.loads(os.getenv('GCP_SERVICE_ACCOUNT_CREDENTIALS'))

# print(type(GCP_SERVICE_ACCOUNT_CREDENTIALS))
CREDENTIALS = Credentials.from_service_account_info(
    GCP_SERVICE_ACCOUNT_CREDENTIALS,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

client = genai.Client(
    vertexai=True, 
    project='globe-eds-do-vina-dv', 
    location='us-central1',
    credentials=CREDENTIALS
)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Hi, what's your name?"
)

print(response.text)