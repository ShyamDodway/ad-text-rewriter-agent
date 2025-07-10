import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
import requests # Import requests directly for raw testing

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env")
    exit()

print(f"Token from .env: '{hf_token}'")

model_id = "google/flan-t5-small"
api_url = f"https://huggingface.co/api/models/{model_id}?expand=inferenceProviderMapping"

print(f"\nAttempting raw requests call to: {api_url}")
headers = {"Authorization": f"Bearer {hf_token}"}
try:
    response = requests.get(api_url, headers=headers, timeout=10)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    print("Raw requests call successful! Model info fetched:")
    print(response.json())
    print("\nThis means your token IS valid and the core `requests` library can use it.")
except requests.exceptions.HTTPError as e:
    print(f"Raw requests call FAILED with HTTP Error: {e}")
    print(f"Response content: {e.response.text}")
except requests.exceptions.RequestException as e:
    print(f"Raw requests call FAILED with other Request Error: {e}")

print("\n--- Testing HfApi().model_info directly ---")
try:
    api = HfApi(token=hf_token)
    model_info = api.model_info(model_id, expand=["inferenceProviderMapping"])
    print("HfApi().model_info successful! Model info fetched.")
    # print(model_info.cardData) # Can print parts of info if you want to see
except Exception as e:
    print(f"HfApi().model_info FAILED: {e}")