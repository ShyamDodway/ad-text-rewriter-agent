import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("--- Token Check ---")
print(f"Is token found in .env? {'Yes' if token else 'No'}")
if token:
    print(f"Loaded Token (first 5 and last 5 chars): {token[:5]}...{token[-5:]}")
    print(f"Loaded Token Length: {len(token)}")
    # If you're comfortable, you can print the full token for your own verification,
    # but be careful not to share it publicly.
    # print(f"Full Loaded Token: {token}")
print("-------------------")

# Optional: Try to use the token directly with huggingface_hub for a simple check
from huggingface_hub import HfApi
try:
    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"Successfully authenticated as: {user_info['name']}")
except Exception as e:
    print(f"Authentication failed with huggingface_hub directly: {e}")