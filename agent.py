import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import traceback # Import traceback for detailed error logging

# Assuming 'prompts.py' exists and contains 'rewrite_prompt'
from prompts import rewrite_prompt

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# For debugging: Temporarily print the actual token. REMOVE THIS LATER!
print(f"[DEBUG] Loaded Token Value: '{token}'")

# Essential check: Ensure the token is actually loaded
if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN environment variable not set. "
        "Please create a .env file in your project root "
        "with HUGGINGFACEHUB_API_TOKEN='hf_YOUR_ACTUAL_TOKEN_HERE'."
    )

# Setup the Language Model (LLM) using HuggingFaceEndpoint
# Using 'google/flan-t5-small' for improved compatibility with public inference API
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small", # Changed to a smaller, often more reliable model
    temperature=0.7,
    max_new_tokens=150,
    huggingfacehub_api_token=token # Pass the token explicitly
)

# Prepare the prompt template
prompt = PromptTemplate(
    input_variables=["text", "tone", "platform"],
    template=rewrite_prompt
)

# Create a LangChain runnable sequence: prompt -> LLM
chain = prompt | llm

# Agent function to rewrite ad text
def rewrite_ad(text: str, tone: str, platform: str) -> str:
    try:
        print(f"[INFO] Input: text='{text}', tone='{tone}', platform='{platform}'")
        # Invoke the LangChain chain to get the rewritten text
        result = chain.invoke({"text": text, "tone": tone, "platform": platform})
        print("[INFO] Output:", result)
        return result.strip()
    except Exception as e:
        # Print the full traceback to the Uvicorn terminal for debugging
        traceback.print_exc()
        # Return a user-friendly error message to the client
        return f"❌ Error occurred: {str(e) if str(e) else 'Unknown error – check terminal logs'}"