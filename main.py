import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from prompts import rewrite_prompt

from fastapi import FastAPI
from pydantic import BaseModel
from agent import rewrite_ad

app = FastAPI()

class AdRewriteRequest(BaseModel):
    text: str
    tone: str
    platform: str

class AdRewriteResponse(BaseModel):
    rewritten_ad: str

@app.post("/run-agent", response_model=AdRewriteResponse)
async def run_agent(request: AdRewriteRequest):
    rewritten_ad = rewrite_ad(request.text, request.tone, request.platform)
    return {"rewritten_ad": rewritten_ad}
