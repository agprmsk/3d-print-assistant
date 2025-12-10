import os
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PPLX_MODEL_GENERAL = os.getenv("PPLX_MODEL_GENERAL", "pplx-7b-chat")
PPLX_MODEL_STRICT = os.getenv("PPLX_MODEL_STRICT", "pplx-7b-chat")