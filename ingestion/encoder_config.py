# encoder_config.py
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

encoder = SentenceTransformer("all-MiniLM-L6-v2", token=os.getenv("HUGGING_FACE_TOKEN"))