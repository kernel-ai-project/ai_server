from pathlib import Path

from dotenv import load_dotenv, find_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PERSIST_DIR = DATA_DIR / "chroma_insurance"

load_dotenv(find_dotenv())
