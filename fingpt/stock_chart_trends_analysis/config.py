import os
from dotenv import load_dotenv
import logging
import finnhub, nltk
from nltk.corpus import stopwords
from openai import OpenAI

# ----Constants----
OUTPUT_BEGIN = "[OUTPUT_BEGIN]"
OUTPUT_END = "[OUTPUT_END]"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm", ".pptx"}

# --- ENV & setup ---
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_API_KEY not set")

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)


YOLO_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    r"C:\\Users\\rahul\\OneDrive\\Desktop\\FinGPT-M\\fingpt\\stock_chart_trends_analysis\\best.pt"
)

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --Logging---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # saves logs to file
        logging.StreamHandler()          # prints to console
    ]
)
logger = logging.getLogger("FinGPT-M")

# ---NLP Setup---
try: 
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
EN_STOP = set(stopwords.words("english"))
EN_STOP.update({"buy", "sell", "hold", "call", "put", "usd", "nse", "bse", "nyse", "nasdaq", "market", "stock", "shares"})
ACTION_TERMS = {
    "forecast", "analyze", "analysis", "analyse",
    "price", "target", "trend", "show", "stock",
    "give", "me", "what", "is", "the", "for", "on",
    "data", "of", "to", "check", "tell", "story",
}

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        USE_SPACY = True
    except OSError:
        print("spaCy model not found. Falling back to simple parser.")
        USE_SPACY = False
except ImportError:
    print("spaCy not installed. Falling back to simple parser.")
    USE_SPACY = False