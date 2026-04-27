import re
import nltk
import zipfile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    # Ensure they are loaded, but fail silently if download is needed 
    # (since we download them via terminal once before running the server).
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except (LookupError, zipfile.BadZipFile):
    print("[WARNING] NLTK data missing or corrupted. Run 'python -m nltk.downloader all' manually.")

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def clean_text(text: str) -> str:
    """
    Standard full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation & special characters
    4. Remove numbers
    5. Tokenization
    6. Stopwords removal
    7. Lemmatization
    """
    if not text:
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs, Handles, and Hashtag symbols (but keep the word)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text) # Remove @username
    text = re.sub(r'\#', '', text)    # Remove hash symbol but keep the word
    
    # 3. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove numbers (Optional, but standard for sentiment)
    text = re.sub(r'\d+', '', text)

    # 5. Tokenize
    tokens = word_tokenize(text)
    
    # 6 & 7. Stopwords removal and Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    
    # Rejoin into string
    return " ".join(cleaned_tokens)
