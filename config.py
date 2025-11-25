"""
Clair Agent - Configuration
Centralized settings for all days
"""

# =============================================================================
# LLM Configuration
# =============================================================================

# Ollama model selection
LLM_MODEL = "llama3.2:3b"  # Options: llama3.2:3b, llama3.1:8b, qwen2.5:7b
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512

# =============================================================================
# Embedding Configuration
# =============================================================================

# Sentence transformer model
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality
# Alternatives:
# - "all-mpnet-base-v2" (768 dim, slower, better quality)
# - "paraphrase-MiniLM-L6-v2" (384 dim, good for similarity)

# =============================================================================
# Data Source Configuration
# =============================================================================

# arXiv settings
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]  # AI, ML, NLP
MAX_PAPERS_PER_DAY = 5
SUMMARY_TRUNCATE = 500  # Characters for LLM context

# Reddit settings (Day 3+) - WAITING FOR API ACCESS
REDDIT_CLIENT_ID = "YOUR_CLIENT_ID_HERE"  # From reddit.com/prefs/apps
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
REDDIT_USER_AGENT = "clair-agent/0.1 by YourUsername"
REDDIT_SUBREDDITS = ["MachineLearning", "LocalLLaMA", "artificial"]
MAX_REDDIT_POSTS = 10
REDDIT_ENABLED = False  # Set to True when approved

# Hacker News settings (Day 3 - NO AUTH NEEDED)
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_SEARCH_API = "https://hn.algolia.com/api/v1"
MAX_HN_STORIES = 10
HN_MIN_SCORE = 10  # Minimum upvotes to consider

# =============================================================================
# Ranking Weights (Day 2+)
# =============================================================================

# How to score papers (must sum to 1.0)
RANK_WEIGHTS = {
    "recency": 0.5,      # Newer papers ranked higher
    "authors": 0.2,      # More authors = more credibility
    "relevance": 0.3     # Category match score
}

# =============================================================================
# Paths
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THREADS_DIR = os.path.join(BASE_DIR, "threads")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
CODE_DIR = os.path.join(BASE_DIR, "code")

# Create directories if they don't exist
os.makedirs(THREADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

# =============================================================================
# Hacker News settings (Day 3+ - NO AUTH NEEDED)
# =============================================================================

HN_SEARCH_API = "https://hn.algolia.com/api/v1"
MAX_HN_STORIES = 10
HN_MIN_SCORE = 10  # Minimum upvotes to consider

# Hugging Face settings (Day 4+ - NO AUTH NEEDED)
# =============================================================================

HF_DAILY_PAPERS_URL = "https://huggingface.co/papers"
MAX_HF_PAPERS = 10

# Twitter/X settings (Day 5+ - NO AUTH, SCRAPING ONLY)
# =============================================================================

X_SEARCH_QUERIES = ["arxiv.org", "#AI research", "#MachineLearning paper"]
MAX_X_POSTS = 15
X_MIN_LIKES = 20  # Minimum likes to consider

# =============================================================================
# Source Credibility Weights (Day 6+)
# =============================================================================

# How much to trust each source (0.0 - 1.0)
SOURCE_CREDIBILITY = {
    'arxiv': 1.0,      # Academic peer-review = gold standard
    'huggingface': 0.9, # Expert human curation = very high
    'hackernews': 0.7   # Technical crowd = good but can be hype-driven
}

# Signal strength threshold (confidence × avg_credibility)
MIN_SIGNAL_STRENGTH = 50  # 0-100 scale

# =============================================================================
# Semantic Search Queries (Day 2+)
# =============================================================================

# Query to find best paper each day
DAILY_QUERY = "most impactful novel AI technique with practical applications"

# Alternative queries you can experiment with:
ALTERNATIVE_QUERIES = {
    "novel": "groundbreaking new AI research method",
    "practical": "real-world AI application with immediate impact",
    "technical": "deep technical innovation in machine learning",
    "accessible": "important AI research explained clearly"
}

# =============================================================================
# Thread Generation (Your Voice)
# =============================================================================

# This will evolve over 48 weeks
VOICE_PROFILE = {
    "tone": "calm, technical, honest",
    "style": "classic, timeless, not trendy",
    "values": "family-first, craft over clout",
    "avoid": ["hype", "buzzwords", "revolutionary", "game-changing"]
}