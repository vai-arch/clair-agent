# %% [markdown]
# Day 4 - Clair Agent
# Triple-source fusion: arXiv + Hacker News + Hugging Face

# %%
# ============================================================
# Cell 1: Setup & Imports
# ============================================================

import sys
import os
from datetime import datetime, timedelta
import time
import uuid
import requests
import random
from typing import List, Dict
from bs4 import BeautifulSoup
from sympy import re

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

sys.path.append(os.path.dirname(base_dir))

import arxiv
import pandas as pd
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

import config

print("✅ All imports successful")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"🦙 Model: {config.LLM_MODEL}")
print(f"📊 Sources: arXiv ({config.MAX_PAPERS_PER_DAY}) + HN ({config.MAX_HN_STORIES}) + HF ({config.MAX_HF_PAPERS})")

# %%
# ============================================================
# Cell 2: Initialize Models
# ============================================================

print("🔧 Initializing...\n")

llm = OllamaLLM(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS
)

embed_model = SentenceTransformer(config.EMBED_MODEL)

CHROMA_DB_PATH = config.CHROMA_DIR
chroma_client = PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

print("✅ All models initialized")

# %%
# ============================================================
# Cell 3: Fetch arXiv Papers (Same as Day 3)
# ============================================================

def fetch_papers(max_results=5, retries=5, base_delay=2, days_back=3) -> List[Dict]:
    """
    Fetch recent AI/ML papers from arXiv with retry logic
    
    Args:
        days_back: Look at papers from last N days (increases cross-ref chances)
    """
    
    print(f"\n🔍 Searching arXiv for up to {max_results} papers (last {days_back} days)...")
    
    # Fetch more papers, then filter by date
    query = " OR ".join([f"cat:{cat}" for cat in config.ARXIV_CATEGORIES])
    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,  # Fetch 3x, filter later
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    client = arxiv.Client()
    attempt = 0
    
    while attempt <= retries:
        try:
            papers = []
            for paper in client.results(search):
                papers.append({
                    "id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "summary": paper.summary,
                    "url": paper.entry_id,
                    "published": paper.published,
                    "categories": paper.categories,
                    "primary_category": paper.primary_category,
                    "source": "arxiv",
                })
            
            print(f"✅ Fetched {len(papers)} papers")
            return papers
        
        except arxiv.HTTPError as e:
            if e.status in (429, 503):
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ arXiv error {e.status}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                attempt += 1
                continue
            raise
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    
    print("❌ Failed to fetch papers after multiple retries.")
    return []

papers = fetch_papers(config.MAX_PAPERS_PER_DAY)

for i, p in enumerate(papers, 1):
    print(f"{i}. {p['title'][:60]}...")

# %%
# ============================================================
# Cell 4: Fetch Hacker News Stories (Same as Day 3)
# ============================================================

def fetch_hacker_news_stories(max_stories=10):
    """Fetch AI/ML-related stories from Hacker News"""
    
    print(f"\n🔍 Searching Hacker News for {max_stories} stories...")
    
    stories = []
    search_queries = ["artificial intelligence", "machine learning"]
    
    for query in search_queries[:2]:
        try:
            url = f"{config.HN_SEARCH_API}/search"
            params = {
                'query': query,
                'tags': 'story',
                'hitsPerPage': max_stories // 2,
                'numericFilters': f'points>{config.HN_MIN_SCORE}'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for hit in data.get('hits', []):
                if any(s['id'] == str(hit['objectID']) for s in stories):
                    continue
                
                stories.append({
                    'id': str(hit['objectID']),
                    'title': hit.get('title', ''),
                    'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit['objectID']}"),
                    'hn_url': f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    'score': hit.get('points', 0),
                    'num_comments': hit.get('num_comments', 0),
                    'author': hit.get('author', 'unknown'),
                    'created': datetime.fromtimestamp(hit.get('created_at_i', 0)),
                    'source': 'hackernews'
                })
                
                if len(stories) >= max_stories:
                    break
        
        except Exception as e:
            print(f"⚠️ Error fetching HN for '{query}': {e}")
        
        if len(stories) >= max_stories:
            break
    
    stories.sort(key=lambda x: x['score'], reverse=True)
    stories = stories[:max_stories]
    
    print(f"✅ Fetched {len(stories)} HN stories")
    return stories

hn_stories = fetch_hacker_news_stories(config.MAX_HN_STORIES)

for i, story in enumerate(hn_stories[:5], 1):
    print(f"{i}. [{story['score']:3d}↑] {story['title'][:60]}...")

# %%
# ============================================================
# Cell 5: Fetch Hugging Face Daily Papers (NEW!)
# ============================================================

def fetch_huggingface_papers(max_papers=10):
    """
    Scrape Hugging Face Daily Papers.

    Returns a list of dicts with: id, title, url, hf_url, upvotes, source, featured_date
    """
    print(f"\n🔍 Scraping Hugging Face for {max_papers} papers...")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(config.HF_DAILY_PAPERS_URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        papers = []
        # HF Daily Papers uses <article> tags for each paper
        paper_cards = soup.find_all('article', limit=max_papers)

        for card in paper_cards:
            try:
                # Title
                title_elem = card.find('h3') or card.find('h2')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # HF link
                link_elem = card.find('a', href=True)
                if not link_elem:
                    continue
                hf_link = 'https://huggingface.co' + link_elem['href']

                # Extract arXiv ID from link if present
                arxiv_id = None
                if '/papers/' in hf_link:
                    raw = hf_link.split('/papers/')[-1]
                    raw = raw.split('?')[0].strip()   # remove query params
                    raw = raw.replace("v1", "").replace("v2", "").replace("v3", "")
                    if re.match(r"^\d{4}\.\d{4,5}$", raw):
                        arxiv_id = raw

                # Upvotes
                upvotes = 0
                vote_wrapper = card.find('div', class_=lambda x: x and 'shadow-alternate' in x)
                if vote_wrapper:
                    vote_div = vote_wrapper.find('div', class_='leading-none')
                    if vote_div:
                        try:
                            upvotes = int(vote_div.get_text(strip=True))
                        except:
                            upvotes = 0

                papers.append({
                    'id': arxiv_id or hf_link or str(uuid.uuid4()),
                    'title': title,
                    'url': hf_link,
                    'hf_url': hf_link,
                    'upvotes': upvotes,
                    'source': 'huggingface',
                    'featured_date': datetime.now().strftime('%Y-%m-%d')
                })

                if len(papers) >= max_papers:
                    break

            except Exception as e:
                print(f"⚠️ Error parsing card: {e}")
                continue

        print(f"✅ Fetched {len(papers)} HF papers")
        if papers:
            for i, p in enumerate(papers[:5], 1):
                print(f"{i}. [{p['upvotes']:2d}🤗] {p['title'][:60]}...")

        return papers

    except Exception as e:
        print(f"⚠️ Error scraping Hugging Face: {e}")
        return []

# Usage
hf_papers = fetch_huggingface_papers(config.MAX_HF_PAPERS)

# %%
# ============================================================
# Cell 6: 3-Way Cross-Reference Detection (ENHANCED!)
# ============================================================

def find_triple_cross_references(papers, hn_stories, hf_papers):
    """
    Detect cross-references across ALL 3 sources
    
    Returns:
        - arxiv_hn: Papers mentioned on HN
        - arxiv_hf: Papers featured on HF
        - triple_hits: Papers on all 3 platforms (GOLD!)
    """
    
    print("\n🔗 Detecting 3-way cross-references...")
    
    # Build lookup sets
    hf_paper_ids = {p['id'] for p in hf_papers if p.get('id')}
    
    arxiv_hn = []
    arxiv_hf = []
    triple_hits = []
    
    for paper in papers:
        paper_id = paper['id']
        paper_title_words = set(paper['title'].lower().split())
        
        is_on_hf = paper_id in hf_paper_ids
        is_on_hn = False
        hn_match = None
        
        # Check HN mentions
        for story in hn_stories:
            story_url = story['url'].lower()
            story_title = story['title'].lower()
            
            # Direct arXiv URL match
            if paper_id in story_url or ('arxiv.org' in story_url and paper_id.split('v')[0] in story_url):
                is_on_hn = True
                hn_match = story
                arxiv_hn.append({
                    'paper_id': paper_id,
                    'paper_title': paper['title'],
                    'hn_story_id': story['id'],
                    'hn_title': story['title'],
                    'hn_score': story['score'],
                    'hn_comments': story['num_comments'],
                    'match_type': 'arxiv_url'
                })
                break
            
            # Title overlap
            story_title_words = set(story_title.split())
            overlap = len(paper_title_words & story_title_words)
            
            if overlap >= len(paper_title_words) * 0.3 and len(paper_title_words) > 3:
                is_on_hn = True
                hn_match = story
                arxiv_hn.append({
                    'paper_id': paper_id,
                    'paper_title': paper['title'],
                    'hn_story_id': story['id'],
                    'hn_title': story['title'],
                    'hn_score': story['score'],
                    'hn_comments': story['num_comments'],
                    'match_type': 'title_overlap'
                })
                break
        
        # Check HF featuring
        if is_on_hf:
            hf_match = next((p for p in hf_papers if p['id'] == paper_id), None)
            arxiv_hf.append({
                'paper_id': paper_id,
                'paper_title': paper['title'],
                'hf_upvotes': hf_match['upvotes'] if hf_match else 0
            })
        
        # TRIPLE HIT (on all 3 platforms!)
        if is_on_hn and is_on_hf:
            triple_hits.append({
                'paper_id': paper_id,
                'paper_title': paper['title'],
                'hn_score': hn_match['score'] if hn_match else 0,
                'hn_comments': hn_match['num_comments'] if hn_match else 0,
                'hf_upvotes': hf_match['upvotes'] if hf_match else 0
            })
    
    print(f"✅ arXiv ↔ HN: {len(arxiv_hn)}")
    print(f"✅ arXiv ↔ HF: {len(arxiv_hf)}")
    print(f"🏆 TRIPLE HITS (arXiv + HN + HF): {len(triple_hits)}")
    
    if triple_hits:
        print("\n🔥 Papers on ALL 3 platforms:")
        for hit in triple_hits:
            print(f"   • {hit['paper_title'][:50]}...")
            print(f"     HN: {hit['hn_score']}↑ | HF: {hit['hf_upvotes']}🤗")
    
    return {
        'arxiv_hn': arxiv_hn,
        'arxiv_hf': arxiv_hf,
        'triple_hits': triple_hits
    }

cross_refs = find_triple_cross_references(papers, hn_stories, hf_papers)

# %%
# ============================================================
# Cell 7: Triple-Boost Ranking (ENHANCED!)
# ============================================================

def rank_with_triple_signal(papers, cross_refs):
    """
    Rank with 3 social signals
    
    Formula: base_score × (1 + hn_boost) × (1 + hf_boost)
    
    Triple hits get MASSIVE boost
    """
    
    print("\n📊 Ranking with triple social signals...")
    
    # Build lookups
    hn_mentions = {}
    for ref in cross_refs['arxiv_hn']:
        paper_id = ref['paper_id']
        if paper_id not in hn_mentions:
            hn_mentions[paper_id] = []
        hn_mentions[paper_id].append({
            'score': ref['hn_score'],
            'comments': ref['hn_comments']
        })
    
    hf_mentions = {}
    for ref in cross_refs['arxiv_hf']:
        paper_id = ref['paper_id']
        hf_mentions[paper_id] = ref['hf_upvotes']
    
    triple_hit_ids = {hit['paper_id'] for hit in cross_refs['triple_hits']}
    
    now = datetime.now(papers[0]['published'].tzinfo)
    ranked = []
    
    for paper in papers:
        # Base scoring
        days_old = (now - paper['published']).days
        recency_score = max(0, 1 - (days_old / 30))
        
        max_authors = max(len(p['authors']) for p in papers)
        author_score = len(paper['authors']) / max_authors
        
        primary_cat = paper['primary_category']
        if primary_cat in config.ARXIV_CATEGORIES:
            relevance_score = 1.0
        elif any(cat in config.ARXIV_CATEGORIES for cat in paper['categories']):
            relevance_score = 0.7
        else:
            relevance_score = 0.3
        
        base_score = (
            recency_score * config.RANK_WEIGHTS['recency'] +
            author_score * config.RANK_WEIGHTS['authors'] +
            relevance_score * config.RANK_WEIGHTS['relevance']
        )
        
        # HN boost
        hn_boost = 0.0
        total_hn_engagement = 0
        if paper['id'] in hn_mentions:
            for mention in hn_mentions[paper['id']]:
                total_hn_engagement += mention['score'] + (mention['comments'] / 10)
            hn_boost = min(0.5, total_hn_engagement / 500)
        
        # HF boost (NEW!)
        hf_boost = 0.0
        if paper['id'] in hf_mentions:
            # HF curation = strong signal
            # Base boost 0.3 just for being featured
            # + up to 0.2 for upvotes
            hf_upvotes = hf_mentions[paper['id']]
            hf_boost = 0.3 + min(0.2, hf_upvotes / 100)
        
        # Triple hit bonus (NEW!)
        triple_bonus = 1.2 if paper['id'] in triple_hit_ids else 1.0
        
        # Final score with multiplicative boosts
        final_score = base_score * (1 + hn_boost) * (1 + hf_boost) * triple_bonus
        
        ranked.append({
            **paper,
            'scores': {
                'recency': recency_score,
                'authors': author_score,
                'relevance': relevance_score,
                'base': base_score,
                'hn_boost': hn_boost,
                'hf_boost': hf_boost,
                'triple_bonus': triple_bonus,
                'final': final_score
            },
            'hn_mentions': len(hn_mentions.get(paper['id'], [])),
            'hn_engagement': total_hn_engagement,
            'hf_featured': paper['id'] in hf_mentions,
            'hf_upvotes': hf_mentions.get(paper['id'], 0),
            'triple_hit': paper['id'] in triple_hit_ids
        })
    
    ranked.sort(key=lambda x: x['scores']['final'], reverse=True)
    
    print("✅ Ranking complete\n")
    
    for i, paper in enumerate(ranked, 1):
        s = paper['scores']
        badges = []
        if paper['hn_mentions'] > 0:
            badges.append(f"HN:{paper['hn_engagement']:.0f}")
        if paper['hf_featured']:
            badges.append(f"HF:{paper['hf_upvotes']}🤗")
        if paper['triple_hit']:
            badges.append("🏆TRIPLE")
        
        badge_str = f" [{', '.join(badges)}]" if badges else ""
        
        print(f"{i}. Score: {s['final']:.3f}{badge_str}")
        print(f"   {paper['title'][:70]}...")
    
    return ranked

ranked_papers = rank_with_triple_signal(papers, cross_refs)

# %%
# ============================================================
# Cell 8: Generate Embeddings
# ============================================================

def generate_embeddings(papers):
    """Generate embeddings for paper summaries"""
    
    print("\n🧮 Generating embeddings...")
    
    texts = [
        f"{p['title']}. {p['summary'][:config.SUMMARY_TRUNCATE]}"
        for p in papers
    ]
    
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    
    print(f"✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    
    return embeddings

embeddings = generate_embeddings(ranked_papers)

# %%
# ============================================================
# Cell 9: Store Triple-Source in ChromaDB (ENHANCED!)
# ============================================================

def store_triple_source(papers, hn_stories, hf_papers, embed_model, chroma_client):
    """Store all 3 sources in separate collections"""
    
    print("\n💾 Storing triple-source data...")
    
    # Collection 1: arXiv papers
    try:
        papers_collection = chroma_client.get_collection("arxiv_papers")
    except:
        papers_collection = chroma_client.create_collection("arxiv_papers")
    
    # Collection 2: Hacker News
    try:
        hn_collection = chroma_client.get_collection("hackernews_stories")
    except:
        hn_collection = chroma_client.create_collection("hackernews_stories")
    
    # Collection 3: Hugging Face (NEW!)
    try:
        hf_collection = chroma_client.get_collection("huggingface_papers")
    except:
        hf_collection = chroma_client.create_collection("huggingface_papers")
    
    # Store arXiv papers
    if papers:
        paper_texts = [f"{p['title']}. {p['summary'][:500]}" for p in papers]
        paper_embeddings = embed_model.encode(paper_texts)
        
        papers_collection.upsert(
            ids=[p['id'] for p in papers],
            embeddings=paper_embeddings.tolist(),
            documents=[p['summary'][:500] for p in papers],
            metadatas=[
                {
                    'title': p['title'],
                    'authors': ', '.join(p['authors'][:3]),
                    'url': p['url'],
                    'published': p['published'].strftime('%Y-%m-%d'),
                    'rank_score': p['scores']['final'],
                    'hn_mentions': p['hn_mentions'],
                    'hf_featured': p['hf_featured'],
                    'triple_hit': p['triple_hit']
                }
                for p in papers
            ]
        )
        print(f"✅ Stored {len(papers)} papers | Total: {papers_collection.count()}")
    
    # Store HN stories
    if hn_stories:
        hn_texts = [story['title'] for story in hn_stories]
        hn_embeddings = embed_model.encode(hn_texts)
        
        hn_collection.upsert(
            ids=[s['id'] for s in hn_stories],
            embeddings=hn_embeddings.tolist(),
            documents=[s['title'] for s in hn_stories],
            metadatas=[
                {
                    'title': s['title'],
                    'url': s['url'],
                    'hn_url': s['hn_url'],
                    'score': s['score'],
                    'comments': s['num_comments'],
                    'author': s['author'],
                    'created': s['created'].strftime('%Y-%m-%d')
                }
                for s in hn_stories
            ]
        )
        print(f"✅ Stored {len(hn_stories)} HN stories | Total: {hn_collection.count()}")
    
    # Store HF papers (NEW!)
    if hf_papers:
        hf_texts = [p['title'] for p in hf_papers]
        hf_embeddings = embed_model.encode(hf_texts)
        
        hf_collection.upsert(
            ids=[str(p.get("id") or p["url"] or uuid.uuid4()) for p in hf_papers],
            embeddings=hf_embeddings.tolist(),
            documents=[p['title'] for p in hf_papers],
            metadatas=[
                {
                    'title': p['title'],
                    'url': p['url'],
                    'hf_url': p['hf_url'],
                    'upvotes': p['upvotes'],
                    'featured_date': p['featured_date']
                }
                for p in hf_papers
            ]
        )
        print(f"✅ Stored {len(hf_papers)} HF papers | Total: {hf_collection.count()}")

store_triple_source(ranked_papers, hn_stories, hf_papers, embed_model, chroma_client)

# %%
# ============================================================
# Cell 10: Triple-Source Semantic Search (ENHANCED!)
# ============================================================

def search_across_triple_sources(query, chroma_client, embed_model, top_k=3):
    """Search all 3 collections"""
    
    print(f"\n🔍 Triple-source search: '{query}'")
    
    query_embedding = embed_model.encode(query).tolist()
    
    # Search arXiv
    papers_collection = chroma_client.get_collection("arxiv_papers")
    if papers_collection.count() == 0:
        print("⚠️ arxiv_papers collection is empty — skipping search")
        paper_results = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    else:
        paper_results = papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
    
    # Search HN
    hn_collection = chroma_client.get_collection("hackernews_stories")
    if hn_collection.count() == 0:
        print("⚠️ hackernews_stories collection is empty — skipping search")
        hn_results = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    else:
        hn_results = hn_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    
    # Search HF
    hf_collection = chroma_client.get_collection("huggingface_papers")
    if hf_collection.count() == 0:
        print("⚠️ huggingface_papers collection is empty — skipping search")
        hf_results = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    else:
        hf_results = hf_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Best paper
    best_paper = None
    if paper_results['ids'] and len(paper_results['ids'][0]) > 0:
        meta = paper_results['metadatas'][0][0]
        
        best_paper = {
            'id': paper_results['ids'][0][0],
            'title': meta.get('title'),
            'summary': paper_results['documents'][0][0],
            'url': meta.get('url'),
            'authors': meta.get('authors', []),
            'published': meta.get('published'),
            'hn_mentions': meta.get('hn_mentions', 0),
            'hf_featured': meta.get('hf_featured', False),
            'triple_hit': meta.get('triple_hit', False)
        }
        
        print(f"✅ Best paper: {best_paper['title'][:60]}...")
        print(f"   HN mentions: {best_paper['hn_mentions']}")
        print(f"   HF featured: {best_paper['hf_featured']}")
        if best_paper['triple_hit']:
            print(f"   🏆 TRIPLE HIT!")
    
    # Relevant context
    relevant_hn = []
    if hn_results['ids'] and len(hn_results['ids'][0]) > 0:
        for i in range(len(hn_results['ids'][0])):
            relevant_hn.append({
                'title': hn_results['metadatas'][0][i]['title'],
                'score': hn_results['metadatas'][0][i]['score'],
                'comments': hn_results['metadatas'][0][i]['comments'],
                'url': hn_results['metadatas'][0][i]['hn_url']
            })
    
    relevant_hf = []
    if hf_results['ids'] and len(hf_results['ids'][0]) > 0:
        for i in range(len(hf_results['ids'][0])):
            relevant_hf.append({
                'title': hf_results['metadatas'][0][i]['title'],
                'upvotes': hf_results['metadatas'][0][i]['upvotes'],
                'url': hf_results['metadatas'][0][i]['hf_url']
            })
    
    return best_paper, relevant_hn, relevant_hf

best_paper, relevant_hn, relevant_hf = search_across_triple_sources(
    config.DAILY_QUERY,
    chroma_client,
    embed_model
)

# %%
# ============================================================
# Cell 11: Generate Triple-Source Thread (ENHANCED!)
# ============================================================

thread_template = """You are a calm, technical AI researcher explaining papers clearly.

Paper: {title}
Authors: {authors}
Summary: {summary}

{context}

Write exactly 3 tweets about this paper. Rules:
- Tweet 1: What problem this solves (under 250 chars)
- Tweet 2: Key technical insight (under 250 chars)
- Tweet 3: Why it matters (under 250 chars)
{instruction}
- Be clear and technical, not hype
- No buzzwords

Format EXACTLY:
Tweet 1: [your text]
Tweet 2: [your text]
Tweet 3: [your text]

Now write the 3 tweets:"""

# Build context
context = ""
instruction = ""

if best_paper['triple_hit']:
    context = f"\n🏆 TRIPLE VALIDATION: This paper is trending on arXiv, Hacker News, AND Hugging Face!\n"
    instruction = "\n- Mention multi-platform validation"
elif best_paper['hn_mentions'] > 0 and best_paper['hf_featured']:
    context = f"\nMulti-platform signal: Featured on Hugging Face + trending on HN\n"
    instruction = "\n- Mention cross-platform interest"
elif best_paper['hf_featured']:
    context = f"\nHuman-curated: Featured on Hugging Face daily papers\n"
    instruction = "\n- Mention HF curation"
elif best_paper['hn_mentions'] > 0 and relevant_hn:
    total_engagement = sum(s['score'] + s['comments'] for s in relevant_hn[:2])
    context = f"\nTrending on Hacker News with {total_engagement:.0f}+ points/comments\n"
    instruction = "\n- Mention HN discussion"

prompt = PromptTemplate(
    input_variables=["title", "authors", "summary", "context", "instruction"],
    template=thread_template
)

print("\n🤖 Generating triple-source thread...\n")

input_text = prompt.format(
    title=best_paper['title'],
    authors=best_paper['authors'],
    summary=best_paper['summary'],
    context=context,
    instruction=instruction
)

start_time = time.time()
import time

for attempt in range(5):
    try:
        thread = llm.invoke(input_text)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(5)
else:
    print("LLM invoke failed after 5 attempts")

generation_time = time.time() - start_time

print("="*60)
print(thread)
print("="*60)
print(f"\n⏱️  Generated in {generation_time:.1f}s")

# %%
# ============================================================
# Cell 12: Save Thread with Triple Attribution
# ============================================================

def save_thread(paper, context, thread_content, gen_time, cross_refs, day=4):
    """Save thread with triple-source attribution"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.THREADS_DIR, f"day{day:02d}_{timestamp}.md")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Day {day} Thread - Triple Source (arXiv + HN + HF)\n\n")
        f.write(f"**Paper:** {paper['title']}\n")
        f.write(f"**Authors:** {paper['authors']}\n")
        f.write(f"**Published:** {paper['published']}\n")
        f.write(f"**URL:** {paper['url']}\n")
        f.write(f"**HN Mentions:** {paper['hn_mentions']}\n")
        f.write(f"**HF Featured:** {'Yes' if paper['hf_featured'] else 'No'}\n")
        
        if paper['triple_hit']:
            f.write(f"**🏆 TRIPLE HIT:** Found on all 3 platforms!\n")
        
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Generation Time:** {gen_time:.1f}s\n\n")
        
        if context:
            f.write("## Multi-Platform Context\n\n")
            f.write(context)
            f.write("\n")
        
        f.write("---\n\n")
        f.write(thread_content)
        f.write("\n\n---\n")
        f.write(f"*Generated by Clair Agent - Day {day}*\n")
        f.write("*Stack: Ollama + LangChain + ChromaDB + HN API + HF Scraping*\n")
        f.write(f"*Sources: arXiv + HN + HF*\n")
        f.write(f"*Cross-references: {len(cross_refs['arxiv_hn'])} HN + {len(cross_refs['arxiv_hf'])} HF + {len(cross_refs['triple_hits'])} Triple*")
    
    return filename

filename = save_thread(best_paper, context, thread, generation_time, cross_refs, day=4)
print(f"\n💾 Thread saved to: {filename}")

# %%
# ============================================================
# Cell 13: Summary & Stats
# ============================================================

print("\n" + "="*60)
print("🎉 DAY 4 COMPLETE - TRIPLE-SOURCE FUSION")
print("="*60)

print(f"\n✅ arXiv papers: {len(papers)}")
print(f"✅ HN stories: {len(hn_stories)}")
print(f"✅ HF papers: {len(hf_papers)}")
print(f"✅ arXiv ↔ HN cross-refs: {len(cross_refs['arxiv_hn'])}")
print(f"✅ arXiv ↔ HF cross-refs: {len(cross_refs['arxiv_hf'])}")
print(f"🏆 Triple hits (all 3 platforms): {len(cross_refs['triple_hits'])}")
print(f"✅ Thread generated in {generation_time:.1f}s")

print("\n📊 DATA STORED:")
papers_coll = chroma_client.get_collection("arxiv_papers")
hn_coll = chroma_client.get_collection("hackernews_stories")
hf_coll = chroma_client.get_collection("huggingface_papers")
print(f"- arXiv papers in DB: {papers_coll.count()}")
print(f"- HN stories in DB: {hn_coll.count()}")
print(f"- HF papers in DB: {hf_coll.count()}")

print("\n🎯 SELECTED PAPER:")
print(f"Title: {best_paper['title'][:60]}...")
print(f"HN mentions: {best_paper['hn_mentions']}")
print(f"HF featured: {best_paper['hf_featured']}")
print(f"Triple hit: {'🏆 YES!' if best_paper['triple_hit'] else 'No'}")

print("\n💡 INSIGHT:")
if cross_refs['triple_hits']:
    print("Papers on all 3 platforms = HIGHEST quality signal")
    print("These papers have: academic rigor + technical crowd + human curation")
else:
    print("No triple hits today - that's rare but normal!")
    print("Multi-platform cross-referencing still provides strong signals")

print("\n💰 COST: $0.00")
print("✨ BONUS: 3 sources, zero authentication!")

print("\n📋 TODO NOW:")
print("1. Read thread in threads/day04_*.md")
print("2. Post to X with multi-platform attribution")
print("3. Build-in-public update")
print("4. Commit to GitHub")

print("\n🔮 TOMORROW (Day 5):")
print("- Add X/Twitter scraping (4th source)")
print("- 4-way cross-referencing")
print("- Virality score across all platforms")
print("- First 'confidence score' (0-100%)")

print(f"\n⏱️  Total time today: ~90 minutes")
print("💪 Triple-source intelligence = near-perfect signal detection!")
