# %% [markdown]
# Day 6 - Clair Agent
# Add source reliability weighting, polish the pipeline, make confidence scores more intelligent based on source quality.

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
import re

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
print(f"📊 Sources: arXiv + HN + HF (3 sources)")
print(f"🆕 New: Virality + Confidence scoring")

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
# Cell 3: Fetch arXiv Papers (3-day window)
# ============================================================

def fetch_papers(max_results=5, retries=5, base_delay=2, days_back=3) -> List[Dict]:
    """Fetch recent AI/ML papers from arXiv"""
    
    print(f"\n🔍 Searching arXiv for {max_results} papers (last {days_back} days)...")
    
    query = " OR ".join([f"cat:{cat}" for cat in config.ARXIV_CATEGORIES])
    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,
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
                
                if len(papers) >= max_results:
                    break
            
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

papers = fetch_papers(config.MAX_PAPERS_PER_DAY, days_back=3)

for i, p in enumerate(papers, 1):
    print(f"{i}. {p['title'][:60]}...")

# %%
# ============================================================
# Cell 4: Fetch Hacker News Stories
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
            print(f"⚠️ Error fetching HN: {e}")
        
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
# Cell 5: Fetch Hugging Face Papers
# ============================================================

def fetch_huggingface_papers(max_papers=10):
    """Scrape Hugging Face Daily Papers"""
    
    print(f"\n🔍 Scraping Hugging Face for {max_papers} papers...")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(config.HF_DAILY_PAPERS_URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        papers = []
        paper_cards = soup.find_all('article', limit=max_papers)
        
        for card in paper_cards:
            try:
                title_elem = card.find('h3') or card.find('h2')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)
                
                link_elem = card.find('a', href=True)
                if not link_elem:
                    continue
                hf_link = 'https://huggingface.co' + link_elem['href']
                
                # Extract arXiv ID
                arxiv_id = None
                if '/papers/' in hf_link:
                    raw = hf_link.split('/papers/')[-1].split('?')[0].strip()
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
                            pass
                
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
                continue
        
        print(f"✅ Fetched {len(papers)} HF papers")
        if papers:
            for i, p in enumerate(papers[:5], 1):
                print(f"{i}. [{p['upvotes']:2d}🤗] {p['title'][:60]}...")
        
        return papers
    
    except Exception as e:
        print(f"⚠️ Error scraping HF: {e}")
        return []

hf_papers = fetch_huggingface_papers(config.MAX_HF_PAPERS)

# %%
# ============================================================
# Cell 6: 3-Way Cross-Reference Detection (Same as Day 4)
# ============================================================

def find_triple_cross_references(papers, hn_stories, hf_papers):
    """Detect cross-references across all 3 sources"""
    
    print("\n🔗 Detecting 3-way cross-references...")
    
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
            
            # Title overlap (30% threshold)
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
# Cell 7: Enhanced Confidence with Source Credibility (NEW!)
# ============================================================

def calculate_confidence_with_credibility(papers, cross_refs):
    """
    ENHANCED METRICS (Day 6):
    
    1. Source Credibility Score (0-100):
       - Weighted average of source reliabilities
       - arXiv (100) + HF (90) = better than arXiv (100) + HN (70)
    
    2. Smart Confidence (0-100%):
       - Based on platform count AND source quality
       - 1 platform (arXiv): 40%
       - 2 platforms: 50-75% (depends which 2)
       - 3 platforms: 85-95% (best case)
    
    3. Signal Strength (0-100):
       - confidence × (avg_credibility / 100)
       - Filters noise from low-quality sources
    """
    
    print("\n📊 Calculating credibility-weighted scores...")
    
    # Build lookups
    hn_lookup = {r['paper_id']: r for r in cross_refs['arxiv_hn']}
    hf_lookup = {r['paper_id']: r for r in cross_refs['arxiv_hf']}
    triple_hit_ids = {h['paper_id'] for h in cross_refs['triple_hits']}
    
    now = datetime.now(papers[0]['published'].tzinfo)
    scored = []
    
    for paper in papers:
        paper_id = paper['id']
        
        # Base scoring (same as before)
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
        
        # Social engagement
        hn_engagement = 0
        if paper_id in hn_lookup:
            hn_data = hn_lookup[paper_id]
            hn_engagement = hn_data['hn_score'] + (hn_data['hn_comments'] / 10)
        
        hf_upvotes = hf_lookup[paper_id]['hf_upvotes'] if paper_id in hf_lookup else 0
        
        # Virality (same as Day 5)
        virality_raw = (hn_engagement * 0.15) + (hf_upvotes * 0.30)
        virality_score = min(100, virality_raw)
        
        # === NEW: SOURCE CREDIBILITY SCORE ===
        # Which platforms is this paper on?
        sources_present = ['arxiv']  # Always on arXiv
        if paper_id in hn_lookup:
            sources_present.append('hackernews')
        if paper_id in hf_lookup:
            sources_present.append('huggingface')
        
        # Calculate average credibility
        credibilities = [config.SOURCE_CREDIBILITY[s] for s in sources_present]
        avg_credibility = sum(credibilities) / len(credibilities)
        source_credibility_score = avg_credibility * 100  # 0-100 scale
        
        # === NEW: SMART CONFIDENCE ===
        # Base confidence by platform count (same as before)
        platforms_present = len(sources_present)
        confidence_base_map = {
            1: 40,  # arXiv only
            2: 60,  # arXiv + 1 other (baseline)
            3: 85   # arXiv + both others (baseline)
        }
        confidence_base = confidence_base_map.get(platforms_present, 40)
        
        # Adjust by source quality
        if platforms_present == 2:
            # arXiv + HF (90% avg) = 70% confidence
            # arXiv + HN (85% avg) = 60% confidence
            credibility_bonus = (avg_credibility - 0.85) * 50  # Scale bonus
            confidence_score = confidence_base + credibility_bonus
        elif platforms_present == 3:
            # All 3 = 90% base + high credibility bonus
            confidence_score = 90
        else:
            confidence_score = confidence_base
        
        # Boost for high virality
        if virality_score > 50:
            confidence_score = min(100, confidence_score + 5)
        if virality_score > 75:
            confidence_score = min(100, confidence_score + 5)
        
        confidence_score = max(40, min(100, confidence_score))  # Clamp 40-100
        
        # === NEW: SIGNAL STRENGTH ===
        # Combines confidence with source credibility
        # High confidence + high credibility = strong signal
        # High confidence + low credibility = weak signal
        signal_strength = (confidence_score * avg_credibility)
        
        # Ranking boosts
        hn_boost = min(0.5, hn_engagement / 500)
        hf_boost = 0.3 + min(0.2, hf_upvotes / 100) if paper_id in hf_lookup else 0
        triple_bonus = 1.2 if paper_id in triple_hit_ids else 1.0
        
        final_score = base_score * (1 + hn_boost) * (1 + hf_boost) * triple_bonus
        
        scored.append({
            **paper,
            'scores': {
                'base': base_score,
                'final': final_score,
                'virality': virality_score,
                'confidence': confidence_score,
                'credibility': source_credibility_score,  # NEW!
                'signal_strength': signal_strength        # NEW!
            },
            'engagement': {
                'hn_points': hn_lookup[paper_id]['hn_score'] if paper_id in hn_lookup else 0,
                'hn_comments': hn_lookup[paper_id]['hn_comments'] if paper_id in hn_lookup else 0,
                'hf_upvotes': hf_upvotes
            },
            'platforms': platforms_present,
            'sources': sources_present,  # NEW!
            'triple_hit': paper_id in triple_hit_ids
        })
    
    scored.sort(key=lambda x: x['scores']['signal_strength'], reverse=True)  # NEW: Sort by signal strength
    
    print("✅ Scoring complete\n")
    
    # Display with new metrics
    for i, paper in enumerate(scored, 1):
        s = paper['scores']
        badges = []
        
        # Signal strength badge (NEW!)
        if s['signal_strength'] >= 70:
            badges.append(f"💎{s['signal_strength']:.0f}")
        elif s['signal_strength'] >= 50:
            badges.append(f"⭐{s['signal_strength']:.0f}")
        else:
            badges.append(f"{s['signal_strength']:.0f}")
        
        # Confidence badge
        if s['confidence'] >= 80:
            badges.append(f"🎯{s['confidence']:.0f}%")
        elif s['confidence'] >= 65:
            badges.append(f"✓{s['confidence']:.0f}%")
        
        # Credibility badge
        if s['credibility'] >= 90:
            badges.append(f"🏅{s['credibility']:.0f}")
        
        # Virality badge
        if s['virality'] >= 50:
            badges.append(f"🔥{s['virality']:.0f}")
        
        badge_str = f" [{', '.join(badges)}]"
        
        print(f"{i}. Score: {s['final']:.3f}{badge_str}")
        print(f"   Sources: {', '.join(paper['sources'])}")
        print(f"   {paper['title'][:70]}...")
    
    # Show signal strength distribution
    strong = len([p for p in scored if p['scores']['signal_strength'] >= 60])
    medium = len([p for p in scored if 45 <= p['scores']['signal_strength'] < 60])
    weak = len([p for p in scored if p['scores']['signal_strength'] < 45])
    
    print(f"\n📊 SIGNAL STRENGTH DISTRIBUTION:")
    print(f"- Strong (≥60): {strong} papers 💎")
    print(f"- Medium (45-59): {medium} papers ⭐")
    print(f"- Weak (<45): {weak} papers")
    
    return scored

ranked_papers = calculate_confidence_with_credibility(papers, cross_refs)

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
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    return embeddings

embeddings = generate_embeddings(ranked_papers)

# %%
# ============================================================
# Cell 9: Store with New Metrics
# ============================================================

def store_with_metrics(papers, hn_stories, hf_papers, embed_model, chroma_client):
    """Store all sources with virality + confidence metrics"""
    
    print("\n💾 Storing with new metrics...")
    
    # Get or create collections
    try:
        papers_collection = chroma_client.get_collection("arxiv_papers")
    except:
        papers_collection = chroma_client.create_collection("arxiv_papers")
    
    try:
        hn_collection = chroma_client.get_collection("hackernews_stories")
    except:
        hn_collection = chroma_client.create_collection("hackernews_stories")
    
    try:
        hf_collection = chroma_client.get_collection("huggingface_papers")
    except:
        hf_collection = chroma_client.create_collection("huggingface_papers")
    
    # Store papers with NEW metrics
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
                    'virality': p['scores']['virality'],  # NEW!
                    'confidence': p['scores']['confidence'],  # NEW!
                    'credibility': p['scores']['credibility'],      # NEW!
                    'signal_strength': p['scores']['signal_strength'],  # NEW!
                    'platforms': p['platforms'],
                    'triple_hit': p['triple_hit']
                }
                for p in papers
            ]
        )
        print(f"✅ Stored {len(papers)} papers with metrics | Total: {papers_collection.count()}")
    
    # Store HN
    if hn_stories:
        hn_texts = [s['title'] for s in hn_stories]
        hn_embeddings = embed_model.encode(hn_texts)
        
        hn_collection.upsert(
            ids=[s['id'] for s in hn_stories],
            embeddings=hn_embeddings.tolist(),
            documents=[s['title'] for s in hn_stories],
            metadatas=[{
                'title': s['title'],
                'url': s['url'],
                'score': s['score'],
                'comments': s['num_comments'],
                'created': s['created'].strftime('%Y-%m-%d')
            } for s in hn_stories]
        )
        print(f"✅ Stored {len(hn_stories)} HN stories | Total: {hn_collection.count()}")
    
    # Store HF
    if hf_papers:
        hf_texts = [p['title'] for p in hf_papers]
        hf_embeddings = embed_model.encode(hf_texts)
        
        hf_collection.upsert(
            ids=[str(p.get("id") or p["url"] or uuid.uuid4()) for p in hf_papers],
            embeddings=hf_embeddings.tolist(),
            documents=[p['title'] for p in hf_papers],
            metadatas=[{
                'title': p['title'],
                'url': p['url'],
                'upvotes': p['upvotes'],
                'featured_date': p['featured_date']
            } for p in hf_papers]
        )
        print(f"✅ Stored {len(hf_papers)} HF papers | Total: {hf_collection.count()}")

store_with_metrics(ranked_papers, hn_stories, hf_papers, embed_model, chroma_client)

# %%
# ============================================================
# Cell 10: Semantic Search with Signal Strength Filter (NEW!)
# ============================================================

def search_with_signal_strength(query, chroma_client, embed_model, min_signal=50):
    """Search and return best paper with ≥min_signal strength"""
    
    print(f"\n🔍 Searching with ≥{min_signal} signal strength filter...")
    
    query_embedding = embed_model.encode(query).tolist()
    
    papers_collection = chroma_client.get_collection("arxiv_papers")
    if papers_collection.count() == 0:
        print("⚠️ No papers in DB")
        return None
    
    # Get top 5, filter by signal strength
    results = papers_collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        signal = meta.get('signal_strength', 0)
        
        if signal >= min_signal:
            best_paper = {
                'id': results['ids'][0][i],
                'title': meta.get('title'),
                'summary': results['documents'][0][i],
                'url': meta.get('url'),
                'authors': meta.get('authors', []),
                'published': meta.get('published'),
                'virality': meta.get('virality', 0),
                'confidence': meta.get('confidence', 40),
                'credibility': meta.get('credibility', 100),
                'signal_strength': signal,
                'platforms': meta.get('platforms', 1),
                'triple_hit': meta.get('triple_hit', False)
            }
            
            print(f"✅ Selected: {best_paper['title'][:60]}...")
            print(f"   Signal Strength: {best_paper['signal_strength']:.0f}/100")
            print(f"   Confidence: {best_paper['confidence']:.0f}%")
            print(f"   Credibility: {best_paper['credibility']:.0f}/100")
            print(f"   Platforms: {best_paper['platforms']}/3")
            
            return best_paper
    
    # If none meet threshold, return best with warning
    print(f"⚠️ No papers ≥{min_signal} signal strength, returning top match")
    meta = results['metadatas'][0][0]
    return {
        'id': results['ids'][0][0],
        'title': meta.get('title'),
        'summary': results['documents'][0][0],
        'url': meta.get('url'),
        'authors': meta.get('authors', []),
        'published': meta.get('published'),
        'virality': meta.get('virality', 0),
        'confidence': meta.get('confidence', 40),
        'credibility': meta.get('credibility', 100),
        'signal_strength': meta.get('signal_strength', 40),
        'platforms': meta.get('platforms', 1),
        'triple_hit': meta.get('triple_hit', False)
    }

best_paper = search_with_signal_strength(
    config.DAILY_QUERY, 
    chroma_client, 
    embed_model, 
    min_signal=config.MIN_SIGNAL_STRENGTH
)

# %%
# ============================================================
# Cell 11: Generate Confidence-Aware Thread
# ============================================================

thread_template = """You are a calm, technical AI researcher.

Paper: {title}
Authors: {authors}
Summary: {summary}

{context}

Write exactly 3 tweets about this paper. Rules:
- Tweet 1: What problem this solves (under 250 chars)
- Tweet 2: Key technical insight (under 250 chars)
- Tweet 3: Why it matters (under 250 chars)
{instruction}
- Be clear and technical
- No buzzwords

Format:
Tweet 1: [text]
Tweet 2: [text]
Tweet 3: [text]

Now write:"""

# Build context based on confidence
context = ""
instruction = ""

if best_paper['triple_hit']:
    context = f"\n🏆 TRIPLE VALIDATION: On arXiv + HN + HF! ({best_paper['confidence']}% confidence)\n"
    instruction = "\n- Mention multi-platform validation"
elif best_paper['platforms'] >= 2:
    context = f"\n🎯 {best_paper['confidence']}% confidence: Validated on {best_paper['platforms']}/3 platforms\n"
    instruction = "\n- Note strong validation signals"
elif best_paper['confidence'] >= 60:
    context = f"\n✓ {best_paper['confidence']}% confidence signal\n"

if best_paper['virality'] >= 50:
    context += f"🔥 High virality: {best_paper['virality']:.0f}/100\n"

prompt = PromptTemplate(
    input_variables=["title", "authors", "summary", "context", "instruction"],
    template=thread_template
)

print("\n🤖 Generating confidence-aware thread...\n")

input_text = prompt.format(
    title=best_paper['title'],
    authors=best_paper['authors'],
    summary=best_paper['summary'],
    context=context,
    instruction=instruction
)

start_time = time.time()

for attempt in range(5):
    try:
        thread = llm.invoke(input_text)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(5)
else:
    print("LLM invoke failed")
    thread = "Error generating thread"

generation_time = time.time() - start_time

print("="*60)
print(thread)
print("="*60)
print(f"\n⏱️  Generated in {generation_time:.1f}s")

# %%
# ============================================================
# Cell 12: Save Thread with Confidence Metrics
# ============================================================

def save_thread(paper, context, thread_content, gen_time, cross_refs, day=5):
    """Save thread with virality + confidence metrics"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.THREADS_DIR, f"day{day:02d}_{timestamp}.md")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Day {day} Thread - Confidence Scoring\n\n")
        f.write(f"**Paper:** {paper['title']}\n")
        f.write(f"**Authors:** {paper['authors']}\n")
        f.write(f"**URL:** {paper['url']}\n\n")
        
        f.write(f"## Quality Metrics (NEW!)\n\n")
        f.write(f"- **Confidence:** {paper['confidence']}% ")
        if paper['confidence'] >= 80:
            f.write("🎯 (High)\n")
        elif paper['confidence'] >= 65:
            f.write("✓ (Medium)\n")
        else:
            f.write("(Low)\n")
        
        f.write(f"- **Virality:** {paper['virality']:.0f}/100 ")
        if paper['virality'] >= 50:
            f.write("🔥 (High engagement)\n")
        elif paper['virality'] >= 20:
            f.write("📈 (Moderate)\n")
        else:
            f.write("(Low)\n")
        
        f.write(f"- **Platforms:** {paper['platforms']}/3 ")
        if paper['triple_hit']:
            f.write("🏆 (Triple hit!)\n")
        elif paper['platforms'] >= 2:
            f.write("(Multi-platform)\n")
        else:
            f.write("(Single source)\n")
        
        f.write(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Generation Time:** {gen_time:.1f}s\n\n")
        
        if context:
            f.write("## Context\n\n")
            f.write(context)
            f.write("\n")
        
        f.write("---\n\n")
        f.write(thread_content)
        f.write("\n\n---\n\n")
        
        f.write(f"*Day {day} - Clair Agent*\n")
        f.write("*Sources: arXiv + HN + HF (3 sources)*\n")
        f.write(f"*Cross-refs: {len(cross_refs['arxiv_hn'])} HN, {len(cross_refs['arxiv_hf'])} HF*\n")
        f.write(f"*Triple hits: {len(cross_refs['triple_hits'])}*\n")
        f.write("*NEW: Virality + Confidence scoring*")
    
    return filename

filename = save_thread(best_paper, context, thread, generation_time, cross_refs, day=5)
print(f"\n💾 Saved: {filename}")

# %%
# ============================================================
# Cell 13: Debug - Show All Papers by Confidence
# ============================================================

print("\n📊 ALL PAPERS BY CONFIDENCE:")
print("="*60)

for i, p in enumerate(ranked_papers, 1):
    s = p['scores']
    badges = []
    
    if s['confidence'] >= 65:
        badges.append("🎯")
    if s['virality'] >= 20:
        badges.append("🔥")
    if p['triple_hit']:
        badges.append("🏆")
    
    badge_str = "".join(badges) if badges else "  "
    
    print(f"{i}. {badge_str} Conf: {s['confidence']:2d}% | Viral: {s['virality']:3.0f} | Platforms: {p['platforms']}/3")
    print(f"   {p['title'][:70]}...")
    print()

print("="*60)
print(f"\n💡 INSIGHT:")
high = [p for p in ranked_papers if p['scores']['confidence'] >= 65]
med = [p for p in ranked_papers if 60 <= p['scores']['confidence'] < 65]
low = [p for p in ranked_papers if p['scores']['confidence'] < 60]

print(f"High confidence (≥65%): {len(high)}")
print(f"Medium confidence (60-64%): {len(med)}")
print(f"Low confidence (<60%): {len(low)}")

# %%
# ============================================================
# Cell 14: Week 1 Complete Summary
# ============================================================

print("\n" + "="*60)
print("🎉 DAY 5 COMPLETE - WEEK 1 DONE!")
print("="*60)

print(f"\n📊 TODAY'S DATA:")
print(f"- arXiv papers: {len(papers)}")
print(f"- HN stories: {len(hn_stories)}")
print(f"- HF papers: {len(hf_papers)}")
print(f"- Total sources: 3 (arXiv, HN, HF)")

print(f"\n🔗 CROSS-REFERENCES:")
print(f"- arXiv ↔ HN: {len(cross_refs['arxiv_hn'])}")
print(f"- arXiv ↔ HF: {len(cross_refs['arxiv_hf'])}")
print(f"- 🏆 Triple hits: {len(cross_refs['triple_hits'])}")

print(f"\n🎯 SELECTED PAPER:")
print(f"- Title: {best_paper['title'][:60]}...")
print(f"- Confidence: {best_paper['confidence']}%")
print(f"- Virality: {best_paper['virality']:.0f}/100")
print(f"- Platforms: {best_paper['platforms']}/3")

papers_coll = chroma_client.get_collection("arxiv_papers")
hn_coll = chroma_client.get_collection("hackernews_stories")
hf_coll = chroma_client.get_collection("huggingface_papers")

print(f"\n💾 CHROMADB:")
print(f"- arXiv papers: {papers_coll.count()}")
print(f"- HN stories: {hn_coll.count()}")
print(f"- HF papers: {hf_coll.count()}")
print(f"- Total entries: {papers_coll.count() + hn_coll.count() + hf_coll.count()}")

print(f"\n🆕 NEW THIS WEEK:")
print("✅ Virality scoring (0-100)")
print("✅ Confidence scoring (0-100%)")
print("✅ Multi-platform validation")
print("✅ Quality thresholds (≥60% confidence)")

print(f"\n📈 WEEK 1 COMPLETE TRAJECTORY:")
print("✅ Day 1: Single paper → local LLM")
print("✅ Day 2: 5 papers → ranking → RAG")
print("✅ Day 3: + HN → social signals")
print("✅ Day 4: + HF → human curation")
print("✅ Day 5: Virality + Confidence metrics")

# Show confidence distribution
high_conf = [p for p in ranked_papers if p['scores']['confidence'] >= 65]
medium_conf = [p for p in ranked_papers if 60 <= p['scores']['confidence'] < 65]
low_conf = [p for p in ranked_papers if p['scores']['confidence'] < 60]

print(f"\n📊 CONFIDENCE DISTRIBUTION:")
print(f"- High (≥65%): {len(high_conf)} papers")
print(f"- Medium (60-64%): {len(medium_conf)} papers")
print(f"- Low (<60%): {len(low_conf)} papers")

print(f"\n💰 COST: $0.00")
print("✨ 3 sources, zero auth, production-grade metrics")

print(f"\n🎯 SELECTED PAPER:")
print(f"- Title: {best_paper['title'][:60]}...")
print(f"- Signal Strength: {best_paper['signal_strength']:.0f}/100 💎")
print(f"- Confidence: {best_paper['confidence']:.0f}%")
print(f"- Source Credibility: {best_paper['credibility']:.0f}/100")
print(f"- Virality: {best_paper['virality']:.0f}/100")
print(f"- Platforms: {best_paper['platforms']}/3")

print(f"\n🆕 DAY 6 IMPROVEMENTS:")
print("✅ Source credibility weighting")
print("✅ Signal strength metric (confidence × credibility)")
print("✅ Smart filtering by source quality")
print("✅ arXiv+HF now ranks higher than arXiv+HN")