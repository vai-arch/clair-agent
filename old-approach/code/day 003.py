# %% [markdown]
# Day 3 - Clair Agent
# Multi-source fusion: arXiv + Hacker News (no auth needed!)

# %%
# ============================================================
# Cell 1: Setup & Imports
# ============================================================

import sys
import os
from datetime import datetime, timedelta
import time
import requests

# Use __file__ for scripts, or Path of notebook if in Jupyter
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ does not exist in Jupyter, fall back to notebook path
    base_dir = os.getcwd()

# Add parent folder
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
print(f"📊 Sources: arXiv ({config.MAX_PAPERS_PER_DAY}) + HN ({config.MAX_HN_STORIES})")

# %%
# ============================================================
# Cell 2: Initialize Models
# ============================================================

print("🔧 Initializing...\n")

# LLM
llm = OllamaLLM(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS
)

# Embedding model
embed_model = SentenceTransformer(config.EMBED_MODEL)

# ChromaDB
CHROMA_DB_PATH = config.CHROMA_DIR
chroma_client = PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(
        anonymized_telemetry=False
    )
)

print("✅ All models initialized")

# %%
# ============================================================
# Cell 3: Fetch arXiv Papers (Same as Day 2)
# ============================================================
import random
from typing import List, Dict

def fetch_papers(max_results=5, retries=5, base_delay=2) -> List[Dict]:
    """
    Fetch recent AI/ML papers from arXiv with retry logic for rate limiting (429)
    and service errors (503).
    """

    print(f"\n🔍 Searching arXiv for up to {max_results} papers...")

    query = " OR ".join([f"cat:{cat}" for cat in config.ARXIV_CATEGORIES])
    search = arxiv.Search(
        query=query,
        max_results=max_results,
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
            # retryable errors
            if e.status in (429, 503):
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ arXiv error {e.status}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                attempt += 1
                continue

            # unknown / non-retryable error → raise immediately
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
# Cell 4: Fetch Hacker News Stories (NEW - NO AUTH!)
# ============================================================

def fetch_hacker_news_stories(max_stories=10):
    """
    Fetch AI/ML-related stories from Hacker News
    
    Uses Algolia HN Search API (no authentication needed)
    """
    
    print(f"\n🔍 Searching Hacker News for {max_stories} stories...")
    
    stories = []
    
    # Search for AI/ML related stories
    search_queries = [
        "artificial intelligence",
        "machine learning",
        "large language model",
        "LLM",
        "transformer"
    ]
    
    for query in search_queries[:2]:  # Just 2 queries to stay fast
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
                # Avoid duplicates
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
            print(f"⚠️  Error fetching HN for '{query}': {e}")
        
        if len(stories) >= max_stories:
            break
    
    # Sort by score (most upvoted first)
    stories.sort(key=lambda x: x['score'], reverse=True)
    stories = stories[:max_stories]
    
    print(f"✅ Fetched {len(stories)} HN stories")
    return stories

hn_stories = fetch_hacker_news_stories(config.MAX_HN_STORIES)

for i, story in enumerate(hn_stories[:5], 1):
    print(f"{i}. [{story['score']:3d}↑] {story['title'][:60]}...")

# %%
# ============================================================
# Cell 5: Cross-Reference Detection (NEW)
# ============================================================

def find_cross_references(papers, hn_stories):
    """
    Detect when HN stories mention arXiv papers
    
    Methods:
    1. arXiv ID in HN URL (e.g., arxiv.org/abs/2311.12345)
    2. Title overlap (>50% words match)
    """
    
    print("\n🔗 Detecting cross-references...")
    
    cross_refs = []
    
    for paper in papers:
        paper_id = paper['id']
        paper_title_words = set(paper['title'].lower().split())
        
        for story in hn_stories:
            # Check 1: arXiv ID in HN URL
            story_url = story['url'].lower()
            story_title = story['title'].lower()
            
            if paper_id in story_url or 'arxiv.org' in story_url and paper_id.split('v')[0] in story_url:
                cross_refs.append({
                    'paper_id': paper_id,
                    'paper_title': paper['title'],
                    'hn_story_id': story['id'],
                    'hn_title': story['title'],
                    'hn_score': story['score'],
                    'hn_comments': story['num_comments'],
                    'match_type': 'arxiv_url'
                })
                continue
            
            # Check 2: Significant title overlap
            story_title_words = set(story_title.split())
            overlap = len(paper_title_words & story_title_words)
            
            if overlap >= len(paper_title_words) * 0.5 and len(paper_title_words) > 3:
                cross_refs.append({
                    'paper_id': paper_id,
                    'paper_title': paper['title'],
                    'hn_story_id': story['id'],
                    'hn_title': story['title'],
                    'hn_score': story['score'],
                    'hn_comments': story['num_comments'],
                    'match_type': 'title_overlap',
                    'overlap_ratio': overlap / len(paper_title_words)
                })
    
    print(f"✅ Found {len(cross_refs)} cross-references")
    
    if cross_refs:
        for ref in cross_refs:
            print(f"   Paper: {ref['paper_title'][:40]}...")
            print(f"   → HN ({ref['hn_score']}↑, {ref['hn_comments']} comments): {ref['hn_title'][:40]}...")
    else:
        print("   (No cross-references today - this is normal!)")
    
    return cross_refs

cross_refs = find_cross_references(papers, hn_stories)

# %%
# ============================================================
# Cell 6: Multi-Source Ranking with HN Signal (ENHANCED)
# ============================================================

def rank_with_social_signal(papers, cross_refs):
    """
    Rank papers with HN mentions as social signal
    
    Formula: base_score × (1 + hn_boost)
    HN boost = min(0.5, (points + comments/10) / 500)
    """
    
    print("\n📊 Ranking with social signals...")
    
    # Build cross-ref lookup
    hn_mentions = {}
    for ref in cross_refs:
        paper_id = ref['paper_id']
        if paper_id not in hn_mentions:
            hn_mentions[paper_id] = []
        hn_mentions[paper_id].append({
            'score': ref['hn_score'],
            'comments': ref['hn_comments']
        })
    
    now = datetime.now(papers[0]['published'].tzinfo)
    
    ranked = []
    for paper in papers:
        # Base scoring (same as Day 2)
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
        
        # NEW: HN social signal boost
        hn_boost = 0.0
        total_hn_engagement = 0
        if paper['id'] in hn_mentions:
            for mention in hn_mentions[paper['id']]:
                # Weight: upvotes + (comments / 10) for engagement
                total_hn_engagement += mention['score'] + (mention['comments'] / 10)
            
            # Boost by 0-50% based on HN engagement
            # 500 points = max boost
            hn_boost = min(0.5, total_hn_engagement / 500)
        
        final_score = base_score * (1 + hn_boost)
        
        ranked.append({
            **paper,
            'scores': {
                'recency': recency_score,
                'authors': author_score,
                'relevance': relevance_score,
                'base': base_score,
                'hn_boost': hn_boost,
                'final': final_score
            },
            'hn_mentions': len(hn_mentions.get(paper['id'], [])),
            'hn_engagement': total_hn_engagement
        })
    
    ranked.sort(key=lambda x: x['scores']['final'], reverse=True)
    
    print("✅ Ranking complete\n")
    
    for i, paper in enumerate(ranked, 1):
        s = paper['scores']
        mentions = f" 🔥 HN: {paper['hn_engagement']:.0f} pts" if paper['hn_mentions'] > 0 else ""
        print(f"{i}. Score: {s['final']:.3f} (base: {s['base']:.3f}, boost: {s['hn_boost']:.2f}){mentions}")
        print(f"   {paper['title'][:70]}...")
    
    return ranked

ranked_papers = rank_with_social_signal(papers, cross_refs)

# %%
# ============================================================
# Cell 7: Generate Embeddings (Same as Day 2)
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
# Cell 8: Store Both Sources in ChromaDB (ENHANCED)
# ============================================================

def store_multi_source(papers, hn_stories, embed_model, chroma_client):
    """Store both arXiv and HN in separate collections"""
    
    print("\n💾 Storing multi-source data...")
    
    # Collection 1: arXiv papers
    try:
        papers_collection = chroma_client.get_collection("arxiv_papers")
    except:
        papers_collection = chroma_client.create_collection("arxiv_papers")
    
    # Collection 2: Hacker News stories (NEW)
    try:
        hn_collection = chroma_client.get_collection("hackernews_stories")
    except:
        hn_collection = chroma_client.create_collection("hackernews_stories")
    
    # Store papers
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
                    'hn_mentions': p['hn_mentions']
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

store_multi_source(ranked_papers, hn_stories, embed_model, chroma_client)

# %%
# ============================================================
# Cell 9: Multi-Collection Semantic Search (ENHANCED)
# ============================================================

def search_across_sources(query, chroma_client, embed_model, top_k=3):
    """
    Search both arXiv and HN collections
    Return best paper + relevant HN context
    """
    
    print(f"\n🔍 Multi-source search: '{query}'")
    
    query_embedding = embed_model.encode(query).tolist()
    
    # Search papers
    papers_collection = chroma_client.get_collection("arxiv_papers")
    paper_results = papers_collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    
    # Search HN
    hn_collection = chroma_client.get_collection("hackernews_stories")
    hn_results = hn_collection.query(
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
            'hn_mentions': meta.get('hn_mentions', 0),  # <-- SAFE DEFAULT
        }

        print(f"✅ Best paper: {best_paper['title'][:60]}...")
        print(f"   HN mentions: {best_paper['hn_mentions']}")
    
    # Relevant HN stories
    relevant_hn = []
    if hn_results['ids'] and len(hn_results['ids'][0]) > 0:
        for i in range(len(hn_results['ids'][0])):
            relevant_hn.append({
                'title': hn_results['metadatas'][0][i]['title'],
                'score': hn_results['metadatas'][0][i]['score'],
                'comments': hn_results['metadatas'][0][i]['comments'],
                'url': hn_results['metadatas'][0][i]['hn_url']
            })
        
        print(f"✅ Relevant HN context: {len(relevant_hn)} stories")
        for story in relevant_hn:
            print(f"   [{story['score']:3d}↑, {story['comments']} comments] {story['title'][:50]}...")
    
    return best_paper, relevant_hn

best_paper, relevant_hn = search_across_sources(
    config.DAILY_QUERY,
    chroma_client,
    embed_model
)

# %%
# ============================================================
# Cell 10: Generate Multi-Source Thread (ENHANCED)
# ============================================================

thread_template = """You are a calm, technical AI researcher explaining papers clearly.

Paper: {title}
Authors: {authors}
Summary: {summary}

{hn_context}

Write exactly 3 tweets about this paper. Rules:
- Tweet 1: What problem this solves (under 250 chars)
- Tweet 2: Key technical insight (under 250 chars)
- Tweet 3: Why it matters (under 250 chars)
{hn_instruction}
- Be clear and technical, not hype
- No buzzwords like "revolutionary" or "game-changing"

Format your response EXACTLY like this:
Tweet 1: [your text]
Tweet 2: [your text]
Tweet 3: [your text]

Now write the 3 tweets:"""

# Build HN context
hn_context = ""
hn_instruction = ""

if best_paper['hn_mentions'] > 0 and relevant_hn:
    total_engagement = sum(s['score'] + s['comments'] for s in relevant_hn[:2])
    hn_context = f"\nHacker News Discussion: This topic has {total_engagement:.0f}+ points/comments on HN:\n"
    for story in relevant_hn[:2]:
        hn_context += f"- [{story['score']}↑, {story['comments']} comments] {story['title'][:60]}...\n"
    hn_instruction = "\n- Mention HN discussion if relevant"

prompt = PromptTemplate(
    input_variables=["title", "authors", "summary", "hn_context", "hn_instruction"],
    template=thread_template
)

print("\n🤖 Generating multi-source thread...\n")

input_text = prompt.format(
    title=best_paper['title'],
    authors=best_paper['authors'],
    summary=best_paper['summary'],
    hn_context=hn_context,
    hn_instruction=hn_instruction
)

start_time = time.time()
thread = llm.invoke(input_text)
generation_time = time.time() - start_time

print("="*60)
print(thread)
print("="*60)
print(f"\n⏱️  Generated in {generation_time:.1f}s")

# %%
# ============================================================
# Cell 11: Save Thread with Source Attribution (ENHANCED)
# ============================================================

def save_thread(paper, hn_context, thread_content, gen_time, day=3):
    """Save thread with multi-source attribution"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.THREADS_DIR, f"day{day:02d}_{timestamp}.md")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Day {day} Thread - Multi-Source (arXiv + HN)\n\n")
        f.write(f"**Paper:** {paper['title']}\n")
        f.write(f"**Authors:** {paper['authors']}\n")
        f.write(f"**Published:** {paper['published']}\n")
        f.write(f"**URL:** {paper['url']}\n")
        f.write(f"**HN Mentions:** {paper['hn_mentions']}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Generation Time:** {gen_time:.1f}s\n\n")
        
        if hn_context:
            f.write("## Hacker News Context\n\n")
            f.write(hn_context)
            f.write("\n")
        
        f.write("---\n\n")
        f.write(thread_content)
        f.write("\n\n---\n")
        f.write(f"*Generated by Clair Agent - Day {day}*\n")
        f.write("*Stack: Ollama + LangChain + ChromaDB + Hacker News API*\n")
        f.write(f"*Sources: arXiv + HN | Cross-references: {len(cross_refs)}*")
    
    return filename

filename = save_thread(best_paper, hn_context, thread, generation_time, day=3)
print(f"\n💾 Thread saved to: {filename}")

# %%
# ============================================================
# Cell 12: Summary & Stats (ENHANCED)
# ============================================================

print("\n" + "="*60)
print("🎉 DAY 3 COMPLETE - MULTI-SOURCE FUSION (arXiv + HN)")
print("="*60)

print(f"\n✅ arXiv papers: {len(papers)}")
print(f"✅ HN stories: {len(hn_stories)}")
print(f"✅ Cross-references: {len(cross_refs)}")
print(f"✅ Papers with HN mentions: {sum(1 for p in ranked_papers if p['hn_mentions'] > 0)}")
print(f"✅ Thread generated in {generation_time:.1f}s")

print("\n📊 DATA STORED:")
papers_coll = chroma_client.get_collection("arxiv_papers")
hn_coll = chroma_client.get_collection("hackernews_stories")
print(f"- arXiv papers in DB: {papers_coll.count()}")
print(f"- HN stories in DB: {hn_coll.count()}")

print("\n🎯 SELECTED PAPER:")
print(f"Title: {best_paper['title'][:60]}...")
print(f"HN mentions: {best_paper['hn_mentions']}")
print(f"Selection: Multi-source semantic search")

print("\n💰 COST: $0.00")
print("✨ BONUS: Zero authentication needed!")

print("\n🔮 TOMORROW (Day 4):")
print("- Add Hugging Face daily papers (3rd source)")
print("- 3-way cross-referencing")
print("- Human curation signal boost")
print("- Enhanced confidence scoring")

print(f"\n⏱️  Total time today: ~90 minutes")
print("💪 Multi-source intelligence with ZERO auth friction!")
