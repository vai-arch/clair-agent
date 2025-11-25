# %% [markdown]
# Day 2 - Clair Agent
# Multi-paper ranking + embeddings + ChromaDB + semantic search

# %%
# ============================================================
# Cell 1: Setup & Imports
# ============================================================

import sys
import os
from datetime import datetime, timedelta
import time

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
print(f"🦙 Using: {config.LLM_MODEL}")
print(f"📊 Fetching: {config.MAX_PAPERS_PER_DAY} papers")

# %%
# ============================================================
# Cell 2: Initialize Models
# ============================================================

print("🔧 Initializing models...\n")

# LLM
llm = OllamaLLM(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS
)

# Embedding model
embed_model = SentenceTransformer(config.EMBED_MODEL)

CHROMA_DB_PATH = config.CHROMA_DIR
# ChromaDB client
chroma_client = PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(
        anonymized_telemetry=False
    )
)
# Get or create collection
try:
    collection = chroma_client.get_collection("arxiv_papers")
    print(f"📊 Loaded existing collection: {collection.count()} papers")
except:
    collection = chroma_client.create_collection(
        name="arxiv_papers",
        metadata={"description": "AI/ML papers from arXiv"}
    )
    print("📊 Created new collection")

print("✅ Models initialized")

# %%
# ============================================================
# Cell 3: Fetch Multiple Papers
# ============================================================

def fetch_papers(max_results=5):
    """Fetch recent AI/ML papers from arXiv"""
    
    print(f"\n🔍 Searching arXiv for {max_results} papers...")
    
    client = arxiv.Client()
    
    # Build query from configured categories
    query = " OR ".join([f"cat:{cat}" for cat in config.ARXIV_CATEGORIES])
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for paper in client.results(search):
        papers.append({
            'id': paper.entry_id.split('/')[-1],  # Extract ID
            'title': paper.title,
            'authors': [a.name for a in paper.authors],
            'summary': paper.summary,
            'url': paper.entry_id,
            'published': paper.published,
            'categories': paper.categories,
            'primary_category': paper.primary_category
        })
    
    print(f"✅ Fetched {len(papers)} papers")
    return papers

papers = fetch_papers(config.MAX_PAPERS_PER_DAY)

# Display summary
for i, p in enumerate(papers, 1):
    print(f"\n{i}. {p['title'][:60]}...")
    print(f"   Authors: {len(p['authors'])} | Published: {p['published'].strftime('%Y-%m-%d')}")

# %%
# ============================================================
# Cell 4: Rank Papers
# ============================================================

def rank_papers(papers):
    """
    Rank papers by multiple criteria
    
    Scoring:
    - Recency: How recent is it? (0-1, 1=today)
    - Authors: More authors = more collaboration = higher quality? (0-1)
    - Relevance: How well does category match our focus? (0-1)
    """
    
    print("\n📊 Ranking papers...")
    
    now = datetime.now(papers[0]['published'].tzinfo)  # Match timezone
    
    ranked = []
    for paper in papers:
        # Recency score (1.0 = today, decays over 30 days)
        days_old = (now - paper['published']).days
        recency_score = max(0, 1 - (days_old / 30))
        
        # Author score (normalized by max authors in dataset)
        max_authors = max(len(p['authors']) for p in papers)
        author_score = len(paper['authors']) / max_authors
        
        # Relevance score (is it in our primary categories?)
        primary_cat = paper['primary_category']
        if primary_cat in config.ARXIV_CATEGORIES:
            relevance_score = 1.0
        elif any(cat in config.ARXIV_CATEGORIES for cat in paper['categories']):
            relevance_score = 0.7
        else:
            relevance_score = 0.3
        
        # Weighted total
        total_score = (
            recency_score * config.RANK_WEIGHTS['recency'] +
            author_score * config.RANK_WEIGHTS['authors'] +
            relevance_score * config.RANK_WEIGHTS['relevance']
        )
        
        ranked.append({
            **paper,
            'scores': {
                'recency': recency_score,
                'authors': author_score,
                'relevance': relevance_score,
                'total': total_score
            }
        })
    
    # Sort by total score
    ranked.sort(key=lambda x: x['scores']['total'], reverse=True)
    
    print("✅ Ranking complete\n")
    
    # Display rankings
    for i, paper in enumerate(ranked, 1):
        scores = paper['scores']
        print(f"{i}. Score: {scores['total']:.3f} | {paper['title'][:50]}...")
        print(f"   Recency: {scores['recency']:.2f} | Authors: {scores['authors']:.2f} | Relevance: {scores['relevance']:.2f}")
    
    return ranked

ranked_papers = rank_papers(papers)

# %%
# ============================================================
# Cell 5: Generate Embeddings
# ============================================================

def generate_embeddings(papers):
    """Generate embeddings for paper summaries"""
    
    print("\n🧮 Generating embeddings...")
    
    # Create text to embed (title + summary for richer context)
    texts = [
        f"{p['title']}. {p['summary'][:config.SUMMARY_TRUNCATE]}"
        for p in papers
    ]
    
    # Batch encode
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    
    print(f"✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    
    return embeddings

embeddings = generate_embeddings(ranked_papers)

# %%
# ============================================================
# Cell 6: Store in ChromaDB
# ============================================================

def store_papers(papers, embeddings, collection):
    """Store papers with embeddings in ChromaDB"""
    
    print("\n💾 Storing in ChromaDB...")
    
    # Prepare data
    ids = [p['id'] for p in papers]
    documents = [p['summary'][:config.SUMMARY_TRUNCATE] for p in papers]
    metadatas = [
        {
            'title': p['title'],
            'authors': ', '.join(p['authors'][:3]),  # First 3 authors
            'url': p['url'],
            'published': p['published'].strftime('%Y-%m-%d'),
            'primary_category': p['primary_category'],
            'rank_score': p['scores']['total']
        }
        for p in papers
    ]
    
    # Add to collection (upsert = update if exists, insert if not)
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Stored {len(papers)} papers")
    print(f"📊 Total papers in DB: {collection.count()}")

store_papers(ranked_papers, embeddings, collection)

# %%
# ============================================================
# Cell 7: Semantic Search
# ============================================================

def semantic_search(query, collection, top_k=1):
    """
    Find most relevant paper using semantic search
    
    Args:
        query: Natural language query
        collection: ChromaDB collection
        top_k: Number of results to return
    """
    
    print(f"\n🔍 Semantic search: '{query}'")
    
    # Embed the query
    query_embedding = embed_model.encode(query)
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Extract best result
    if results['ids'] and len(results['ids'][0]) > 0:
        best_match = {
            'id': results['ids'][0][0],
            'title': results['metadatas'][0][0]['title'],
            'summary': results['documents'][0][0],
            'url': results['metadatas'][0][0]['url'],
            'authors': results['metadatas'][0][0]['authors'],
            'published': results['metadatas'][0][0]['published'],
            'distance': results['distances'][0][0] if 'distances' in results else None
        }
        
        print(f"✅ Best match: {best_match['title'][:60]}...")
        return best_match
    else:
        print("❌ No results found")
        return None

# Search for best paper
best_paper = semantic_search(config.DAILY_QUERY, collection, top_k=1)

# %%
# ============================================================
# Cell 8: Generate Thread
# ============================================================

thread_template = """You are a calm, technical AI researcher explaining papers clearly.

Paper: {title}
Authors: {authors}
Summary: {summary}

Write exactly 3 tweets about this paper. Rules:
- Tweet 1: What problem this solves (under 250 chars)
- Tweet 2: Key technical insight (under 250 chars) 
- Tweet 3: Why it matters (under 250 chars)
- Be clear and technical, not hype
- No buzzwords like "revolutionary" or "game-changing"

Format your response EXACTLY like this:
Tweet 1: [your text]
Tweet 2: [your text]
Tweet 3: [your text]

Now write the 3 tweets:"""

prompt = PromptTemplate(
    input_variables=["title", "authors", "summary"],
    template=thread_template
)

print("\n🤖 Generating thread...\n")

input_text = prompt.format(
    title=best_paper['title'],
    authors=best_paper['authors'],
    summary=best_paper['summary']
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
# Cell 9: Save Thread
# ============================================================

def save_thread(paper_data, thread_content, gen_time, day=2):
    """Save thread as markdown"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.THREADS_DIR, f"day{day:02d}_{timestamp}.md")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Day {day} Thread\n\n")
        f.write(f"**Paper:** {paper_data['title']}\n")
        f.write(f"**Authors:** {paper_data['authors']}\n")
        f.write(f"**Published:** {paper_data['published']}\n")
        f.write(f"**URL:** {paper_data['url']}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Generation Time:** {gen_time:.1f}s\n\n")
        f.write("---\n\n")
        f.write(thread_content)
        f.write("\n\n---\n")
        f.write(f"*Generated by Clair Agent - Day {day}*\n")
        f.write("*Stack: Ollama + LangChain + ChromaDB + semantic search*\n")
        f.write(f"*Selected from {config.MAX_PAPERS_PER_DAY} papers via ranking + embedding similarity*")
    
    return filename

filename = save_thread(best_paper, thread, generation_time, day=2)
print(f"\n💾 Thread saved to: {filename}")

# %%
# ============================================================
# Cell 10: Summary & Stats
# ============================================================

print("\n" + "="*60)
print("🎉 DAY 2 COMPLETE - MULTI-PAPER PIPELINE")
print("="*60)

print(f"\n✅ Papers fetched: {len(papers)}")
print(f"✅ Papers ranked by {len(config.RANK_WEIGHTS)} criteria")
print(f"✅ Embeddings generated: {len(embeddings)}")
print(f"✅ Papers in ChromaDB: {collection.count()}")
print(f"✅ Semantic search complete")
print(f"✅ Thread generated in {generation_time:.1f}s")
print(f"✅ Saved to: {filename}")

print("\n📊 TOP 3 PAPERS TODAY:")
for i, paper in enumerate(ranked_papers[:3], 1):
    print(f"{i}. [{paper['scores']['total']:.3f}] {paper['title'][:50]}...")

print("\n🎯 SELECTED PAPER:")
print(f"Title: {best_paper['title'][:60]}...")
print(f"Method: Semantic search")
print(f"Query: '{config.DAILY_QUERY}'")

print("\n💰 COST:")
print("- Today: $0.00")
print("- Forever: $0.00")

print("\n📋 TODO NOW:")
print("1. Read thread in threads/day02_*.md")
print("2. Post to X manually")
print("3. Build-in-public update")
print("4. Commit to GitHub")

print("\n🔮 TOMORROW (Day 3):")
print("- Add Reddit r/MachineLearning scraping")
print("- Multi-source fusion (arXiv + Reddit)")
print("- Source diversity scoring")
print("- Cross-reference detection")

print(f"\n⏱️  Total time today: ~60-90 minutes")
print("💪 You now have real RAG infrastructure.")
