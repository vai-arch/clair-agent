# Clair Agent — 48 Weeks LLM Mastery Plan

**Start Date:** November 25, 2025  
**End Date:** November 18, 2026  
**Your Mentor:** Claude (that's me!)

---

## 🎯 Project Vision

Build **Clairvoyant (Clair) Agent**: an open-source agent that wakes up every morning, scans arXiv/X/HN/Reddit/HuggingFace, picks the 5 real trends, writes viral threads in your calm/classic voice, analyzes your posting history, and optionally auto-posts with one-click approval.

**But more importantly:** Become a true LLM expert who understands transformers, fine-tuning, alignment, inference optimization, and cutting-edge research from the ground up.

---

## 🧭 Core Philosophy

- **Ship to learn, don't learn to ship**
- LLM topics start Day 1, every week touches them
- Prioritize: understanding internals > perfect product
- Scraping/RAG/agents are servants to LLM mastery, not the main event
- Real experiments > polished UI
- Windows-native, 60–120 min/day, family-first, zero hype

---

## 📋 How to Use This Plan with Claude (Your Mentor)

### When you need BIG PICTURE guidance:
Say: **"Vibe Plan Phase X"**  
→ I'll give you strategy, motivation, potential pitfalls, emotional energy for that whole phase

### When you need HANDS-ON implementation:
Say: **"Vibe Code Week Y"** or **"Day Z"**  
→ I'll give you exact commands, file structure, code snippets, debugging tips, Windows-specific fixes

### When you're stuck:
Say: **"Debug [problem]"** or **"Explain [concept] like I'm 5"**  
→ I'll break it down in your calm, classic voice

### When you finish something:
Say: **"Ship Week X"**  
→ I'll help you write the thread, review your code, and prep for next week

---

## 🗓️ Daily Structure (60-120 min)

1. **20-30 min:** Read/watch material on today's LLM topic
2. **40-60 min:** Code + experiment (always ship something visible)
3. **10-20 min:** Write thread explaining what you learned
4. **5-10 min:** Commit + post on X

**Reality check:** Life happens. Family first. Some days you'll do 30 min, some days 0. The plan survives if you're honest about it.

---

## 📚 Phase 1 — Days 1–28 | Foundation: Transformers from Scratch + Local LLMs

**Goal by Day 28:** You understand how transformers actually work (attention, embeddings, tokenization) and can run/modify local LLMs confidently.

### Week 1 (Days 1–7): Hello Transformers

**Focus:** Build attention from NumPy, tokenization basics, load Llama 3.2 with transformers library

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 1 | What is attention? | Scaled dot-product attention in NumPy | "Attention is just matrix multiplication (here's why)" |
| 2 | Multi-head attention | Multi-head attention from scratch | "Why models need multiple attention heads" |
| 3 | Tokenization basics | BPE tokenizer exploration | "How LLMs see text (spoiler: not words)" |
| 4 | Positional encoding | Sinusoidal position embeddings | "Why order matters in transformers" |
| 5 | Load Llama 3.2 locally | Run inference with transformers library | "I just loaded a 3B param model on my laptop" |
| 6 | Model architecture | Inspect Llama's layers | "What's actually inside an LLM" |
| 7 | Week 1 integration | Mini pipeline: tokenize → attend → generate | Week 1 recap thread |

**Ship:** Working attention mechanism + 7 threads

---

### Week 2 (Days 8–14): Local LLM Mastery

**Focus:** Ollama vs llama.cpp vs transformers, GGUF formats, quantization (what's actually happening), context windows

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 8 | Ollama setup | Install & run models via Ollama | "Ollama makes local LLMs stupid easy" |
| 9 | llama.cpp deep dive | Compile & run llama.cpp on Windows | "Why llama.cpp is fast (C++ magic)" |
| 10 | GGUF format | Explore .gguf file structure | "What's in a GGUF file?" |
| 11 | Quantization math | Compare FP16 vs Q4_K_M outputs | "Quantization: 16GB → 4GB without breaking" |
| 12 | Context windows | Test 2K vs 8K vs 32K context | "Why context length matters (and costs)" |
| 13 | Speed benchmarks | Ollama vs llama.cpp vs transformers | "Which local LLM setup is fastest?" |
| 14 | Week 2 integration | Build a CLI tool to compare all 3 | Week 2 recap thread |

**Ship:** CLI comparison tool + 7 threads

---

### Week 3 (Days 15–21): Inference Deep Dive

**Focus:** KV cache, temperature/top-p/top-k math, sampling strategies, batching, why 3B vs 7B vs 70B matters

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 15 | KV cache explained | Visualize KV cache savings | "KV cache: why LLMs don't recompute everything" |
| 16 | Temperature math | Test temperature 0.1 → 2.0 | "Temperature controls creativity (here's the math)" |
| 17 | Top-p vs top-k | Implement both sampling methods | "Top-p vs top-k: which sampling wins?" |
| 18 | Sampling strategies | Beam search vs nucleus sampling | "How LLMs choose the next token" |
| 19 | Batching | Batch inference for 10 prompts | "Batching: 10x faster inference trick" |
| 20 | Model size tradeoffs | 3B vs 7B vs 70B comparison | "Why I don't always use the biggest model" |
| 21 | Week 3 integration | Custom inference engine with all controls | Week 3 recap thread |

**Ship:** Custom inference tool + 7 threads

---

### Week 4 (Days 22–28): First arXiv Thread Generator

**Focus:** Combine Week 1-3 knowledge: fetch papers, embed with SentenceTransformers, generate threads, analyze what the model is doing

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 22 | arXiv API | Fetch recent ML papers | "How to scrape arXiv like a pro" |
| 23 | SentenceTransformers | Embed paper abstracts | "Embeddings: turn text into math" |
| 24 | Similarity search | Find similar papers with cosine sim | "Finding related papers with vectors" |
| 25 | Thread generation | Generate thread from paper | "I built an AI that reads papers for me" |
| 26 | Prompt engineering | Improve thread quality with prompting | "Prompt engineering is just compiler design" |
| 27 | Output analysis | Analyze model attention on key sentences | "What the model focuses on when writing" |
| 28 | Week 4 integration | End-to-end arXiv → thread pipeline | Week 4 recap + Phase 1 summary |

**Ship:** Ugly but working thread generator + 7 threads + Phase 1 retrospective

---

## 📚 Phase 2 — Days 29–56 | Embeddings + RAG + Vector Math

**Goal by Day 56:** You deeply understand embeddings, similarity search, and why RAG works (and when it fails).

### Week 5 (Days 29–35): Embedding Models

**Focus:** How BERT/SentenceTransformers work, cosine vs dot product, embedding dimensions, normalization, when to use different models

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 29 | BERT architecture | Load & inspect BERT layers | "BERT: the model that changed NLP" |
| 30 | Sentence embeddings | Compare 3 embedding models | "Not all embedding models are equal" |
| 31 | Cosine vs dot product | Math + speed comparison | "Cosine similarity explained (with code)" |
| 32 | Embedding dimensions | 384 vs 768 vs 1024 tradeoffs | "Why embedding size matters" |
| 33 | Normalization | L2 norm and why it helps | "Normalize your embeddings (here's why)" |
| 34 | Model selection | When to use which model | "Choosing the right embedding model" |
| 35 | Week 5 integration | Embedding benchmark suite | Week 5 recap thread |

**Ship:** Embedding comparison tool + 7 threads

---

### Week 6 (Days 36–42): Vector Databases

**Focus:** ChromaDB vs FAISS vs Qdrant internals, HNSW algorithm, indexing strategies, metadata filtering

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 36 | ChromaDB setup | Store & query embeddings in Chroma | "ChromaDB: SQLite for vectors" |
| 37 | FAISS basics | Build FAISS index | "FAISS: Facebook's vector search" |
| 38 | Qdrant exploration | Deploy Qdrant locally | "Qdrant: the Postgres of vector DBs" |
| 39 | HNSW algorithm | Visualize HNSW graph | "HNSW: how vector search gets fast" |
| 40 | Indexing strategies | IVF vs HNSW vs flat index | "Vector index types compared" |
| 41 | Metadata filtering | Add metadata to vectors | "Filtering vectors with metadata" |
| 42 | Week 6 integration | Vector DB benchmark (speed + recall) | Week 6 recap thread |

**Ship:** Vector DB comparison + 7 threads

---

### Week 7 (Days 43–49): RAG Architectures

**Focus:** Naive RAG, reranking (why ColBERT matters), query decomposition, context window management

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 43 | Naive RAG | Simple retrieve → generate pipeline | "RAG in 50 lines of code" |
| 44 | Retrieval quality | Measure precision/recall | "Why most RAG systems suck" |
| 45 | Reranking | Add cross-encoder reranker | "Reranking: the secret to better RAG" |
| 46 | ColBERT | Implement ColBERT-style reranking | "ColBERT: attention-based reranking" |
| 47 | Query decomposition | Break complex queries into sub-queries | "Query decomposition for better retrieval" |
| 48 | Context management | Fit retrieved docs in context window | "Context window math for RAG" |
| 49 | Week 7 integration | Advanced RAG pipeline | Week 7 recap thread |

**Ship:** Advanced RAG system + 7 threads

---

### Week 8 (Days 50–56): Multi-Source RAG

**Focus:** Combine arXiv + Reddit + HN, deduplicate, rank by recency + relevance, understand retrieval-generation tradeoffs

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 50 | Reddit scraping | Fetch ML subreddit posts | "Scraping Reddit for AI trends" |
| 51 | HN scraping | Fetch Hacker News stories | "HN scraper in Python" |
| 52 | Multi-source ingestion | Unified pipeline for 3 sources | "Combining arXiv + Reddit + HN" |
| 53 | Deduplication | Near-duplicate detection with embeddings | "Deduplication with vector similarity" |
| 54 | Ranking algorithm | Recency + relevance scoring | "Ranking algorithm for multi-source RAG" |
| 55 | Retrieval-generation tradeoffs | When to retrieve vs when to generate | "RAG tradeoffs: retrieval vs generation" |
| 56 | Week 8 integration | Production multi-source RAG | Week 8 recap + Phase 2 summary |

**Ship:** Multi-source RAG system + 7 threads + Phase 2 retrospective

---

## 📚 Phase 3 — Days 57–84 | Prompt Engineering + Function Calling

**Goal by Day 84:** Master advanced prompting, understand how function calling works under the hood, build your first agent from scratch.

### Week 9 (Days 57–63): Prompt Engineering Deep Dive

**Focus:** Few-shot vs zero-shot, chain-of-thought, ReAct, DSPy basics, why prompting is compiler design

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 57 | Zero-shot prompting | Test zero-shot on 5 tasks | "Zero-shot prompting: when it works" |
| 58 | Few-shot learning | 1-shot vs 5-shot comparison | "Few-shot learning explained" |
| 59 | Chain-of-thought | Implement CoT prompting | "Chain-of-thought makes models smarter" |
| 60 | ReAct pattern | Build ReAct agent from scratch | "ReAct: reasoning + acting" |
| 61 | DSPy introduction | Basic DSPy program | "DSPy: prompting as programming" |
| 62 | Prompt optimization | A/B test 10 prompt variations | "How I optimized a prompt 10x" |
| 63 | Week 9 integration | Prompt engineering toolkit | Week 9 recap thread |

**Ship:** Prompt engineering library + 7 threads

---

### Week 10 (Days 64–70): Function Calling Internals

**Focus:** How tool use actually works, JSON mode, structured outputs, constrained decoding, grammar-based generation

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 64 | Function calling basics | Simple tool use with OpenAI-style API | "Function calling explained" |
| 65 | JSON mode | Force JSON outputs | "JSON mode: structured outputs guaranteed" |
| 66 | Structured outputs | Pydantic models for validation | "Structured outputs with Pydantic" |
| 67 | Constrained decoding | Implement grammar constraints | "Constrained decoding: control the output" |
| 68 | Grammar-based generation | Use GBNF grammars | "Grammar-based LLM generation" |
| 69 | Tool selection | Multi-tool agent | "How models choose which tool to use" |
| 70 | Week 10 integration | Function calling framework | Week 10 recap thread |

**Ship:** Function calling library + 7 threads

---

### Week 11 (Days 71–77): First Real Agent

**Focus:** LangGraph basics, state management, loops, memory, tool integration

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 71 | LangGraph setup | First LangGraph graph | "LangGraph: agents as graphs" |
| 72 | State management | Persistent state across steps | "Managing state in agents" |
| 73 | Loops & cycles | Build agent with cycles | "Agent loops: when to stop" |
| 74 | Memory integration | Add vector memory to agent | "Giving agents memory" |
| 75 | Tool integration | Connect 5 tools to agent | "Multi-tool agent architecture" |
| 76 | Error handling | Graceful failure & retries | "Making agents robust" |
| 77 | Week 11 integration | Production agent v1 | Week 11 recap thread |

**Ship:** Working agent + 7 threads

---

### Week 12 (Days 78–84): Agent Testing & Evaluation

**Focus:** How to measure agent performance, eval datasets, failure modes, debugging tools

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 78 | Eval frameworks | LangSmith for agent evals | "How to evaluate agents" |
| 79 | Eval datasets | Create 50 test cases | "Building agent eval datasets" |
| 80 | Success metrics | Define success criteria | "Metrics for agent performance" |
| 81 | Failure analysis | Categorize failure modes | "Why agents fail (and how to fix it)" |
| 82 | Debugging tools | Build agent debugger | "Debugging agents is hard" |
| 83 | Regression testing | Prevent regressions | "Regression testing for agents" |
| 84 | Week 12 integration | Full agent eval suite | Week 12 recap + Phase 3 summary |

**Ship:** Agent with eval suite + 7 threads + Phase 3 retrospective

---

## 📚 Phase 4 — Days 85–112 | Fine-Tuning Fundamentals

**Goal by Day 112:** Understand the math behind fine-tuning, run your first successful fine-tune on your writing style.

### Week 13 (Days 85–91): Fine-Tuning Theory

**Focus:** What actually changes when you fine-tune, LoRA/QLoRA math, why rank matters, where adapters go in the model

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 85 | Fine-tuning overview | What changes during fine-tuning | "Fine-tuning: what actually happens" |
| 86 | Full fine-tuning | Understand backprop through transformer | "Why full fine-tuning is expensive" |
| 87 | LoRA math | Implement LoRA from scratch | "LoRA: fine-tune with 0.1% parameters" |
| 88 | QLoRA explained | Quantization + LoRA | "QLoRA: fine-tune 70B on 24GB" |
| 89 | Rank parameter | Test rank 4 vs 8 vs 64 | "Why LoRA rank matters" |
| 90 | Adapter placement | Where adapters go in the model | "Where to put LoRA adapters" |
| 91 | Week 13 integration | LoRA implementation from scratch | Week 13 recap thread |

**Ship:** LoRA implementation + 7 threads

---

### Week 14 (Days 92–98): Dataset Curation

**Focus:** What makes good training data, format your tweets, data cleaning, train/val splits, overfitting signs

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 92 | Data collection | Export all your tweets | "How to export your Twitter data" |
| 93 | Data cleaning | Remove URLs, fix formatting | "Cleaning text data for fine-tuning" |
| 94 | Data formatting | Convert to instruction format | "Formatting data for fine-tuning" |
| 95 | Quality filtering | Remove low-quality tweets | "Quality > quantity in training data" |
| 96 | Train/val splits | 90/10 split + stratification | "Train/val splits for fine-tuning" |
| 97 | Data augmentation prep | Plan augmentation strategies | "Data augmentation for small datasets" |
| 98 | Week 14 integration | Clean, formatted dataset ready | Week 14 recap thread |

**Ship:** Clean training dataset + 7 threads

---

### Week 15 (Days 99–105): First Fine-Tune

**Focus:** Unsloth + Llama 3.2 8B on your voice, learning rates, batch sizes, gradient accumulation, loss curves

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 99 | Unsloth setup | Install & configure Unsloth | "Unsloth: 2x faster fine-tuning" |
| 100 | First training run | Train for 1 epoch | "My first fine-tune (it's running!)" |
| 101 | Learning rate tuning | Test 1e-5 to 1e-3 | "Learning rate: the most important HP" |
| 102 | Batch size experiments | Batch size vs memory tradeoffs | "Batch size tuning for fine-tuning" |
| 103 | Gradient accumulation | Simulate larger batches | "Gradient accumulation explained" |
| 104 | Loss curves | Analyze training loss | "Reading loss curves like a pro" |
| 105 | Week 15 integration | Optimal fine-tune v1 | Week 15 recap thread |

**Ship:** Fine-tuned model v1 + 7 threads

---

### Week 16 (Days 106–112): Evaluation & Iteration

**Focus:** Compare base vs fine-tuned, quantitative metrics (perplexity, BLEU) vs qualitative (does it sound like you?)

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 106 | Base vs fine-tuned | Generate 50 samples from each | "Fine-tuned vs base: side by side" |
| 107 | Perplexity | Calculate perplexity on test set | "Perplexity: measuring model confidence" |
| 108 | BLEU score | BLEU for generation quality | "BLEU score explained (with skepticism)" |
| 109 | Human eval | Rate 100 generations | "Why human eval matters most" |
| 110 | Voice consistency | Does it sound like you? | "Does my fine-tune sound like me?" |
| 111 | Iteration plan | What to improve in v2 | "Lessons from fine-tune v1" |
| 112 | Week 16 integration | Full evaluation report | Week 16 recap + Phase 4 summary |

**Ship:** Evaluated model + report + 7 threads + Phase 4 retrospective

---

## 📚 Phase 5 — Days 113–140 | Advanced Fine-Tuning + Merging

**Goal by Day 140:** Master LoRA merging, multi-task fine-tuning, and data augmentation. Your model now writes convincingly.

### Week 17 (Days 113–119): LoRA Merging & PEFT

**Focus:** Merge multiple LoRAs, task arithmetic, model soups, when to merge vs stack

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 113 | LoRA merging basics | Merge 2 LoRAs | "Merging LoRAs: combine superpowers" |
| 114 | Task arithmetic | Add/subtract task vectors | "Task arithmetic: LoRA math" |
| 115 | Model soups | Average multiple fine-tunes | "Model soups: ensembling via merging" |
| 116 | Merge vs stack | When to merge vs stack adapters | "Merge vs stack: which is better?" |
| 117 | PEFT library | Explore PEFT features | "PEFT: the fine-tuning Swiss Army knife" |
| 118 | Multi-adapter loading | Load multiple adapters dynamically | "Dynamic adapter loading" |
| 119 | Week 17 integration | Multi-adapter system | Week 17 recap thread |

**Ship:** Multi-adapter framework + 7 threads

---

### Week 18 (Days 120–126): Data Augmentation

**Focus:** Synthetic data generation, back-translation, paraphrasing, quality filtering

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 120 | Synthetic data generation | Generate tweets from prompts | "Synthetic data for fine-tuning" |
| 121 | Back-translation | Translate to Spanish & back | "Back-translation augmentation" |
| 122 | Paraphrasing | Paraphrase with GPT-4 | "Paraphrasing for data augmentation" |
| 123 | Quality filtering | Filter low-quality synthetic data | "Filtering synthetic data" |
| 124 | Diversity metrics | Measure dataset diversity | "Dataset diversity matters" |
| 125 | Augmentation pipeline | Automated augmentation | "Automated data augmentation" |
| 126 | Week 18 integration | 10K augmented dataset | Week 18 recap thread |

**Ship:** Augmented dataset + 7 threads

---

### Week 19 (Days 127–133): Multi-Task Fine-Tuning

**Focus:** Train on threads + analysis + replies simultaneously, task balancing, catastrophic forgetting

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 127 | Multi-task setup | Prepare 3 task datasets | "Multi-task fine-tuning setup" |
| 128 | Task balancing | Balance loss across tasks | "Balancing multiple tasks in training" |
| 129 | Catastrophic forgetting | Measure forgetting on task A | "Catastrophic forgetting is real" |
| 130 | Replay buffers | Prevent forgetting with replay | "Preventing catastrophic forgetting" |
| 131 | Task-specific adapters | Separate adapters per task | "Task-specific LoRA adapters" |
| 132 | Multi-task eval | Evaluate on all 3 tasks | "Evaluating multi-task models" |
| 133 | Week 19 integration | Multi-task fine-tuned model | Week 19 recap thread |

**Ship:** Multi-task model + 7 threads

---

### Week 20 (Days 134–140): Hyperparameter Deep Dive

**Focus:** Learning rate schedules, warmup, weight decay, gradient clipping, why they matter mathematically

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 134 | LR schedules | Cosine vs linear vs constant | "Learning rate schedules compared" |
| 135 | Warmup | Why warmup prevents instability | "Warmup: the training stabilizer" |
| 136 | Weight decay | L2 regularization in practice | "Weight decay explained" |
| 137 | Gradient clipping | Prevent exploding gradients | "Gradient clipping: when and why" |
| 138 | Adam optimizer | How Adam actually works | "Adam optimizer internals" |
| 139 | HP optimization | Grid search vs random search | "Finding optimal hyperparameters" |
| 140 | Week 20 integration | Optimally tuned model v2 | Week 20 recap + Phase 5 summary |

**Ship:** Final fine-tuned model + 7 threads + Phase 5 retrospective

---

## 📚 Phase 6 — Days 141–168 | Quantization + Inference Optimization

**Goal by Day 168:** Understand quantization deeply, run fast inference, deploy your model efficiently.

### Week 21 (Days 141–147): Quantization Theory

**Focus:** INT8/INT4/FP16 math, GPTQ vs AWQ vs GGUF, accuracy-speed tradeoffs, calibration datasets

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 141 | Quantization basics | FP32 → INT8 by hand | "Quantization: from 32 bits to 8" |
| 142 | INT4 quantization | Understanding 4-bit precision | "4-bit quantization explained" |
| 143 | GPTQ | Post-training quantization with GPTQ | "GPTQ: accurate 4-bit models" |
| 144 | AWQ | Activation-aware quantization | "AWQ vs GPTQ: which is better?" |
| 145 | GGUF deep dive | GGUF format internals | "Inside GGUF files" |
| 146 | Calibration data | Choose calibration data | "Calibration data for quantization" |
| 147 | Week 21 integration | Quantize your fine-tuned model | Week 21 recap thread |

**Ship:** Quantized models + 7 threads

---

### Week 22 (Days 148–154): Inference Engines

**Focus:** vLLM, TensorRT-LLM, llama.cpp internals, continuous batching, PagedAttention

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 148 | vLLM setup | Deploy model with vLLM | "vLLM: fast LLM inference" |
| 149 | Continuous batching | How vLLM batches requests | "Continuous batching explained" |
| 150 | PagedAttention | Memory-efficient attention | "PagedAttention: KV cache optimization" |
| 151 | TensorRT-LLM | Compile model for TensorRT | "TensorRT-LLM: NVIDIA's inference engine" |
| 152 | llama.cpp optimization | Optimize llama.cpp for your CPU | "llama.cpp performance tuning" |
| 153 | Benchmark comparison | vLLM vs TensorRT vs llama.cpp | "Inference engine benchmark" |
| 154 | Week 22 integration | Choose best inference engine | Week 22 recap thread |

**Ship:** Inference engine comparison + 7 threads

---

### Week 23 (Days 155–161): Deployment

**Focus:** RunPod setup, cost analysis ($30/month goal), API design, rate limiting

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 155 | RunPod setup | Deploy model to RunPod | "Deploying LLMs to the cloud" |
| 156 | Cost analysis | Calculate $/1K tokens | "LLM inference cost breakdown" |
| 157 | API design | FastAPI wrapper for model | "Building an LLM API" |
| 158 | Rate limiting | Implement rate limits | "Rate limiting for LLM APIs" |
| 159 | Caching | Cache common requests | "Caching strategies for LLMs" |
| 160 | Monitoring setup | Basic logging & metrics | "Monitoring LLM deployments" |
| 161 | Week 23 integration | Production API live | Week 23 recap thread |

**Ship:** Deployed API + 7 threads

---

### Week 24 (Days 162–168): Optimization Project

**Focus:** Make your fine-tuned model run <500ms on your hardware, measure latency/throughput

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 162 | Latency baseline | Measure current latency | "Latency benchmarking 101" |
| 163 | Prompt caching | Cache prompt prefixes | "Prompt caching for faster inference" |
| 164 | Speculative decoding | Implement speculative decoding | "Speculative decoding: 2x speedup" |
| 165 | Batch optimization | Optimize batch sizes | "Batching for throughput" |
| 166 | Memory optimization | Reduce memory footprint | "Memory optimization tricks" |
| 167 | Final optimization | Hit <500ms target | "I got my model under 500ms" |
| 168 | Week 24 integration | Optimized production model | Week 24 recap + Phase 6 summary |

**Ship:** Optimized model + benchmarks + 7 threads + Phase 6 retrospective

---

## 📚 Phase 7 — Days 169–196 | Alignment & RLHF Basics

**Goal by Day 196:** Understand how models are aligned (SFT → RLHF → DPO), run your first DPO training.

### Week 25 (Days 169–175): Alignment Fundamentals

**Focus:** Why models need alignment, instruction tuning, safety concerns, Constitutional AI basics

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 169 | Alignment overview | Why alignment matters | "Why LLMs need alignment" |
| 170 | Instruction tuning | SFT on instruction dataset | "Instruction tuning explained" |
| 171 | Safety concerns | Test model safety | "LLM safety: what could go wrong" |
| 172 | Constitutional AI | Implement basic Constitutional AI | "Constitutional AI basics" |
| 173 | Value alignment | Align model to your values | "Aligning AI to human values" |
| 174 | Alignment eval | Measure alignment quality | "How to evaluate alignment" |
| 175 | Week 25 integration | Instruction-tuned model | Week 25 recap thread |

**Ship:** Instruction-tuned model + 7 threads

---

### Week 26 (Days 176–182): RLHF Theory

**Focus:** Reward models, PPO algorithm (conceptually), preference learning, KL divergence penalty

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 176 | RLHF overview | How RLHF works end-to-end | "RLHF explained in plain English" |
| 177 | Reward models | Train simple reward model | "Reward models: teaching preferences" |
| 178 | PPO algorithm | Understand PPO conceptually | "PPO: the RLHF workhorse" |
| 179 | Preference learning | Collect preference data | "Learning from preferences" |
| 180 | KL divergence | Why KL penalty matters | "KL divergence in RLHF" |
| 181 | RLHF challenges | Why RLHF is hard | "RLHF: harder than it looks" |
| 182 | Week 26 integration | RLHF theory mastery | Week 26 recap thread |

**Ship:** Reward model + 7 threads

---

### Week 27 (Days 183–189): DPO (Direct Preference Optimization)

**Focus:** DPO vs RLHF, preference dataset creation, loss function, training loop

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 183 | DPO overview | DPO vs RLHF tradeoffs | "DPO: RLHF without the RL" |
| 184 | Preference pairs | Create chosen/rejected pairs | "Building preference datasets" |
| 185 | DPO loss function | Implement DPO loss | "DPO loss function explained" |
| 186 | Training setup | Set up DPO training | "Setting up DPO training" |
| 187 | Hyperparameters | Beta parameter tuning | "DPO hyperparameters" |
| 188 | DPO training | Train on preferences | "My first DPO training run" |
| 189 | Week 27 integration | DPO theory + practice | Week 27 recap thread |

**Ship:** DPO loss implementation + 7 threads

---

### Week 28 (Days 190–196): First DPO Run

**Focus:** Create preference pairs from your good/bad threads, train on them, evaluate changes

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 190 | Data collection | Label your tweets good/bad | "Creating my own preference data" |
| 191 | Preference generation | Generate alternatives with model | "Synthetic preference pairs" |
| 192 | Dataset quality | Validate preference data | "Quality control for preferences" |
| 193 | DPO training run | Full DPO training | "Training with DPO (live thread)" |
| 194 | Before/after comparison | Compare pre/post DPO | "DPO results: before and after" |
| 195 | Behavior analysis | What changed? | "What DPO actually changed" |
| 196 | Week 28 integration | DPO-aligned model | Week 28 recap + Phase 7 summary |

**Ship:** DPO-trained model + 7 threads + Phase 7 retrospective

---

## 📚 Phase 8 — Days 197–224 | Advanced Alignment + Steering

**Goal by Day 224:** Master advanced alignment techniques, steering vectors, and model behavior control.

### Week 29 (Days 197–203): Iterative Alignment

**Focus:** Multi-round DPO, online learning, active learning for preferences

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 197 | Multi-round DPO | Train → eval → retrain | "Iterative DPO training" |
| 198 | Online learning | Update model from live feedback | "Online learning for alignment" |
| 199 | Active learning | Ask user for hard examples | "Active learning: query the right data" |
| 200 | Curriculum learning | Easy → hard preference training | "Curriculum learning for alignment" |
| 201 | Ensemble alignment | Combine multiple aligned models | "Ensemble alignment strategies" |
| 202 | Alignment stability | Prevent alignment collapse | "Maintaining alignment over time" |
| 203 | Week 29 integration | Iteratively aligned model v2 | Week 29 recap thread |

**Ship:** Improved aligned model + 7 threads

---

### Week 30 (Days 204–210): Steering Vectors

**Focus:** Activation engineering, representation control, how to make models more/less of something

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 204 | Activation analysis | Analyze layer activations | "Inside transformer activations" |
| 205 | Steering vectors | Compute steering directions | "Steering vectors explained" |
| 206 | Representation control | Steer model toward/away from trait | "Controlling model behavior with vectors" |
| 207 | Multi-vector steering | Combine multiple steering vectors | "Multi-dimensional steering" |
| 208 | Steering strength | Calibrate steering intensity | "How much steering is too much?" |
| 209 | Evaluation | Does steering work? | "Evaluating steering interventions" |
| 210 | Week 30 integration | Steerable model | Week 30 recap thread |

**Ship:** Steering framework + 7 threads

---

### Week 31 (Days 211–217): Hallucination Reduction

**Focus:** Why models hallucinate, grounding techniques, citation training, uncertainty estimation

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 211 | Hallucination analysis | Categorize hallucination types | "Why LLMs hallucinate" |
| 212 | Grounding techniques | Ground outputs in retrieved docs | "Grounding: reduce hallucinations" |
| 213 | Citation training | Train model to cite sources | "Teaching models to cite sources" |
| 214 | Uncertainty estimation | Detect uncertain generations | "Uncertainty estimation in LLMs" |
| 215 | Factuality metrics | Measure factual accuracy | "Measuring LLM factuality" |
| 216 | Hallucination mitigation | Combine all techniques | "My hallucination reduction pipeline" |
| 217 | Week 31 integration | Low-hallucination model | Week 31 recap thread |

**Ship:** Hallucination-reduced model + 7 threads

---

### Week 32 (Days 218–224): Final Alignment

**Focus:** Combine all techniques, achieve <1% hallucination rate on threads

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 218 | Alignment stack | DPO + steering + grounding | "My complete alignment stack" |
| 219 | Comprehensive eval | Test on 1000 examples | "Comprehensive alignment evaluation" |
| 220 | Edge case handling | Fix remaining failures | "Handling alignment edge cases" |
| 221 | Robustness testing | Adversarial prompts | "Testing alignment robustness" |
| 222 | Final tuning | Polish the aligned model | "Final alignment tuning" |
| 223 | Benchmark results | <1% hallucination achieved | "I hit <1% hallucination rate" |
| 224 | Week 32 integration | Production-ready aligned model | Week 32 recap + Phase 8 summary |

**Ship:** Final aligned model + eval report + 7 threads + Phase 8 retrospective

---

## 📚 Phase 9 — Days 225–252 | Multi-Agent Systems + Memory

**Goal by Day 252:** Build a multi-agent system where agents have memory and collaborate effectively.

### Week 33 (Days 225–231): LangGraph Advanced

**Focus:** Conditional edges, checkpoints, streaming, human-in-the-loop

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 225 | Conditional edges | Dynamic graph routing | "Conditional edges in LangGraph" |
| 226 | Checkpoints | Save & restore agent state | "Checkpointing for agents" |
| 227 | Streaming | Stream agent outputs | "Streaming agent responses" |
| 228 | Human-in-the-loop | Add approval steps | "Human-in-the-loop agents" |
| 229 | Error recovery | Graceful error handling | "Error recovery in agents" |
| 230 | Graph visualization | Visualize agent flow | "Visualizing agent graphs" |
| 231 | Week 33 integration | Advanced LangGraph patterns | Week 33 recap thread |

**Ship:** Advanced agent patterns + 7 threads

---

### Week 34 (Days 232–238): Agent Memory Systems

**Focus:** Vector memory, episodic memory, semantic memory, when to retrieve vs forget

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 232 | Memory types | Vector vs episodic vs semantic | "Types of agent memory" |
| 233 | Vector memory | Add vector memory to agent | "Vector memory for agents" |
| 234 | Episodic memory | Store conversation episodes | "Episodic memory in agents" |
| 235 | Semantic memory | Extract facts & store | "Semantic memory: long-term facts" |
| 236 | Memory retrieval | When to retrieve from memory | "Memory retrieval strategies" |
| 237 | Memory forgetting | Prune irrelevant memories | "When agents should forget" |
| 238 | Week 34 integration | Full memory system | Week 34 recap thread |

**Ship:** Memory-enabled agent + 7 threads

---

### Week 35 (Days 239–245): Multi-Agent Coordination

**Focus:** 4 agents (Researcher, Strategist, Writer, Critic), communication protocols, consensus

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 239 | Multi-agent architecture | Design 4-agent system | "Multi-agent system architecture" |
| 240 | Researcher agent | Build researcher agent | "Researcher agent: find the trends" |
| 241 | Strategist agent | Build strategist agent | "Strategist agent: pick the best" |
| 242 | Writer agent | Build writer agent | "Writer agent: craft the thread" |
| 243 | Critic agent | Build critic agent | "Critic agent: quality control" |
| 244 | Communication protocol | Agents communicate | "Agent communication protocols" |
| 245 | Week 35 integration | 4 agents working together | Week 35 recap thread |

**Ship:** Multi-agent system + 7 threads

---

### Week 36 (Days 246–252): Reflection & Metacognition

**Focus:** Self-correction loops, critic agents, quality gates, max iteration limits

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 246 | Self-reflection | Agents evaluate own output | "Self-reflection in agents" |
| 247 | Critic loops | Critic → Writer → Critic | "Critic loops for quality" |
| 248 | Quality gates | Define quality thresholds | "Quality gates in multi-agent systems" |
| 249 | Iteration limits | Prevent infinite loops | "When to stop iterating" |
| 250 | Metacognition | Agents reason about reasoning | "Metacognition in agents" |
| 251 | Final integration | Complete Clair agent system | "Clair agent: all pieces together" |
| 252 | Week 36 integration | Production multi-agent system | Week 36 recap + Phase 9 summary |

**Ship:** Complete Clair system + 7 threads + Phase 9 retrospective

---

## 📚 Phase 10 — Days 253–280 | Production ML + Monitoring

**Goal by Day 280:** Your system runs reliably in production with proper monitoring and failure handling.

### Week 37 (Days 253–259): Production Best Practices

**Focus:** Error handling, retries, fallbacks, graceful degradation

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 253 | Error handling | Comprehensive error handling | "Error handling in production AI" |
| 254 | Retry logic | Exponential backoff | "Retry strategies for AI systems" |
| 255 | Fallbacks | Graceful degradation | "Fallback strategies for agents" |
| 256 | Circuit breakers | Prevent cascade failures | "Circuit breakers for AI systems" |
| 257 | Health checks | API health monitoring | "Health checks for production AI" |
| 258 | Deployment strategies | Blue-green deployment | "Deploying AI systems safely" |
| 259 | Week 37 integration | Production-ready error handling | Week 37 recap thread |

**Ship:** Robust production system + 7 threads

---

### Week 38 (Days 260–266): Monitoring & Observability

**Focus:** LangSmith, logging, metrics, alerts, debugging production issues

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 260 | LangSmith setup | Integrate LangSmith | "LangSmith for agent observability" |
| 261 | Structured logging | Add structured logs | "Logging best practices for AI" |
| 262 | Metrics | Track key metrics | "Metrics for production AI" |
| 263 | Alerting | Set up alerts | "Alerting for AI systems" |
| 264 | Debugging production | Debug live issues | "Debugging production AI issues" |
| 265 | Dashboards | Build monitoring dashboard | "Monitoring dashboards for AI" |
| 266 | Week 38 integration | Full observability stack | Week 38 recap thread |

**Ship:** Monitoring system + 7 threads

---

### Week 39 (Days 267–273): A/B Testing & Experimentation

**Focus:** Compare model versions, statistical significance, user feedback loops

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 267 | A/B testing setup | Set up A/B tests | "A/B testing for AI models" |
| 268 | Traffic splitting | Route traffic to variants | "Traffic splitting strategies" |
| 269 | Statistical significance | Calculate significance | "Statistical significance in A/B tests" |
| 270 | User feedback | Collect user feedback | "User feedback loops for AI" |
| 271 | Experiment analysis | Analyze experiment results | "Analyzing A/B test results" |
| 272 | Rollout strategy | Gradual rollout plan | "Safe rollout strategies" |
| 273 | Week 39 integration | Experimentation framework | Week 39 recap thread |

**Ship:** A/B testing framework + 7 threads

---

### Week 40 (Days 274–280): Cost Optimization

**Focus:** Token usage tracking, caching strategies, model selection heuristics

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 274 | Cost tracking | Track token usage & costs | "Tracking AI system costs" |
| 275 | Caching strategies | Implement smart caching | "Caching strategies to reduce costs" |
| 276 | Model selection | Route to optimal model | "Model routing for cost optimization" |
| 277 | Prompt optimization | Shorter prompts, same quality | "Prompt optimization for costs" |
| 278 | Batch processing | Reduce costs with batching | "Batching to reduce inference costs" |
| 279 | Cost analysis | Monthly cost breakdown | "My AI system costs $X/month" |
| 280 | Week 40 integration | Cost-optimized production system | Week 40 recap + Phase 10 summary |

**Ship:** Cost-optimized system + analysis + 7 threads + Phase 10 retrospective

---

## 📚 Phase 11 — Days 281–308 | Advanced Topics + Research

**Goal by Day 308:** Explore cutting-edge LLM research and implement novel techniques.

### Week 41 (Days 281–287): Long Context

**Focus:** RoPE, ALiBi, streaming LLMs, context compression techniques

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 281 | Long context challenges | Why long context is hard | "Long context: the next frontier" |
| 282 | RoPE | Rotary positional embeddings | "RoPE: relative position encoding" |
| 283 | ALiBi | Attention with linear biases | "ALiBi: simple long context solution" |
| 284 | Streaming LLMs | Implement streaming inference | "Streaming LLMs: infinite context?" |
| 285 | Context compression | Compress long contexts | "Context compression techniques" |
| 286 | Long context eval | Test on long documents | "Evaluating long context models" |
| 287 | Week 41 integration | Long context system | Week 41 recap thread |

**Ship:** Long context experiments + 7 threads

---

### Week 42 (Days 288–294): Mixture of Experts

**Focus:** How MoE works, router training, expert specialization

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 288 | MoE overview | How MoE works | "Mixture of Experts explained" |
| 289 | Router training | Train expert router | "Training MoE routers" |
| 290 | Expert specialization | Analyze expert specialization | "What do MoE experts learn?" |
| 291 | Sparse activation | Understand sparse activation | "Sparse activation in MoE" |
| 292 | MoE fine-tuning | Fine-tune MoE model | "Fine-tuning Mixture of Experts" |
| 293 | MoE efficiency | Compare MoE vs dense | "MoE: cheaper than dense models?" |
| 294 | Week 42 integration | MoE experiments complete | Week 42 recap thread |

**Ship:** MoE analysis + 7 threads

---

### Week 43 (Days 295–301): Multimodal

**Focus:** Vision-language models basics, CLIP, image embeddings for threads

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 295 | Multimodal overview | Vision + language models | "Multimodal AI: vision meets language" |
| 296 | CLIP basics | How CLIP works | "CLIP: connecting vision and text" |
| 297 | Image embeddings | Embed images for similarity | "Image embeddings with CLIP" |
| 298 | Vision-language retrieval | Image search with text | "Multimodal retrieval systems" |
| 299 | Image captioning | Generate captions for images | "Image captioning with LLMs" |
| 300 | Visual threads | Add images to thread generator | "Visual threads: images + text" |
| 301 | Week 43 integration | Multimodal Clair agent | Week 43 recap thread |

**Ship:** Multimodal features + 7 threads

---

### Week 44 (Days 302–308): Novel Techniques

**Focus:** Whatever's hot on arXiv that week — implement a paper from scratch

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 302 | arXiv scan | Find 5 interesting papers | "This week on arXiv" |
| 303 | Paper deep dive | Read & understand 1 paper | "Paper explained: [title]" |
| 304 | Implementation Day 1 | Start implementing | "Implementing [paper] from scratch - Day 1" |
| 305 | Implementation Day 2 | Continue implementation | "Implementing [paper] from scratch - Day 2" |
| 306 | Implementation Day 3 | Finish implementation | "Implementing [paper] from scratch - Day 3" |
| 307 | Evaluation | Test implementation | "Does [technique] actually work?" |
| 308 | Week 44 integration | Novel technique integrated | Week 44 recap + Phase 11 summary |

**Ship:** Paper implementation + 7 threads + Phase 11 retrospective

---

## 📚 Phase 12 — Days 309–336 | Polish + Knowledge Sharing

**Goal by Day 336:** Finalize the system, write comprehensive documentation, create educational content.

### Week 45 (Days 309–315): Code Quality

**Focus:** Refactor, tests, type hints, documentation, make it beautiful

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 309 | Code review | Audit entire codebase | "Code review: what needs fixing" |
| 310 | Refactoring | Clean up messy code | "Refactoring: making code beautiful" |
| 311 | Type hints | Add comprehensive type hints | "Type hints for production AI" |
| 312 | Unit tests | Write critical tests | "Testing AI systems" |
| 313 | Documentation | Write API documentation | "Documentation best practices" |
| 314 | Code formatting | Format with black/ruff | "Code formatting matters" |
| 315 | Week 45 integration | Clean, documented codebase | Week 45 recap thread |

**Ship:** Production-quality code + 7 threads

---

### Week 46 (Days 316–322): Educational Content

**Focus:** Write the 98-tweet mega-thread explaining everything you learned

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 316 | Content outline | Plan the mega-thread | "Planning my 98-tweet mega-thread" |
| 317 | Weeks 1-12 | Write threads covering Phases 1-3 | "Mega-thread Part 1: Foundations" |
| 318 | Weeks 13-24 | Write threads covering Phases 4-6 | "Mega-thread Part 2: Fine-tuning" |
| 319 | Weeks 25-36 | Write threads covering Phases 7-9 | "Mega-thread Part 3: Alignment & Agents" |
| 320 | Weeks 37-48 | Write threads covering Phases 10-12 | "Mega-thread Part 4: Production & Research" |
| 321 | Edit & polish | Refine the mega-thread | "Editing the mega-thread" |
| 322 | Week 46 integration | Complete educational content | Week 46 recap thread |

**Ship:** Epic mega-thread + 7 threads

---

### Week 47 (Days 323–329): Open Source Launch

**Focus:** GitHub repo with excellent README, Hugging Face models, Gumroad templates

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 323 | GitHub README | Write killer README | "Writing a great README" |
| 324 | Hugging Face upload | Upload models to HF | "Publishing models on Hugging Face" |
| 325 | Documentation site | Build docs with MkDocs | "Documentation sites with MkDocs" |
| 326 | Gumroad templates | Package as templates | "Monetizing open source" |
| 327 | Launch prep | Final prep for launch | "Preparing for open source launch" |
| 328 | Launch day | Ship Clair v1.0 | "🚀 Clair v1.0 is live!" |
| 329 | Week 47 integration | Open source launch complete | Week 47 recap thread |

**Ship:** Open source Clair + 7 threads

---

### Week 48 (Days 330–336): Reflection & Next Steps

**Focus:** What you learned, what surprised you, where to go deeper

| Day | Topic | What You'll Build | Thread Topic |
|-----|-------|-------------------|--------------|
| 330 | Learning reflection | What did you actually learn? | "48 weeks of LLM learning: what I learned" |
| 331 | Surprises | What surprised you? | "The most surprising things I learned" |
| 332 | Challenges | What was hardest? | "The hardest parts of learning LLMs" |
| 333 | Wins | What worked best? | "My biggest wins in 48 weeks" |
| 334 | Future topics | Where to go deeper? | "Next steps in my LLM journey" |
| 335 | Advice to past self | What would you tell Day 1 you? | "Advice to Day 1 me" |
| 336 | Final reflection | Closing thoughts | "Day 336: The end is the beginning" |

**Ship:** Reflection threads + complete roadmap + 7 threads + Final retrospective

---

## 🎓 Educational Philosophy

**This plan makes you:**
- An LLM expert who understands transformers, attention, embeddings, fine-tuning, alignment, quantization, and inference optimization from first principles
- Someone who can read arXiv papers and implement them from scratch
- A practitioner who knows when to use which technique and why
- A builder who happens to have an amazing agent as proof of learning

**You're not building an agent and learning LLMs on the side. You're learning LLMs deeply and building an agent as the vehicle for that learning.**

---

## 📊 Progress Tracking

Create a simple tracking sheet with:
- [ ] Day completed
- [ ] Code shipped (GitHub link)
- [ ] Thread posted (X link)
- [ ] Key learning
- [ ] Time spent

**Be honest.** Some days you'll do 30 minutes. Some days zero. That's life. The plan survives reality.

---

## 🔥 Rules (Repeat for Emphasis)

1. **Windows only** (no Linux/WSL instructions ever)
2. **LLM topics from Day 1** (never delay the good stuff)
3. **Ship daily** (visible code + public thread)
4. **Calm, classic voice** (zero hype)
5. **Max 120 min/day** (usually 60-90)
6. **Real code, no placeholders** (always deliverable)
7. **Family first** (life happens, be honest)

---

## 🤝 How to Work with Claude (Me!)

I'm your mentor, not a motivational speaker. I'll:
- Give you exact code, not pseudo-code
- Debug Windows-specific issues
- Explain the math when you need it
- Keep you honest about time budgets
- Remind you that some days you'll ship nothing (and that's okay)
- Help you write threads that teach what you learned
- Push back when you're over-engineering

**Let's make you an LLM expert. One day at a time. Starting today.**

---

**Ready to start Day 1? Say "Vibe Code Day 1" and let's build some attention from scratch.**
