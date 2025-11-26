# 🔮 Clair Agent — 48 Weeks of LLM Mastery

**An open-source AI agent built from the ground up, not as a product, but as a vehicle for deep learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-in%20progress-orange)](https://img.shields.io/badge/build-in%20progress-orange)

## 🎯 What is Clair?

**Clair (Clairvoyant Agent)** is an AI system that:

- Wakes up every morning and scans arXiv, X (Twitter), Hacker News, Reddit, and HuggingFace
- Identifies the 5 real AI/ML trends (not hype)
- Writes viral threads in a calm, classic voice
- Analyzes posting history to maintain consistent tone
- Optionally auto-posts with one-click approval

**But more importantly:** This is a 48-week journey to become a true LLM expert who understands transformers, fine-tuning, alignment, inference optimization, and cutting-edge research from first principles.

## 🧭 Philosophy

> **"Ship to learn, don't learn to ship"**

This project prioritizes:

- ✅ Understanding LLM internals over using black-box APIs
- ✅ Real experiments over polished products
- ✅ Daily shipping over perfect code
- ✅ LLM mastery over agent features
- ✅ Learning in public over stealth mode

**You're not watching someone build an agent. You're watching someone become an LLM expert, with the agent as proof of learning.**

## 📚 The 48-Week Curriculum

### 🏗️ **Phase 1: Foundation (Weeks 1-4)** ✅ In Progress

- ✅ Week 1: Build transformers from NumPy (attention, tokenization, positional encoding)
- 🚧 Week 2: Local LLM mastery (Ollama, llama.cpp, GGUF, quantization)
- ⏳ Week 3: Inference deep dive (KV cache, sampling, batching)
- ⏳ Week 4: First arXiv thread generator

### 🧠 **Phase 2: Embeddings + RAG (Weeks 5-8)**

- Week 5: Embedding models (BERT, SentenceTransformers, similarity math)
- Week 6: Vector databases (ChromaDB, FAISS, Qdrant, HNSW)
- Week 7: RAG architectures (naive RAG, reranking, ColBERT)
- Week 8: Multi-source RAG (arXiv + Reddit + HN)

### 🎨 **Phase 3: Prompting + Agents (Weeks 9-12)**

- Week 9: Prompt engineering (few-shot, CoT, ReAct, DSPy)
- Week 10: Function calling internals (JSON mode, structured outputs)
- Week 11: First real agent (LangGraph, state, memory)
- Week 12: Agent evaluation (evals, metrics, debugging)

### 🔧 **Phase 4: Fine-Tuning (Weeks 13-16)**

- Week 13: Fine-tuning theory (LoRA/QLoRA math, adapters)
- Week 14: Dataset curation (Twitter export, cleaning, formatting)
- Week 15: First fine-tune (Unsloth + Llama 3.2 8B)
- Week 16: Evaluation (perplexity, BLEU, human eval)

### 🚀 **Phase 5: Advanced Fine-Tuning (Weeks 17-20)**

- Week 17: LoRA merging & PEFT (task arithmetic, model soups)
- Week 18: Data augmentation (synthetic data, back-translation)
- Week 19: Multi-task fine-tuning (catastrophic forgetting)
- Week 20: Hyperparameter optimization (LR schedules, warmup)

### ⚡ **Phase 6: Quantization + Inference (Weeks 21-24)**

- Week 21: Quantization theory (INT8/INT4, GPTQ, AWQ)
- Week 22: Inference engines (vLLM, TensorRT-LLM, llama.cpp)
- Week 23: Deployment (RunPod, cost analysis, API design)
- Week 24: Optimization (<500ms inference target)

### 🎭 **Phase 7: Alignment Basics (Weeks 25-28)**

- Week 25: Alignment fundamentals (instruction tuning, Constitutional AI)
- Week 26: RLHF theory (reward models, PPO, KL divergence)
- Week 27: DPO theory (Direct Preference Optimization)
- Week 28: First DPO run (preference pairs from tweets)

### 🧪 **Phase 8: Advanced Alignment (Weeks 29-32)**

- Week 29: Iterative alignment (multi-round DPO, online learning)
- Week 30: Steering vectors (activation engineering, representation control)
- Week 31: Hallucination reduction (grounding, citations, uncertainty)
- Week 32: Final alignment (<1% hallucination target)

### 🤖 **Phase 9: Multi-Agent Systems (Weeks 33-36)**

- Week 33: LangGraph advanced (conditional edges, checkpoints)
- Week 34: Agent memory (vector, episodic, semantic memory)
- Week 35: Multi-agent coordination (4 agents working together)
- Week 36: Reflection & metacognition (self-correction loops)

### 🏭 **Phase 10: Production (Weeks 37-40)**

- Week 37: Production best practices (error handling, retries)
- Week 38: Monitoring & observability (LangSmith, logging, metrics)
- Week 39: A/B testing & experimentation (statistical significance)
- Week 40: Cost optimization (token tracking, caching)

### 🔬 **Phase 11: Research (Weeks 41-44)**

- Week 41: Long context (RoPE, ALiBi, streaming LLMs)
- Week 42: Mixture of Experts (MoE internals, router training)
- Week 43: Multimodal (vision-language models, CLIP)
- Week 44: Novel techniques (implement latest arXiv paper)

### 🎓 **Phase 12: Polish + Share (Weeks 45-48)**

- Week 45: Code quality (refactor, tests, docs)
- Week 46: Educational content (98-tweet mega-thread)
- Week 47: Open source launch (GitHub, HuggingFace, Gumroad)
- Week 48: Reflection & next steps

## 📂 Repository Structure

clair-agent/
├── week-01/                    # Week 1: Hello Transformers
│   ├── day-01-attention/
│   │   ├── src/
│   │   │   └── attention.py
│   │   ├── tests/
│   │   └── README.md
│   ├── day-02-multihead/
│   └── ...
├── week-02/                    # Week 2: Local LLM Mastery
├── .../
├── week-48/                    # Week 48: Final Reflection
├── docs/
│   ├── Clair-LLM_Expert_Plan.md
│   └── progress-tracker.md
├── experiments/                # One-off experiments
├── models/                     # Fine-tuned models (gitignored)
├── data/                       # Datasets (gitignored)
└── README.md

**Why this structure?**

- Each week is self-contained (can jump to any week)
- Easy to follow the learning journey chronologically
- Code evolves organically (see the progression)
- No premature abstractions (refactor at phase boundaries)

## 🚀 Getting Started

### Prerequisites

- **OS:** Windows 11 (all commands are Windows-native)
- **Python:** 3.11+
- **Hardware:** 16GB+ RAM recommended (32GB for fine-tuning weeks)
- **GPU:** Optional for Weeks 1-12, required for Weeks 13+ (NVIDIA preferred)

### Installation

```powershell
# Clone the repo
git clone https://github.com/vai-arch/clair-agent
cd clair-agent

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies for current week
cd week-01/day-01-attention
pip install -r requirements.txt
```

### Run Today's Code

```powershell
# Each day has a standalone script
python src/attention.py

# Or run tests
python -m pytest tests/
```

## 📖 Following Along

### As a Learner

Each week's folder contains:

- ✅ **Complete working code** (no TODOs or placeholders)
- ✅ **README.md** explaining concepts
- ✅ **requirements.txt** with exact versions
- ✅ **Tests** showing expected behavior
- ✅ **Thread summary** (what was learned publicly)

**How to use:**

1. Start at `week-01/day-01-attention/`
2. Read the README
3. Run the code
4. Modify and experiment
5. Read the thread to see public explanation
6. Move to next day

### As a Recruiter/Evaluator

Looking at this repo, you'll see:

- **336 days of consistent shipping** (with honest gaps for life)
- **Progression from NumPy attention to production LLM systems**
- **Deep understanding:** not just using libraries, but implementing from scratch
- **Public learning:** threads explain concepts in simple terms
- **Real experiments:** loss curves, benchmarks, A/B tests
- **Production mindset:** testing, monitoring, cost optimization

## 🧪 Key Projects by Phase

| Phase | Week | Project | Status |
|-------|------|---------|--------|
| 1 | 1 | Attention mechanism from NumPy | ✅ Complete |
| 1 | 2 | CLI tool comparing Ollama/llama.cpp/transformers | 🚧 In Progress |
| 1 | 4 | arXiv → thread generator (v1) | ⏳ Planned |
| 2 | 8 | Multi-source RAG system | ⏳ Planned |
| 3 | 11 | First LangGraph agent | ⏳ Planned |
| 4 | 15 | Fine-tuned Llama 3.2 on my voice | ⏳ Planned |
| 6 | 24 | <500ms inference optimized model | ⏳ Planned |
| 7 | 28 | DPO-aligned model | ⏳ Planned |
| 9 | 36 | 4-agent Clair system (complete) | ⏳ Planned |
| 12 | 48 | Open-source launch + mega-thread | ⏳ Planned |

## 📊 Progress Tracking

**Current Status:**

- 📅 **Week:** 1/48
- 📈 **Days Completed:** 1/336 (0%)
- 🔥 **Streak:** 1 days
- ⏱️ **Avg Time/Day:** 75 min

**Completed Milestones:**

- ✅ Scaled dot-product attention (NumPy)
- ✅ Multi-head attention (NumPy)
- ✅ BPE tokenizer exploration
- ✅ Positional encoding implementation
- ✅ Llama 3.2 local inference
- ✅ Model architecture inspection
- ✅ Week 1 integration pipeline

[See full progress tracker →](docs/progress-tracker.md)

## 🎓 Learning Resources

### Papers Implemented (So Far)

- [ ] Attention Is All You Need (Vaswani et al., 2017) — *Week 1*
- [ ] LoRA (Hu et al., 2021) — *Week 13*
- [ ] DPO (Rafailov et al., 2023) — *Week 27*
- [ ] Many more...

### Key Concepts Mastered

- ✅ Scaled dot-product attention
- ✅ Multi-head attention
- ✅ Tokenization (BPE)
- ✅ Positional encoding
- 🚧 KV cache
- 🚧 Quantization
- ⏳ LoRA fine-tuning
- ⏳ DPO alignment
- ⏳ Many more...

## 🧵 Public Learning (Threads)

Following along on X (Twitter): [@vai_arch](https://x.com/vai_arch)

## 🤝 Contributing

This is primarily a **learning journey**, not a collaborative project. However:

**You can:**

- ⭐ Star if you're following along
- 🐛 Open issues if you find bugs in the code
- 💡 Share your own implementations in Discussions
- 📖 Suggest improvements to READMEs

**Please don't:**

- ❌ Submit PRs to "improve" the learning code (it's intentionally incremental)
- ❌ Suggest skipping weeks (the progression is deliberate)
- ❌ Ask to collaborate on Clair itself (it's a solo learning project)

**Want to learn alongside?** Fork the repo and start your own 48-week journey! Tag me in your threads.

## 📝 License

MIT License — feel free to use this code for learning.

**However:**

- The fine-tuned models (Weeks 15+) trained on my writing are **not** included
- The full Clair agent (Week 36+) will be open-sourced separately upon completion

## 🎯 Why This Exists

Most LLM tutorials:

- Show you how to use APIs
- Skip the math
- Give you completed code
- Don't show the messy learning process

This repo:

- Builds everything from scratch
- Shows the math when it matters
- Starts from zero (NumPy attention)
- Documents failures, bugs, and confusion
- Proves learning in public works

**Goal:** By Week 48, you'll see someone go from "what is attention?" to "here's my production-grade, aligned, optimized LLM agent."

## 🌟 Milestones

- [x] **Day 1:** First line of code (NumPy attention)
- [ ] **Week 1:** Complete transformer understanding
- [ ] **Week 4:** First thread generated by my code
- [ ] **Week 12:** First working agent
- [ ] **Week 15:** First fine-tuned model in my voice
- [ ] **Week 24:** Sub-500ms inference
- [ ] **Week 28:** First DPO-aligned model
- [ ] **Week 36:** Complete 4-agent Clair system
- [ ] **Week 48:** Open-source launch + mega-thread

## 📧 Contact

- **X (Twitter):** [@vai_arch](https://x.com/vai_arch)
- **GitHub:** [@vai_arch](https://github.com/vai-arch)

**Questions?** DM me on X or open a GitHub Discussion.

## 🙏 Acknowledgments

**Inspired by:**

- Andrej Karpathy's "Neural Networks: Zero to Hero" series
- Jeremy Howard's fast.ai philosophy
- Anthropic's Claude (my mentor throughout this journey)

**Special thanks to:**

- Everyone who follows along and shares feedback
- The open-source LLM community
- My family for supporting this 48-week journey

## ⏭️ What's Next?

**Next week (Week 2):** Local LLM mastery

- Day 8: Ollama setup
- Day 9: llama.cpp deep dive
- Day 10: GGUF format exploration
- Day 11: Quantization experiments
- Day 12: Context window testing
- Day 13: Speed benchmarks
- Day 14: CLI comparison tool

**Follow along:** Star this repo and check back weekly for updates.

**Last Updated:** November 25, 2025  
**Current Week:** 1/48  
**Next Milestone:** Week 2 — First thread generator
