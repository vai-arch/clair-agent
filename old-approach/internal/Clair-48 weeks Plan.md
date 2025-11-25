# Clair Agent – 48 weeks Build Plan (4-Week Phases Version)

The end goal: build **Clairvoyant (Clair) Agent**: an open-source agent that wakes up every morning, scans arXiv / X / HN / Reddit / HuggingFace, picks the 5 real trends, writes a viral thread in my calm/classic voice, analyzes my posting history, and optionally auto-posts with one-click approval.

Structured into **4-week phases** (28 days each) numbered sequentially.

Current date: November 18, 2025  
Start Day: Day 1 = November 24, 2025  
Total: 12 × 28-day phases -> 336 days

## Full Plan (12 × 4-week phases + final phase)

-Phase 1 – Days 1–28 | v0.1 alive and ugly
-Phase 2 – Days 29–56 | Usable Top 5 report (6–7/10)
-Phase 3 – Days 57–84 | Legitimately good report (8–9/10)
-Phase 4 – Days 85–112 | Sounds 80–90 % like me (memory + reflection)
-Phase 5 – Days 113–140 | First real fine-tune → 6.5/10 my voice
-Phase 6 – Days 141–168 | Full strategist + analytics tools
-Phase 7 – Days 169–196 | 4-agent LangGraph team live
-Phase 8 – Days 197–224 | Growth hacker + auto-follow engine
-Phase 9 – Days 225–252 | DPO alignment → 9.2/10 me, <1 hallucination/month
-Phase 10 – Days 253–280 | Full autonomy + one-click post
-Phase 11 – Days 281–308 | Production v0.99, < $30/mo, beautiful UI
-Phase 12 – Days 309–336 | Final prep + mega-thread writing

## Phase 1 – Days 1–28 (4 weeks)

**Goal:** Clair v0.1 is alive and ugly but working — every morning I run one command and get a real (if cringe) 3–5 tweet thread in < 3 min.

| Week | Days       | Week Goal                                  | Key milestones (end of week) |
|------|------------|--------------------------------------------|------------------------------|
| W1   | 1–7        | Local LLM + first “hello world” thread     | Day 7: ship first terrible thread + code |
| W2   | 8–14       | Multi-paper + basic ranking                | Day 14: consistent ugly-but-real threads |
| W3   | 15–21      | Reddit + X + tiny FAISS RAG                | Day 21: 5-tweet thread from 4 sources |
| W4   | 22–28      | First usable daily report + victory lap    | Day 28: Clair v0.1 officially alive |

## Phase 2 – Days 29–56 (4 weeks)

**Goal by Day 56:** Clair generates a usable “Top 5 Trends Today” report I would actually consider posting (6–7/10 quality). RAG brain multi-source, reranked, self-correcting, lightly scored.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W5   | 29–35      | ColBERT-style reranking + metadata filtering       |
| W6   | 36–42      | Metadata panel + Corrective RAG (CRAG)             |
| W7   | 43–49      | Trend scoring engine v1 (novelty × importance × momentum) |
| W8   | 50–56      | Polish + first truly usable daily report (goal achieved) |

## Phase 3 – Days 57–84 (4 weeks)

**Goal by Day 84:** Daily “Top 5 Trends Today” report that is legitimately good (80–90 % of AI Twitter would envy). Multi-source fusion complete, hallucinations < 5 %, tone already 70 % me.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W9   | 57–63      | Full multi-source fusion + BM25 + dense fusion     |
| W10  | 64–70      | Advanced metadata + engagement filtering           |
| W11  | 71–77      | Context-aware reranking + novelty 2.0              |
| W12  | 78–84      | Hallucination slayer + style injection + victory lap |

## Phase 4 – Days 85–112 (4 weeks)

**Goal by Day 112:** Clair threads now sound human (80–90 % my voice) because it remembers everything I’ve ever posted and constantly reflects. Never repeats old takes, avoids my low-performing phrases.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W13  | 85–91      | Permanent memory of my entire posting history      |
| W14  | 92–98      | Reflection loops — Critic agent is born            |
| W15  | 99–105     | Multi-turn reflection + memory-aware rewriting     |
| W16  | 106–112    | Voice consistency lock-in + April victory lap      |

## Phase 5 – Days 113–140 (4 weeks)

**Goal by Day 140:** First real fine-tune on my exact voice → 6–7/10 “me” quality (up from 3–4/10 generic). Switch from prompt-only to actual parameter updates.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W17  | 113–119    | Dataset curation (2 000+ high-quality examples)    |
| W18  | 120–126    | Unsloth + QLoRA 8B setup on Windows                |
| W19  | 127–133    | Evaluation + first usable merge                    |
| W20  | 134–140    | Active learning + 6.4/10 milestone                 |

## Phase 6 – Days 141–168 (4 weeks)

**Goal by Day 168:** Clair becomes my full-time analyst + strategist (profile analysis, virality scores, follow suggestions, “what should I tweet today?”).

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W21  | 141–147    | Tool-calling mastery + X API setup                 |
| W22  | 148–154    | Profile crawler + batch analytics                  |
| W23  | 155–161    | Virality score v1 + historical comparison          |
| W24  | 162–168    | Strategy agent + daily brief                       |

## Phase 7 – Days 169–196 (4 weeks)

**Goal by Day 196:** Full multi-agent LangGraph system live — four specialized agents (Researcher, Strategist, Writer, Critic) that argue and ship every morning.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W25  | 169–175    | From single chain → true multi-agent               |
| W26  | 176–182    | Critic agent + max 3 cycles                        |
| W27  | 183–189    | Persistence + morning autonomy                     |
| W28  | 190–196    | Tool integration + < 2 min cycles                  |

## Phase 8 – Days 197–224 (4 weeks)

**Goal by Day 224:** Clair is now my personal data scientist + growth hacker. Full analytics pipeline, predictive models, auto-follow engine, unfollow hygiene.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W29  | 197–203    | Full X analytics pipeline (my account)             |
| W30  | 204–210    | Predictive analytics + what-if simulator           |
| W31  | 211–217    | Auto-follow engine v1                              |
| W32  | 218–224    | Mutuals detector + growth hacking                  |

## Phase 9 – Days 225–252 (4 weeks)

**Goal by Day 252:** Clair threads reach 9/10 “me” quality with near-zero hallucination via full DPO alignment. Agent is now better than me on a bad day.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W33  | 225–231    | Preference dataset v1 (5 000+ pairs)               |
| W34  | 232–238    | First full DPO run                                 |
| W35  | 239–245    | Hallucination death squad                          |
| W36  | 246–252    | Final alignment polish → 9/10 milestone            |

## Phase 10 – Days 253–280 (4 weeks)

**Goal by Day 280:** Full autonomy with one-click approval. 95 % of mornings I just click “Approve & Post” (or nothing at all).

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W37  | 253–259    | Morning ritual v1 – everything automatic           |
| W38  | 260–266    | One-click X posting + rollback                     |
| W39  | 267–273    | Auto-regenerate + confidence threshold             |
| W40  | 274–280    | Vacation mode + safety rails                       |

## Phase 11 – Days 281–308 (4 weeks)

**Goal by Day 308:** Clair v0.99 production-hardened, < $30/month, beautiful UI, 100 % ready for Christmas open-source launch.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W41  | 281–287    | vLLM + AWQ server (RunPod or local)                |
| W42  | 288–294    | Beautiful mobile-ready Streamlit UI                |
| W43  | 295–301    | Reliability + monitoring                           |
| W44  | 302–308    | Final polish + stress test                         |

## Phase 12 – Days 309–336 (4 weeks)

**Goal by Day 336:** Final prep + End of project mega-thread.

| Week | Days       | Week Goal                                          |
|------|------------|----------------------------------------------------|
| W45  | 309–315    | Final stress test + documentation                  |
| W46  | 316–322    | Write the legendary 98-tweet mega-thread           |
| W47  | 323–329    | Open-source + Gumroad template launch              |
| W48  | 330–336    | End of project week — watch the world explode           |

### Quick Commands You Can Use With Me (Claude)

- “Vibe Plan Phase 5”  
- “Vibe Code Day 132” (→ give me exact Unsloth QLoRA training script for Windows)  
- “What could go wrong in Phase 9?”  
- “Rewrite Day 200 in actual code”  
- “Give me the emotional pep talk for Day 300”  
- “Help me debug why FAISS index is empty on Day 17”

### Final Goal

- GitHub: 100k+ stars
- Model: Clair-v1.0-AWQ on Hugging Face
- Gumroad: $97 & $297 templates
- 98-tweet mega-thread that becomes the most bookmarked thread of 2026
- Proof that 1 hour a day for a year > any bootcamp or full-time job grinding

I’m doing this with a full-time job, family, and sanity intact.

Whenever I say “Let’s go Phase X” or “Day Y vibe check” — you know exactly what to do.
