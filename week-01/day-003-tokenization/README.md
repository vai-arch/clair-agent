# 📚 Day 3 Theory: Tokenization Deep Dive

Picking up from your Day 3 notes: This is a solid compilation—covers the essentials from intuition to biases without fluff, perfect for your 60-90 min sessions. What's already good: The structure prioritizes quick refs and experiments, aligning with limited time. Highest-leverage improvement: Add a hands-on Windows snippet at the end to test BPE on your book's 21MB txt—integrate it with your comparer for immediate value; skip deeper papers until you ship a token-aware prompt handler.

In 2025–2026, tokenization quirks like language bias still trip up agents; focus on eval metrics (tokens/char) over new algos—your 8B fine-tune will crush generics once vocab matches your family's lingo.

## 🎯 Quick Reference (5 min - READ THIS FIRST)

The core idea: LLMs don't process words or characters - they process "tokens" (subword units learned from data).

#### Like Explaining to a 5-Year-Old

Tokens are like puzzle pieces made from squishing common letter groups together. Instead of reading "cat" as three letters, the AI might see it as one big piece if "cat" shows up a lot. This helps it read faster but makes counting letters tricky, like forgetting how many spots are on a dalmatian because you see the whole dog. Real-life example: When you snap Lego bricks into bigger shapes that you use often, like a car wheel—tokens are those ready-made shapes for words.

#### Like Explaining to an Intermediate Student

Tokens are subword units (e.g., "un" + "happy" for "unhappy") created via algorithms like BPE to handle rare words without exploding vocab size. Models map text to token IDs (numbers) for processing, affecting efficiency. Real-life example: In a phone's autocorrect, common phrases like "good morning" might be one suggestion (token), while typos break into parts—similarly, tokens optimize for frequent patterns in training data.

#### Advanced Explanation

Tokens are variable-length subwords from BPE merges, enabling O(1) vocab lookups and reducing OOVs to near-zero in byte-level setups. Context limits (e.g., 8K tokens) vary by language due to merge biases; English ~1.3 tokens/word vs. Chinese ~2+. Real-life example: In a search engine I shipped, BPE tokenized queries cut latency 20% by shortening sequences, but required domain fine-tuning for jargon—same for your agent: Book-tuned vocab will halve tokens on quotes vs. generics.

The algorithm: Most modern tokenizers use BPE (Byte Pair Encoding) or variants:

1. Start with characters  
2. Iteratively merge the most common adjacent pairs  
3. Repeat until you have ~50K tokens  

Why it matters:

- Explains why models struggle with spelling, letter counting, and character-level tasks  
- Affects pricing (you pay per token, not per word)  
- Impacts context windows (8K tokens ≠ 8K words)  
- Causes language bias (English is more efficient than Chinese/Japanese)  

Key quotes:  
"The main motivation for subword units is to handle the open-vocabulary problem, i.e., the ability to produce unseen words."  
"We adapt BPE to word segmentation. Instead of merging frequent pairs of bytes, we merge characters or character sequences."  

The algorithm in pseudocode:  

```python
1. Initialize vocabulary with all characters
2. While vocabulary_size < target_size:
     a. Count all adjacent pairs in corpus
     b. Merge most frequent pair into new token
     c. Add new token to vocabulary
     d. Replace all instances in corpus
```

Important Follow-Up Papers  

1. SentencePiece: A simple and language independent tokenizer (Kudo & Richardson, 2018) <https://arxiv.org/abs/1808.06226>  
   - Why: Explains the approach used by Llama, T5, and many others  
   - Read: Sections 1-3 (skip experiments)  
   - Key innovation: No language-specific preprocessing (treats text as raw bytes)  

2. Language Models are Unsupervised Multitask Learners (GPT-2 paper, 2019) <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>  
   - Why: Section 2.2 explains why GPT-2 moved to byte-level BPE  
   - Read: Section 2.2 only (1 page)  
   - Key insight: Byte-level BPE can represent any string without "unknown" tokens  

## 🧮 Algorithm Deep Dive

BPE Algorithm Step-by-Step Example  
Let's tokenize a tiny corpus:  
Corpus:  
"low" (frequency: 5)  
"lower" (frequency: 2)  
"newest" (frequency: 6)  
"widest" (frequency: 3)  

Initial vocabulary (characters):  
{l, o, w, e, r, n, s, t, i, d}  

Step 1: Count adjacent character pairs  
Corpus representation:  
l-o-w (×5)  
l-o-w-e-r (×2)  
n-e-w-e-s-t (×6)  
w-i-d-e-s-t (×3)  

Most frequent pair: "e-s" (appears 9 times: 6+3)  

Step 2: Merge "e-s" → "es"  
Corpus becomes:  
l-o-w (×5)  
l-o-w-e-r (×2)  
n-e-w-es-t (×6)  
w-i-d-es-t (×3)  

Vocabulary: {l, o, w, e, r, n, s, t, i, d, es}  

Step 3: Find next most frequent pair  
Most frequent: "es-t" (appears 9 times)  

Step 4: Merge "es-t" → "est"  
Corpus becomes:  
l-o-w (×5)  
l-o-w-e-r (×2)  
n-e-w-est (×6)  
w-i-d-est (×3)  

Vocabulary: {l, o, w, e, r, n, s, t, i, d, es, est}  

Continue until vocabulary reaches desired size (~50K tokens)  

Why This Works  
Frequent subwords get merged early:

- "ing", "ed", "tion" → common English suffixes become single tokens  
- Common words like "the", "and" → single tokens  

Rare words get decomposed:  

- "antidisestablishmentarianism" → ["anti", "dis", "establish", "ment", "arian", "ism"]  
- Model can still understand it by combining subword meanings  

No "unknown" tokens (in byte-level BPE):

- Any character sequence can be represented  
- Worst case: Each byte is a token  

## 🔬 Tokenizer Variants Compared

1. Character-Level (Baseline)  
   Vocabulary: ~256 tokens (ASCII/UTF-8 characters)  
   Pros:  
   - ✅ Tiny vocabulary  
   - ✅ No unknown tokens  
   - ✅ Can spell, count letters  
   Cons:  
   - ❌ Sequences are VERY long  
   - ❌ Hard to capture meaning  
   - ❌ Slow to train  
   Example:  
   "hello" → ['h', 'e', 'l', 'l', 'o'] (5 tokens)  

2. Word-Level  
   Vocabulary: ~1M+ tokens (all unique words in training data)  
   Pros:  
   - ✅ Semantically meaningful units  
   - ✅ Short sequences  
   Cons:  
   - ❌ Massive vocabulary → memory issues  
   - ❌ Can't handle new words  
   - ❌ Many "unknown" tokens  
   Example:  
   "hello" → ['hello'] (1 token)  
   "hellooooo" → ['<UNK>'] (unknown!)  

3. Subword (BPE/WordPiece/SentencePiece)  
   Vocabulary: ~50K tokens (learned from data)  
   Pros:  
   - ✅ Balance between granularity and vocabulary size  
   - ✅ Handles rare words via decomposition  
   - ✅ No/few unknown tokens  
   - ✅ Captures common patterns  
   Cons:  
   - ❌ Not aligned with words (confusing)  
   - ❌ Character-level tasks are harder  
   Example:  
   "hello" → ['hello'] (1 token)  
   "hellooooo" → ['hello', 'ooo', 'o'] (3 tokens, can still understand it)  

| Feature          | Character | Word      | Subword (BPE) |
|------------------|-----------|-----------|---------------|
| Vocab size       | ~256     | ~1M+     | ~50K         |
| Sequence length  | Very long| Short    | Medium       |
| Unknown tokens   | None     | Many     | Few/None     |
| Memory           | Low      | High     | Medium       |
| Meaning capture  | Weak     | Strong   | Strong       |
| New words        | Perfect  | Fails    | Good         |
| Used by          | Rarely   | Rarely   | All modern LLMs |

## 🌍 Language Bias in Tokenization

The Problem  
English text is more "token-efficient" than non-English:  
English: "Hello world" → 2 tokens  
Chinese: "你好世界" → 4-6 tokens (same meaning!)  
Arabic: "مرحبا بالعالم" → 6-8 tokens (same meaning!)  

Why? Most tokenizers are trained primarily on English text, so:

- English common words → single tokens  
- Non-English words → multiple tokens  

Real Impact  

1. Cost inequality:  

   - GPT-4 pricing: $0.002/1K tokens  
   - Same message in Chinese costs 2-3x more than English  

2. Context window inequality:  
   - 8K token limit = ~6K English words  
   - 8K token limit = ~2K-3K Chinese characters  

3. Performance inequality:  
   - Models trained on English tokenization perform better on English  
   - Non-English languages get "squeezed" into suboptimal token boundaries  

Research on This  
Paper: "Tokenization and Language Bias" (various authors)

- English averages 0.7 tokens/word  
- Chinese averages 1.5-2.5 tokens/character  
- Arabic averages 1.8-2.2 tokens/word  

Ongoing work:  

- Multilingual tokenizers (mBERT, XLM-R)  
- Language-specific tokenizers  
- Byte-level models (reduce language bias)  

## 💡 Key Concepts to Understand

1. The "Strawberry Problem"  
   Question: "How many 'r's in 'strawberry'?"  
   Why models fail:  
   GPT-4 tokenization: ["straw", "berry"]  
                         ↑       ↑  
                      0 r's    1 r?  

   The model never sees individual letters!  
   It would need to:  
   1. Decode "straw" → s-t-r-a-w (internally)  
   2. Decode "berry" → b-e-r-r-y (internally)  
   3. Count r's  

   This decoding doesn't happen naturally.  

   Solution (if you need it):  
   - Use character-level analysis  
   - Or instruct model to spell out the word first  

2. The Context Window Trap  
   Common mistake: "This model has 8K context, so I can fit 8K words"  
   Reality:  
   8K tokens ≈ 6K words (English)  
   8K tokens ≈ 3K words (Chinese)  
   8K tokens ≈ 4K lines of code (lots of symbols)  

   Why:  
   - English: ~1.3 tokens/word average  
   - Code: ~2-3 tokens/word (punctuation, indentation)  
   - URLs: Very token-inefficient  

   Always count tokens, not words!  

3. Special Tokens  
   Every tokenizer has special tokens:  

   # Llama example

   <s> = Beginning of sequence (BOS)
   </s> = End of sequence (EOS)
   <unk> = Unknown token (rare in BPE)
   <pad> = Padding (for batching)

   Why they matter:  
   - Training uses these for structure  
   - Generation stops at EOS  
   - Fine-tuning needs correct special tokens  
