"""
Compare tokenizer behavior on edge cases and interesting patterns.
"""

import tiktoken
from transformers import AutoTokenizer


def analyze_edge_cases():
    """Explore tokenization edge cases."""
    
    gpt4_enc = tiktoken.get_encoding("cl100k_base")
    
    print("=" * 70)
    print("EDGE CASES & INTERESTING PATTERNS")
    print("=" * 70)
    
    # Edge case 1: Repeated characters
    print("\n1. Repeated characters:")
    texts = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"]
    for text in texts:
        tokens = gpt4_enc.encode(text)
        print(f"  '{text}' → {len(tokens)} tokens")
    
    # Edge case 2: URLs
    print("\n2. URLs (high token count):")
    url = "https://www.example.com/path/to/page?param=value"
    tokens = gpt4_enc.encode(url)
    decoded = [gpt4_enc.decode([t]) for t in tokens]
    print(f"  URL: {url}")
    print(f"  Tokens ({len(tokens)}): {decoded}")
    
    # Edge case 3: Code
    print("\n3. Code (special handling):")
    code = "def hello_world():\n    print('Hello, world!')"
    tokens = gpt4_enc.encode(code)
    decoded = [gpt4_enc.decode([t]) for t in tokens]
    print(f"  Code:\n{code}")
    print(f"  Tokens ({len(tokens)}): {decoded}")
    
    # Edge case 4: Emojis
    print("\n4. Emojis:")
    emoji_text = "Hello 👋 World 🌍"
    tokens = gpt4_enc.encode(emoji_text)
    decoded = [gpt4_enc.decode([t]) for t in tokens]
    print(f"  Text: {emoji_text}")
    print(f"  Tokens ({len(tokens)}): {decoded}")
    
    # Edge case 5: Whitespace
    print("\n5. Whitespace handling:")
    texts = ["hello", " hello", "  hello", "hello ", "hello  "]
    for text in texts:
        tokens = gpt4_enc.encode(text)
        print(f"  '{text}' → {len(tokens)} tokens: {tokens}")
    
    # Edge case 6: Numbers
    print("\n6. Number tokenization:")
    numbers = ["1", "12", "123", "1234", "12345", "123456"]
    for num in numbers:
        tokens = gpt4_enc.encode(num)
        decoded = [gpt4_enc.decode([t]) for t in tokens]
        print(f"  '{num}' → {decoded}")
    
    # Edge case 7: Letter counting challenge
    print("\n7. Why GPT struggles with 'How many r's in strawberry?':")
    word = "strawberry"
    tokens = gpt4_enc.encode(word)
    decoded = [gpt4_enc.decode([t]) for t in tokens]
    print(f"  Word: '{word}'")
    print(f"  Tokens: {decoded}")
    print(f"  Actual 'r' count: {word.count('r')}")
    print(f"  But model sees {len(decoded)} tokens, not {len(word)} letters!")


def analyze_efficiency():
    """Compare tokenization efficiency across languages and domains."""
    
    gpt4_enc = tiktoken.get_encoding("cl100k_base")
    
    print("\n" + "=" * 70)
    print("TOKENIZATION EFFICIENCY")
    print("=" * 70)
    
    test_cases = [
        ("English", "The quick brown fox jumps over the lazy dog."),
        ("Spanish", "El rápido zorro marrón salta sobre el perro perezoso."),
        ("German", "Der schnelle braune Fuchs springt über den faulen Hund."),
        ("French", "Le rapide renard brun saute par-dessus le chien paresseux."),
        ("Japanese", "素早い茶色のキツネが怠惰な犬を飛び越える。"),
        ("Code", "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr"),
        ("Math", "∫₀^∞ e^(-x²) dx = √π/2"),
        ("Technical", "The GPU processes tensors using CUDA kernels for parallel computation."),
    ]
    
    print("\nTokens per character (lower is more efficient):")
    print()
    
    for name, text in test_cases:
        tokens = gpt4_enc.encode(text)
        ratio = len(tokens) / len(text)
        print(f"  {name:12s}: {len(text):3d} chars → {len(tokens):3d} tokens ({ratio:.2f} tokens/char)")


def count_special_tokens():
    """Show special tokens in different tokenizers."""
    
    print("\n" + "=" * 70)
    print("SPECIAL TOKENS")
    print("=" * 70)
    
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    print("\nLlama special tokens:")
    print(f"  BOS (beginning of sequence): '{llama_tokenizer.bos_token}' → {llama_tokenizer.bos_token_id}")
    print(f"  EOS (end of sequence): '{llama_tokenizer.eos_token}' → {llama_tokenizer.eos_token_id}")
    print(f"  PAD (padding): '{llama_tokenizer.pad_token}' → {llama_tokenizer.pad_token_id}")
    print(f"  UNK (unknown): '{llama_tokenizer.unk_token}' → {llama_tokenizer.unk_token_id}")
    
    # Show encoding with and without special tokens
    text = "Hello, world!"
    
    print(f"\nEncoding '{text}':")
    tokens_without = llama_tokenizer.encode(text, add_special_tokens=False)
    tokens_with = llama_tokenizer.encode(text, add_special_tokens=True)
    
    print(f"  Without special tokens: {tokens_without} ({len(tokens_without)} tokens)")
    print(f"  With special tokens: {tokens_with} ({len(tokens_with)} tokens)")


if __name__ == "__main__":
    analyze_edge_cases()
    analyze_efficiency()
    count_special_tokens()