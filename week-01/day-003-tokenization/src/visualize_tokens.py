"""
Visualize tokenization with colors.
"""

import tiktoken


def visualize_tokens(text):
    """
    Print text with each token in a different color/bracket.
    
    Args:
        text: Input text to tokenize and visualize
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    tokens = enc.encode(text)
    decoded_tokens = [enc.decode([token]) for token in tokens]
    
    print("\n" + "=" * 70)
    print(f"TEXT: {text}")
    print("=" * 70)
    print(f"Token count: {len(tokens)}")
    print()
    
    # Print with brackets around each token
    print("Tokenization (each token in brackets):")
    result = "".join(f"[{token}]" for token in decoded_tokens)
    print(result)
    print()
    
    # Print tokens with IDs
    print("Token breakdown:")
    for i, (token, token_id) in enumerate(zip(decoded_tokens, tokens)):
        # Replace newlines and tabs for visibility
        display_token = token.replace('\n', '\\n').replace('\t', '\\t').replace(' ', '␣')
        print(f"  Token {i}: '{display_token}' (ID: {token_id})")


if __name__ == "__main__":
    examples = [
        "Hello, world!",
        "The cat sat on the mat.",
        "unhappiness",
        "ChatGPT is amazing!",
        "def hello_world():\n    print('Hello!')",
        "strawberry has 3 r's",
    ]
    
    for example in examples:
        visualize_tokens(example)