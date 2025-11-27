from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
import time
import os

start_time = time.time()

# ------------------------
# Setup tokenizer
# ------------------------
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Lowercase()  # Optional: normalize text
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=50000,  # adjust depending on corpus size
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2
)

# ------------------------
# Collect all book files
# ------------------------
book_folder = "week-01/day-003-tokenization/examples/"
files = [
    os.path.join(book_folder, f)
    for f in os.listdir(book_folder)
    if f.endswith(".txt")
]

print(f"Found {len(files)} books to train tokenizer on.")

# ------------------------
# Train tokenizer on all books
# ------------------------
tokenizer.train(files, trainer=trainer)

# ------------------------
# Save tokenizer
# ------------------------
tokenizer.save("book_tokenizer.json")

end_time = time.time()
print(f"Tokenizer training completed in {end_time - start_time:.2f} seconds")
