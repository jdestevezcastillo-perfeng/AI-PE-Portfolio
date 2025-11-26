"""
BYTE-PAIR ENCODING (BPE) DIAGRAM
--------------------------------

   [Raw Text]       "h u g g i n g"
       |
       v
   [Base Vocab]     ['h', 'u', 'g', 'i', 'n'] (Characters)
       |
       v
   [Count Pairs]    ('g', 'g') appears most often
       |
       v
   [Merge Rule]     'g' + 'g' -> 'gg' (New Token)
       |
       v
   [Iteration]      Repeat until vocab_size reached
       |
       v
   [Final Vocab]    Includes 'hug', 'ging', 'transformer', etc.

   >>> AIPE NOTE: Why BPE?
   - Character-level (Lab 02): Too long sequences, weak semantic meaning.
   - Word-level: Vocab too huge (millions of words), "OOV" (Out Of Vocabulary) issues.
   - Subword (BPE): The "Goldilocks" zone. Common words are single tokens, rare words are split.
     Example: "unaffordability" -> "un", "afford", "ability"
"""


from typing import Dict, List, Tuple

# ==========================================
# 1. CONFIGURATION & DATA
# ==========================================

# We'll use the same raw text as Lab 02 for consistency, plus some modern terms.
TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Transformation is key. The transformer architecture relies on tokenization.
"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Counts the frequency of all adjacent pairs in the list of integers.
    e.g., [1, 2, 1, 2, 3] -> {(1, 2): 2, (2, 3): 1, (2, 1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    Replaces all occurrences of `pair` with the new token `idx`.
    e.g., ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # Check if we are at the start of the pair and not at the very end of list
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# ==========================================
# 3. BPE TRAINING CLASS
# ==========================================

class SimpleBPE:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocab = {}  # int -> bytes
        self.special_tokens = {} # str -> int (not implemented for this simple demo)

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        """
        Trains the BPE tokenizer on the given text until vocab_size is reached.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Input text preprocessing
        # In a real tokenizer (like GPT-4), we use regex to split text into words first
        # to prevent merging across punctuation (e.g., "dog." -> "dog" + ".").
        # For simplicity here, we just convert raw bytes.
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # Convert bytes to list of integers [0..255]

        print(f"Original length (bytes): {len(ids)}")
        print(f"Vocab size target: {vocab_size} (256 base + {num_merges} merges)")
        print("-" * 40)

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                print("No more pairs to merge!")
                break

            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            
            # Mint a new token ID
            idx = 256 + i
            
            # Apply the merge
            ids = merge(ids, pair, idx)
            
            # Save the rule
            self.merges[pair] = idx
            
            if verbose:
                # Decode the pair to see what text it represents
                # Note: This simple decode might fail for partial utf-8 bytes, but works for ASCII
                try:
                    p1_bytes = bytes([pair[0]]) if pair[0] < 256 else self.vocab.get(pair[0], b"")
                    p2_bytes = bytes([pair[1]]) if pair[1] < 256 else self.vocab.get(pair[1], b"")
                    merged_bytes = p1_bytes + p2_bytes
                    # Store in vocab for later use
                    self.vocab[idx] = merged_bytes
                    
                    print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} | '{merged_bytes.decode('utf-8', errors='replace')}' (count: {stats[pair]})")
                except:
                    print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")

        print("-" * 40)
        print(f"Final length (tokens): {len(ids)}")
        print(f"Compression ratio: {len(text_bytes) / len(ids):.2f}X")

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of integers using learned merges.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair with the lowest merge index (earliest learned rule)
            # This ensures we apply merges in the same order we learned them
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # No more applicable merges
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of integers back to a string.
        """
        # 1. Build the vocab mapping from ID -> Bytes
        # (In a real class, we'd maintain this during training)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            
        # 2. Concatenate bytes
        tokens = b"".join(vocab[idx] for idx in ids)
        
        # 3. Decode to string
        text = tokens.decode("utf-8", errors="replace")
        return text

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    print(">>> AIPE NOTE: BPE Training")
    print("We start with raw bytes (UTF-8) and iteratively merge the most common pairs.")
    print("Watch how it learns common words like 'the', 'and', 'ing'.\n")

    tokenizer = SimpleBPE()
    
    # Train on our corpus
    # We choose a small vocab size for demonstration (256 base + 20 merges = 276)
    tokenizer.train(TEXT, vocab_size=300)

    print("\n>>> AIPE NOTE: Verification")
    test_str = "The transformer architecture is efficient."
    print(f"Input: '{test_str}'")
    
    encoded = tokenizer.encode(test_str)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    assert decoded == test_str, "Error: Decoded string does not match input!"
    print("SUCCESS: Encoding <-> Decoding cycle verified.")

if __name__ == "__main__":
    main()
