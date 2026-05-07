# Transformer: BPE Tokenizer

## 🧠 Core Concepts

*   **BPE (Byte Pair Encoding):** A data compression algorithm adapted for AI tokenization. It starts with a base vocabulary of single bytes and repeatedly merges the most frequent adjacent pairs into single, larger tokens.
*   **UTF-8 Bytes:** Modern tokenizers operate on raw bytes (e.g., `b'hello'`) instead of characters. This ensures the tokenizer can handle *any* text, including emojis or foreign languages, without ever throwing an "Unknown" (`<UNK>`) token error.
*   **Why UTF-8 (instead of UTF-16 or UTF-32)?** UTF-8 is highly space-efficient. It uses only 1 byte for standard ASCII characters. UTF-16 and UTF-32 use 2 and 4 bytes minimum per character respectively, which would inject a massive amount of redundant zero-bytes (`\x00`) into common text. Additionally, UTF-8 is the de facto standard of the internet (~98% of webpages), making it the native format for most training corpora.
*   **Pre-tokenization (The Regex Chopper):** Before BPE merges happen, text is split using a Regular Expression (Regex). This enforces boundaries, preventing the tokenizer from accidentally merging words with punctuation (e.g., preventing `"world!"` from becoming a single token). Note that spaces are usually attached to the *start* of the following word (e.g., `" world"`).
*   **Why keep spaces attached?** Keeping the space attached to the word makes the tokenizer **lossless**. When decoding, you simply concatenate the bytes—no guessing where spaces belong. It also saves the model from having to predict a separate "space token" between every single word, which would double the sequence length.
*   **Why do `"hello"` and `" hello"` have different IDs?** Because they are different byte sequences, BPE merges them differently. `"hello"` typically appears at the start of a sentence, while `" hello"` appears in the middle. Assigning different IDs helps the model implicitly learn capitalization and formatting rules!
*   **Vocabulary (`vocab` / `encoder` / `decoder`):** The "translation dictionary" mapping integer IDs to raw bytes and vice versa. Base bytes (0-255) come first, followed by special tokens, followed by the learned BPE merges.
*   **Merges (`merges` list vs. `merge_ranks` dict):** The `train_bpe` algorithm outputs an ordered list of `merges`. When initializing the tokenizer, this list is converted into a `merge_ranks` dictionary that maps each byte pair to its index (rank) for O(1) lookups. A lower rank (e.g., 0) means the pair was learned earlier and has a **higher priority** to be merged first.
*   **Special Tokens:** Strings like `<|endoftext|>` that have a dedicated ID. They completely bypass the regex chopper and BPE algorithms.

---

## 🔄 The Three Phases of Tokenization

A BPE tokenizer's lifecycle is broken down into three distinct phases:

### 1. Training Phase (`train_bpe`)
**Goal:** Learn the vocabulary and merge rules from a massive dataset.
*   **Step 1:** Initialize the base vocabulary (bytes 0-255) and add special tokens.
*   **Step 2:** Stream a large text corpus, safely chunking it to avoid splitting words.
*   **Step 3:** Use the regex chopper to split the text into words and count the frequency of each pre-tokenized word (represented as a sequence of raw bytes).
*   **Step 4 (Pair Frequencies):** Iterate through the `word_counts` map. For every adjacent pair of bytes in a word, add that word's frequency to the pair's overall tally to find the most frequent pair globally.
*   **Step 5 (Merging & Updating):** Merge the most frequent pair into a new token, add it to the vocabulary, and create a *new* `word_counts` map. If a word contains the winning pair, replace the separate bytes with the new merged token (e.g., `(b'h', b'e', b'l', b'l', b'o')` becomes `(b'he', b'l', b'l', b'o')`) and keep its count.
*   **Step 6 (Stopping Condition):** Repeat Steps 4 and 5 exactly `vocab_size - len(base_vocab)` times. The loop stops when the vocabulary reaches the target size, or early if no more pairs can be merged.

### 2. Encoding Phase (`encode` / `encode_iterable`)
**Goal:** Convert human-readable text into a sequence of integer IDs for the AI model.
*   **Step 1:** Stream the input text and safely chunk it.
*   **Step 2:** Isolate special tokens and instantly map them to their dedicated IDs.
*   **Step 3:** Pass the remaining normal text through the regex chopper to get word chunks.
*   **Step 4 (The `_bpe` Merge Engine):** Convert each chunk to raw bytes, then break it down into a tuple of individual 1-byte symbols. Scan adjacent pairs, look them up in `merge_ranks`, and find the valid pair with the **lowest rank**. Merge that pair into a single symbol, and repeat the loop. The loop stops when no adjacent pairs exist in the `merge_ranks` rulebook.
*   **Step 5:** Map the final, merged byte sequences to their integer IDs and return the list.

### 3. Decoding Phase (`decode`)
**Goal:** Convert the AI model's output IDs back into human-readable text.
*   **Step 1:** Loop through the list of predicted integer IDs.
*   **Step 2:** Look up the raw bytes for each ID in the vocabulary dictionary.
*   **Step 3:** Concatenate all the bytes together into a single byte string (`b"".join(...)`).
*   **Step 4:** Decode the continuous byte string back into a standard UTF-8 string, using `errors="replace"` to safely handle any invalid or partial characters.

---

## 🔍 End-to-End Example Trace: Training Phase

Let's trace exactly how `train_bpe` learns its rules using a tiny dataset: `"cat cat cat car car bat"`.
Assume our target `vocab_size` is `258` (which means 256 base bytes + 2 BPE merges).

### Step 1: Initial Word Counts
The regex chopper breaks the text into words, converts them to bytes, and counts them:
```python
word_counts = {
    (b'c', b'a', b't'): 3,  # "cat" appears 3 times
    (b'c', b'a', b'r'): 2,  # "car" appears 2 times
    (b'b', b'a', b't'): 1   # "bat" appears 1 time
}
```

### Step 2: Merge Iteration 1
*   **Pair Frequencies:** The algorithm scans `word_counts` for adjacent pairs:
    *   `(b'c', b'a')` appears 5 times (3 in cat + 2 in car).
    *   `(b'a', b't')` appears 4 times (3 in cat + 1 in bat).
    *   `(b'a', b'r')` appears 2 times (in car).
*   **The Merge:** `(b'c', b'a')` is the most frequent. It becomes the new token `b'ca'` and gets assigned ID `256`.
*   **Updating Counts:** We replace `b'c'` and `b'a'` in our dictionary:
```python
word_counts = {
    (b'ca', b't'): 3,  
    (b'ca', b'r'): 2,  
    (b'b', b'a', b't'): 1
}
```

### Step 3: Merge Iteration 2
*   **Pair Frequencies:** Scanning again, `(b'ca', b't')` appears 3 times, making it the most frequent pair.
*   **The Merge:** `(b'ca', b't')` becomes `b'cat'` and gets ID `257`.
*   **Updating Counts:** `(b'ca', b't')` becomes a single element tuple `(b'cat',)`.

### Step 4: The Stopping Condition
Our vocabulary now has exactly 258 tokens (256 base bytes + 2 learned merges). The target `vocab_size` is reached, so the `_perform_merges` loop terminates! The populated `vocab` dictionary and the ordered `merges` list are returned.

---

## 🔍 End-to-End Example Trace: Encoding/Decoding Phase (`"hello world hello<|endoftext|>"`)

Let's trace exactly what happens when you call `tokenizer.encode("hello world hello<|endoftext|>")`.

### Step 1: `encode(text)`
**Function Call:** `encode("hello world hello<|endoftext|>")`
*   **What it does:** Wraps the string in a list to make it an iterable and passes it to the streaming function.
*   **Code:** `return list(self.encode_iterable(["hello world hello<|endoftext|>"]))`

### Step 2: `encode_iterable(iterable)`
**Function Call:** `encode_iterable(["hello world hello<|endoftext|>"])`
*   **What it does:** The main engine. It streams chunks of text. It immediately hands the iterable to a chunking helper to safely process the text without splitting words in half.

### Step 3: `_extract_processable_chunks`
**Function Call:** `_extract_processable_chunks(["hello world hello<|endoftext|>"], self.special_pattern)`
*   **What it does:** Scans the text for special tokens using `special_pattern.split()`. It safely isolates the special token from the normal text.
*   **Yields:** Three separate strings: `"hello world hello"`, `"<|endoftext|>"`, and `""` (empty string).

### Step 4: The Regex Chopper (`pattern.finditer`)
Back inside `encode_iterable`, it processes the yielded chunks one by one.

**Chunk 1: `"hello world hello"`**
*   It is not a special token, so it hits the `else` block to be chopped:
```python
for match in self.pattern.finditer(part):
```
*   **What it does:** The regex chops the sentence into distinct pre-tokens. 
*   **Matches Found:**
    1.  `"hello"`
    2.  `" world"` *(Notice the leading space is preserved!)*
    3.  `" hello"` *(Again, leading space preserved)*

**Chunk 2: `"<|endoftext|>"`**
*   **What it does:** It checks `if part in self.special_tokens_set:`. Since it **is** a special token, it completely bypasses the regex chopper and BPE algorithms! It directly yields the mapped ID (e.g., `256`).

**Chunk 3: `""` (Empty String)**
*   **What it does:** It hits `if not part: continue` and is ignored.

---

### Step 5: UTF-8 Encoding & `_bpe` Merging
For each regex match in Chunk 1, it converts the string to bytes and passes it to `_bpe()`.

#### Match 1: `"hello"`
**Function Call:** `_bpe(b'hello')`
*   **Initialization:** Breaks into a tuple of bytes: `symbols = (b'h', b'e', b'l', b'l', b'o')`
*   **Merge Loop:** Scans adjacent pairs and checks their priority in `self.merge_ranks`. 
    *   *Iteration 1:* `(b'h', b'e')` has the lowest rank. Merges to `(b'he', b'l', b'l', b'o')`.
    *   *Iteration 2:* Finds `(b'l', b'l')` is the best pair. Merges to `(b'he', b'll', b'o')`.
    *   *Iteration 3:* Finds `(b'he', b'll')` is the best pair. Merges to `(b'hell', b'o')`.
    *   *Iteration 4:* Finds `(b'hell', b'o')` is the best pair. Merges to `(b'hello',)`
*   **Lookup:** Converts the final tuples to IDs via `self.encoder`. 
*   **Returns:** `[31373]` *(Example ID for "hello")*

#### Match 2: `" world"`
**Function Call:** `_bpe(b' world')`
*   **Initialization:** `symbols = (b' ', b'w', b'o', b'r', b'l', b'd')`
*   **Merge Loop:** Applies learned rules.
    *   Merges `(b'o', b'r')` -> `(b' ', b'w', b'or', b'l', b'd')`
    *   Merges `(b'w', b'or')` -> `(b' ', b'wor', b'l', b'd')`
    *   Merges `(b'wor', b'l')` -> `(b' ', b'worl', b'd')`
    *   Merges `(b'worl', b'd')` -> `(b' ', b'world')`
    *   Merges `(b' ', b'world')` -> `(b' world',)`
*   **Lookup:** Finds ID for `b' world'` in `self.encoder`.
*   **Returns:** `[995]` *(Example ID for " world")*

#### Match 3: `" hello"`
**Function Call:** `_bpe(b' hello')`
*   **Initialization:** `symbols = (b' ', b'h', b'e', b'l', b'l', b'o')`
*   **Merge Loop:** Merges similarly to the first match, but incorporates the leading space.
    *   ... eventually merges down to `(b' hello',)`
*   **Lookup:** Finds ID for `b' hello'` in `self.encoder`.
*   **Returns:** `[25089]` *(Example ID for " hello". Note it is different from "hello" without a space!)*

### Step 6: Flattening the Output
As `_bpe` (and the special token handler) yields these lists of IDs back up the chain, `encode_iterable` yields them one by one. The `list(...)` constructor in the original `encode()` method collects them all into a final, flat Python list.

**Final Output:**
`[31373, 995, 25089, 256]`

### Step 7: Decoding (Translating back to text)
**Function Call:** `decode([31373, 995, 25089, 256])`
*   **What it does:** Reverses the entire process to get the original string back from the AI's output.
*   **Lookup:** It looks up each ID in `self.decoder` (the `vocab` dictionary):
    *   `31373` -> `b'hello'`
    *   `995` -> `b' world'`
    *   `25089` -> `b' hello'`
    *   `256` -> `b'<|endoftext|>'`
*   **Join & Decode:** It glues the bytes together into `b'hello world hello<|endoftext|>'` and decodes them using UTF-8.
*   **Final Output:** `"hello world hello<|endoftext|>"`
