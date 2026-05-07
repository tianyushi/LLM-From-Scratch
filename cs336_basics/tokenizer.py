import os
import re
import regex
import collections
from typing import Any, Iterator


def _build_regexes(special_tokens: list[str] | None) -> tuple[regex.Pattern, regex.Pattern | None]:
    """
    Build the standard GPT-2 regex pattern and the special tokens pattern.
    
    Args:
        special_tokens: A list of special token strings to recognize, or None.
        
    Returns:
        A tuple containing the GPT-2 regex pattern and the special tokens regex pattern (or None).
    """
    # This regex splits text into categories: contractions ('s, 't), letters, numbers,
    # punctuation, and whitespace, keeping preceding spaces with the words.
    pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    if special_tokens:
        # Sort special tokens by length descending so longer tokens are matched first
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        # Create a single regex grouping all special tokens separated by OR (|)
        special_pattern = regex.compile("(" + "|".join(map(regex.escape, sorted_special)) + ")")
    else:
        special_pattern = None
    return pattern, special_pattern


def _extract_processable_chunks(iterable: Any, special_pattern: regex.Pattern | None) -> Iterator[str]:
    """
    Yield processable text chunks from an iterable, avoiding splitting words.
    
    Args:
        iterable: An iterable yielding text chunks (e.g., a file object).
        special_pattern: Optional regex pattern used to identify special tokens.
        
    Yields:
        Chunks of text safely split at whitespace boundaries.
    """
    buffer = ""
    for chunk in iterable:
        buffer += chunk
        # Wait until we have a decently sized chunk (64KB) to process
        if len(buffer) < 65536:
            continue
            
        if special_pattern:
            # Split by special tokens; the last part might be an incomplete word/token
            parts = special_pattern.split(buffer)
            last_part = parts[-1]
        else:
            last_part = buffer
            
        # Find the last whitespace character followed by a non-whitespace character
        # We look a bit before the very end ([:-1000]) to ensure we don't cut mid-word
        matches = list(re.finditer(r'\S\s', last_part[:-1000]))
        if matches:
            # Split right after the whitespace
            split_point = matches[-1].start() + 1
            to_process_len = len(buffer) - len(last_part) + split_point
            to_process = buffer[:to_process_len]
            buffer = buffer[to_process_len:]
            yield from (special_pattern.split(to_process) if special_pattern else [to_process])
        elif len(buffer) > 1000000:
            # Fallback: if no good boundary is found and the buffer is huge, just split it
            to_process = buffer[:-1000]
            buffer = buffer[-1000:]
            yield from (special_pattern.split(to_process) if special_pattern else [to_process])
            
    # Yield whatever is left over at the end
    if buffer:
        yield from (special_pattern.split(buffer) if special_pattern else [buffer])


def _get_word_counts(
    input_path: str | os.PathLike,
    pattern: regex.Pattern,
    special_pattern: regex.Pattern | None,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Read the input file and count the frequency of each pre-tokenized word.
    
    Args:
        input_path: Path to the text file to be processed.
        pattern: The compiled GPT-2 regex pattern to split words.
        special_pattern: The compiled regex pattern for special tokens.
        special_tokens: A list of special token strings to ignore during sub-word counting.
        
    Returns:
        A dictionary mapping pre-tokenized words (as tuples of bytes) to their occurrence counts.
    """
    word_counts = collections.defaultdict(int)
    special_tokens_set = set(special_tokens) if special_tokens else set()
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    if special_pattern:
        parts = special_pattern.split(text)
    else:
        parts = [text]
        
    for part in parts:
        if not part or part in special_tokens_set:
            continue
        
        for match in pattern.finditer(part):
            # Convert the string match into individual UTF-8 bytes
            token_bytes = match.group().encode("utf-8")
            word_counts[tuple(bytes([b]) for b in token_bytes)] += 1
                    
    return dict(word_counts)


def _perform_merges(
    word_counts: dict[tuple[bytes, ...], int],
    num_merges: int,
    vocab: dict[int, bytes]
) -> list[tuple[bytes, bytes]]:
    """
    Perform BPE merges based on word counts, updating vocab and returning merges.

    Uses incremental pair-count updates: pair_counts is built once up front,
    then only the pairs belonging to words affected by each merge are
    subtracted (old word) and re-added (merged word).  This avoids the O(N)
    full rescan that the naive approach performs on every iteration.
    """
    # 1. Build initial pair counts once
    pair_counts = collections.defaultdict(int)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += count

    merges = []
    for _ in range(num_merges):
        # Remove stale zero-count entries so max() stays correct
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # 2. Apply merge only to affected words and update pair_counts incrementally
        new_word_counts = {}
        for word, count in word_counts.items():
            if len(word) < 2 or best_pair[0] not in word or best_pair[1] not in word:
                new_word_counts[word] = count
                continue

            # Subtract this word's contribution to pair_counts
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] -= count

            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_t = tuple(new_word)
            new_word_counts[new_word_t] = count

            # Add the new word's contribution back to pair_counts
            for i in range(len(new_word_t) - 1):
                pair_counts[(new_word_t[i], new_word_t[i + 1])] += count

        word_counts = new_word_counts

        # Prune dead entries periodically to keep max() fast
        pair_counts = collections.defaultdict(
            int, {p: c for p, c in pair_counts.items() if c > 0}
        )

    return merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer and output its vocabulary and merges.
    """
    # Initialize vocabulary with the 256 base byte values
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens to the vocabulary
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in vocab.values():
            vocab[len(vocab)] = st_bytes

    pattern, special_pattern = _build_regexes(special_tokens)
    word_counts = _get_word_counts(input_path, pattern, special_pattern, special_tokens)
    
    # Calculate how many BPE merges we need to perform
    num_merges = vocab_size - len(vocab)
    merges = _perform_merges(word_counts, num_merges, vocab)

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_set = set(self.special_tokens)
        
        self.decoder = self.vocab
        # Create reverse mapping for fast token -> ID lookups
        self.encoder = {v: k for k, v in self.vocab.items()}
        # Create a mapping of pair -> rank (index in merges) for O(1) lookups
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.cache = {}
        
        self.pattern, self.special_pattern = _build_regexes(self.special_tokens)

    def _bpe(self, chunk_bytes: bytes) -> list[int]:
        """
        Apply Byte Pair Encoding to a chunk of bytes.
        """
        if chunk_bytes in self.cache:
            return self.cache[chunk_bytes]
            
        symbols = list(bytes([b]) for b in chunk_bytes)
        if not symbols:
            return []
        
        while len(symbols) > 1:
            min_rank = float('inf')
            best_pair = None
            
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            new_symbols = []
            i = 0
            new_token = best_pair[0] + best_pair[1]
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            
        result = [self.encoder[token] for token in symbols]
        self.cache[chunk_bytes] = result
        return result

    def encode(self, text: str) -> list[int]:
        """
        Encode a string of text into a list of token IDs.
        """
        return list(self.encode_iterable([text]))

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        b = b"".join([self.decoder[i] for i in ids])
        return b.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Any) -> Iterator[int]:
        """
        Stream chunks from an iterable and encode them efficiently.
        """
        for part in _extract_processable_chunks(iterable, self.special_pattern):
            if not part:
                continue
            if part in self.special_tokens_set:
                yield self.encoder[part.encode("utf-8")]
            else:
                for match in self.pattern.finditer(part):
                    yield from self._bpe(match.group().encode("utf-8"))
