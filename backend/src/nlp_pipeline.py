"""
nlp_pipeline.py — Comprehensive NLP pipeline for the DGS text-to-sign system.

Implements course requirements across four exercises:

Lab Experiment 1  — Tokenization & Text Normalization
  • Word, Sentence, Character, Subword (BPE-style) tokenization
  • Stop-word removal, Stemming (SnowballStemmer German), Lemmatization

Feature Engineering Exercise
  • Bag-of-Words (BoW) document-term matrix
  • N-gram features (unigram / bigram / trigram)
  • TF-IDF representation
  • Statistical co-occurrence and PMI / PPMI
  • Word embeddings (via sentence-transformers)

Exercise 3 — N-Gram Language Models for Translation
  • Unigram / Bigram / Trigram language models over gloss sequences
  • Laplace (add-one) smoothing
  • Translation candidate scoring
  • Perplexity computation

Exercise 4 — Sequence Labeling (HMM POS Tagging)
  • HMM estimation (transition + emission probabilities)
  • Viterbi decoding
  • Applied to German text to improve gloss ordering
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GERMAN_STOPWORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein",
    "unser", "euer", "ja", "nein", "nicht", "auch", "aber", "und", "oder",
    "weil", "dass", "als", "wenn", "dann", "so", "wie", "was", "wo", "wann",
    "warum", "der", "die", "das", "ein", "eine", "einen", "einem", "einer",
    "ist", "bin", "bist", "sind", "war", "hatte", "hat", "haben", "mir",
    "dir", "ihm", "uns", "euch", "ihnen", "na", "mal", "doch", "noch",
    "schon", "hier", "da", "jetzt", "immer", "sehr", "mehr", "nur", "man",
    "an", "auf", "bei", "bis", "durch", "für", "gegen", "in", "mit", "nach",
    "ohne", "über", "um", "unter", "vor", "von", "zu", "zum", "zur",
    "sich", "worden", "werden", "wurde", "kann", "muss", "soll", "will",
    "darf", "mag", "diesem", "dieser", "diesen", "dieses", "diese", "jeder",
    "jeden", "jede", "jedes", "all", "alle", "allem", "allen", "alles",
}

# Simple German word boundary pattern
_WORD_RE = re.compile(r"[a-zA-ZäöüÄÖÜß]+", flags=re.UNICODE)

# ============================================================================
# Lab 1 — Tokenization & Text Normalization
# ============================================================================


def word_tokenize(text: str) -> List[str]:
    """Word-level tokenization for German text."""
    return _WORD_RE.findall(text)


def sentence_tokenize(text: str) -> List[str]:
    """Sentence tokenization using punctuation heuristics."""
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text, language="german")
        except Exception:
            return nltk.sent_tokenize(text)
    except ImportError:
        # Fallback: split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]


def char_tokenize(text: str) -> List[str]:
    """Character-level tokenization (useful for noisy/misspelled text)."""
    return list(text)


def subword_tokenize_bpe(text: str, vocab_size: int = 50) -> List[str]:
    """
    Byte-Pair Encoding (BPE) style subword tokenization.
    Implementation follows the original BPE paper (Sennrich et al., 2016).
    Starts from character-level and merges most frequent pairs iteratively.
    Returns subword tokens for the given text.
    """
    words = word_tokenize(text.lower())
    if not words:
        return []

    # Build character-split representation with end-of-word marker
    word_freqs: Dict[Tuple[str, ...], int] = Counter()
    for w in words:
        chars = tuple(list(w) + ["</w>"])
        word_freqs[chars] += 1

    def get_pairs(vocab: Dict[Tuple[str, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_pair(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]):
        new_vocab: Dict[Tuple[str, ...], int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    # Perform merges
    num_merges = min(vocab_size, len(word_freqs) * 3)
    for _ in range(num_merges):
        pairs = get_pairs(word_freqs)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        word_freqs = merge_pair(best, word_freqs)

    # Tokenize text using learned vocab
    tokens = []
    for w in words:
        best_toks = tuple(list(w) + ["</w>"])
        # Replace learned merges: just reconstruct from word_freqs keys
        for merged_word in word_freqs:
            if "".join(merged_word).replace("</w>", "") == w:
                best_toks = merged_word
                break
        tokens.extend([t.replace("</w>", "") for t in best_toks if t != "</w>"])
    return tokens


def remove_stopwords(tokens: List[str], language: str = "german") -> List[str]:
    """Remove German stop words from a token list."""
    return [t for t in tokens if t.lower() not in _GERMAN_STOPWORDS]


def stem_tokens(tokens: List[str], language: str = "german") -> List[str]:
    """Apply SnowballStemmer (German) to a list of tokens."""
    try:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer(language)
        return [stemmer.stem(t) for t in tokens]
    except Exception:
        return [t.lower() for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatization via rule-based suffix stripping for German
    (lightweight, no spaCy dependency required).
    Common German inflectional endings are stripped to approximate lemmas.
    """
    suffixes = [
        ("ungen", "ung"), ("enden", "end"), ("enden", "end"),
        ("ieren", "ieren"), ("ierte", "ieren"), ("ierten", "ieren"),
        ("ungen", "ung"), ("ungen", "ung"),
        ("sten", "st"), ("sten", ""),
        ("enden", "en"), ("endem", "en"),
        ("heit", "heit"), ("keit", "keit"),
        ("lich", "lich"), ("isch", "isch"), ("haft", "haft"),
        ("en", "en"), ("em", ""), ("er", ""), ("es", ""), ("est", ""),
        ("ten", "t"), ("nen", "n"),
        ("le", "l"), ("re", "r"),
    ]
    result = []
    for tok in tokens:
        t = tok.lower()
        lemma = t
        for suffix, replacement in suffixes:
            if len(t) > len(suffix) + 2 and t.endswith(suffix):
                lemma = t[: -len(suffix)] + replacement
                break
        result.append(lemma)
    return result


def full_tokenization_analysis(text: str) -> dict:
    """
    Lab 1 complete analysis: all tokenization and normalization stages.
    Returns a structured dict for tabular display.
    """
    words = word_tokenize(text)
    words_lower = [w.lower() for w in words]
    sentences = sentence_tokenize(text)
    chars = char_tokenize(text)
    subwords = subword_tokenize_bpe(text)
    no_stop = remove_stopwords(words_lower)
    stems = stem_tokens(words_lower)
    lemmas = lemmatize_tokens(words_lower)

    return {
        "original": text,
        "word_tokens": words,
        "sentence_tokens": sentences,
        "char_tokens": chars,
        "subword_tokens_bpe": subwords,
        "after_stopword_removal": no_stop,
        "stemmed_tokens": stems,
        "lemmatized_tokens": lemmas,
        "stats": {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "char_count": len(chars),
            "subword_count": len(subwords),
            "unique_words": len(set(words_lower)),
        },
    }


# ============================================================================
# Feature Engineering — BoW, N-gram, TF-IDF, PMI, Embeddings
# ============================================================================


def build_bow(corpus: List[str]) -> Tuple[List[List[int]], List[str]]:
    """
    Build a Bag-of-Words document-term matrix from a corpus.
    Returns (matrix, vocabulary) where matrix[i][j] = count of vocab[j] in doc[i].
    """
    vocab_set: set = set()
    tokenized = []
    for doc in corpus:
        tokens = [t.lower() for t in word_tokenize(doc)]
        tokenized.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    matrix = []
    for tokens in tokenized:
        row = [0] * len(vocab)
        for t in tokens:
            if t in vocab_idx:
                row[vocab_idx[t]] += 1
        matrix.append(row)
    return matrix, vocab


def build_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-gram tuples from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_tfidf(corpus: List[str]) -> Tuple[List[List[float]], List[str]]:
    """
    TF-IDF representation using sklearn's TfidfVectorizer.
    Returns (matrix as list-of-lists, feature names).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-ZäöüÄÖÜß]+\b")
        X = vectorizer.fit_transform(corpus)
        return X.toarray().tolist(), list(vectorizer.get_feature_names_out())
    except Exception:
        # Fallback: manual TF-IDF
        bow, vocab = build_bow(corpus)
        n_docs = len(corpus)
        doc_freq = [
            sum(1 for row in bow if row[j] > 0) for j in range(len(vocab))
        ]
        result = []
        for row in bow:
            total = sum(row) or 1
            tfidf_row = []
            for j, cnt in enumerate(row):
                tf = cnt / total
                idf = math.log((n_docs + 1) / (doc_freq[j] + 1)) + 1
                tfidf_row.append(round(tf * idf, 4))
            result.append(tfidf_row)
        return result, vocab


def compute_pmi(
    corpus: List[str], window: int = 2, positive_only: bool = True
) -> Dict[Tuple[str, str], float]:
    """
    Compute Pointwise Mutual Information (PMI) / PPMI between word pairs.
    Uses a sliding window approach over all documents.
    PPMI clamps negative values to 0 (default).
    """
    word_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    total_tokens = 0

    for doc in corpus:
        tokens = [t.lower() for t in word_tokenize(doc)]
        for i, w in enumerate(tokens):
            word_counts[w] += 1
            total_tokens += 1
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    pair_counts[(w, tokens[j])] += 1

    pmi: Dict[Tuple[str, str], float] = {}
    pair_total = sum(pair_counts.values()) or 1
    for (w1, w2), cnt in pair_counts.items():
        p_w1 = word_counts[w1] / total_tokens
        p_w2 = word_counts[w2] / total_tokens
        p_w1w2 = cnt / pair_total
        if p_w1 > 0 and p_w2 > 0 and p_w1w2 > 0:
            raw = math.log2(p_w1w2 / (p_w1 * p_w2))
            pmi[(w1, w2)] = max(0.0, raw) if positive_only else raw

    return pmi


def top_pmi_pairs(
    corpus: List[str], n: int = 20, window: int = 2
) -> List[Tuple[Tuple[str, str], float]]:
    """Return the top-n PPMI word pairs sorted by score."""
    pmi = compute_pmi(corpus, window=window)
    return sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:n]


def feature_summary(corpus: List[str]) -> dict:
    """
    Feature Engineering summary for a corpus:
    BoW sparsity, N-gram vocab sizes, TF-IDF top terms, top PMI pairs.
    """
    bow, bow_vocab = build_bow(corpus)
    total_cells = len(bow) * len(bow_vocab)
    zero_cells = sum(1 for row in bow for v in row if v == 0)
    sparsity = zero_cells / total_cells if total_cells else 0.0

    all_tokens: List[str] = []
    for doc in corpus:
        all_tokens.extend(t.lower() for t in word_tokenize(doc))

    unigram_vocab = len(set(all_tokens))
    bigrams = build_ngrams(all_tokens, 2)
    trigrams = build_ngrams(all_tokens, 3)
    bigram_vocab = len(set(bigrams))
    trigram_vocab = len(set(trigrams))

    tfidf, tfidf_vocab = compute_tfidf(corpus)
    # Top 10 TF-IDF terms by average score
    avg_tfidf = [
        (tfidf_vocab[j], sum(row[j] for row in tfidf) / len(tfidf))
        for j in range(len(tfidf_vocab))
    ]
    top_tfidf = sorted(avg_tfidf, key=lambda x: x[1], reverse=True)[:10]

    top_pmi = top_pmi_pairs(corpus, n=10)

    return {
        "corpus_size": len(corpus),
        "bow": {
            "vocab_size": len(bow_vocab),
            "sparsity_pct": round(sparsity * 100, 1),
            "top_20_words": [
                w for w, _ in Counter(all_tokens).most_common(20)
            ],
        },
        "ngrams": {
            "unigram_vocab": unigram_vocab,
            "bigram_vocab": bigram_vocab,
            "trigram_vocab": trigram_vocab,
        },
        "tfidf": {
            "top_10_terms": [{"word": w, "score": round(s, 4)} for w, s in top_tfidf],
        },
        "pmi": {
            "top_10_pairs": [
                {"pair": list(p), "ppmi": round(v, 4)} for p, v in top_pmi
            ],
        },
    }


# ============================================================================
# Exercise 3 — N-Gram Language Model for Gloss Sequences
# ============================================================================


class NGramLM:
    """
    N-gram language model with Laplace (add-one) smoothing.
    Trained on gloss sequences from the DGS-Korpus manifest.
    Supports unigram, bigram, and trigram models.
    """

    BOS = "<s>"   # sentence start
    EOS = "</s>"  # sentence end

    def __init__(self, n: int = 2) -> None:
        assert 1 <= n <= 3, "Only n=1,2,3 supported"
        self.n = n
        self._counts: Dict[tuple, int] = defaultdict(int)
        self._context_counts: Dict[tuple, int] = defaultdict(int)
        self._vocab: set = set()
        self._trained = False

    def train(self, sequences: List[List[str]]) -> None:
        """Train the n-gram model on a list of gloss sequences."""
        padded = []
        for seq in sequences:
            if not seq:
                continue
            padded.append(
                [self.BOS] * (self.n - 1) + list(seq) + [self.EOS]
            )

        for tokens in padded:
            self._vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n])
                context = ngram[:-1]
                self._counts[ngram] += 1
                self._context_counts[context] += 1

        self._trained = True

    def log_prob(self, ngram: tuple) -> float:
        """
        Log-probability of an n-gram with Laplace smoothing.
        P(w_n | w_{n-1}, ...) = (count(ngram) + 1) / (count(context) + V)
        """
        V = len(self._vocab) or 1
        context = ngram[:-1]
        numerator = self._counts[ngram] + 1
        denominator = self._context_counts[context] + V
        return math.log(numerator / denominator)

    def score_sequence(self, sequence: List[str]) -> float:
        """
        Return total log-probability of a gloss sequence.
        Uses appropriate BOS padding.
        """
        if not self._trained or not sequence:
            return float("-inf")
        tokens = [self.BOS] * (self.n - 1) + list(sequence) + [self.EOS]
        total = 0.0
        for i in range(self.n - 1, len(tokens)):
            ngram = tuple(tokens[i - self.n + 1 : i + 1])
            total += self.log_prob(ngram)
        return total

    def perplexity(self, test_sequences: List[List[str]]) -> float:
        """
        Perplexity = exp(-1/N * sum log P(w_i | context)).
        Lower = better fit.
        """
        total_log_prob = 0.0
        total_tokens = 0
        for seq in test_sequences:
            if not seq:
                continue
            tokens = [self.BOS] * (self.n - 1) + list(seq) + [self.EOS]
            for i in range(self.n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.n + 1 : i + 1])
                total_log_prob += self.log_prob(ngram)
                total_tokens += 1
        if total_tokens == 0:
            return float("inf")
        avg_ll = total_log_prob / total_tokens
        return math.exp(-avg_ll)

    def score_candidates(
        self, candidates: List[List[str]]
    ) -> List[Tuple[List[str], float, float]]:
        """
        Score a list of candidate gloss sequences.
        Returns list of (sequence, log_prob, avg_log_prob_per_token).
        Sorted best-first.
        """
        scored = []
        for cand in candidates:
            lp = self.score_sequence(cand)
            n_tok = max(len(cand), 1)
            scored.append((cand, lp, lp / n_tok))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def get_counts_table(self, max_rows: int = 20) -> List[dict]:
        """Return top n-gram counts as a list of dicts for display."""
        top = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[
            :max_rows
        ]
        return [{"ngram": list(ng), "count": cnt} for ng, cnt in top]


class GlossLanguageModel:
    """
    Trains and exposes unigram / bigram / trigram language models over the
    gloss sequences found in the segments manifest.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.sequences: List[List[str]] = []
        self.unigram = NGramLM(n=1)
        self.bigram = NGramLM(n=2)
        self.trigram = NGramLM(n=3)
        self._load_and_train()

    def _load_and_train(self) -> None:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            gs = seg.get("gloss_sequence") or []
            if gs:
                self.sequences.append(gs)

        self.unigram.train(self.sequences)
        self.bigram.train(self.sequences)
        self.trigram.train(self.sequences)

    def score_all(self, gloss_sequence: List[str]) -> dict:
        """Score a gloss sequence with all three LM orders."""
        return {
            "unigram_log_prob": round(self.unigram.score_sequence(gloss_sequence), 4),
            "bigram_log_prob": round(self.bigram.score_sequence(gloss_sequence), 4),
            "trigram_log_prob": round(self.trigram.score_sequence(gloss_sequence), 4),
            "bigram_perplexity": round(self.bigram.perplexity([gloss_sequence]), 2),
        }

    def select_best_translation(
        self, candidates: List[List[str]]
    ) -> Tuple[List[str], dict]:
        """
        Given candidate gloss sequences, return the one ranked highest by
        bigram LM (best-effort fluency proxy for sign language).
        """
        if not candidates:
            return [], {}
        scored = self.bigram.score_candidates(candidates)
        best_seq, best_lp, _ = scored[0]
        all_scores = {
            str(i): {
                "sequence": cand,
                "bigram_log_prob": round(lp, 4),
                "avg_per_token": round(avg, 4),
            }
            for i, (cand, lp, avg) in enumerate(scored)
        }
        return best_seq, all_scores


# ============================================================================
# Exercise 4 — HMM POS Tagger with Viterbi Decoding
# ============================================================================

# Universal POS tags (simplified for German)
_UNI_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "NUM", "PUNCT", "X"]

# Simple heuristic rules for German word → POS (used as emission prior)
_GERMAN_POS_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^[A-ZÄÖÜ][a-zäöü]+$"), "NOUN"),         # Capitalized in mid-sentence
    (re.compile(r"\b(ist|bin|bist|sind|war|hatte|hat|haben|werden|wurde|sein)\b"), "VERB"),
    (re.compile(r"\b(ich|du|er|sie|es|wir|ihr|mich|mir|dich|dir)\b"), "PRON"),
    (re.compile(r"\b(der|die|das|ein|eine|einen|dem|des|eines)\b"), "DET"),
    (re.compile(r"\b(auf|in|an|bei|von|mit|für|über|unter|vor|nach|zu|durch|ohne|gegen)\b"), "ADP"),
    (re.compile(r"\b(und|oder|aber|denn|weil|dass|wenn|als|ob)\b"), "CONJ"),
    (re.compile(r"\b(sehr|auch|nicht|noch|schon|nur|immer|hier|da|jetzt|dann|so|wie)\b"), "ADV"),
    (re.compile(r"\b\d+\b"), "NUM"),
    (re.compile(r"[.!?,;:]"), "PUNCT"),
    (re.compile(r"\b(gut|schlecht|groß|klein|alt|neu|schön|lang|kurz|wichtig|schwer|leicht)\b"), "ADJ"),
    (re.compile(r"(ung|heit|keit|schaft|tion|tät)$"), "NOUN"),
    (re.compile(r"(lich|isch|haft|sam|bar|los)$"), "ADJ"),
    (re.compile(r"(en|eln|ern|ieren)$"), "VERB"),
    (re.compile(r"(lich|weise|wärts|mals)$"), "ADV"),
]


def _heuristic_pos(word: str) -> str:
    """Heuristic German POS tag for a word (emission fallback)."""
    for pattern, tag in _GERMAN_POS_RULES:
        if pattern.search(word):
            return tag
    return "X"


class HMMPOSTagger:
    """
    Hidden Markov Model POS tagger trained on a synthetic 'corpus' derived
    from the DGS-Korpus German sentence texts with heuristic POS labels.

    Training data is auto-labeled using heuristic rules; this provides
    a reasonable baseline without an external annotated treebank.

    Implements Viterbi decoding (Exercise 4 requirement).
    """

    def __init__(self) -> None:
        self.tags = _UNI_TAGS
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        # Transition: count[prev_tag][curr_tag]
        self._trans_counts: Dict[str, Counter] = defaultdict(Counter)
        # Emission: count[tag][word]
        self._emit_counts: Dict[str, Counter] = defaultdict(Counter)
        # Initial: count[first_tag_of_sentence]
        self._init_counts: Counter = Counter()
        self._trained = False

    def _label_sentence(self, tokens: List[str]) -> List[str]:
        """Auto-label a German sentence with heuristic POS tags."""
        return [_heuristic_pos(t) for t in tokens]

    def train(self, sentences: List[str]) -> None:
        """Train the HMM on a list of German sentences."""
        for sent in sentences:
            tokens = word_tokenize(sent)
            if not tokens:
                continue
            tags = self._label_sentence(tokens)
            self._init_counts[tags[0]] += 1
            for i, (token, tag) in enumerate(zip(tokens, tags)):
                self._emit_counts[tag][token.lower()] += 1
                if i > 0:
                    self._trans_counts[tags[i - 1]][tag] += 1

        self._trained = True

    def _trans_prob(self, prev: str, curr: str) -> float:
        """Transition probability P(curr | prev) with smoothing."""
        total = sum(self._trans_counts[prev].values()) + len(self.tags)
        return (self._trans_counts[prev][curr] + 1) / total

    def _emit_prob(self, tag: str, word: str) -> float:
        """Emission probability P(word | tag) with smoothing."""
        total = sum(self._emit_counts[tag].values()) + len(self._emit_counts[tag]) + 1
        return (self._emit_counts[tag][word.lower()] + 1) / total

    def _init_prob(self, tag: str) -> float:
        """Initial probability P(tag at position 0) with smoothing."""
        total = sum(self._init_counts.values()) + len(self.tags)
        return (self._init_counts[tag] + 1) / total

    def viterbi(self, tokens: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Viterbi algorithm for POS tag sequence decoding.

        Returns:
          best_tags  — sequence of POS tags for the input tokens
          viterbi_matrix — probability matrix (log-scale) [n_tags × n_tokens]
                           for Exercise 4 Viterbi matrix display
        """
        n = len(tokens)
        T = len(self.tags)
        if n == 0:
            return [], []

        # viterbi[t][i] = log-prob of best path ending at tag i at position t
        viterbi_mat = [[-math.inf] * T for _ in range(n)]
        backpointer = [[0] * T for _ in range(n)]

        # Initialization
        for i, tag in enumerate(self.tags):
            viterbi_mat[0][i] = (
                math.log(self._init_prob(tag)) +
                math.log(self._emit_prob(tag, tokens[0]))
            )

        # Recursion
        for t in range(1, n):
            for curr_i, curr_tag in enumerate(self.tags):
                emit_lp = math.log(self._emit_prob(curr_tag, tokens[t]))
                best_prev = max(
                    range(T),
                    key=lambda pi: viterbi_mat[t - 1][pi] + math.log(self._trans_prob(self.tags[pi], curr_tag)),
                )
                viterbi_mat[t][curr_i] = (
                    viterbi_mat[t - 1][best_prev]
                    + math.log(self._trans_prob(self.tags[best_prev], curr_tag))
                    + emit_lp
                )
                backpointer[t][curr_i] = best_prev

        # Backtrack
        best_last = int(np.argmax(viterbi_mat[n - 1]))
        tag_indices = [0] * n
        tag_indices[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            tag_indices[t] = backpointer[t + 1][tag_indices[t + 1]]

        best_tags = [self.tags[i] for i in tag_indices]
        return best_tags, viterbi_mat

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Tag a German text string. Returns list of (word, POS_tag)."""
        tokens = word_tokenize(text)
        if not tokens:
            return []
        best_tags, _ = self.viterbi(tokens)
        return list(zip(tokens, best_tags))

    def get_transition_table(self) -> List[dict]:
        """Return transition probability table for display."""
        rows = []
        for prev_tag in self.tags:
            row = {"from_tag": prev_tag}
            for curr_tag in self.tags:
                row[curr_tag] = round(self._trans_prob(prev_tag, curr_tag), 4)
            rows.append(row)
        return rows

    def get_emission_table(self, top_n: int = 5) -> dict:
        """Return top-n emission words per POS tag."""
        result = {}
        for tag in self.tags:
            top = self._emit_counts[tag].most_common(top_n)
            result[tag] = [{"word": w, "count": c} for w, c in top]
        return result

    def get_viterbi_matrix_display(
        self, text: str
    ) -> dict:
        """
        Return the Viterbi matrix formatted for display.
        Shows log-probabilities per (tag, token) position.
        """
        tokens = word_tokenize(text)
        if not tokens:
            return {}
        best_tags, viterbi_mat = self.viterbi(tokens)
        return {
            "tokens": tokens,
            "tags": self.tags,
            "matrix": [
                # column per token
                {tokens[t]: {self.tags[i]: round(viterbi_mat[t][i], 3) for i in range(len(self.tags))}}
                for t in range(len(tokens))
            ],
            "predicted_tags": best_tags,
            "tagged_pairs": [{"word": w, "tag": t} for w, t in zip(tokens, best_tags)],
        }


# ============================================================================
# Singleton accessors — lazy-load and cache
# ============================================================================

_gloss_lm: Optional[GlossLanguageModel] = None
_hmm_tagger: Optional[HMMPOSTagger] = None

_MANIFEST = Path(__file__).resolve().parents[2] / "backend" / "data" / "segments_manifest.json"


def get_gloss_lm() -> GlossLanguageModel:
    global _gloss_lm
    if _gloss_lm is None:
        _gloss_lm = GlossLanguageModel(_MANIFEST)
    return _gloss_lm


def get_hmm_tagger(manifest_path: Optional[Path] = None) -> HMMPOSTagger:
    """Return a trained HMM POS tagger (trained on German sentences from manifest)."""
    global _hmm_tagger
    if _hmm_tagger is None:
        tagger = HMMPOSTagger()
        mp = manifest_path or _MANIFEST
        sentences: List[str] = []
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for seg in data.get("segments", []):
                txt = seg.get("german_text", "")
                if txt:
                    sentences.append(txt)
        tagger.train(sentences)
        _hmm_tagger = tagger
    return _hmm_tagger


# ============================================================================
# High-level analysis endpoint helper
# ============================================================================


def analyse_text(text: str, corpus: Optional[List[str]] = None) -> dict:
    """
    Full NLP analysis for a given German text.
    Returns all pipeline stages for display in the API / frontend.
    """
    # Lab 1 — Tokenization
    tok_analysis = full_tokenization_analysis(text)

    # POS tagging via HMM
    tagger = get_hmm_tagger()
    pos_pairs = tagger.tag(text)
    viterbi_display = tagger.get_viterbi_matrix_display(text)

    # Gloss LM scoring (if text has been translated already)
    lm = get_gloss_lm()
    lm_summary = {
        "trained_on_sequences": len(lm.sequences),
        "bigram_perplexity_self": (
            round(lm.bigram.perplexity(lm.sequences[:5]), 2)
            if lm.sequences
            else None
        ),
        "unigram_ngram_table": lm.unigram.get_counts_table(10),
        "bigram_ngram_table": lm.bigram.get_counts_table(10),
    }

    # Feature engineering over a small corpus (use manifest sentences if none given)
    if corpus is None:
        if _MANIFEST.exists():
            with _MANIFEST.open("r", encoding="utf-8") as f:
                mdata = json.load(f)
            corpus = [
                s.get("german_text", "") for s in mdata.get("segments", [])[:50] if s.get("german_text")
            ]
        else:
            corpus = [text]

    fe = feature_summary(corpus) if corpus else {}

    return {
        "tokenization": tok_analysis,
        "pos_tagging": {
            "tagged_pairs": [{"word": w, "tag": t} for w, t in pos_pairs],
            "viterbi_matrix": viterbi_display,
            "transition_table": tagger.get_transition_table(),
            "emission_table": tagger.get_emission_table(),
        },
        "language_model": lm_summary,
        "feature_engineering": fe,
    }
