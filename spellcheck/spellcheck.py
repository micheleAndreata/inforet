from editdistpy import damerau_osa
from collections import defaultdict
from functools import cache, reduce
from spellcheck.utils import load_counts, splits, tokens
import math


class SpellCheck():
    def __init__(self, words_path):
        self._words = load_counts(words_path)
        self._num_words = sum(self._words.values())

    def prob(self, word):
        "Probability of `word`."
        return self._words[word] / self._num_words

    def correct(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.prob)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or
                self.known(self.edits1(word)) or
                self.known(self.edits2(word)) or
                [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of self._words."
        return set(w for w in words if w in self._words)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + replaces + inserts + transposes)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


class SpellCheckFast():
    def __init__(self, words_path, bigrams_path, max_dictionary_edit_distance=2, prefix_length=7):
        self._words = load_counts(words_path)
        self._bigrams = load_counts(bigrams_path)

        self._num_words = sum(self._words.values())
        self._num_bigrams = sum(self._bigrams.values())
        self._max_dictionary_edit_distance = max_dictionary_edit_distance
        self._prefix_length = prefix_length
        self._max_length = 0

        self._deletes = defaultdict(list)
        for key in self._words.keys():
            if len(key) > self._max_length:
                self._max_length = len(key)
            edits = self._edits_prefix(key)
            for delete in edits:
                self._deletes[delete].append(key)

    def correct(self, word, max_edit_distance=None, top_n=1):
        "Most probable spelling correction for word."
        if top_n == 1:
            return max(self.suggestions(word, max_edit_distance), key=self._words.get)
        else:
            part_sugg = self.suggestions(
                word, max_edit_distance=max_edit_distance, return_all=True)
            part_sugg = map(lambda sugg_list: sorted(
                sugg_list, key=self._words.get, reverse=True), part_sugg)
            suggestions = reduce(lambda a, b: a+b, part_sugg, [])
            if not suggestions:
                return [word]
            return suggestions[:min(top_n, len(suggestions))]

    def correct_text(self, text, max_edit_distance=None):
        "Most probable spelling correction for a sentence, using bigrams."
        prev = '<S>'
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):
            corrected = max(self.suggestions(
                tok_text[i], max_edit_distance=max_edit_distance),
                key=lambda w:
                    self.cond_prob_word(w, prev))
            prev = corrected
            corrected_text.append(corrected)
        return " ".join(corrected_text)

    def correct_text_simple(self, text, max_edit_distance=None):
        "Most probable spelling correction for a sentence, uses only monograms."
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):
            corrected = max(self.suggestions(
                tok_text[i], max_edit_distance=max_edit_distance),
                key=self._words.get)
            corrected_text.append(corrected)
        return " ".join(corrected_text)

    def p1w(self, word):
        "Probability of `word`."
        return self._words[word] / self._num_words if word in self._words else 10./(self._num_words * 10**len(word))

    def p2w(self, bigram):
        "Probability of `bigram`."
        return self._bigrams[bigram] / self._num_bigrams

    def cond_prob_word(self, word, prev):
        "Conditional probability of word, given previous word."
        bigram = prev + ' ' + word
        if self.p2w(bigram) > 0 and self.p1w(prev) > 0:
            return self.p2w(bigram) / self.p1w(prev)
        else:
            return self.p1w(word)

    def cond_prob_words(self, words, prev='<S>'):
        "The probability of a sequence of words, using bigram data, given prev word."
        return reduce(lambda a, b: a*b, (
            self.cond_prob_word(w, (prev if (i == 0) else words[i-1])) for (i, w) in enumerate(words)), 1)

    def log_cond_prob_words(self, words, prev='<S>'):
        "The log probability of a sequence of words, using bigram data, given prev word."
        return reduce(lambda a, b: a+b, (
            math.log10(self.cond_prob_word(w, (prev if (i == 0) else words[i-1]))) for (i, w) in enumerate(words)), 0)

    @cache
    def segment(self, text, prev='<S>'):
        "Return (log P(words), words), where words is the best segmentation."
        if not text:
            return 0.0, []
        candidates = []
        for first, rem in splits(text, 1):
            first_log_prob = math.log10(self.cond_prob_word(first, prev))
            rem_log_prob, rem_words = self.segment(rem, first)
            candidates.append(
                (first_log_prob + rem_log_prob, [first] + rem_words))
        return max(candidates)

    def suggestions(self, word, return_all=False, max_edit_distance=None):
        "Generate possible spelling corrections for word."
        if word in self._words:
            if return_all:
                return [[word]]
            return [word]
        word_len = len(word)

        if max_edit_distance is None:
            max_edit_distance = self._max_dictionary_edit_distance

        # early exit - word is too big to possibly match any words
        if word_len - max_edit_distance > self._max_length:
            if return_all:
                return [[word]]
            return [word]

        candidate_pointer = 0
        candidates = []
        hash_candidates = set()
        suggestions = set()
        partitioned_suggestions = [[] for _ in range(max_edit_distance + 1)]
        partitioned_suggestions[-1].append(word)

        word_prefix_len = word_len
        if word_prefix_len > self._prefix_length:
            word_prefix_len = self._prefix_length
            candidates.append(word[:word_prefix_len])
            hash_candidates.add(word[:word_prefix_len])
        else:
            candidates.append(word)
            hash_candidates.add(word)

        while candidate_pointer < len(candidates):
            candidate = candidates[candidate_pointer]
            candidate_pointer += 1
            candidate_len = len(candidate)
            len_diff = word_prefix_len - candidate_len

            if len_diff > max_edit_distance:
                break

            for suggestion in self._deletes[candidate]:
                if abs(len(suggestion) - word_len) > max_edit_distance:
                    continue

                if suggestion in suggestions:
                    continue

                suggestions.add(suggestion)
                dist = damerau_osa.distance(
                    word, suggestion, max_edit_distance)
                if dist != -1:
                    partitioned_suggestions[dist-1].append(suggestion)

            # add edits: derive edits (deletes) from candidate (word) and add
            # them to candidates list. this is a recursive process until the
            # maximum edit distance has been reached
            if len_diff < max_edit_distance and candidate_len <= self._prefix_length:
                for i in range(candidate_len):
                    delete = candidate[:i] + candidate[i + 1:]
                    if delete not in hash_candidates:
                        hash_candidates.add(delete)
                        candidates.append(delete)

        if return_all:
            return partitioned_suggestions
        else:
            for s in partitioned_suggestions:
                if len(s) > 0:
                    return s

    def _edits_prefix(self, key):
        hash_set = set()
        if len(key) <= self._max_dictionary_edit_distance:
            hash_set.add("")
        if len(key) > self._prefix_length:
            key = key[: self._prefix_length]
        hash_set.add(key)
        return self._edits(key, 0, hash_set)

    def _edits(self, word, edit_distance, delete_words, current_distance=0):
        edit_distance += 1
        if not word:
            return delete_words
        for i in range(current_distance, len(word)):
            delete = word[:i] + word[i + 1:]
            if delete not in delete_words:
                delete_words.add(delete)
            # recursion, if maximum edit distance not yet reached
            if edit_distance < self._max_dictionary_edit_distance:
                self._edits(delete, edit_distance,
                            delete_words, current_distance=i)
        return delete_words
