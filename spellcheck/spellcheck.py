from editdistpy import damerau_osa
from collections import defaultdict
from functools import cache, reduce
from spellcheck.utils import load_counts, splits, tokens, combine
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

    def correct(self, word, max_edit_distance=None):
        "Most probable spelling correction for word."
        return max(self._suggestions(word, max_edit_distance), key=self._words.get)

    def correct_text_best(self, text, max_edit_distance=None):
        "Corrects spelling and wrong word segmentation. A very naive approach."
        prev = '<S>'
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):

            # case 1: word is misspelled
            corr1 = max(self._suggestions(
                tok_text[i], max_edit_distance=max_edit_distance),
                key=lambda w:
                    self.log_cond_prob_words([w], prev))
            prob1 = self.log_cond_prob_words([corr1], prev)

            # case 2: word is segmented wrong
            prob2, corr2 = self.segment(tok_text[i], prev)

            if prob1 > prob2:
                prev = corr1
                corrected_text.append(corr1)
            else:
                prev = corr2[-1]
                corrected_text += corr2

        return " ".join(corrected_text)

    def correct_text_better(self, text, prev='<S>', max_edit_distance=None):
        "Most probable spelling correction for a sentence, using bigrams, bigger window."
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):
            corrected = max(self._suggestions(
                tok_text[i], max_edit_distance=max_edit_distance),
                key=lambda w:
                    self.cond_prob_words([w]+([tok_text[i+1]] if i != len(tok_text)-1 else []), prev))
            prev = corrected
            corrected_text.append(corrected)
        return " ".join(corrected_text)

    def correct_text(self, text, max_edit_distance=None):
        "Most probable spelling correction for a sentence, using bigrams."
        prev = '<S>'
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):
            corrected = max(self._suggestions(
                tok_text[i], max_edit_distance=max_edit_distance),
                key=lambda w:
                    self.cond_prob_word(w, prev))
            prev = corrected
            corrected_text.append(corrected)
        return " ".join(corrected_text)

    def correct_text_dumb(self, text, max_edit_distance=None):
        "Most probable spelling correction for a sentence."
        corrected_text = []
        tok_text = tokens(text)
        for i in range(len(tok_text)):
            corrected = self.correct(tok_text[i], max_edit_distance)
            corrected_text.append(corrected)
        return " ".join(corrected_text)

    def _p1w(self, word):
        "Probability of `word`."
        return self._words[word] / self._num_words if word in self._words else 10./(self._num_words * 10**len(word))

    def _p2w(self, bigram):
        "Probability of `bigram`."
        return self._bigrams[bigram] / self._num_bigrams

    def cond_prob_word(self, word, prev):
        "Conditional probability of word, given previous word."
        bigram = prev + ' ' + word
        if self._p2w(bigram) > 0 and self._p1w(prev) > 0:
            return self._p2w(bigram) / self._p1w(prev)
        else:
            return self._p1w(word)

    def cond_prob_words(self, words, prev='<S>'):
        "The probability of a sequence of words, using bigram data, given prev word."
        return reduce(lambda a, b: a*b, (
            self.cond_prob_word(w, (prev if (i == 0) else words[i-1])) for (i, w) in enumerate(words)), 1)

    def log_cond_prob_words(self, words, prev='<S>'):
        "The probability of a sequence of words, using bigram data, given prev word."
        return reduce(lambda a, b: a+b, (
            math.log10(self.cond_prob_word(w, (prev if (i == 0) else words[i-1]))) for (i, w) in enumerate(words)), 0)

    @cache
    def segment(self, text, prev='<S>'):
        "Return (log P(words), words), where words is the best segmentation."
        if not text:
            return 0.0, []
        candidates = [combine(math.log10(self.cond_prob_word(first, prev)), first, *self.segment(rem, first))
                      for first, rem in splits(text, 1)]
        return max(candidates)

    def _suggestions(self, word, max_edit_distance=None):
        "Generate possible spelling corrections for word."
        if word in self._words:
            return [word]
        word_len = len(word)

        if max_edit_distance is None:
            max_edit_distance = self._max_dictionary_edit_distance

        # early exit - word is too big to possibly match any words
        if word_len - max_edit_distance > self._max_length:
            return [word]

        candidate_pointer = 0
        candidates = []
        suggestions = set()

        word_prefix_len = word_len
        if word_prefix_len > self._prefix_length:
            word_prefix_len = self._prefix_length
            candidates.append(word[:word_prefix_len])
        else:
            candidates.append(word)

        while candidate_pointer < len(candidates):
            candidate = candidates[candidate_pointer]
            candidate_pointer += 1
            candidate_len = len(candidate)
            len_diff = word_prefix_len - candidate_len

            if len_diff > max_edit_distance:
                # print("cand:", candidate, len_diff)
                break

            for suggestion in self._deletes[candidate]:
                if abs(len(suggestion) - word_len) > max_edit_distance:
                    continue

                suggestions.add(suggestion)

            # add edits: derive edits (deletes) from candidate (word) and add
            # them to candidates list. this is a recursive process until the
            # maximum edit distance has been reached
            if len_diff < max_edit_distance and candidate_len <= self._prefix_length:
                for i in range(candidate_len):
                    delete = candidate[:i] + candidate[i + 1:]
                    if delete not in candidates:
                        candidates.append(delete)

        partitioned_suggestions = [[] for _ in range(max_edit_distance)]
        for s in suggestions:
            dist = damerau_osa.distance(word, s, max_edit_distance)
            if dist != -1:
                partitioned_suggestions[dist-1].append(s)

        for s in partitioned_suggestions:
            if len(s) > 0:
                return s

        return [word]

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
