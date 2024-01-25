import random
import re
from editdistpy import damerau_osa
from collections import Counter

WORD_PATTERN = r"(([^\W_]|['’])+)"


def load_counts(filename, sep='\t'):
    "Return a Counter initialized from key-value pairs, one on each line of filename."
    C = Counter()
    for line in open(filename):
        key, count = line.split(sep)
        C[key] = int(count)
    return C


def splits(text, start=0, L=20):
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    return [(text[:i], text[i:])
            for i in range(start, min(len(text), L)+1)]


def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return list(map(lambda t: t[0], re.findall(WORD_PATTERN, text.lower())))


def combine(Pfirst, first, Prem, rem):
    "Combine first and rem results into one (probability, words) pair."
    return Pfirst + Prem, [first] + rem


def string_similarity(s1, s2):
    """Calculates the similarity between two strings using the Damerau-Levenshtein distance.
    The similarity is a number between 0 and 1, where 1 means the strings are equal.
    From https://stats.stackexchange.com/questions/158279/how-i-can-convert-distance-euclidean-to-similarity-score#answer-630118
    """
    max_dist = max(len(s1), len(s2))
    dist = damerau_osa.distance(s1, s2, max_dist)
    if dist == -1:
        return 0
    return 1 - min(dist / max_dist, 1)


def spelltest(test_set, spell_check_func, spell_dict, verbose=False):
    """Run a spellchecker on a test set of (misspelled, correct) pairs; report results.
    The score is reported as the percentage of correct answers, and the percentage of unknown words.
    """
    import time
    start = time.time()
    good, unknown = 0, 0
    n = len(test_set)
    for wrong, right in test_set:
        w = spell_check_func(wrong)
        mmatch = (w.lower() == right.lower())
        good += mmatch
        unknown += (not mmatch) and (right not in spell_dict)
        if verbose and (not mmatch):
            print('correct({}) => {}, expected {}'
                  .format(wrong, w, right))
    dt = time.time() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))


def add_spelling_errors(phrase, error_probability=0.1, segment=False):
    "Adds random spelling errors to a phrase."
    modified_word = ""
    for char in phrase:
        if random.random() < error_probability:
            error_type = random.choice(
                ["delete", "add", "replace", "transpose", "segment"])
            if error_type != "segment" and char == " ":
                modified_word += char
            if error_type == "delete":
                continue
            elif error_type == "add":
                modified_word += random.choice(
                    "abcdefghijklmnopqrstuvwxyz") + char
            elif error_type == "replace":
                modified_word += random.choice("abcdefghijklmnopqrstuvwxyz")
            elif error_type == "transpose":
                if len(modified_word) > 0:
                    modified_word = modified_word[:-
                                                  1] + char + modified_word[-1]
                else:
                    modified_word += char
            elif error_type == "segment" and segment:
                if char == " ":
                    continue
                else:
                    modified_word += " " + char
            else:
                modified_word += char
        else:
            modified_word += char

    return modified_word


def generate_test_set(corpus):
    "Generate a test set from an nltk corpus."
    test_corpus_right = list(
        map(lambda t: t.lower(), filter(lambda t: t.isalpha(), corpus.words())))
    test_corpus_wrong = list(
        map(lambda t: add_spelling_errors(t, 0.1), test_corpus_right))
    test_set = zip(test_corpus_wrong, test_corpus_right)
    test_set = list(filter(lambda c: c[0] != c[1], test_set))
    return test_set


def import_test_set(path):
    "Import a test set from a file. The file should be tab-separated with one (wrong, right) pair per line."
    test = []
    with open(path) as f:
        for l in f.readlines():
            maybe_comma = l.find(",")
            if maybe_comma != -1:
                l = l[:maybe_comma]
            test.append(tuple(l.strip().split("\t")))
    return test


def test_segmenter(segmenter, tests, verbose=False):
    """Try segmenter on tests; report failures; return fraction correct.
    Tests should be a list of sentences; each sentence should be a string.
    The function concatenates the words in each sentence, and applies the segmenter.
    """
    result = 0
    for test in tests:
        words = tokens(test)
        result = segmenter("".join(words))
        correct = (result == words)
        result += correct
        if not correct and verbose:
            print('expected', words)
            print('got     ', result)
    return result / len(tests)


def test_sentence_correction(correct_text, test, verbose):
    """Try correct_text on tests; report failures; return average similarity score.
    Tests should be a list of (wrong, right) pairs.
    """
    result = 0
    for wrong, right in test:
        corrected = correct_text(wrong)
        score = string_similarity(corrected, right) * 100
        result += score
        if verbose:
            print('score: {:.2f}%'.format(score))
            print('\texpected', right)
            print('\tgot     ', corrected)
    return round(result / len(test), 2)