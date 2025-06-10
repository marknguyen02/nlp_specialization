import numpy as np
from collections import Counter
from collections import defaultdict


def preprocess_corpus(corpus):
    preproc_corpus = []

    for tagged_sent in corpus:
        preproc_corpus.append([(word.lower(), tag) for word, tag in tagged_sent])

    for i in range(len(preproc_corpus)):
        preproc_corpus[i].insert(0, ('<s>', '--s--'))

    return preproc_corpus


def exact_rare_words(train_corpus, *, rare_thresh_hold=1):
    all_words = []
    for tagged_sent in train_corpus:
        for word, _ in tagged_sent:
            all_words.append(word)

    word_counts = Counter(all_words)
    unknown_words = {word for word, count in word_counts.items() if count == rare_thresh_hold}
    return unknown_words


def replace_rare_words(train_corpus, unknown_words, *, unknown_token='<unk>'):
    new_corpus = []
    for tagged_sent in train_corpus:
        new_tagged_sent = [(unknown_token if word in unknown_words else word, tag) for word, tag in tagged_sent]
        new_corpus.append(new_tagged_sent)
    return new_corpus


def build_vocab(new_train_corpus):
    all_words = set()
    for tagged_sent in new_train_corpus:
        for word, _ in tagged_sent:
            all_words.add(word)

    vocab_words = sorted(all_words)
    vocab = {word: i for i, word in enumerate(vocab_words)}
    return vocab



def create_dictonaries(train_corpus):
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for tagged_sent in train_corpus:
        emission_counts[tagged_sent[0][0], tagged_sent[0][1]] += 1
        tag_counts[tagged_sent[0][1]] += 1
        for i in range(1, len(tagged_sent)):
            prev_tag = tagged_sent[i - 1][1]
            word, tag = tagged_sent[i][0], tagged_sent[i][1]
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    return transition_counts, emission_counts, tag_counts


def create_transition_matrix(transition_counts, tag_counts, *, alpha=0.001):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags, num_tags))
    for i in range(num_tags):
        for j in range(num_tags):
            key = (all_tags[i], all_tags[j])
            count = transition_counts.get(key, 0)
            A[i, j] = (count + alpha) / (tag_counts[all_tags[i]] + alpha * len(all_tags))

    return A


def create_emission_matrix(emission_counts, tag_counts, vocab, *, alpha=0.001):
    vocab_l = list(vocab)
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab_l)
    B = np.zeros((num_tags, num_words))
        
    for i in range(num_tags):
        for j in range(num_words):      
            key = (all_tags[i], vocab_l[j])
            count = emission_counts.get(key, 0)
            count_tag = tag_counts[all_tags[i]]
            B[i,j] = (count + alpha) / (count_tag + alpha * num_words)

    return B