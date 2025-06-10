from utils import (
    preprocess_corpus, 
    exact_rare_words,
    replace_rare_words,
    build_vocab,
    create_dictonaries,
    create_transition_matrix, 
    create_emission_matrix
)
import numpy as np
from collections import defaultdict


class HiddenMarkovModel():
    def __init__(self, *, alpha=0.001, rare_threshold=1):
        self._alpha = alpha
        self.rare_threshold = rare_threshold
        self._all_tags = []
        self._vocab = {}
        self._A = None
        self._B = None
        self._tag_counts = defaultdict(int)
        self._unknown_token = '<unk>'
        self._is_trained = False

    def fit(self, train_corpus):
        corpus = preprocess_corpus(train_corpus)
        unknown_words = exact_rare_words(
            train_corpus, 
            rare_thresh_hold=self.rare_threshold
        )
        corpus = replace_rare_words(
            corpus, 
            unknown_words,
            unknown_token=self._unknown_token
        )

        self._vocab = build_vocab(corpus)
        transition_counts, emission_counts, self._tag_counts = create_dictonaries(corpus)

        self._all_tags = sorted(self._tag_counts.keys())

        self._A = create_transition_matrix(
            transition_counts, 
            self._tag_counts, 
            alpha=self._alpha
        )

        self._B = create_emission_matrix(
            emission_counts,
            self._tag_counts,
            self._vocab, 
            alpha=self._alpha
        )

        self._is_trained = True

    def _initialize(self, words):
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call `train()` before using `predict()`.")
            
        assert self._A is not None, "Transition matrix (A) must be initialized"
        assert self._B is not None, "Emission matrix (B) must be initialized"
        
        num_tags = len(self._tag_counts)
        num_words = len(words)
        best_probs = np.zeros((num_tags, num_words))
        best_paths = np.zeros((num_tags, num_words), dtype=int)
        s_idx = self._all_tags.index('--s--')
        for i in range(num_tags):
            best_probs[i,0] = np.log(self._A[s_idx, i]) + \
                np.log(self._B[i, self._vocab.get(words[0], self._vocab[self._unknown_token])])

        return best_probs, best_paths
    

    def _forward(self, words, best_probs, best_paths):
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call `train()` before using `predict()`.")
        
        assert self._A is not None, "Transition matrix (A) must be initialized"
        assert self._B is not None, "Emission matrix (B) must be initialized"
    
        num_tags = best_probs.shape[0]
        for i in range(1, len(words)): 
            for j in range(num_tags):
                best_prob_i = float("-inf")
                best_path_i = None

                for k in range(num_tags):
                    prob = best_probs[k, i - 1] + np.log(self._A[k, j]) + \
                        np.log(self._B[j, self._vocab.get(words[i], self._vocab[self._unknown_token])])
                    if prob > best_prob_i:
                        best_prob_i = prob
                        best_path_i = k

                best_probs[j,i] = best_prob_i
                best_paths[j,i] = best_path_i

    def _backward(self, best_probs, best_paths):
        m = best_paths.shape[1] 
        z = [None] * m
        num_tags = best_probs.shape[0]
        best_prob_for_last_word = float('-inf')  
        pred_tags = [None] * m
        skip_idx_tag = self._all_tags.index('--s--')

        for k in range(num_tags):
            if k == skip_idx_tag:
                continue
            
            if best_probs[k, m - 1] > best_prob_for_last_word:
                best_prob_for_last_word = best_probs[k, m - 1]
                z[m - 1] = k
                
        pred_tags[m - 1] = self._all_tags[z[m - 1]]

        for i in range(m - 1, 0, -1):
            pos_tag_for_word_i = best_paths[z[i], i]
            z[i - 1] = pos_tag_for_word_i
            pred_tags[i - 1] = self._all_tags[pos_tag_for_word_i]        
        return pred_tags
    
    def predict(self, test_corpus):
        pred_corpus = []

        for tagged_sent in test_corpus:
            words = [pair[0] for pair in tagged_sent]
            best_probs, best_paths = self._initialize(words)
            self._forward(words, best_probs, best_paths)
            pred_tags = self._backward(best_probs, best_paths)
            pred_corpus.append([(words[i], pred_tags[i]) for i in range(len(words))])

        return pred_corpus
