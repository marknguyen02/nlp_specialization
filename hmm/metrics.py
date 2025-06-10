import numpy as np


def compute_accuracy(origin_corpus, pred_corpus):
    if len(origin_corpus) != len(pred_corpus):
        raise ValueError(
            f"Length mismatch: origin_corpus has {len(origin_corpus)} elements, "
            f"but pred_corpus has {len(pred_corpus)}. Ensure both corpora have the same number of sentences."
        )

    accuracies = []
    for i in range(len(origin_corpus)):
        len_sent_i = len(origin_corpus[i])
        if len_sent_i != len(pred_corpus[i]):
            raise ValueError(f"Sentence {i} has length mismatch: {len_sent_i} vs {len(pred_corpus[i])}")

        correct = sum(
            1 for j in range(len_sent_i)
            if origin_corpus[i][j][1] == pred_corpus[i][j][1]
        )
        accuracies.append(correct / len_sent_i)

    return np.mean(accuracies)
