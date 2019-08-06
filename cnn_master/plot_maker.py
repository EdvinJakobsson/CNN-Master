import os
import sys
import numpy as np
import csv
import reader_full
import functions
from BenHamner.score import quadratic_weighted_kappa
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.text import text_to_word_sequence
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# import models
# from keras.layers import Dense, Input, GlobalMaxPooling1D
# from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.models import Model
# from keras.initializers import Constant


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "{}".format(f)
    if "e" in s or "E" in s:
        return "{0:.{1}f}".format(f, n)
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])


asap_ranges = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}
human_range = False

# hyper-parameters
output = "linear"  # linear, sigmoid or softmax
essaysets = [[1], [2], [3], [4], [5], [6], [7], [8]]
essaysets = [[3]]

for essayset in essaysets:
    set = essayset[0]
    print("Essayset:", set)
    max_score = asap_ranges[set][1] - asap_ranges[set][0]
    essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
    wordvectorfile = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"

    data = reader_full.read_dataset(essayset, filepath=essayfile)
    data = data[: int(len(data) * 0.7)]  # save 30% of essays for final evaluation
    texts, essaysetlist, essaynumber, targets = functions.process_texts(
        data, output, essayset, asap_ranges, human_range
    )
    targets = targets + asap_ranges[set][0]

    count = collections.Counter(targets)
    ordered_count = []
    percents1 = []
    for i in range(max_score + 1):
        percents1.append(int(0.5 + count[i] * 100 / len(targets)))
        ordered_count.append(count[i])

    print("count: ", ordered_count)

    bin = np.arange(0, 4, 1)
    print(bin)
    plt.hist(
        targets,
        bins=bin,
        density=True,
        align="left",
        range=(asap_ranges[set][0], asap_ranges[set][1] + 1),
    )

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    title = "Essayset " + str(set)
    plt.title(title)
    plt.xlabel("Grade")
    plt.ylabel("Percentage of all essays")
    savefile = str(set) + ".png"
    # plt.savefig(savefile)
    plt.show()
    plt.close()
