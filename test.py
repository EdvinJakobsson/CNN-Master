import reader_full
import numpy as np
#from BenHamner.score import mean_quadratic_weighted_kappa
#import functions
#from keras.preprocessing.sequence import pad_sequences
#import keras

asap_ranges = {
0: (0, 60),
1: (2, 12),
2: (1, 6),
3: (0, 3),
4: (0, 3),
5: (0, 4),
6: (0, 4),
7: (0, 30),
8: (0, 60)
}

essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
wordvectorfile = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"
data = reader_full.read_dataset([8], filepath=essayfile)

a = int(len(data)*0.7)
data = data[:a]
b = asap_ranges[2][1]
targets = [x[6] for x in data]


list1 = []
for i in range(len(targets)):
    list1.append(targets[i])
instances = list(dict.fromkeys(list1))
#print(instances)

kappas = [0, 0.99]
#c = mean_quadratic_weighted_kappa(kappas)



for i in range(2):

    sequences = [[1,2,3,4],[1,5,4],[2,3]]
    MAX_SEQUENCE_LENGTH = 3

    #pad_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #print(pad_sequences)

essaysets = [[1],[2],[3],[4],[5],[6],[7],[8]]

essays = [i[0] for i in essaysets]
print(essays)
