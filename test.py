import reader_full
import numpy
from BenHamner.score import mean_quadratic_weighted_kappa
#import functions

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
data = reader_full.read_dataset([3], filepath=essayfile)

a = int(len(data)*0.7)
data = data[:a]
b = asap_ranges[2][1]
list1 = [x[6] for x in data]
instances = list(dict.fromkeys(list1))

kappas = [0, 0.99]
c = mean_quadratic_weighted_kappa(kappas)

list = [0.424, 0.76, 6.012, 6.99]
listb = [int(x+0.5) for x in list]
print(listb)
