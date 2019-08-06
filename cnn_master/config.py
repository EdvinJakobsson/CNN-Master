class Config:

        ESSAY_FILE = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
        WORDVECTOR_FILE = "C:/Users/Edvin/Projects/Data/glove.6B/glove.6B.100d.txt"
        ASAP_RANGES = {
            0: (0, 60),
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60),
        }
         # set to true if comparison should be made with one expert human grader
         # instead of the resolved score of the two experts
        HUMAN_RANGE = False
        if HUMAN_RANGE == True:
            print("WARNING: human_range is set to True!")
            ASAP_RANGES = {
                0: (0, 60),
                1: (1, 6),
                2: (1, 6),
                3: (0, 3),
                4: (0, 3),
                5: (0, 4),
                6: (0, 4),
                7: (0, 15),
                8: (0, 30),

            }

        # word embeddings
        MAX_SEQUENCE_LENGTH = 1000
        MAX_NUM_WORDS = 100000
        EMBEDDING_DIM = 100
        VALIDATION_SPLIT = 0.2
        RANDOMIZE_DATA = False
        NUM_WORD_EMBEDDINGS = -1  # -1 = all of them
        TRAINABLE_EMBEDDINGS = False

        # hyper-parameters
        MODEL_OUTPUT = "linear"  # linear, sigmoid or softmax
        MODEL_NUMBER = 8
        DROPOUT = 0.5
        LEARNING_RATE = 1e-4
        DENSE = 100
        KERNELS = 100
        KERNEL_LENGTH = 3
        ITERATIONS = 3

        # training
        EPOCHS_BETWEEN_KAPPA = 1
        ESSAYSETS = [[1], [2], [3], [4], [5], [6], [7], [8]]
        L_TWO = 1e-4

        TESTING = True
        #test values only for my computer
        if TESTING:
            NUM_WORD_EMBEDDINGS = 1
            DENSE = 1
            KERNELS = 1
            KERNEL_LENGTH = 1
            ESSAYSETS = [[1]]
            ITERATIONS = 2
