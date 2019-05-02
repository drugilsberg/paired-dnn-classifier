"""Generate synthetic data for PairedDNNClassifier."""
import os
import random
import string
import numpy as np
import pandas as pd

SEED = 123
NUMBER_OF_WORDS = 300
NUMBER_OF_DIMENSIONS = 500
NUMBER_OF_PAIRS = 1000
MAXIMUM_WORD_LENGTH = 10
NUMBER_OF_TRAINING_PAIRS = int(NUMBER_OF_PAIRS*.5)

# set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
# generate random vectors
vectors = pd.DataFrame(np.random.rand(
    NUMBER_OF_WORDS, NUMBER_OF_DIMENSIONS
))
# generate random words
vectors.index = [
    ''.join(
        random.choice(string.ascii_letters)
        for i in range(random.randint(1,MAXIMUM_WORD_LENGTH))
    )
    for _ in range(NUMBER_OF_WORDS)

]
# write word vectors
vectors.to_csv(
    os.path.join('data', 'vectors.csv')
)

# generate word pairs
pairs = pd.DataFrame(np.random.randint(
    0, high=NUMBER_OF_WORDS,
    size=(NUMBER_OF_PAIRS,2)
), columns=['first', 'second'])
# add label
pairs.insert(2, 'label', pairs['first'] % 2)
# replace indexes with words
pairs['first'] = vectors.index[pairs['first']]
pairs['second'] = vectors.index[pairs['second']]
# write training pairs
pairs[:NUMBER_OF_TRAINING_PAIRS].reindex().to_csv(
    os.path.join('data', 'train_pairs.csv')
)
# write test pairs
pairs[NUMBER_OF_TRAINING_PAIRS:].reindex().to_csv(
    os.path.join('data', 'test_pairs.csv')
)