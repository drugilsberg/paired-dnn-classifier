"""
Train a DNN classifier for paired words using pre-trained vectors.
"""
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class WordPairsDataset(Dataset):
    """Word pairs dataset."""

    def __init__(self, filepath, vectors_filepath):
        """
        Initialize the WordPairsDataset.

        Args:
            filepath (string): path to the csv file with the pairs.
            vectors_filepath (string): path to the csv file with the vectors.
                Used to map the words to indexes.
        """
        self.pairs = pd.read_csv(filepath)
        self.word_to_index = {
            word: index
            for index, word in enumerate(
                pd.read_csv(vectors_filepath, index_col=0).index.tolist()
            )
        }

    def __len__(self):
        """
        Get number of pairs.
        
        Returns:
            the number of pairs.
        """
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Get a pair and the associated label.

        Args:
            - index (int): the index of a pair.

        Returns:
            a tuple with two torch.Tenors:
                - the first containing the pair indexes.
                - the second containing the label.
        """
        row = self.pairs.iloc[index]
        pair = torch.from_numpy(
            np.array([
                [self.word_to_index[row['first']]],
                [self.word_to_index[row['second']]]
            ])
        )
        label = torch.tensor([row['label']], dtype=torch.float)
        return pair, label


def create_dense_layer(
    input_size, output_size,
    activation_fn=nn.ReLU(), dropout=.5
):
    """
    Create a dense layer.

    Args:
        - input_size (int): size of the input.
        - output_size (int): size of the output.
        - activation_fn (an activation): activation function.
            Defaults to ReLU.
        - dropout: dropout rate. Defaults to 0.5.

    Returns:
        a nn.Sequential.
    """
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(input_size, output_size)),
        ('activation_fn', activation_fn),
        ('dropout', nn.Dropout(p=dropout)),
]))


def create_embedding_layer(vectors, non_trainable=False):
    """
    Create an embedding layer.

    Args:
        - vectors (np.ndarray): word vectors in numpy format.
        - non_trainable (bool): non trainable vectors. Defaults to False.

    Returns:
        a nn.Embedding layer.
    """
    number_of_vectors, vector_dimension = vectors.shape
    embedding_layer = nn.Embedding(number_of_vectors, vector_dimension)
    embedding_layer.load_state_dict({'weight': torch.from_numpy(vectors)})
    if non_trainable:
        embedding_layer.weight.requires_grad = False
    return embedding_layer


class PairedDNNClassifier(nn.Module):
    """
    Binary classification of paired words using pre-trained word vectors.
    """

    def __init__(
        self, vectors, units=[64, 16],
        dropout=.5, trainable_vectors=True
    ):
        """
        Initialize a PairedDNNClassifier.

        Args:
             - vectors (np.ndarray): word vectors in numpy format.
             - units (list): list of units of the DNN. The length of the
                list determine the number of layers. Defaults to [64, 16].
            - dropout (float): dropout rate. Defaults to 0.5.
            - trainable_vectors (bool): trainable vectors. Defaults to True.
        """
        super(PairedDNNClassifier, self).__init__()
        # create embedding with the pretrained vectors.
        self.embedding_layer = create_embedding_layer(
            vectors, not trainable_vectors
        )
        # add to the hidden units the first layer for the paired words.
        self.units = [2*vectors.shape[1]] + units
        self.number_of_layers = len(units)
        self.stacked_dense_layers = nn.Sequential(*[
            create_dense_layer(input_size, output_size)
            for input_size, output_size in zip(self.units, self.units[1:])
        ])
        # add the binary classification layer
        self.output = create_dense_layer(
            self.units[-1], 1,
            activation_fn=nn.Sigmoid(),
            dropout=0.0
        )
        
    def forward(self, pair):
        """
        Apply the forward pass of the model.

        Args:
            - pair: a torch.Tensor containing the indexes of the
                paired words.
        Returns:
            a torch.Tensor with the score for the pairs.
        """
        embedded_pair =  self.embedding_layer(pair).view(-1, self.units[0])
        encoded_pair = self.stacked_dense_layers(embedded_pair)
        return self.output(encoded_pair)


# setup
BATCH_SIZE = 128
EPOCHS = 100
VECTORS_FILEPATH = os.path.join('data', 'vectors.csv')
TRAIN_FILEPATH = os.path.join('data', 'train_pairs.csv')
TEST_FILEPATH = os.path.join('data', 'test_pairs.csv')
# prepare the datasets
vectors = pd.read_csv(VECTORS_FILEPATH, index_col=0)
train_dataset = WordPairsDataset(TRAIN_FILEPATH, VECTORS_FILEPATH)
test_dataset = WordPairsDataset(TEST_FILEPATH, VECTORS_FILEPATH)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE
)
# initialize model with default parameters
# see docstring for customization
model = PairedDNNClassifier(vectors.values)
# binary classification so binary cross entropy loss
criterion = nn.BCELoss()
# adam with standard parameters
optimizer = optim.Adam(
    model.parameters(), lr=0.001, betas=(0.9, 0.999)
)
# train the model
model.train()
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    losses = []
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs
        samples, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # collect loss
        losses.append(loss.item())
    print(
        'epoch={}\tloss={}'.format(
            epoch + 1,
            sum(losses) / float(len(losses))
        )
    )
# evaluate the model
model.eval()
accuracies = []
for data in test_dataloader:
        # get the inputs
        samples, labels = data
        outputs = model(samples)
        # collect accuracy
        accuracies.append(
            (
                (outputs >.5).numpy().flatten() ==
                labels.numpy().flatten()
            ).astype(int).mean()
        )
print('accuracy={}'.format(sum(accuracies) / len(accuracies)))
# get the fine-tuned vectors
finetuned_vectors = pd.DataFrame(
    model.embedding_layer.weight.data.numpy(),
    index=vectors.index,
    columns=vectors.columns
)
