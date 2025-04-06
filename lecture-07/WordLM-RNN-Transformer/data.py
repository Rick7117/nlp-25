import os
from io import open
import torch

class Dictionary(object):
    """A class for creating and managing word-to-index and index-to-word mappings.
    
    This class maintains two data structures:
    1. word2idx (dict): Maps words to their corresponding indices
    2. idx2word (list): Maps indices back to their corresponding words
    
    The dictionary is used for converting text data into numerical format
    that can be processed by neural networks.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """Add a word to the dictionary if it doesn't exist.
        
        Args:
            word (str): The word to be added to the dictionary
            
        Returns:
            int: The index of the word in the dictionary
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        """Return the size of the dictionary.
        
        Returns:
            int: Number of unique words in the dictionary
        """
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
