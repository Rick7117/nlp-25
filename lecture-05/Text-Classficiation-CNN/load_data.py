# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    # Define tokenizer function to split text into words
    tokenize = lambda x: x.split()
    # Create text field processor with configurations:
    # - sequential: Process as sequence data
    # - tokenize: Use the defined tokenizer
    # - lower: Convert to lowercase
    # - include_lengths: Keep sequence length info
    # - batch_first: Put batch dimension first
    # - fix_length: Pad all sequences to 200 words
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    # Create label field processor for sentiment labels (0/1)
    LABEL = data.LabelField(dtype=torch.long)
    # Load IMDB dataset and split into train/test sets
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # Build vocabulary using training data and load GloVe embeddings
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
