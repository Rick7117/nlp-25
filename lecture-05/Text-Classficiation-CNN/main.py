import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.CNN import CNN

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] != 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] != 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	

learning_rate = 2e-5
batch_size = 32 # Batch size for training and evaluation
output_size = 2 # Number of classes for classification: 2
in_channels = 1 # Number of input channels for CNN: 1
out_channels = 10 # Number of output channels/feature maps for CNN
kernel_heights = [5, 7, 9] # Kernel heights for CNN: 5, 7, 9
stride = 1 # Stride for CNN
padding = 0 # Padding for CNN
keep_probab = 0.5 # Probability of retaining an activation node during dropout operation
embedding_length = 300

# Load the dataset
# We use pre-trained word embeddings as weights for our CNN
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

model = CNN(batch_size,
            output_size, 
            in_channels, 
            out_channels, 
            kernel_heights, 
            stride, 
            padding, 
            keep_probab, 
            vocab_size, 
            embedding_length, 
            weights = word_embeddings
            )
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

with torch.no_grad():
    test_sen = np.asarray(test_sen1)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen)
    test_tensor = test_tensor.cuda()
    model.eval()
    output = model(test_tensor, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print ("Sentiment: Positive")
    else:
        print ("Sentiment: Negative")
