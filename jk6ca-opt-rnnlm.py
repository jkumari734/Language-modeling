#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "on device")

# In[ ]:


def file_read(filename):
    f = open(filename, "r",encoding="utf8")
    sentences = []

    for x in f:
        sentences.append(x.split())
        
    return sentences


# In[3]:


def generate_tags(sentences):
    tags = {}
    
    c = 0
    for i in sentences:
        for j in i:
            if j not in tags:
                tags[j] = c
                c += 1
    return tags


# In[4]:


def data_preprocess(tags, sentences):
    training_data = []
    
    for i in sentences:
        temp = []
        for j in i:
#             x = tags[j]
            temp.append(tags[j])
        
        training_data.append(temp)
        
    return training_data
    


# In[5]:


sentences = file_read('trn-wiki.txt')
sentences_dev = file_read('dev-wiki.txt')


# In[6]:


tags = generate_tags(sentences)


# In[7]:


training_data = data_preprocess(tags, sentences)
data_dev = data_preprocess(tags, sentences_dev)


# In[8]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[11]:


def perplexity(data, model, tag_scores):
    s = 0
    gt_length = 0
    for i in data:
        with torch.no_grad():
            inputs = i[:-1]
            inputs = torch.LongTensor(inputs).to(device)
            tag_scores = model(inputs) 
            gt = i[1:]
            gt_length += len(gt)
            x = 0
            for j in gt:
                s += tag_scores[x][j]      #check for log math domain error
            x += 1
                
    perplex = math.exp(-(s/gt_length))
    
    return perplex


# In[12]:


def train(epochs, model, data):

    for epoch in range(epochs): 
        print(epoch,"epoch")
        for sentence in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            sentence = torch.LongTensor(sentence).to(device)

            tag_scores = model(sentence[:-1])

            loss = loss_function(tag_scores, sentence[1:])
            loss.backward()
            optimizer.step() 

    return tag_scores


# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores), "dev accuracy")


# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores),"dev accuracy")



# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores),"dev accuracy")



# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores),"dev accuracy")



# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores),"dev accuracy")



# In[ ]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False)
tag_scores = train(10, model, training_data)
print(perplexity(training_data, model, tag_scores),"training accuracy")
print(perplexity(data_dev, model, tag_scores),"dev accuracy")


