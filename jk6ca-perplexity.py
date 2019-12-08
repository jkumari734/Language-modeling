#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


# In[3]:


def file_read(filename):
    f = open(filename, "r",encoding="utf8")
    sentences = []

    for x in f:
        sentences.append(x.split())
        
    return sentences


# In[4]:


def generate_tags(sentences):
    tags = {}
    
    c = 0
    for i in sentences:
        for j in i:
            if j not in tags:
                tags[j] = c
                c += 1
    return tags


# In[5]:


def data_preprocess(tags, sentences):
    training_data = []
    
    for i in sentences:
        temp = []
        for j in i:
#             x = tags[j]
            temp.append(tags[j])
        
        training_data.append(temp)
        
    return training_data
    


# In[6]:


sentences = file_read('trn-wiki.txt')


# In[7]:


tags = generate_tags(sentences)


# In[8]:


training_data = data_preprocess(tags, sentences)


# In[39]:


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


# In[40]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(1): 
    for sentence in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        sentence = torch.LongTensor(sentence)
    
        tag_scores = model(sentence[:-1])

        loss = loss_function(tag_scores, sentence[1:])
        loss.backward()
        optimizer.step()
        break


# In[41]:


def perplexity(data):
    s = 0
    gt_length = 0
    for i in data:
        with torch.no_grad():
            inputs = i[:-1]
            inputs = torch.LongTensor(inputs)
            tag_scores = model(inputs) 
            gt = i[1:]
            gt_length += len(gt)
            x = 0
            for j in gt:
                s += math.log(tag_scores[x][j])      #check for log math domain error
            x += 1
                
        break
    perplex = math.exp(-(s/gt_length))
    
    return perplex


# In[42]:


perplexity(training_data)


# In[44]:


sentences_dev = file_read('dev-wiki.txt')


# In[45]:


data_dev = data_preprocess(tags, sentences_dev)
perplexity(data_dev)

