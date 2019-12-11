#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "on device")

# In[2]:


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


###################### For stacked layers = 3

#class LSTMTagger(nn.Module):
 #   def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
  #      super(LSTMTagger, self).__init__()
   #     self.hidden_dim = hidden_dim

    #    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
     #   self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
      #  self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
       # self.lstm3 = nn.LSTM(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
     #   self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

   # def forward(self, sentence):
    #    embeds = self.word_embeddings(sentence)
    #    lstm_out1, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
     #   lstm_out2, _ = self.lstm2(lstm_out1.view(len(sentence), 1, -1))
     #   lstm_out3, _ = self.lstm3(lstm_out2.view(len(sentence), 1, -1))
     #   tag_space = self.hidden2tag(lstm_out3.view(len(sentence), -1))
     #   tag_scores = F.log_softmax(tag_space, dim=1)
     #   return tag_scores


# In[ ]:

###############  For stacked layers = 2

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out1, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1.view(len(sentence), 1, -1))
        lstm_out3, _ = self.lstm3(lstm_out2.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out2.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[9]:


EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tags), len(tags)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(10): 
    print(epoch,"epoch")
    for sentence in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
<<<<<<< HEAD
<<<<<<< HEAD
        sentence = torch.LongTensor(sentence).to(device)
=======
        sentence = torch.LongTensor(sentence)to(device)
>>>>>>> b83b46e8e8e99d98e816e8b16c13bf195a3b83f7
=======
        sentence = torch.LongTensor(sentence).to(device)
>>>>>>> bc0666b0797e9e55668546f56eaefd4186ff4362
    
        tag_scores = model(sentence[:-1])

        loss = loss_function(tag_scores, sentence[1:])
        loss.backward()
        optimizer.step()


# In[12]:


def perplexity(data):
    s = 0
    gt_length = 0
    for i in data:
        with torch.no_grad():
            inputs = i[:-1]
<<<<<<< HEAD
<<<<<<< HEAD
            inputs = torch.LongTensor(inputs).to(device)
=======
            inputs = torch.LongTensor(inputs)to(device)
>>>>>>> b83b46e8e8e99d98e816e8b16c13bf195a3b83f7
=======
            inputs = torch.LongTensor(inputs).to(device)
>>>>>>> bc0666b0797e9e55668546f56eaefd4186ff4362
            tag_scores = model(inputs) 
            gt = i[1:]
            gt_length += len(gt)
            x = 0
            for j in gt:
                s += tag_scores[x][j]      #check for log math domain error
            x += 1
                
    perplex = math.exp(-(s/gt_length))
    
    return perplex


# In[13]:


print(perplexity(training_data),"train")


# In[15]:


print(perplexity(data_dev),"dev")

