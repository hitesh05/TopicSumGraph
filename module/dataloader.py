import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle
import torch
import argparse
import time
from transformers import BertTokenizer, BertModel
import networkx as nx
import dgl

from .vocabulary import Vocab

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

# utils
def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

class Example(object):
    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the label
        self.labels = np.array([1 if i in label else 0 for i in range(len(self.original_article_sents))])
        
    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)
            
class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, num_topics):

        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        self.num_topics = num_topics
        with open(data_path, "r", encoding="utf-8") as f:
            self.example_list = json.load(f)
        self.size = len(self.example_list)

        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break
        self.filterids = list(set(self.filterids))
        self.filterwords = list(set(self.filterwords))
        with open(w2s_path, "r") as f:
            self.w2s_tfidf = json.load(f)  
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')      
            
    def get_example(self, index):
        file_name, new_index  = self.example_list[str(index)]
        e = readJson(file_name)
        e = e[new_index]
        e["summary"] = e.setdefault("summary", [])
        self.text = e['text']
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example
    
    def get_bow_rep(self, input_pad):
        vocab_len = self.vocab.size()
        num_sents = input_pad.shape[0]
        bow_rep = np.zeros((num_sents, vocab_len))
        for i, sent in enumerate(input_pad):
            for word in sent:
                if word not in self.filterids:
                    bow_rep[i][int(word)]+=1
        return bow_rep

    def get_bert_tokenizer(self):
        inputs = self.tokenizer(self.text, padding='max_length', truncation=True, return_tensors='pt', max_length=100)
        return inputs
    
    def AddTopicNode(self, G, num_topics):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for k in range(num_topics):
            node_id = k 
            wid2nid[node_id] = nid
            nid2wid[nid] = node_id
            nid += 1

        G.add_nodes(nid)
        G.ndata["unit"] = torch.zeros(nid)
        G.ndata["dtype"] = torch.zeros(nid)
        G.ndata['id']= torch.LongTensor(list(nid2wid.values()))
        return wid2nid, nid2wid 
    
    def create_graph(self, input_pad, bow_rep, inputs, labels, num_topics):
        G = dgl.graph(([], []))
        _, _ = self.AddTopicNode(G, num_topics)
        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][num_topics:] = torch.ones(N)
        G.ndata["dtype"][num_topics:] = torch.ones(N)
        sentids = [i+num_topics for i in range(N)]
        G.nodes[sentids].data['bert_input_ids'] = inputs['input_ids'][:len(sentids)]
        G.nodes[sentids].data['bert_attention_mask'] = inputs['attention_mask'][:len(sentids)]
        G.nodes[sentids].data['bert_token_type_ids'] = inputs['token_type_ids'][:len(sentids)]
        G.nodes[sentids].data['bow'] = bow_rep[:len(sentids)]
        G.nodes[sentids].data['label'] = labels[:len(sentids)]
        for i in range(N): 
            G.nodes[i+num_topics].data['id'] = torch.LongTensor([i])
        for i in range(num_topics):
            for j in range(N):
                G.add_edges([i], [j+num_topics], data={'tfidfembed': torch.tensor([1.0]), 'dtype': torch.tensor([1.0])})
                G.add_edges([j+num_topics], [i], data={'tfidfembed': torch.tensor([1.0]), 'dtype': torch.tensor([0.0])})
        
        return G
        
    def __getitem__(self, index):
        item = self.get_example(index)
        sents = np.array(item.enc_sent_input_pad)
        labels = item.labels
        input_pad = np.array(item.enc_sent_input_pad)
        bow_rep = torch.tensor(self.get_bow_rep(input_pad), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)
        inputs = self.get_bert_tokenizer()
        G = self.create_graph(input_pad, bow_rep, inputs, labels, self.num_topics)
        return G, index
    
    def __len__(self):
        return self.size
    
def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    # print(samples)
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]