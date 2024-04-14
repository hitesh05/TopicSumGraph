import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel

from module.GAT import WSWGAT
from module.ntm import NTM_Module

class TopicSumGraph(nn.Module):
    def __init__(self, hps, vocab_size):
        """
        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self.embed_size = hps.word_emb_dim
        self.ntm = NTM_Module(dt=hps.dt, K=hps.num_topics, V=vocab_size)
        
        # bert init
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        # self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        
        self.n_feature_proj = nn.Linear(self.embed_size, hps.hidden_size, bias=False) # bert embed dim = 768
        
        # word -> sent
        self.word2sent = WSWGAT(in_dim=hps.dt,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=hps.dt,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )

        # node classification
        self.n_feature = hps.dt
        self.wh = nn.Linear(2*self.n_feature, 2) # classifier
        # self.TFEmbed = nn.Linear(1, hps.feat_embed_size)
        
    def forward(self, graph):
        
        # word node init
        # pass through ntm
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        xbow_reconstructed, theta = self.ntm.forward(graph,snode_id)
        topic_features = self.ntm.get_ti() # [wnode, dt]
        
        # pass through bert
        
        input_ids = graph.nodes[snode_id].data['bert_input_ids']
        attention_mask = graph.nodes[snode_id].data['bert_attention_mask']
        
        with torch.no_grad():
            sent_embeds = self.bertmodel(input_ids=input_ids, attention_mask=attention_mask)
        sent_embeds = sent_embeds.last_hidden_state[:, 0, :] # [snode, 768]
        sent_feature = self.n_feature_proj(sent_embeds)
        
        # the start state
        word_state = topic_features
        sent_state = sent_feature
        
        for i in range(self._n_iter): # n_iter = 1 generally
            # sent -> word
            
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        topic_state = torch.matmul(theta, word_state)
        final_state = torch.cat((topic_state, sent_state), axis=-1)
        result = self.wh(final_state)

        return xbow_reconstructed,result
