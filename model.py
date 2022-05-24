import torch.nn as nn
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Word2Vec_emb import Word_Embedding
from utils.vocabulary import Vocab
import os
import sys
import numpy as np
import torch.nn.functional as F
from AF_LSTM import AF_LSTM


class seq_generation_loss(nn.Module):
    ''' sequence generation loss
        model_out: (seq,batch,vocab_size)
        tgt: (batch,seq)       type(tgt[i,j])==int  and  tgt[i,j]<vocab_size
        tgt_freq: (batch,seq)       type(tgt_freq[i,j]==word freq)
    '''
    def __init__(self,device='cpu',numda=0.4,gama=0.4):
        super(seq_generation_loss,self).__init__()
        self.device=device
        self.numda=numda
        self.gama=gama
    def forward(self,for_out,back_out,back_tgt_batch,for_tgt_batch,back_tgt_freq=None,for_tgt_freq=None,back_tgt_pos=None,for_tgt_pos=None):
        for_seq_size,batch_size,vocab_size=for_out.shape
        back_seq_size,_,_=back_out.shape

        alpha=0.05
        '''
        前向损失
        '''

        tgt_vocab=torch.zeros((batch_size,for_seq_size,vocab_size))
        tgt_vocab=tgt_vocab.to(self.device)

        # tgt相比于原序列偏移前进一位
        tgt_shift=torch.roll(for_tgt_batch,-1,dims=1)
        tgt_shift[:,-1]=0
        tgt_shift=tgt_shift.unsqueeze(-1)
        tgt_vocab=tgt_vocab.scatter_(2,tgt_shift,1-alpha)
        tgt_vocab += alpha/vocab_size
        tgt_vocab=tgt_vocab.permute(1,0,2)

        # 输出概率分布
        model_out = F.log_softmax(for_out,dim=2)

        # 根据是否为0区分mask
        tgt_shift = tgt_shift.permute(1, 0, 2)
        pad_mask  = tgt_shift!=0
        tgt_mask  = pad_mask.expand(for_seq_size,batch_size,vocab_size)

        # 仅选择没有被mask的区域计算损失
        out=torch.masked_select(model_out, tgt_mask)
        tgt=torch.masked_select(tgt_vocab, tgt_mask)
        # tgt_freq=tgt_freq.permute(1,0,2)
        tgt_f_vocab=torch.ones((batch_size,for_seq_size,vocab_size)).to(self.device)
        if self.numda!=0:
            if for_tgt_freq!=None:
                for_tgt_freq=for_tgt_freq.unsqueeze(-1)
                tgt_freq_vocab=tgt_f_vocab*for_tgt_freq
                tgt_freq_vocab=tgt_freq_vocab.permute(1,0,2)
                w=torch.masked_select(tgt_freq_vocab, tgt_mask)
                b=torch.pow(w, self.numda)
                c=1/b
                loss = out*tgt*c
            else:
                loss = out*tgt
        else:
            loss = out*tgt

        # 根据其位置添加损失，本论文没有做到这一步
        if self.gama!=0:
            for_tgt_pos=for_tgt_pos.unsqueeze(-1)
            tgt_pos_vocab=tgt_f_vocab*for_tgt_pos
            tgt_pos_vocab=tgt_pos_vocab.permute(1,0,2)
            w=torch.masked_select(tgt_pos_vocab, tgt_mask)
            b=torch.pow(w, self.gama)
            loss=loss*b

        # 序列生成平均损失
        loss1 = -loss.sum() / pad_mask.sum()

        '''
        后向损失
        '''
        tgt_vocab=torch.zeros((batch_size,back_seq_size,vocab_size))
        tgt_vocab=tgt_vocab.to(self.device)
        tgt_shift=torch.roll(back_tgt_batch,-1,dims=1)
        tgt_shift[:,-1]=0
        tgt_shift=tgt_shift.unsqueeze(-1)
        tgt_vocab=tgt_vocab.scatter_(2,tgt_shift,1-alpha)
        tgt_vocab += alpha/vocab_size
        tgt_vocab=tgt_vocab.permute(1,0,2)

        model_out = F.log_softmax(back_out,dim=2)

        tgt_shift = tgt_shift.permute(1,0,2)
        pad_mask  = tgt_shift!=0
        tgt_mask  = pad_mask.expand(back_seq_size,batch_size,vocab_size)
        out=torch.masked_select(model_out, tgt_mask)
        tgt=torch.masked_select(tgt_vocab, tgt_mask)
        # tgt_freq=tgt_freq.permute(1,0,2)
        tgt_f_vocab=torch.ones((batch_size,back_seq_size,vocab_size)).to(self.device)
        if self.numda!=0:
            if back_tgt_freq!=None:
                back_tgt_freq=back_tgt_freq.unsqueeze(-1)
                tgt_freq_vocab=tgt_f_vocab*back_tgt_freq
                tgt_freq_vocab=tgt_freq_vocab.permute(1,0,2)
                w=torch.masked_select(tgt_freq_vocab, tgt_mask)
                b=torch.pow(w, self.numda)
                c=1/b
                loss = out*tgt*c
            else:
                loss = out*tgt
        else:
            loss = out*tgt
        if self.gama!=0:
            back_tgt_pos=back_tgt_pos.unsqueeze(-1)
            tgt_pos_vocab=tgt_f_vocab*back_tgt_pos
            tgt_pos_vocab=tgt_pos_vocab.permute(1,0,2)
            w=torch.masked_select(tgt_pos_vocab, tgt_mask)
            b=torch.pow(w, self.gama)
            loss=loss*b
        loss2 = -loss.sum() / pad_mask.sum()

        loss=loss1+loss2
        return loss
    
    
        



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class S2sTransformer(nn.Module):
    def __init__(self,vocab,senti_model=None, word_emb_dim = 128,nhead = 4,pretrained_weight=None,num_encoder_layers=12,
                 num_decoder_layers=6,dim_feedforward=2048,dropout=0.1):
        super(S2sTransformer,self).__init__()
        # embedding 层
        self.vocab_size=vocab.size()
        
        if pretrained_weight is None:
            self.embedding = nn.Embedding(vocab.size(), word_emb_dim)
        else:
            self.embedding = nn.Embedding(vocab.size(), word_emb_dim)
            self.embedding.weight.data.copy_(pretrained_weight)
            self.embedding.weight.requires_grad = True

        # word emb dim = 128
        self.model_dim=word_emb_dim

        # 可学习的PositionEncoding层
        self.pos_encoder = LearnedPositionEncoding(self.model_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(self.model_dim,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(self.model_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,encoder_norm)
        
        # 加载情感分析模型
        self.senti_model = senti_model
        
        self.memory_linear = nn.Linear(2*word_emb_dim,word_emb_dim)
        self.memory_linear_dropout = nn.Dropout(p=0.1)  

        # Decoder 分为前向和后向
        back_decoder_layer = nn.TransformerDecoderLayer(self.model_dim,nhead,dim_feedforward,dropout)
        back_decoder_norm = nn.LayerNorm(self.model_dim)
        for_decoder_layer = nn.TransformerDecoderLayer(self.model_dim,nhead,dim_feedforward,dropout)
        for_decoder_norm = nn.LayerNorm(self.model_dim)
        self.back_decoder = nn.TransformerDecoder(back_decoder_layer,num_decoder_layers,back_decoder_norm)
        self.for_decoder = nn.TransformerDecoder(for_decoder_layer,num_decoder_layers,for_decoder_norm)

        # 输出层
        self.output_layer = nn.Linear(self.model_dim,self.vocab_size)
        # self.for_output_layer = nn.Linear(self.model_dim,self.vocab_size)
        # self.back_output_layer = nn.Linear(self.model_dim,self.vocab_size)
        self._reset_parameters()

        self.nhead = nhead


    def forward(self, src_ids,back_tgt_ids,for_tgt_ids,src_pad_mask=None,back_tgt_pad_mask=None,for_tgt_pad_mask=None,back_tgt_mask=None,for_tgt_mask=None,src_mask=None,new_list=None,new_mask=None, entity=None,memory_mask=None,gpunum = 0):

        # embed层输入
        src = self.embedding(src_ids)
        back_tgt = self.embedding(back_tgt_ids)
        for_tgt = self.embedding(for_tgt_ids)

        # shape check
        if src.size(2) != self.model_dim or for_tgt.size(2) != self.model_dim:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # seq_len在第一维度
        src = src.transpose(0, 1)
        back_tgt = back_tgt.transpose(0, 1)
        for_tgt = for_tgt.transpose(0, 1)

        # position encoding 层
        src = self.pos_encoder(src)
        back_tgt = self.pos_encoder(back_tgt)
        for_tgt = self.pos_encoder(for_tgt)

        # 内容编码结果
        content_memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        #print(content_memory.shape)
        
        # 情感编码结果
        _, senti_memory = self.senti_model(new_list, new_mask, entity, gpunum)
        #print(senti_memory.shape)
        
        # 扩展维度
        senti_memory = torch.repeat_interleave(senti_memory.unsqueeze(0), repeats=content_memory.shape[0], dim=0)
        #print(senti_memory.shape)
        
        # 组合链接
        memory = self.memory_linear(torch.cat((content_memory,senti_memory) , dim=2))
        memory = self.memory_linear_dropout(memory)
        #print(memory.shape)
        

        # 不采用稀疏注意力矩阵，memory_key_padding_mask置为None
        back_output = self.back_decoder(back_tgt, memory, tgt_mask=back_tgt_mask, memory_mask=memory_mask,
                                        tgt_key_padding_mask=back_tgt_pad_mask,
                                        memory_key_padding_mask=src_pad_mask)
        for_output = self.for_decoder(for_tgt, memory, tgt_mask=for_tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=for_tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)

        # print("decode shape:",output.shape)
        for_output = self.output_layer(for_output)
        back_output = self.output_layer(back_output)
        # for_output = self.for_output_layer(for_output)
        # back_output = self.back_output_layer(back_output)
        # return output
        return for_output,back_output
    
    def saveAFLSTM(self,path):
        torch.save(self.senti_model.state_dict(),path)
    
    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)