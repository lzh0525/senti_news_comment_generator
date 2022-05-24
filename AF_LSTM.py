from scipy.fftpack import fft,ifft
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import json
import re


class Self_Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Self_Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(self.hidden_size,self.hidden_size,bias = False)
        self.tanh = nn.Tanh()
        self.AWeight = nn.Linear(self.hidden_size,1,bias = False)
        self.softmax = nn.Softmax(dim = 1)
        
    '''
    key 形状 ：(batch_size,seq_len,hidden_size)  M
    value 形状：(batch_size,seq_len,hidden_size) hidden_output
    keymask 形状：(batch_size,seq_len)
    
    返回r形状：(batch_size,hidden_size)
    '''
    def forward(self, key, value, key_mask):
        
        QK = self.tanh(self.linear_in(key))
        
        # attention mask处理
        QK = key_mask.unsqueeze(-1) * QK
        
        # weight(batch_size,seq_len,1) ,在dim = 1 作softmax运算
        weight = self.softmax(self.AWeight(QK))
        
        # r = (batch_size,1,seq_len) * (batch_size,seq_len,hidden_size) = (batch_size,1,hidden_size)
        r = torch.matmul(weight.permute(0, 2, 1), value)
        
        # r = (batch_size, hidden_size)
        return r.squeeze(1)

'''
ifNorm : 指定AF_LSTM网络是否需要添加归一化层
num_layers : 指定AF_LSTM网络中LSTM层的层数
'''
class AF_LSTM(nn.Module):

    def __init__(self, vocab, ifNorm = True, num_layers=1, pretrained_embeddings=None):
        print("Model : AF_LSTM is ready to run")
        super(AF_LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.vocab = vocab
        self.ifNorm = ifNorm
        
        if pretrained_embeddings is None:
            self.emb_size = 128
            self.word_emb = nn.Embedding(self.vocab.size(), self.emb_size)
        else:
            self.emb_size = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(self.vocab.size(), self.emb_size)
            self.word_emb.weight.data.copy_(pretrained_embeddings)
        
        # 这里LSTM的隐藏状态长度必须和输入emb维度相同
        self.hidden_size = self.emb_size
        self.LSTM_layer = nn.LSTM(input_size=self.emb_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers)
        
        
        self.AspectNorm = nn.LayerNorm(self.emb_size)
        self.HiddenNorm = nn.LayerNorm(self.hidden_size)
        
        self.Attention = Self_Attention(self.hidden_size)
        self.rout = nn.Linear(self.hidden_size,self.hidden_size,bias = False)
        self.hout = nn.Linear(self.hidden_size,self.hidden_size,bias = False)
        self.tanh = nn.Tanh()
        self.classification = nn.Linear(self.hidden_size,2)
        self.last_Softmax = nn.Softmax(-1)


    # Associative Memory Operator ：循环卷积 circular convolution
    def correlation(self, h, s, gpunum):
        h = h.cpu().data.numpy()
        s = s.cpu().data.numpy()
        result = []
        for i in range(len(s)):
            c = ifft(fft(h[i]) * fft(s[i])).real
            result.append(list(c))
        return torch.from_numpy(np.array(result)).cuda(gpunum)

    def forward(self,new_list,new_mask,aspect, gpunum):
        new_inputs_emb = self.word_emb(new_list.permute(1, 0))
        # 得到 hidden_output形状（seq_len,batch_size,hidden_size） 
        # h_L为最后一个时间步的隐藏状态，形状(batch_size,hidden_size)
        hidden_output, (h_L,_) = self.LSTM_layer(new_inputs_emb)
        # 得到 aspect_emb 形状 （batch_size,1,emb_size）
        aspect_emb = self.word_emb(aspect)
    
        # 是否使用归一化层属于超参数范畴
        if self.ifNorm == True:
            aspect_emb = self.AspectNorm(aspect_emb)
            hidden_output = self.HiddenNorm(hidden_output)
        '''
        correlation 循环卷积 circular convolution
        输入形状
        h ：(batch_size,seq_len,hidden_size)
        s ：(batch_size,1,emb_size)
        emb_size == hidden_size 否则运算报错
        '''
        M = self.correlation(hidden_output.permute(1, 0, 2), aspect_emb, gpunum)
        
        # 通过M计算注意力权重的self-attention模块，得到attention计算后的向量 r
        r = self.Attention(M,hidden_output.permute(1, 0, 2),new_mask)
        
        r = self.tanh(self.rout(r) + self.hout(h_L.squeeze(0)))
        
        result = self.last_Softmax(self.classification(r))
        
        # result 二分类结果
        # r memory (batch_size, self.hidden_size)
        return result, r