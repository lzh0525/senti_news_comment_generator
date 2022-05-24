import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
import torch
from model.dialogue_dataset import Exampledataset,collate_func
from model.transformer_base import transformer_base,S2sTransformer
from model.vocabulary import Vocab
from utils.loss import seq_generation_loss
from utils._utils import reset_log
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import textstat
from model.embedding import Word_Embedding
from rouge import Rouge
import json
import os
import numpy as np
import torch.nn.functional as F
import math
import heapq
from torch.nn.utils.rnn import pad_sequence
from nltk.translate import bleu_score
os.environ['CUDA_VISIBLE_DEVICES'] ='5'

def preprocess(or_query):
    return or_query.strip().replace(' ','')
def jsonloader(filename):
    # 将数据加载到一个列表中
    file = open(filename, 'r', encoding='utf-8')
    entity_list=[]#实体
    news_list = []#新闻
    date_list = []#日期
    vnames_list = []#新闻所含实体列表
    label_list=[]#评论
    title_list=[]#标题
    label_score_list=[]#评论情感分数
    for line in file.readlines():
        pop_dict = json.loads(line)
        entity=pop_dict['entity']
        date = pop_dict['date']
        news = pop_dict['news']
        vnames = pop_dict['v_names']
        label=pop_dict['label']
        title=pop_dict['title']
        #label_score=pop_dict['label_score']


        news_list.append(news)
        date_list.append(date)
        entity_list.append(entity)
        vnames_list.append(vnames)
        label_list.append(label)
        title_list.append(title)
        #label_score_list.append(label_score)
    return entity_list,news_list, date_list, vnames_list,label_list,label_score_list,title_list

def searchcomment(label_list):
    cur_label_list=[]
    for label in label_list:
        cur_label_list.append(label[0])
    return cur_label_list
def changenews2list(news_list):
    cur_news_list=[]
    for news in news_list:
        tmp=[]
        for sent in news:
            tmp.extend(sent)
        news_token=[]
        for word in tmp:
            if vocab.word2id(word)!=1:
                news_token.append(word)
        if len(news_token)>500:
            news_token=news_token[0:500]
        cur_news_list.append(news_token)
    return cur_news_list


def evalreadlibility(text):
    #text=[word,word-.....]为word的列表
    test_data = ' '.join(text)
    flesch_reading_ease=textstat.flesch_reading_ease(test_data)
    smog_index=textstat.smog_index(test_data)
    flesch_kincaid_grade=textstat.flesch_kincaid_grade(test_data)
    coleman_liau_index=textstat.coleman_liau_index(test_data)
    automated_readability_index=textstat.automated_readability_index(test_data)
    dale_chall_readability_score=textstat.dale_chall_readability_score(test_data)
    difficult_words=textstat.difficult_words(test_data)
    linsear_write_formula=textstat.linsear_write_formula(test_data)
    gunning_fog=textstat.gunning_fog(test_data)
    text_standard=textstat.text_standard(test_data,float_output=True)

    return flesch_reading_ease,smog_index,flesch_kincaid_grade,coleman_liau_index,automated_readability_index,dale_chall_readability_score,difficult_words,linsear_write_formula,gunning_fog,text_standard

'''
给定a、b两句句子(不能是单词列表)：
a = ["i am a student from china"]  # 预测摘要 
b = ["i am student from school on japan"] #真实摘要
生成 Rouge-1，Rouge-2，Rouge-L
'''
def Rouge_score(a,b):
    a = ' '.join(a)
    b = ' '.join(b)
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b)
    rouge_1=rouge_score[0]["rouge-1"]['r']
    rouge_2=rouge_score[0]["rouge-2"]['r']
    rouge_L=rouge_score[0]["rouge-l"]['r']
    return rouge_1,rouge_2,rouge_L
def Blue_score(a, b):

    b = [b]
    smooth = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu(b,a,smoothing_function=smooth.method2)

def greedysearch(decoder, max_len, query,vocab):
    """
    a greedy search implementation about seq2seq transformer
    :param decoder:
    :param max_len: max length of result
    :param query: input of Encoder
    :param vocab: vocab
    :return: list of index
    """
    cnt=0
    res=[]
    src_ids=[vocab.word2id(word) for word in query]
    src = torch.tensor(src_ids).unsqueeze(0)
    tgt_ids=[vocab.word2id('[START]')]
    while cnt<max_len:
        tgt=torch.tensor(tgt_ids).unsqueeze(0)
        out=decoder.forward(src,tgt)
        cur=out[-1,0,:].argmax().item()
        if vocab.id2word(cur)=='[STOP]':
            break
        tgt_ids.append(cur)
        res.append(vocab.id2word(cur))
        cnt+=1
    return ''.join(res),res

def starattentionmask(length):
    global_mask= \
        [torch.tensor([True]*(length))]+ \
        [torch.tensor([True]) for _ in range(length-1)]
    local_mask= \
        [torch.tensor([True]*(2))]+ \
        [torch.tensor([False]*(i)+[True]*(3)) for i in range(0,length-2)]+ \
        [torch.tensor([False]*(length-2)+[True]*(2))]
    global_attention_mask=pad_sequence(global_mask,batch_first=True)
    local_attention_mask=pad_sequence(local_mask,batch_first=True)
    attention_mask=global_attention_mask+local_attention_mask
    return attention_mask

class beamheap:
    def __init__(self,vocab,model_ckpt_path,model_params,pretrained_weight,Beamsize,backaerfa,foraerfa):
        self.vocab=vocab
        #self.model=transformer_base(self.vocab,model_params['embed_dim'],model_params['nheads'],pretrained_weight)
        self.model=S2sTransformer(self.vocab,model_params['embed_dim'],model_params['nheads'],pretrained_weight)
        self.model.load_state_dict(torch.load(model_ckpt_path))
        self.Beamsize=Beamsize
        self.backaerfa=backaerfa#长度惩罚因子
        self.foraerfa=foraerfa
    def back_decode(self,query,tgt_ids):
        src_ids=[vocab.word2id(word) for word in query]
        src = torch.tensor(src_ids).unsqueeze(0)
        self.model.eval()
        #attention_mask=starattentionmask(src.size(1))
        tgt=torch.tensor(tgt_ids).unsqueeze(0)
        _,back_out=self.model(src,for_tgt_ids=tgt,back_tgt_ids=tgt)
        return back_out # 输出维度维度是V*1 ，代表每个词出现的概率
    def for_decode(self,query,tgt_ids):
        src_ids=[vocab.word2id(word) for word in query]
        src = torch.tensor(src_ids).unsqueeze(0)
        self.model.eval()
        #attention_mask=starattentionmask(src.size(1))
        tgt=torch.tensor(tgt_ids).unsqueeze(0)
        for_out,_=self.model(src,for_tgt_ids=tgt,back_tgt_ids=tgt)
        return for_out # 输出维度维度是V*1 ，代表每个词出现的概率

    def fun(self,que,x,dir=None):# 这块维持一个大小为k（Beam search的宽度）的小顶堆，使得复杂度降到log(k)
        k=self.Beamsize
        if dir == "back":
            aerfa=self.backaerfa
        elif dir == "for":
            aerfa=self.foraerfa
        else:
            raise RuntimeError("the direction of decode is unlogic")
        que.append(x)
        def second(key):# 以序列出现的概率为依据进行排序
            p=key[1]
            L=len(key[0])
            M=math.log(p)/(math.pow( L,aerfa))
            return M
        que.sort(key=second)
        if len(que)>k:
            que=que[-k:]
        return que
def updatafreq(beams,beamsfreq):
    for dic in beams:
        word=dic
        beamsfreq[word]+=1

class beam_search_decoder:
    def __init__(self,beamheap,maxlen,numda,minlen):
        self.back_decode = beamheap.back_decode#前向decode和后向decode
        self.for_decode = beamheap.for_decode
        self.topk = beamheap.fun# 这是用于更新当前保留较大的K个备选项的函数
        self.vocab = beamheap.vocab
        self.maxlen = maxlen
        self.beamsize = beamheap.Beamsize
        self.numda = numda
        self.minlen = minlen
    def back_forward(self,src,entity):# 模型输入为初始向量，一般是编码器的输出
        beams = [([self.vocab.word2id(entity)],1.0)]# 首先把初始向量填入beam中 第一值是输出的序列列表，第二值是该序列出现的概率
        beamsfreq=[1 for _ in range(self.vocab.size())]
        for itername in range(self.maxlen):# 自回归式迭代生成输出序列 最大输出序列长度为max_len
            que = [] # 临时beam缓存
            for x,score in beams:# 遍历Beam中所有备选项
                if x[-1]==2:#BOS id = 2 如果已经输出了开始字符 则该序列直接用于更新，不再进行解码
                    que = self.topk(que,(x,score),'back')
                else:
                    output = self.back_decode(src,x) # 以Beam中已生成的序列为输入，生成下一token的概率分布
                    output = output[-1,0,:]
                    output = F.softmax(output)
                    if self.numda!=0:
                        tgt_freq=torch.FloatTensor(beamsfreq)
                        tgt_freq=torch.pow(tgt_freq, self.numda)
                        tgt_f_vocab=1/tgt_freq
                        output=output*tgt_f_vocab
                    output=output.tolist()
                    beammax=heapq.nlargest(self.beamsize,range(len(output)),output.__getitem__)
                    # if itername<self.minlen:

                    updatafreq(beammax,beamsfreq)
                    for wid in beammax:# 遍历输出选中的beamsize个词
                        o_score=output[wid]
                        que = self.topk(que,(x+[wid],score*o_score),'back')# 假设改词为输出的词，那么可以得到一个新的序列以及该序列出现的概率
            beams = que # 更新Beam
        return beams[-1][0]
    def for_forward(self,src,back_seq):
        beams = [(back_seq,1.0)]# 首先把初始向量填入beam中 第一值是输出的序列列表，第二值是该序列出现的概率
        beamsfreq=[1 for _ in range(self.vocab.size())]
        for itername in range(self.maxlen):# 自回归式迭代生成输出序列 最大输出序列长度为max_len
            que = [] # 临时beam缓存
            for x,score in beams:# 遍历Beam中所有备选项
                if x[-1]==3:#EOS id = 3 如果已经输出了结束字符 则该序列直接用于更新，不再进行解码
                    que = self.topk(que,(x,score),'for')
                else:
                    output = self.for_decode(src,x) # 以Beam中已生成的序列为输入，生成下一token的概率分布
                    output = output[-1,0,:]
                    output = F.softmax(output)
                    if self.numda!=0:
                        tgt_freq=torch.FloatTensor(beamsfreq)
                        tgt_freq=torch.pow(tgt_freq, self.numda)
                        tgt_f_vocab=1/tgt_freq
                        output=output*tgt_f_vocab
                    output=output.tolist()
                    beammax=heapq.nlargest(self.beamsize,range(len(output)),output.__getitem__)
                    updatafreq(beammax,beamsfreq)
                    for wid in beammax:# 遍历输出选中的beamsize个词
                        o_score=output[wid]
                        que = self.topk(que,(x+[wid],score*o_score),'for')# 假设改词为输出的词，那么可以得到一个新的序列以及该序列出现的概率
            beams = que # 更新Beam
        return beams[-1][0]
    

if __name__=="__main__":
    vocab_list=[]
    #for line in open("./data/backdata/entertainment_entity_vocab4.txt", "r"):
    for line in open("./data/sportdata/sport_vocab_49500_2gram.txt", "r"):  # 设置文件对象并读取每一行文件
        vocab_list.append(line[:-1])
    print("[INFO] vocab_list读取成功！")
    # 创建vocab类
    vocab = Vocab(vocab_list, 111298)
    print("[INFO] vocab_size:" , vocab.size())
    
    #ckpt_path='./ckpt/trans_xhj_v2_steps_283000.pkl'
    #ckpt_path='./ckpt/trans_xhj_v2_best.pkl'
    #ckpt_path='./ckpt/transentity_back300000.pkl'
    ckpt_path='./ckpt/transback_49_2gram_itf_sport_steps_250000.pkl'
    embedding_path='./data/word_embedding'
    #pretrain_path='data/backdata/pretrained_weight_entertainment_entity_vocab4.npy'
    pretrain_path='data/sportdata/pretrained_weight_sport_vocab_49500_2gram.npy'
    #test_path='./data/backdata/entertainment_test_entity4.json'
    test_path='./data/sportdata/sport_test_49500_2gram.json'
    word_emb_dim=128

    if not os.path.exists(pretrain_path):
        print("无pretrain,加载中")
        embed_loader = Word_Embedding(embedding_path, vocab)
        vectors = embed_loader.load_my_vecs()
        pretrained_weight = embed_loader.add_unknown_words_by_uniform(vectors, word_emb_dim)
        np.save(pretrain_path,pretrained_weight)
        print("save完成")
    pretrained_weight = np.load(pretrain_path)
    print("pretrain_weight load 成功！")
    print("model name",ckpt_path)
    model_params={"embed_dim":128,"nheads":4}
    comment_maxlen=20
    #test_path='./data/entity2/entertainment_test_entity3.json'

    Beamsize=6
    backaerfa=1.7
    foraerfa=1.38
    # backaerfa=1.9
    # foraerfa=1.6
    #长度惩罚因子，越大越鼓励长一点的句子
    numda=0.6
    minlen=5
    #freq惩罚，避免重复单词
    batch_size=5
    entity_list,\
    news_list, \
    date_list, \
    vnames_list,\
    label_list,\
    label_score_list,\
    title_list= jsonloader(test_path)
    print("测试集样本数：%d"%(len(news_list)))
    news_list=changenews2list(news_list)
    #label_list=searchcomment(label_list)
    #label_list=changenews2list(label_list)
    print(label_list[0:10])
    beamsearchheap=beamheap(vocab,ckpt_path,model_params,pretrained_weight,Beamsize,backaerfa,foraerfa)
    beam_search_decode=beam_search_decoder(beamsearchheap,comment_maxlen,numda,minlen)


    flesch_reading_ease_score=0
    smog_index_score=0
    flesch_kincaid_grade_score=0
    coleman_liau_index_score=0
    automated_readability_index_score=0
    dale_chall_readability_score_score=0
    difficult_words_score=0
    linsear_write_formula_score=0
    gunning_fog_score=0
    text_standard_score=0
    rouge_1_list,rouge_2_list,rouge_L_list=0,0,0
    bleu_list=0
    for i in range(len(news_list)):
        news=news_list[i]
        label=label_list[i]
        title=title_list[i]
        entity=entity_list[i]
        src=[entity]+title

        print('news',i,''.join(src))

        prefix_comment=beam_search_decode.back_forward(src,entity)
        prefix_comment=prefix_comment[::-1]
        complete_comment=beam_search_decode.for_forward(src,prefix_comment)
        if complete_comment[-1]!=3:
            complete_comment.append(3)
        complete_comment=[vocab.id2word(cur) for cur in complete_comment]

        print('comment'+str(i)+' predict:',''.join(complete_comment))

        label.insert(0,"[START]")
        label.append("[STOP]")
        predict.insert(0,"[START]")
        predict.append("[STOP]")

        print('labels',i,''.join(label))
        flesch_reading_ease,\
        smog_index,\
        flesch_kincaid_grade,\
        coleman_liau_index,\
        automated_readability_index,\
        dale_chall_readability_score,\
        difficult_words,\
        linsear_write_formula,\
        gunning_fog,\
        text_standard=evalreadlibility(complete_comment)
        rouge_1,rouge_2,rouge_L=Rouge_score(complete_comment,label)
        bleu=Blue_score(complete_comment,label)
        bleu_list+=bleu
        rouge_1_list+=rouge_1
        rouge_2_list+=rouge_2
        rouge_L_list+=rouge_L
        flesch_reading_ease_score+=flesch_reading_ease
        smog_index_score+=smog_index
        flesch_kincaid_grade_score+=flesch_kincaid_grade
        coleman_liau_index_score+=coleman_liau_index
        automated_readability_index_score+=automated_readability_index
        dale_chall_readability_score_score+=dale_chall_readability_score
        difficult_words_score+=difficult_words
        linsear_write_formula_score+=linsear_write_formula
        gunning_fog_score+=gunning_fog
        text_standard_score+=text_standard
    n_iter=len(news_list)
    print("rouge_1:",rouge_1_list/n_iter)
    print("rouge_2:",rouge_2_list/n_iter)
    print("rouge_L:",rouge_L_list/n_iter)
    print("bleu:",bleu_list/n_iter)
    print("flesch_reading_ease_score:",flesch_reading_ease_score/n_iter)
    print("smog_score:",smog_index_score/n_iter)
    print("flesch_kincaid_grade_score:",flesch_kincaid_grade_score/n_iter)
    print("coleman_liau_index_score:",coleman_liau_index_score/n_iter)
    print("automated_readability_score:",automated_readability_index_score/n_iter)
    print("dale_chall_readability_score:",dale_chall_readability_score_score/n_iter)
    print("difficult_words_score:",difficult_words_score/n_iter)
    print("gunning_fog_score:",gunning_fog_score/n_iter)
    print("linsear_write_formula_score:",linsear_write_formula_score/n_iter)
    print("text_standard_score:",text_standard_score/n_iter)