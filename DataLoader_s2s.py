import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
import sys
from utils.vocabulary import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def jsonfreqread(freq_path,vocab):
    file = open(freq_path, 'r', encoding='utf-8')
    id2freq={}
    word2freq={}
    filereader = file.read()
    pop_dict = json.loads(filereader)
    for word in list(pop_dict.keys()):
        word2freq[word]=pop_dict[word]
        id2freq[vocab.word2id(word)]=pop_dict[word]
    return word2freq

def jsonloader(filename,flag):
    # 将数据加载到一个列表中
    file = open(filename, 'r', encoding='utf-8')
    entity_list=[]#实体
    news_list = []#包含实体的新闻新闻
    date_list = []#日期
    vnames_list = []#新闻所含实体列表
    label_list=[]#评论
    title_list=[]#标题
    # no use
    label_score_list=[]#评论情感分数
    for line in file.readlines():
        pop_dict = json.loads(line)
        if 'entity' not in pop_dict and flag=='train':
            continue
        entity = pop_dict['entity']
        date = pop_dict['date']
        news = pop_dict['news']
        
        newlist = []
        for new in news:
            if entity in new:
                newlist.extend(new)
                
        vnames = pop_dict['v_names']
        if flag=='train':
            label=pop_dict['label']
        else:
            label=pop_dict['label']
        title=pop_dict['title']
        #label_score=pop_dict['label_score']


        news_list.append(newlist)
        date_list.append(date)
        entity_list.append(entity)
        vnames_list.append(vnames)
        label_list.append(label)
        title_list.append(title)
        #label_score_list.append(label_score)
    return entity_list,news_list, date_list, vnames_list,label_list,label_score_list,title_list

class Exampledataset(Dataset):
    def __init__(self, data_file,vocab,senti_vocab,flag,freq_path=None):
        """
        :param data_root:   数据集路径
        """
        self.vocab = vocab
        self.senti_vocab = senti_vocab
        self.data_root = data_file
        #----------read process-----------
        self.entity_list,self.news_list, time_lists, \
        self.vnames_list ,\
        self.label_list,\
        self.label_score_list,\
        self.title_list= jsonloader(data_file,flag)
        if freq_path!=None:
            self.word2freq=jsonfreqread(freq_path,vocab)

        if flag=="valid":
            self.label_list=self.changenews2list(self.label_list)

        #----------data preprocess--------
        # self.news_list = self.changenews2list(news_list)
        self.title_list = self.delword(self.title_list)
        self.for_label_list,self.back_label_list = self.entitylabelcreat(self.label_list,self.entity_list)
    
    def delword(self,label_list):
        cur_news_list=[]
        for label in label_list:
            comment=[]
            for word in label:
                if self.vocab.word2id(word)!=1:
                    comment.append(word)
            #因为加了开始符号和结束符号
            if len(comment)>497:
                comment=comment[0:497]
            else:
                comment=comment
            cur_news_list.append(comment)
        return cur_news_list

    def entitylabelcreat(self,labels,entitys):
        forward_context_list=[]
        backward_context_list=[]
        for i in range(len(labels)):
            label=labels[i]
            entity=entitys[i]
            #entity必在label中出现，不然不构成上下文
            target_index=label.index(entity)
            # 根据entity位置划分前后文
            backward_context=label[0:target_index+1]
            # 逆序
            backward_context=backward_context[::-1]
            #forward_context=label[target_index:]
            forward_context_list.append(label)
            backward_context_list.append(backward_context)
        return forward_context_list,backward_context_list

    def changenews2list(self,news_list):
        cur_news_list=[]
        for news in news_list:
            tmp=[]
            for sent in news:
                tmp.extend(sent)
            news_token=[]
            for word in tmp:
                if self.vocab.word2id(word)!=1:
                    news_token.append(word)
            if len(news_token)>499:
                news_token=news_token[0:499]
            cur_news_list.append(news_token)
        return cur_news_list

    def __len__(self):
        return len(self.title_list)
        #return len(self.news_list)

    def __getitem__(self, index):
        #news = self.news_list[index]
        title= self.title_list[index]
        entity=self.entity_list[index]
        newlist = self.news_list[index]
        for_label=self.for_label_list[index]
        back_label=self.back_label_list[index]
        for_freq=[self.word2freq[word] for word in for_label]
        back_freq=[self.word2freq[word] for word in back_label]
        
        # print(entity)
        # print(title)
        # print(for_label)
        # print(back_label)
        new_token = []
        for word in newlist:
            if self.senti_vocab.word2id(word) != 1:
                new_token.append(self.senti_vocab.word2id(word))
        # 将entity作为首位的输入
        sample = {'title_token':[self.vocab.word2id(entity)]+[self.vocab.word2id(word) for word in title] ,
                  'new_token':new_token ,
                  'entity':[self.senti_vocab.word2id(entity)] ,
                  # 添加结束符
                  'for_comment_token':[self.vocab.word2id(word) for word in for_label]+[self.vocab.word2id('[STOP]')],
                  # 添加开始符
                  'back_comment_token':[self.vocab.word2id(word) for word in back_label]+[self.vocab.word2id('[START]')],
                  # 开始和STOP频率设置为1
                  'for_comment_freq':for_freq+[1],
                  'back_comment_freq':back_freq+[1]}

        return sample
    
def starattentionmask(length):
    global_mask=\
        [torch.tensor([True]*(length))]+\
        [torch.tensor([True]) for _ in range(length-1)]
    local_mask=\
        [torch.tensor([True]*(2))]+\
        [torch.tensor([False]*(i)+[True]*(3)) for i in range(0,length-2)]+\
        [torch.tensor([False]*(length-2)+[True]*(2))]
    global_attention_mask=pad_sequence(global_mask,batch_first=True)
    local_attention_mask=pad_sequence(local_mask,batch_first=True)
    attention_mask=global_attention_mask+local_attention_mask
    return attention_mask

def collate_func(batch_dic):

    batch_len=len(batch_dic)
    src_ids_batch = []
    new_ids_batch = []
    entity_batch = []
    back_tgt_ids_batch = []
    for_tgt_ids_batch = []
    src_pad_mask_batch = []
    back_tgt_pad_mask_batch = []
    for_tgt_pad_mask_batch = []
    back_tgt_freq_batch = []
    for_tgt_freq_batch = []
    for i in range(batch_len):
        dic=batch_dic[i]
        entity_batch.append(torch.tensor(dic['entity']))
        src_ids_batch.append(torch.tensor(dic['title_token']))
        new_ids_batch.append(torch.tensor(dic['new_token']))
        back_tgt_ids_batch.append(torch.tensor(dic['back_comment_token']))
        for_tgt_ids_batch.append(torch.tensor(dic['for_comment_token']))
        src_pad_mask_batch.append(torch.tensor([True]*len(dic['title_token'])))
        back_tgt_pad_mask_batch.append(torch.tensor([True]*len(dic['back_comment_token'])))
        for_tgt_pad_mask_batch.append(torch.tensor([True]*len(dic['for_comment_token'])))
        back_tgt_freq_batch.append(torch.tensor(dic['back_comment_freq']))
        for_tgt_freq_batch.append(torch.tensor(dic['for_comment_freq']))
    res={}
    
    # 内容标题id batch
    res['src_ids'] = pad_sequence(src_ids_batch,batch_first=True)
    # 新闻列表id batch
    res['new_ids'] = pad_sequence(new_ids_batch,batch_first=True)
    res['new_ids'] = torch.LongTensor(res['new_ids'].numpy())
    
    # 新闻列表mask
    res['new_ids_mask'] = res['new_ids'] != 0
    
    # 实体集合
    res['entity'] = torch.LongTensor(torch.stack(entity_batch,dim = 0))
    
    # 后向目标序列
    res['back_tgt_ids']=pad_sequence(back_tgt_ids_batch,batch_first=True)
    
    # 前向目标序列
    res['for_tgt_ids']=pad_sequence(for_tgt_ids_batch,batch_first=True)

    # 新闻 padmask
    res['src_pad_mask']=~pad_sequence(src_pad_mask_batch,batch_first=True)

    # 后向 padmask
    res['back_tgt_pad_mask']=~pad_sequence(back_tgt_pad_mask_batch,batch_first=True)
    # 前向 padmask
    res['for_tgt_pad_mask']=~pad_sequence(for_tgt_pad_mask_batch,batch_first=True)

    # 后向下三角mask矩阵
    back_tgt_length=res['back_tgt_pad_mask'].shape[1]
    back_tgt_mask_batch=[torch.tensor([True]*(i+1)) for i in range(back_tgt_length)]
    res['back_tgt_mask']=~pad_sequence(back_tgt_mask_batch,batch_first=True)

    # 下三角mask矩阵
    for_tgt_length=res['for_tgt_pad_mask'].shape[1]
    for_tgt_mask_batch=[torch.tensor([True]*(i+1)) for i in range(for_tgt_length)]
    res['for_tgt_mask']=~pad_sequence(for_tgt_mask_batch,batch_first=True)

    # 稀疏注意力矩阵
    src_length=res['src_pad_mask'].shape[1]
    res['src_mask']=starattentionmask(src_length)

    # frq 等长pad
    res['back_tgt_freq']=pad_sequence(back_tgt_freq_batch,batch_first=True)
    res['for_tgt_freq']=pad_sequence(for_tgt_freq_batch,batch_first=True)

    # 位置索引 【1,2,3,4,pad,pad,pad...】
    for_tgt_pos_batch = [torch.LongTensor([i+1 for i in range(for_tgt_length)]) for _ in range(res['for_tgt_pad_mask'].shape[0])]
    res['for_tgt_pos'] = pad_sequence(for_tgt_pos_batch, batch_first=True)

    back_tgt_pos_batch = [torch.LongTensor([i+1 for i in range(back_tgt_length)]) for _ in range(res['back_tgt_pad_mask'].shape[0])]
    res['back_tgt_pos'] = pad_sequence(back_tgt_pos_batch, batch_first=True)

    return res




if __name__ == "__main__":
    vocab_list=[]
    for line in open("../data/5Wdata/entertainment_vocab_50000_B.txt", "r"):  # 设置文件对象并读取每一行文件
        vocab_list.append(line[:-1])
    print("[INFO] vocab_list读取成功！")
    print("[INFO] vocab_size:" , len(vocab_list))
    # 创建vocab类
    vocab = Vocab(vocab_list, 110000)
    # print(vocab.id2word(4466))
    # print(vocab.id2word(62102))
    print(vocab.word2id('迪丽热巴'))
    #freq_path='../data/entity2/entertainment_entity_labelFre3.json'
    freq_path='../data/5Wdata/entertainment_49500_2-gram_labelFre.json'
    train_dataset = Exampledataset('../data/5Wdata/entertainment_train_49500_2gram.json',vocab,"train",freq_path)

    batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in tqdm(enumerate(train_loader),total=int(len(train_loader.dataset) / batch_size) + 1):
        src_batch, \
        back_tgt_batch, \
        for_tgt_batch, \
        src_pad_batch, \
        back_tgt_pad_batch, \
        for_tgt_pad_batch, \
        back_tgt_mask_batch, \
        for_tgt_mask_batch, \
        src_mask_batch, \
        back_tgt_freq_batch, \
        for_tgt_freq_batch , \
        back_tgt_pos_batch, \
        for_tgt_pos_batch= \
            batch['src_ids'], \
            batch['back_tgt_ids'], \
            batch['for_tgt_ids'], \
            batch['src_pad_mask'], \
            batch['back_tgt_pad_mask'], \
            batch['for_tgt_pad_mask'], \
            batch['back_tgt_mask'], \
            batch['for_tgt_mask'], \
            batch['src_mask'], \
            batch['back_tgt_freq'], \
            batch['for_tgt_freq'], \
            batch['back_tgt_pos'], \
            batch['for_tgt_pos']
        print(src_pad_batch)
        for i in range(len(src_batch)):
            title=[]
            src_ids=src_batch[i].tolist()
            for s in src_ids:
                title.append(vocab.id2word(int(s)))
            for_comment=[]
            for s in for_tgt_batch[i].tolist():
                for_comment.append(vocab.id2word(int(s)))
            back_comment=[]
            for s in back_tgt_batch[i].tolist():
                back_comment.append(vocab.id2word(int(s)))

            print("title:",''.join(title))
            print("back_comment:",''.join(back_comment))
            print("for_comment:",''.join(for_comment))

        # print(src_pad_batch[0])
        # print(tgt_pad_batch[0])
        # print(tgt_mask_batch)
        break