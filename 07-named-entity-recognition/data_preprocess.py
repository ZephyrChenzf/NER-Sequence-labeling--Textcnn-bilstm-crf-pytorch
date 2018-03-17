import numpy as np
import torch
from torch.utils.data import Dataset


word2index={'pad':0,'unknow':1}
index2word={0:'pad',1:'unknow'}
tag2index={'pad':0}
index2tag={0:'pad'}
MAX_LEN=100

data=open('./data/train_data','r')
train_x=[]#总x训练集
train_y=[]#总y训练集
sen_x=[]#每次存一句话的id组
sen_y=[]#每次存一句话的标签id组

#将数据按每句话分出来
for line in data:
    line=line.strip()
    if(line=="" or line=="\n" or line=="\r\n"):#一句话结束了
        train_x.append(sen_x)
        sen_x=[]
        train_y.append(sen_y)
        sen_y=[]
        continue
    line=line.split(' ')
    if(len(line)<2):
        continue
    if line[0] in word2index:#如果在词典中有该词，将id给sen_x
        sen_x.append(word2index[line[0]])
    else:#如果没有则加入字典，并将id给sen_x
        word2index[line[0]]=len(word2index)
        index2word[len(index2word)]=line[0]
        sen_x.append(word2index[line[0]])
    if line[1] in tag2index:#同理，注意不同标签对应的id与初始碰到的标签有关
        sen_y.append((tag2index[line[1]]))
    else:
        tag2index[line[1]]=len(tag2index)
        index2tag[len(index2tag)]=line[1]
        sen_y.append(tag2index[line[1]])

#开始对每句话进行裁剪，主要是最大长度的限制
train_x_cut=[]
train_y_cut=[]
train_mask=[]
for i in range(len(train_x)):
    if len(train_x[i])<=MAX_LEN:#如果句子长度小于max_sen_len
        train_x_cut.append(train_x[i])
        train_y_cut.append(train_y[i])
        train_mask.append([1]*len(train_x[i]))
        continue
    while len(train_x[i])>MAX_LEN:#超过100，使用标点符号拆分句子，将前面部分加入训练集，若后面部分仍超过100，继续拆分
        flag=False
        for j in reversed(range(MAX_LEN)):#反向访问，99、98、97...
            if train_x[i][j]==word2index['，'] or train_x[i][j]==word2index['、']:
                train_x_cut.append(train_x[i][:j+1])
                train_y_cut.append(train_y[i][:j+1])
                train_mask.append([1]*(j+1))
                train_x[i]=train_x[i][j+1:]
                train_y[i]=train_y[i][j+1:]
                break
            if j==0:
                flag=True
        if flag:
            train_x_cut.append(train_x[i][:MAX_LEN])
            train_y_cut.append(train_y[i][:MAX_LEN])
            train_mask.append([1]*MAX_LEN)
            train_x[i]=train_x[i][MAX_LEN:]
            train_y[i]=train_y[i][MAX_LEN:]
    if len(train_x[i])<=MAX_LEN:#如果句子长度小于max_sen_len，最后没有超过100的直接加入
        train_x_cut.append(train_x[i])
        train_y_cut.append(train_y[i])
        train_mask.append([1]*len(train_x[i]))

#给每段分割填充0
for i in range(len(train_x_cut)):
    if len(train_x_cut[i])<MAX_LEN:
        tlen=len(train_x_cut[i])
        for j in range(MAX_LEN-tlen):
            train_x_cut[i].append(0)

for i in range(len(train_y_cut)):
    if len(train_y_cut[i])<MAX_LEN:
        tlen=len(train_y_cut[i])
        for j in range(MAX_LEN-tlen):
            train_y_cut[i].append(0)

for i in range(len(train_mask)):
    if len(train_mask[i])<MAX_LEN:
        tlen=len(train_mask[i])
        for j in range(MAX_LEN-tlen):
            train_mask[i].append(0)

#将以上数据划分为训练集和验证集
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y,train_mask,val_mask=train_test_split(train_x_cut,train_y_cut,train_mask,test_size=0.2,random_state=20180306)

#将数据转化为LongTensor
train_x=torch.LongTensor(train_x)
val_x=torch.LongTensor(val_x)
train_y=torch.LongTensor(train_y)
val_y=torch.LongTensor(val_y)
train_mask=torch.ByteTensor(train_mask)
val_mask=torch.ByteTensor(val_mask)
#返回字典
def get_dic():
    return word2index,index2word,tag2index,index2tag

#返回训练集和验证集
def get_data():
    return train_x,val_x,train_y,val_y,train_mask,val_mask

#定义TextDataSet类
class TextDataSet(Dataset):
    def __init__(self,inputs,outputs,masks):
        self.inputs,self.outputs,self.masks=inputs,outputs,masks
    def __getitem__(self, item):
        return self.inputs[item],self.outputs[item],self.masks[item]
    def __len__(self):
        return len(self.inputs)

#对测试集处理的函数
def getTest_x(filepath):
    data=open(filepath,'r')
    test_x = []  # 总x测试集
    test_word=[] #所有句话的词
    sen_x = []  # 每次存一句话的id组
    sen_word=[]# 一句话的词

    # 将数据按每句话分出来
    for line in data:
        line = line.strip()
        if (line == "" or line == "\n" or line == "\r\n"):  # 一句话结束了
            test_x.append(sen_x)
            sen_x = []
            test_word.append(sen_word)
            sen_word=[]
            continue
        line = line.split(' ')
        sen_word.append(line[0])
        if line[0] in word2index:  # 如果在词典中有该词，将id给sen_x
            sen_x.append(word2index[line[0]])
        else:  # 如果没有则设为未识别
            sen_x.append(1)

    # 开始对每句话进行裁剪，主要是最大长度的限制
    test_x_cut = []#每个分割的词id
    test_x_len=[]#每句话本身的长度（不填充的长度）
    test_x_cut_word=[]#所有分割出的词
    count=0#用于样本计数
    test_x_fenge=[]#用于记分割了的样本序号
    for i in range(len(test_x)):
        if len(test_x[i]) <= MAX_LEN:  # 如果句子长度小于max_sen_len
            test_x_cut.append(test_x[i])
            test_x_len.append(len(test_x[i]))
            test_x_cut_word.append(test_word[i])
            count+=1
            continue
        while len(test_x[i]) > MAX_LEN:  # 超过100，使用标点符号拆分句子，将前面部分加入训练集，若后面部分仍超过100，继续拆分
            flag = False
            for j in reversed(range(MAX_LEN)):  # 反向访问，99、98、97...
                if test_x[i][j] == word2index['，'] or test_x[i][j] == word2index['、']:
                    test_x_cut.append(test_x[i][:j + 1])
                    test_x_len.append(j+1)
                    test_x_cut_word.append(test_word[i][:j+1])
                    test_x[i] = test_x[i][j + 1:]
                    test_x_cut_word[i]=test_word[i][j+1:]
                    test_x_fenge.append(count)
                    count+=1
                    break
                if j == 0:
                    flag = True
            if flag:
                test_x_cut.append(test_x[i][:MAX_LEN])
                test_x_len.append(MAX_LEN)
                test_x_cut_word.append(test_word[i][:MAX_LEN])
                test_x[i] = test_x[i][MAX_LEN:]
                test_x_cut_word[i]=test_word[i][MAX_LEN:]
                test_x_fenge.append(count)
                count+=1
        if len(test_x[i]) <= MAX_LEN:  # 如果句子长度小于max_sen_len，最后没有超过100的直接加入
            test_x_cut.append(test_x[i])
            test_x_len.append(len(test_x[i]))
            test_x_cut_word.append(test_word[i])
            count += 1

    # 给每段分割填充0
    for i in range(len(test_x_cut)):
        if len(test_x_cut[i]) < MAX_LEN:
            tlen = len(test_x_cut[i])
            for j in range(MAX_LEN - tlen):
                test_x_cut[i].append(0)
    #转化LongTensor
    test_x_cut=torch.LongTensor(test_x_cut)
    return test_x_cut,test_x_len,test_x_cut_word,test_x_fenge

def write_result_to_file(filepath,y_pred,test_x_len,test_x_cut_word,test_x_fenge):
    pass