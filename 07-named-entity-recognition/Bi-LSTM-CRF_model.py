import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim,nn
import data_preprocess
import os
torch.manual_seed(1)

os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda=torch.cuda.is_available()

word2index,index2word,tag2index,index2tag=data_preprocess.get_dic()
train_x,val_x,train_y,val_y,train_mask,val_mask=data_preprocess.get_data()
trainDataSet=data_preprocess.TextDataSet(train_x,train_y,train_mask)
valDataSet=data_preprocess.TextDataSet(val_x,val_y,val_mask)
trainDataLoader=DataLoader(trainDataSet,batch_size=16,shuffle=True)
valDataLoader=DataLoader(valDataSet,batch_size=16,shuffle=False)

MAXLEN=100
vcab_size=len(word2index)
emb_dim=128
hidden_dim=256
num_epoches=20
batch_size=16


class BILSTM_CRF(nn.Module):
    def __init__(self,vcab_size,tag2index,emb_dim,hidden_dim,batch_size):
        super(BILSTM_CRF,self).__init__()
        self.vcab_size=vcab_size
        self.tag2index=tag2index
        self.tag_size=len(tag2index)
        self.emb_dim=emb_dim
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.embed=nn.Embedding(num_embeddings=vcab_size,embedding_dim=emb_dim)#b,100,128
        #->100,b,128
        self.bilstm=nn.LSTM(input_size=emb_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True)#100,b,256*2
        #->100*b,256*2
        self.classfy=nn.Linear(hidden_dim*2,self.tag_size+2)#100*b,10
        #->100,b,10
        #->b,100,10
        #存储crf层中的转移概率,大小为10*10
        #i,j代表从j转移到i
        self.start,self.end=-2,-1
        self.transitions=torch.zeros(self.tag_size+2,self.tag_size+2)
        self.transitions[:,self.start]=-1000.#转移到start的阻力
        self.transitions[self.end,:]=-1000.#从stop转移出的阻力
        if use_cuda:
            self.transitions=self.transitions.cuda()
        self.transitions=nn.Parameter(self.transitions)

    def init_hidden(self):#作为初始化传入lstm的隐含变量
        h_h=Variable(torch.randn(2,self.batch_size,self.hidden_dim))
        h_c=Variable(torch.randn(2,self.batch_size,self.hidden_dim))
        if use_cuda:
            h_h=h_h.cuda()
            h_c=h_c.cuda()
        return (h_h,h_c)

    def get_bilstm_out(self,x):#计算bilstm的输出
        x=self.embed(x)
        x=x.permute(1,0,2)
        init_hidden=self.init_hidden()
        out,hidden=self.bilstm(x,init_hidden)
        s,b,h=out.size()#长宽高
        out=out.view(s*b,-1)
        out=self.classfy(out)
        out=out.view(s,b,-1)
        out=out.permute(1,0,2)
        return out

    def log_sum_exp(self,vec,m_size):#b,tag_size,tag_size
        max_value,idx=torch.max(vec,1)#b,m
        max_score=max_value.view(-1,1,m_size)#b,1,m
        return max_score.view(-1,m_size)+torch.log(torch.sum(torch.exp(vec-max_score.expand_as(vec)),1)).view(-1,m_size)# 为了让exp的值不过大，先剪max再加回来

    def get_all_score(self,feats,mask):#计算所有路径的总分#s,b,h
        """
        feats: size=(batch_size, seq_len, self.tag_size+2)
        mask: size=(batch_size, seq_len)
        """
        batch_size=feats.size(0)
        seq_len=feats.size(1)
        tag_size=feats.size(-1)
        mask=mask.transpose(1,0).contiguous()#变为s，b
        b_s_num=batch_size*seq_len
        feats=feats.transpose(1,0).contiguous().view(b_s_num,1,tag_size).expand(b_s_num,tag_size,tag_size)#存储着所有发射概率，并将其广播
        trans=self.transitions.view(1,tag_size,tag_size).expand(b_s_num,tag_size,tag_size)#存储着所有的转移概率，并将其广播
        scores=feats+trans# 将发射概率和转移概率预先全部加在一起，不用循环计算
        scores=scores.view(seq_len,batch_size,tag_size,tag_size)

        seq_iter=enumerate(scores)#step迭代器
        _,initvalues=seq_iter.__next__()#初始值
        partition=initvalues[:,self.start,:].clone().view(batch_size,tag_size,1)#第0次，从start开始转换即(s0=e0.broadcast+t)['start'],并转化为竖行相加
        for idx,cur_values in seq_iter:
            cur_values=cur_values+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size)#相当于(e[i].broadcats+t)+s[i-1].broadcast
            cur_partition=self.log_sum_exp(cur_values,tag_size)#b,tag_size
            mask_idx=mask[idx,:].view(batch_size,1).expand(batch_size,tag_size)#用mask去掉padding的计算
            masked_cur_partition=cur_partition.masked_select(mask_idx)
            mask_idx=mask_idx.contiguous().view(batch_size,tag_size,1)
            partition.masked_scatter_(mask_idx,masked_cur_partition)#将mask为1的值复制到本tensor中,即padding的不在增加分数
        cur_values=self.transitions.view(1,tag_size,tag_size).expand(batch_size,tag_size,tag_size)+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size)
        cur_partition=self.log_sum_exp(cur_values,tag_size)
        final_partition=cur_partition[:,self.end]#b,1
        return final_partition.sum(),scores

    def get_real_score(self,scores,mask,tags):#计算真实路径得分
        """
        scores: size=(seq_len, batch_size, self.tag_size+2, self.tag_size+2)
        mask: size=(batch_size, seq_len)
        target: size=(batch_size, seq_len)
        """
        batch_size=scores.size(1)
        seq_len=scores.size(0)
        tag_size=scores.size(-1)
        new_tags=Variable(torch.LongTensor(batch_size,seq_len))
        if use_cuda:
            new_tags=new_tags.cuda()
        for idx in range(seq_len):#当前的序列id=前序列id*tag_size+当前序列id
            if idx==0:
                new_tags[:,0]=(tag_size-2)*tag_size+tags[:,0]
            else:
                new_tags[:,idx]=tags[:,idx-1]*tag_size+tags[:,idx]
        end_transition=self.transitions[:,self.end].contiguous().view(1,tag_size).expand(batch_size,tag_size)
        length_mask=torch.sum(mask,dim=1).view(batch_size,1).long()# 真实序列长度b,1
        end_ids=torch.gather(tags,1,length_mask-1)#取最后一个有效id（非padding）b,1
        end_energy=torch.gather(end_transition,1,end_ids)#从最后一个有效id到end的转移概率b,1
        new_tags=new_tags.transpose(1,0).contiguous().view(seq_len,batch_size,1)
        tg_energy=torch.gather(scores.view(seq_len,batch_size,-1),2,new_tags).view(seq_len,batch_size)#所有分数,未view前的 s,b,1代表每个序列每个batch的发射和转移概率之和，
        tg_energy=tg_energy.masked_select(mask.transpose(1,0))
        real_score=tg_energy.sum()+end_energy.sum()
        return real_score

    def neg_log_likelihood(self,x,tags,mask):
        feats=self.get_bilstm_out(x)#bilstm的输出#b,100,10
        batch_size=feats.size(0)
        forward_score,scores=self.get_all_score(feats,mask)#所有路径总分以及发射概率和转移概率的所有组合和（scores）
        real_score=self.get_real_score(scores,mask,tags)#真实路径得分
        loss=(forward_score-real_score)/batch_size
        return loss #目标是最小化这个值，即最大化没log前的真实占总的比例

    def viterbi_decode(self,feats,mask):#使用viterbi算法计算最优路径和最优得分
        """
        feats: size=(batch_size, seq_len, self.target_size+2)
        mask: size=(batch_size, seq_len)
        """
        batch_size=feats.size(0)
        seq_len=feats.size(1)
        tag_size=feats.size(-1)
        length_mask=torch.sum(mask,dim=1).view(batch_size,1).long()#真实序列长度b,1
        mask=mask.transpose(1,0).contiguous()#s,b将padding置为1
        b_s_num=seq_len*batch_size
        feats=feats.transpose(1,0).contiguous().view(b_s_num,1,tag_size).expand(b_s_num,tag_size,tag_size)#存储着所有发射概率，并将其广播
        trans=self.transitions.view(1,tag_size,tag_size).expand(b_s_num,tag_size,tag_size)#存储着所有的转移概率，并将其广播
        scores = feats + trans  # 将发射概率和转移概率预先全部加在一起，不用循环计算
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter=enumerate(scores)
        back_points=[]#记录最佳路径
        partition_history=[]
        mask=Variable(1-mask.data)
        _,initvalues=seq_iter.__next__()#初始值
        partition=initvalues[:,self.start,:].clone().view(batch_size,tag_size,1)#第0次，从start开始转换即(s0=e0.broadcast+t)['start'],并转化为竖行相加
        partition_history.append(partition)
        for idx,cur_values in seq_iter:
            cur_values=cur_values+partition.contiguous().view(batch_size,tag_size,1).expand(batch_size,tag_size,tag_size)#相当于(e[i].broadcats+t)+s[i-1].broadcast
            partition,cur_bp=torch.max(cur_values,1)#partition获取到达不同标签各自的最大值，cur_bp获取到达不同标签各自的来源标签b,tag_size
            partition_history.append(partition.view(batch_size,tag_size,1))
            cur_bp.masked_fill_(mask[idx].view(batch_size,1).expand(batch_size,tag_size),0)#将序列已经为padding的来源标签即上一个序列的到达标签用0填充
            back_points.append(cur_bp)

        partition_history=torch.cat(partition_history).view(seq_len,batch_size,-1).transpose(1,0).contiguous()#b,s,t
        last_position=length_mask.view(batch_size,1,1).expand(batch_size,1,tag_size)-1# 最后一个非padding的位置b,1
        last_partition=torch.gather(partition_history,1,last_position).view(batch_size,tag_size,1)#最后一个非padding序列的所有标签的score#b,tag_size,1
        last_values=last_partition.expand(batch_size,tag_size,tag_size)+self.transitions.view(1,tag_size,tag_size).expand(batch_size,tag_size,tag_size)#last_score+t，最后非padding序列再加一次
        max_value,last_bp=torch.max(last_values,1)#b,tag_size, 用last_bp获取出发标签
        pad_zero=Variable(torch.zeros(batch_size,tag_size)).long()
        if use_cuda:
            pad_zero=pad_zero.cuda()
        back_points.append(pad_zero)
        back_points=torch.cat(back_points).view(seq_len,batch_size,tag_size)#s,b,tag_size
        pointer=last_bp[:,self.end]#b,1，获取到达end标签的来源标签
        insert_last=pointer.contiguous().view(batch_size,1,1).expand(batch_size,1,tag_size)
        back_points=back_points.transpose(1,0).contiguous()#b,s,tag_size，广播为都一样的id
        back_points.scatter_(1,last_position,insert_last)#将最后的路径统一改为相同路径
        back_points=back_points.transpose(1,0).contiguous()#s,b,tag_size
        decode_idx=Variable(torch.LongTensor(seq_len,batch_size))
        if use_cuda:
            decode_idx=decode_idx.cuda()
        decode_idx[-1]=pointer.data
        for idx in range(len(back_points)-2,-1,-1):
            pointer=torch.gather(back_points[idx],1,pointer.contiguous().view(batch_size,1))
            decode_idx[idx]=pointer.data
        path_score=max_value[:,self.end]#b,1
        decode_idx=decode_idx.transpose(1,0)#b,s
        return path_score,decode_idx

    def forward(self, x,mask):
        feats=self.get_bilstm_out(x)#bilstm的输出#b,100,10
        path_score,best_path=self.viterbi_decode(feats,mask)#最佳得分和最佳路径
        return path_score,best_path


if use_cuda:
    model=BILSTM_CRF(vcab_size,tag2index,emb_dim,hidden_dim,batch_size).cuda()
else:
    model=BILSTM_CRF(vcab_size,tag2index,emb_dim,hidden_dim,batch_size)

optimzier=optim.Adam(model.parameters(),lr=1e-3)

best_acc=0
best_model=None
for epoch in range(num_epoches):
    train_loss=0
    train_acc=0
    model.train()
    for i,data in enumerate(trainDataLoader):
        x,y,mask=data
        if use_cuda:
            x=Variable(x).cuda()
            y=Variable(y).cuda()
            mask=Variable(mask).cuda()
        else:
            x=Variable(x)
            y=Variable(y)
            mask=Variable(mask)
        loss=model.neg_log_likelihood(x,y,mask)
        train_loss+=loss.data[0]
        _,prepath=model(x,mask)
        acc_num=(prepath==y).data.sum()
        train_acc+=acc_num
        #backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        if (i + 1) % 10 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(trainDataLoader),
                                                                            train_loss / (i),
                                                                            train_acc / (i * batch_size*MAXLEN)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(trainDataLoader)),
                                                                     train_acc / (len(trainDataLoader) * batch_size*MAXLEN)))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(valDataLoader):
        x, y,mask = data
        if use_cuda:
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
            mask=Variable(mask,volatile=True).cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            mask = Variable(mask, volatile=True)
        loss=model.neg_log_likelihood(x,y,mask)
        eval_loss += loss.data[0]
        _, prepath = model(x,mask)
        acc_num = (prepath == y).data.sum()
        eval_acc += acc_num
    print('val loss is:{:.6f},test acc is:{:.6f}'.format(
        eval_loss / (len(valDataLoader) ),
        eval_acc / (len(valDataLoader) * batch_size*MAXLEN)))
    if best_acc < (eval_acc / (len(valDataLoader) * batch_size*MAXLEN)):
        best_acc = eval_acc / (len(valDataLoader) * batch_size*MAXLEN)
        best_model = model.state_dict()
        print('best acc is {:.6f},best model is changed'.format(best_acc))

torch.save(best_model.state_dict(),'./model/best_model.pth')
torch.save(model.state_dict(),'./model/last_model.pth')
