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
        self.num_tags=len(tag2index)
        self.emb_dim=emb_dim
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.use_cuda=torch.cuda.is_available()
        self.embed=nn.Embedding(num_embeddings=vcab_size,embedding_dim=emb_dim)#b,100,128
        #->100,b,128
        self.bilstm=nn.LSTM(input_size=emb_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True,dropout=0.1)#100,b,256*2
        self.conv1 = nn.Sequential(
            #b,1,100,128
            nn.Conv2d(1, 128, (1,emb_dim),padding=0),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 128, (3,emb_dim+2), padding=1),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 128, (5,emb_dim+4), padding=2),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        #b,128*3,100,1->100,b,128*3
        self.linear1 = nn.Linear(hidden_dim * 2+128*3,hidden_dim)
        self.drop=nn.Dropout(0.2)
        self.classfy=nn.Linear(hidden_dim,self.num_tags)#100*b,10
        #->100,b,10
        # init transitions
        self.start_transitions = nn.Parameter(torch.Tensor(self.num_tags))#i表示出发，j表示到达
        self.end_transitions = nn.Parameter(torch.Tensor(self.num_tags))#i表示到达，j表示出发
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))#i表示出发，j表示到达
        nn.init.uniform(self.start_transitions, -0.1, 0.1)
        nn.init.uniform(self.end_transitions, -0.1, 0.1)
        nn.init.uniform(self.transitions, -0.1, 0.1)

    def init_hidden(self,batch_size):#作为初始化传入lstm的隐含变量
        h_h=Variable(torch.randn(2,batch_size,self.hidden_dim))
        h_c=Variable(torch.randn(2,batch_size,self.hidden_dim))
        if use_cuda:
            h_h=h_h.cuda()
            h_c=h_c.cuda()
        return (h_h,h_c)

    def get_bilstm_out(self,x):#计算bilstm的输出
        batch_size = x.size(0)
        emb=self.embed(x)

        #cnn输出
        emb_cnn=emb.unsqueeze(1)
        cnn1=self.conv1(emb_cnn)
        cnn2=self.conv2(emb_cnn)
        cnn3=self.conv3(emb_cnn)
        cnn_cat=torch.cat((cnn1,cnn2,cnn3),1)
        cnn_out=cnn_cat.squeeze().permute(2,0,1)#100,b,128*3

        emb_rnn=emb.permute(1,0,2)
        init_hidden=self.init_hidden(batch_size)
        lstm_out,hidden=self.bilstm(emb_rnn,init_hidden)

        cat_out=torch.cat((cnn_out,lstm_out),2)#100,b,128*3+256*2
        s,b,h=cat_out.size()
        cat_out=cat_out.view(s*b,h)
        cat_out=self.linear1(cat_out)
        cat_out=self.drop(cat_out)
        cat_out=self.classfy(cat_out)
        cat_out=cat_out.view(s,b,-1)
        # out=out.permute(1,0,2)
        return cat_out

    def _log_sum_exp(self,tensor,dim):
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)#b,m
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)#b,1,m
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))#b,m
        # Add offset back
        return offset + safe_log_sum_exp

    def get_all_score(self,emissions,mask):#计算所有路径的总分#s,b,h
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (batch_size,seq_length)
        seq_length = emissions.size(0)
        mask = mask.permute(1,0).contiguous().float()

        log_prob = self.start_transitions.view(1, -1) + emissions[0]  # b,m,所有从start出发的路径s0

        for i in range(1, seq_length):
            broadcast_log_prob = log_prob.unsqueeze(2)  # b,m,1
            broadcast_transitions = self.transitions.unsqueeze(0)  #1,m,m
            broadcast_emissions = emissions[i].unsqueeze(1)  # b,1,m

            score = broadcast_log_prob + broadcast_transitions \
                    + broadcast_emissions  # b,m,m

            score = self._log_sum_exp(score, 1)  # b,m即为si

            log_prob = score * mask[i].unsqueeze(1) + log_prob * (1. - mask[i]).unsqueeze(
                1)  # mask为0的保持不变，mask为1的更换score

        # End transition score
        log_prob += self.end_transitions.view(1, -1)
        # Sum (log-sum-exp) over all possible tags
        return self._log_sum_exp(log_prob, 1)  # (batch_size,)返回最终score

    def get_real_score(self,emissions,mask,tags):#计算真实路径得分
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (batch_size,seq_length)
        # mask: (batch_size,seq_length)
        seq_length = emissions.size(0)#s
        mask = mask.permute(1,0).contiguous().float()
        tags=tags.permute(1,0).contiguous()

        # Start transition score
        llh = self.start_transitions[tags[0]]  # (batch_size,),T(start->firstTag)

        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i+1]
            # Emission score for current tag
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]#(b,1)->b->b*mask，上一轮score+当前发射概率
            # Transition score to next tag
            transition_score = self.transitions[cur_tag.data, next_tag.data]#当前到下一轮的转换概率
            # Only add transition score if the next tag is not masked (mask == 1)
            llh += transition_score * mask[i+1]#若下一轮为padding则不转换

        # Find last tag index
        last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)计算每个序列真实长度
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)#b,最后一个非padding的标签id

        # End transition score
        llh += self.end_transitions[last_tags]#加上从最后一个非padding标签到end的转换概率
        # Emission score for the last tag, if mask is valid (mask == 1)
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]#考虑最后一个seq为有效id

        return llh#b

    def neg_log_likelihood(self,feats,tags,mask):
        #feats:  bilstm的输出#100,b,10
        batch_size=feats.size(1)
        all_score=self.get_all_score(feats,mask)#所有路径总分b
        real_score=self.get_real_score(feats,mask,tags)#真实路径得分b
        loss=(all_score.view(batch_size,1)-real_score.view(batch_size,1)).sum()/batch_size
        return loss #目标是最小化这个值，即最大化没log前的真实占总的比例

    def viterbi_decode(self, emissions,mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (batch_size,seq_length)
        seq_length=emissions.size(0)
        batch_size=emissions.size(1)
        num_tags=emissions.size(2)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()  # 真实序列长度b,1
        mask=mask.permute(1,0).contiguous().float()#s,b

        viterbi_history=[]
        viterbi_score = self.start_transitions.view(1, -1) + emissions[0]  # b,m,所有从start出发的路径s0

        for i in range(1, seq_length):
            broadcast_viterbi_score = viterbi_score.unsqueeze(2)  # b,m,1
            broadcast_transitions = self.transitions.unsqueeze(0)  #1,m,m
            broadcast_emissions = emissions[i].unsqueeze(1)  # b,1,m

            score = broadcast_viterbi_score + broadcast_transitions \
                    + broadcast_emissions  # b,m,m

            best_score,best_path = torch.max(score, 1)  # b,m即为si
            viterbi_history.append(best_path*mask[i].long().unsqueeze(1))#将带0pading的路径加进来
            viterbi_score = best_score * mask[i].unsqueeze(1) + viterbi_score * (1. - mask[i]).unsqueeze(
                1)  # mask为0的保持不变，mask为1的更换score
        viterbi_score+=self.end_transitions.view(1,-1)#b,m
        best_score,last_path=torch.max(viterbi_score,1)#b
        last_path=last_path.view(-1,1)#b,1
        last_position = (length_mask.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, num_tags) - 1).contiguous()  # 最后一个非padding的位置b,1->b,1,m
        pad_zero = Variable(torch.zeros(batch_size, num_tags)).long()
        if use_cuda:
            pad_zero = pad_zero.cuda()
        viterbi_history.append(pad_zero)#(s-1,b,m)->(s,b,m)
        viterbi_history = torch.cat(viterbi_history).view(-1, batch_size, num_tags)  # s,b,m
        insert_last = last_path.view(batch_size, 1, 1).expand(batch_size, 1, num_tags) #要将最后的路径插入最后的真实位置b,1,m
        viterbi_history = viterbi_history.transpose(1, 0).contiguous()  # b,s,m
        viterbi_history.scatter_(1, last_position, insert_last)  # 将最后位置的路径统一改为相同路径b,s,m（back_points中的某些值改变了）
        viterbi_history = viterbi_history.transpose(1, 0).contiguous()  # s,b,m
        decode_idx = Variable(torch.LongTensor(seq_length, batch_size))#最后用来记录路径的s,b
        if use_cuda:
            decode_idx = decode_idx.cuda()
        # decode_idx[-1] = 0
        for idx in range(len(viterbi_history)-2,-1,-1):
            last_path=torch.gather(viterbi_history[idx],1,last_path)
            decode_idx[idx]=last_path.data
        decode_idx=decode_idx.transpose(1,0)#b,s
        return decode_idx

    def forward(self, feats,mask):
        #feats    #bilstm的输出#100.b.10
        best_path=self.viterbi_decode(feats,mask)#最佳路径b,s
        return best_path


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
    batch_len_all=0
   # model.train()
    for i,data in enumerate(trainDataLoader):
        x,y,mask=data
        batch_len = len(x)
        batch_len_all += batch_len
        if use_cuda:
            x=Variable(x).cuda()
            y=Variable(y).cuda()
            mask=Variable(mask).cuda()
        else:
            x=Variable(x)
            y=Variable(y)
            mask=Variable(mask)
        feats=model.get_bilstm_out(x)
        loss=model.neg_log_likelihood(feats,y,mask)
        train_loss+=loss.data[0]
        prepath=model(feats,mask)#b,s
        pre_y=prepath.masked_select(mask)
        true_y=y.masked_select(mask)
        acc_num=(pre_y==true_y).data.sum()
        # acc_num=(pre_y==true_y).sum()
        acc_pro=float(acc_num)/len(pre_y)
        train_acc+=acc_pro
        #backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        if (i + 1) % 100 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i+1, len(trainDataLoader),
                                                                            train_loss / (i+1),
                                                                            train_acc / (i+1)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(trainDataLoader)),
                                                                     train_acc / (len(trainDataLoader))))
    # model.eval()
    eval_loss = 0
    eval_acc = 0
    batch_len_all = 0
    for i, data in enumerate(valDataLoader):
        x, y,mask = data
        batch_len = len(x)
        batch_len_all += batch_len
        if use_cuda:
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
            mask=Variable(mask,volatile=True).cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            mask = Variable(mask, volatile=True)
        feats=model.get_bilstm_out(x)
        loss=model.neg_log_likelihood(feats,y,mask)
        eval_loss += loss.data[0]
        prepath = model(feats, mask)  # b,s
        pre_y = prepath.masked_select(mask)
        true_y = y.masked_select(mask)
        acc_num = (pre_y == true_y).data.sum()
        acc_pro = float(acc_num) / len(pre_y)
        eval_acc += acc_pro
    print('val loss is:{:.6f},val acc is:{:.6f}'.format(
        eval_loss / (len(valDataLoader) ),
        eval_acc / (len(valDataLoader))))
    if best_acc < (eval_acc / (len(valDataLoader))):
        best_acc = eval_acc / (len(valDataLoader))
        best_model = model.state_dict()
        print('best acc is {:.6f},best model is changed'.format(best_acc))

torch.save(best_model,'./model/best_model.pth')
torch.save(model.state_dict(),'./model/last_model.pth')