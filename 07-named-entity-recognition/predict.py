import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim, nn
import data_preprocess
import os

torch.manual_seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

word2index, index2word, tag2index, index2tag = data_preprocess.get_dic()
test_x_cut, test_y_cut, test_mask, test_x_len, test_x_cut_word, test_x_fenge = data_preprocess.getTest_xy(
    './data/test_data')
testDataSet = data_preprocess.TextDataSet(test_x_cut, test_y_cut, test_mask)

testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False)

MAXLEN = 100
vcab_size = len(word2index)
emb_dim = 128
hidden_dim = 256
num_epoches = 20
batch_size = 16


class BILSTM_CRF(nn.Module):
    def __init__(self, vcab_size, tag2index, emb_dim, hidden_dim, batch_size):
        super(BILSTM_CRF, self).__init__()
        self.vcab_size = vcab_size
        self.tag2index = tag2index
        self.num_tags = len(tag2index)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.embed = nn.Embedding(num_embeddings=vcab_size, embedding_dim=emb_dim)  # b,100,128
        # ->100,b,128
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True,
                              dropout=0.1)  # 100,b,256*2
        self.conv1 = nn.Sequential(
            # b,1,100,128
            nn.Conv2d(1, 128, (1, emb_dim), padding=0),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 128, (3, emb_dim + 2), padding=1),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 128, (5, emb_dim + 4), padding=2),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # b,128*3,100,1->100,b,128*3
        self.linear1 = nn.Linear(hidden_dim * 2 + 128 * 3, hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.classfy = nn.Linear(hidden_dim, self.num_tags)  # 100*b,10
        # ->100,b,10
        # init transitions
        self.start_transitions = nn.Parameter(torch.Tensor(self.num_tags))  # i表示出发，j表示到达
        self.end_transitions = nn.Parameter(torch.Tensor(self.num_tags))  # i表示到达，j表示出发
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))  # i表示出发，j表示到达
        nn.init.uniform(self.start_transitions, -0.1, 0.1)
        nn.init.uniform(self.end_transitions, -0.1, 0.1)
        nn.init.uniform(self.transitions, -0.1, 0.1)

    def init_hidden(self, batch_size):  # 作为初始化传入lstm的隐含变量
        h_h = Variable(torch.randn(2, batch_size, self.hidden_dim))
        h_c = Variable(torch.randn(2, batch_size, self.hidden_dim))
        if use_cuda:
            h_h = h_h.cuda()
            h_c = h_c.cuda()
        return (h_h, h_c)

    def get_bilstm_out(self, x):  # 计算bilstm的输出
        batch_size = x.size(0)
        emb = self.embed(x)

        # cnn输出
        emb_cnn = emb.unsqueeze(1)
        cnn1 = self.conv1(emb_cnn)
        cnn2 = self.conv2(emb_cnn)
        cnn3 = self.conv3(emb_cnn)
        cnn_cat = torch.cat((cnn1, cnn2, cnn3), 1)
        cnn_out = cnn_cat.squeeze().permute(2, 0, 1)  # 100,b,128*3

        emb_rnn = emb.permute(1, 0, 2)
        init_hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.bilstm(emb_rnn, init_hidden)

        cat_out = torch.cat((cnn_out, lstm_out), 2)  # 100,b,128*3+256*2
        s, b, h = cat_out.size()
        cat_out = cat_out.view(s * b, h)
        cat_out = self.linear1(cat_out)
        cat_out = self.drop(cat_out)
        cat_out = self.classfy(cat_out)
        cat_out = cat_out.view(s, b, -1)
        # out=out.permute(1,0,2)
        return cat_out

    def _log_sum_exp(self, tensor, dim):
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)  # b,m
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)  # b,1,m
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))  # b,m
        # Add offset back
        return offset + safe_log_sum_exp

    def get_all_score(self, emissions, mask):  # 计算所有路径的总分#s,b,h
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)
        mask = mask.permute(1, 0).contiguous().float()

        log_prob = self.start_transitions.view(1, -1) + emissions[0]  # b,m,所有从start出发的路径s0

        for i in range(1, seq_length):
            broadcast_log_prob = log_prob.unsqueeze(2)  # b,m,1
            broadcast_transitions = self.transitions.unsqueeze(0)  # 1,m,m
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

    def get_real_score(self, emissions, mask, tags):  # 计算真实路径得分
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)  # s
        mask = mask.permute(1, 0).contiguous().float()
        tags = tags.permute(1, 0).contiguous()

        # Start transition score
        llh = self.start_transitions[tags[0]]  # (batch_size,),T(start->firstTag)

        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i + 1]
            # Emission score for current tag
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]  # (b,1)->b->b*mask，上一轮score+当前发射概率
            # Transition score to next tag
            transition_score = self.transitions[cur_tag.data, next_tag.data]  # 当前到下一轮的转换概率
            # Only add transition score if the next tag is not masked (mask == 1)
            llh += transition_score * mask[i + 1]  # 若下一轮为padding则不转换

        # Find last tag index
        last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)计算每个序列真实长度
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)  # b,最后一个非padding的标签id

        # End transition score
        llh += self.end_transitions[last_tags]  # 加上从最后一个非padding标签到end的转换概率
        # Emission score for the last tag, if mask is valid (mask == 1)
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]  # 考虑最后一个seq为有效id

        return llh  # b

    def neg_log_likelihood(self, feats, tags, mask):
        # feats:  bilstm的输出#100,b,10
        batch_size = feats.size(1)
        all_score = self.get_all_score(feats, mask)  # 所有路径总分b
        real_score = self.get_real_score(feats, mask, tags)  # 真实路径得分b
        loss = (all_score.view(batch_size, 1) - real_score.view(batch_size, 1)).sum() / batch_size
        return loss  # 目标是最小化这个值，即最大化没log前的真实占总的比例

    def _viterbi_decode(self, emission):  # 使用viterbi算法计算最优路径和最优得分
        # emission: (seq_length, num_tags)
        seq_length = emission.size(0)

        # Start transition
        viterbi_score = self.start_transitions + emission[0]  # s0
        viterbi_path = []
        # Here, viterbi_score has shape of (num_tags,) where value at index i stores
        # the score of the best tag sequence so far that ends with tag i
        # viterbi_path saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = viterbi_score.view(-1, 1)  # m,1
            # Broadcast emission score for every possible current tag
            broadcast_emission = emission[i].view(1, -1)  # 1,m
            # Compute the score matrix of shape (num_tags, num_tags) where each entry at
            # row i and column j stores the score of transitioning from tag i to tag j
            # and emitting
            score = broadcast_score + self.transitions + broadcast_emission  # m,m
            # Find the maximum score over all possible current tag
            best_score, best_path = score.max(0)  # m
            # Save the score and the path
            viterbi_score = best_score
            viterbi_path.append(best_path)

        # End transition
        viterbi_score += self.end_transitions  # m

        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = viterbi_score.max(0)  # 1
        best_tags = [best_last_tag[0]]  # 最后到达的一个tag
        best_tags_pad = [best_last_tag[0]]
        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for path in reversed(viterbi_path):
            best_last_tag = path[best_tags[-1]]
            best_tags.append(best_last_tag)
            best_tags_pad.append(best_last_tag)

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_pad.reverse()
        pad = Variable(torch.LongTensor([0]))
        if use_cuda:
            pad = pad.cuda()
        for i in range(MAXLEN - seq_length):
            best_tags_pad.append(pad)
        return torch.cat(best_tags), torch.cat(best_tags_pad)

    def decode(self, emissions, mask):
        # Transpose batch_size and seq_length
        batch_len = emissions.size(1)
        emissions = emissions.transpose(0, 1)
        # mask = mask.transpose(0, 1)

        best_tags = []
        best_tags_pad = []
        for emission, mask_ in zip(emissions, mask):
            seq_length = mask_.data.int().sum()  # 真实长度
            # e=emission[:seq_length]
            best_t, best_t_pad = self._viterbi_decode(emission[:seq_length])
            best_tags.append(best_t)
            best_tags_pad.append(best_t_pad)
        best_tags_pad = torch.cat(best_tags_pad).view(batch_len, MAXLEN)
        return best_tags, best_tags_pad

    def forward(self, feats, mask):
        # feats    #bilstm的输出#100.b.10
        best_path, best_path_pad = self.decode(feats, mask)  # 最佳路径
        return best_path, best_path_pad


if use_cuda:
    model = BILSTM_CRF(vcab_size, tag2index, emb_dim, hidden_dim, batch_size).cuda()
else:
    model = BILSTM_CRF(vcab_size, tag2index, emb_dim, hidden_dim, batch_size)

model.load_state_dict(torch.load('./model/best_model.pth'))

# model.eval()
test_loss = 0
test_acc = 0
batch_len_all = 0
for i, data in enumerate(testDataLoader):
    x, y, mask = data
    batch_len = len(x)
    batch_len_all += batch_len
    if use_cuda:
        x = Variable(x, volatile=True).cuda()
        y = Variable(y, volatile=True).cuda()
        mask = Variable(mask, volatile=True).cuda()
    else:
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)
        mask = Variable(mask, volatile=True)
    feats = model.get_bilstm_out(x)
    loss = model.neg_log_likelihood(feats, y, mask)
    test_loss += loss.data[0]
    prepath, prepath_pad = model(feats, mask)
    pre_y = torch.cat(prepath)
    true_y = y.masked_select(mask)
    acc_num = (pre_y == true_y).data.sum()
    acc_pro = float(acc_num) / len(pre_y)
    test_acc += acc_pro
print('test loss is:{:.6f},test acc is:{:.6f}'.format(test_loss / (len(testDataLoader)),test_acc / (len(testDataLoader))))

