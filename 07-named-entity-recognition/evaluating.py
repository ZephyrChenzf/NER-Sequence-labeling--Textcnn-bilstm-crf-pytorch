import numpy as np
import data_preprocess


word2index,index2word,tag2index,index2tag=data_preprocess.get_dic()

def evaluate(sourcePath,resultPath):
    f_s=open(sourcePath,'r')
    f_r=open(resultPath,'r')
    source_data=[]
    result_data=[]
    table_eval=np.zeros((len(tag2index),len(tag2index)))  #横轴表示真实值，纵轴表示预测值
    for line in f_s:
        source_data.append(line)
    for line in f_r:
        result_data.append(line)
    length=len(source_data)
    for i in range(length):
        if source_data[i]=='\n':
            continue
        tag_t=source_data[i].split()[1]
        tag_p=result_data[i].split()[1]
        tag_t_inx=tag2index[tag_t]
        tag_p_inx=tag2index[tag_p]
        table_eval[tag_p_inx][tag_t_inx]+=1
    #print(table_eval)
    # 评测
    all_p_numerator=0
    all_p_denominator=0
    all_r_denominator = 0
    #具体评测内容自行添加
    # for i in range(2,len(tag2index)-1,2):
    #     print('############'+index2tag[i]+'##############')
    #     precision=(table_eval[i,i]+table_eval[i+1,i+1])/(table_eval[i,:].sum()+table_eval[i+1,:].sum())#precision
    #     recall=(table_eval[i,i]+table_eval[i+1,i+1])/(table_eval[:,i].sum()+table_eval[:,i+1].sum())#recall
    #     f1=2*precision*recall/(precision+recall)#f1
    #     print("num: "+str(table_eval[i,i])+' '+str(table_eval[i+1,i+1]))
    #     print(precision)
    #     print(recall)
    #     print(f1)
    #     print("#########################")
    #     all_p_numerator+=table_eval[i,i]+table_eval[i+1,i+1]
    #     all_p_denominator+=table_eval[i,:].sum()+table_eval[i+1,:].sum()
    #     all_r_denominator+=table_eval[:,i].sum()+table_eval[:,i+1].sum()
    # print("##########all################")
    all_p=all_p_numerator/all_p_denominator
    all_r=all_p_numerator/all_r_denominator
    all_f1=2*all_p*all_r/(all_p+all_r)
    print(all_p)
    print(all_r)
    print(all_f1)

evaluate('./data/test_data','./data/result_data')
print(tag2index)
