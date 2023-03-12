import logging
import sys
sys.path.append("/data/honglifeng/ljqtools")
import ljqpy
import collections
from tqdm import trange
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

#
textkeys = ['sentText', 'text', 'sent', 'sentence']
spokeys = ['relationMentions', 'spo_list']
def convjson_spo(z):
    r = {}
    r['label'] = z.get('label', z.get('p'))
    r['em1Text'] = z.get('em1Text', z.get('s', z.get('subject')))
    r['em2Text'] = z.get('em2Text', z.get('o', z.get('object')))
    return r  
def convjson(z):
    for x in textkeys:
        if x in z: 
            z[textkeys[0]] = z[x]
            if x != textkeys[0]: z.pop(x)
            break
    for x in spokeys:
        if x in z:
            z[spokeys[0]] = list(map(convjson_spo, z[x]))
            if x != spokeys[0]: z.pop(x)
            break  
    return z

def prompt_train_data(data):
    p_data=[]  #prompt data
    for item in data:
        item = convjson(item)#统一标签
        text, spo_list = item['sentText'], item.get('relationMentions', [])   
        
        #label->em1Text
        label_dict=collections.defaultdict(list)
        for spo in spo_list:
            #spo  {label  em1Text  em2Text}
            label_dict[spo['label']].append(spo['em1Text'])
        for key,value in label_dict.items():
            p_data.append({'prompt':key,
                            'text':text,
                            'label':'<i>'.join(value)})
        
        #label:em1Text->em1Text:
        labelem1Text_dict=collections.defaultdict(list)
        for spo in spo_list:
            #spo  {label  em1Text  em2Text}
            labelem1Text_dict[spo['label']+':'+spo['em1Text']].append(spo['em2Text'])
        for key,value in labelem1Text_dict.items():
            p_data.append({'prompt':key,
                            'text':text,
                            'label':'<i>'.join(value)})
    return p_data
def prompt_test_data(data,predict_list=[]):
    p_data=[]
    if not predict_list:  #before first time predict
        for item in data:
            item = convjson(item)#统一标签
            text, spo_list = item['sentText'], item.get('relationMentions', [])   
            
            #label->em1Text
            label_dict=collections.defaultdict(int)
            for spo in spo_list:
                #spo  {label  em1Text  em2Text}
                label_dict[spo['label']]=1
            for label,_ in label_dict.items():
                p_data.append({
                    'prompt':label,
                    'text':text
                    })
    else:     #before second time predict
        for d,preds in zip(data,predict_list):
            pred_list=preds.split('<i>')
            #print(pred_list)
            for pred in pred_list:
                if pred!='':
                    p_data.append({
                    'prompt':d['prompt']+':'+pred,
                    'text':d['text']
                    })
    return p_data    

def get_predict_triple(data,predict_list):
    predict_triple=[]
    print(len(data),len(predict_list))
    for d,preds in zip(data,predict_list):
        split_res=d['prompt'].split(':')
        if len(split_res)>2:
            label,em1Text=split_res[0],''.join(split_res[1:])
        else:
            label,em1Text=split_res
        pred_list=preds.split('<i>')
        for pred in pred_list:
            if pred!='':
                predict_triple.append([label,em1Text,pred])
    return predict_triple
def get_truth_triple(data):
    truth_triple=[]
    for item in data:
        item = convjson(item)#统一标签
        text, spo_list = item['sentText'], item.get('relationMentions', [])
        for spo in spo_list:
            truth_triple.append([spo['label'],spo['em1Text'],spo['em2Text']])
    return truth_triple


def pred_deal_evaluate(results,labels,filename):
    #print(results)
    import collections
    len_predict=len(results)
    len_truth=len(labels)
    predict_true=0
    predict_set=[]
    with open(f'../results/{filename}.txt','a+',encoding='utf-8') as f:
        for item in results:
            if item in labels and item not in predict_set:
                predict_true+=1
                f.write('yes\t'+str(item)+'\n')
                predict_set.append(item)
            else:
                f.write('no\t'+str(item)+'\n') 
        print(f'predTrue {predict_true},len_predict {len_predict},len_truth {len_truth}')             
        p=predict_true/max(1,len_predict)
        r=predict_true/max(1,len_truth)
        f1=2*p*r/max(1,p+r)
        f.write('precision:{:.4f}\trecall:{:.4f}\tF1:{:.4f}\n'.format(p,r,f1))
    return p,r,f1