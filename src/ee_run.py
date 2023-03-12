import sys,os
sys.path.append("/data/honglifeng/ljqtools")
import ljqpy
import torch
import warnings
import argparse
import transformers
from sklearn.utils import shuffle
from utils import get_logger,prompt_train_data,prompt_test_data,get_predict_triple,get_truth_triple,pred_deal_evaluate
from tqdm import trange
from transformers import BertTokenizer,BartForConditionalGeneration,AdamW,get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
parser=argparse.ArgumentParser()
parser.add_argument('--cuda_id',default=3,type=int)
parser.add_argument('--multi_gpu',action='store_true')
parser.add_argument('--device_ids',default='3,4,5')
parser.add_argument('--do_train',action='store_true')
parser.add_argument('--do_eval',action='store_true')
parser.add_argument('--do_predict',action='store_true')
parser.add_argument('--pretrain_mode',action='store_true')
parser.add_argument('--finetune_mode',action='store_true')
parser.add_argument('--use_new_model',action='store_true')
parser.add_argument('--use_model',default='None',type=str)
parser.add_argument('--save_model',default='_cndb_ske20_hac',type=str)
parser.add_argument('--data_path',default='/mnt/data122/hlf/new_rere/dataset',type=str)
parser.add_argument('--dataset',default='ske2019',type=str)
parser.add_argument('--max_source_length',default=512,type=int)
parser.add_argument('--max_target_length',default=128,type=int,help='如果一条样本中某标签对应的实体很多，这个参数要调大点。')
parser.add_argument("--log_level",default=1,type=int)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--do_schedule",action='store_true',help='小数据集不用')
parser.add_argument("--batch_size",default=64,type=int)
parser.add_argument("--save_interval",default=10,type=int)
parser.add_argument("--epochs",default=40,type=int)
parser.add_argument("--describe",default='why for this run',type=str)
args=parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda_id}'
logger = get_logger(filename=f"../logs/{args.save_model}.log", verbosity=args.log_level)
logger.info(f"-------------------------------------------new_log------------------------------------------")
logger.info(f"{args.describe}")
logger.info(f"input args:{args}")
logger.info(f"using model:{args.use_model}")
torch.cuda.device_count()
model_name='fnlp/bart-base-chinese'
#prepare tokenizer
tokenizer_kwargs = {
    "use_fast": True,
    "additional_special_tokens": ['<i>'],
}
tokenizer=BertTokenizer.from_pretrained(model_name,**tokenizer_kwargs)
#如果中文文本中含有英文，basic tokenizer 会将英文识别为单词，BertTokenizerFast会将英文识别为英文单词本身，或者##xxx之类

#eval
def evaluate(model,tokenizer,args,device):
    #prepare data
    eval_data=ljqpy.LoadJsons(os.path.join(args.data_path,args.dataset+'/dev.json'))
    #prompt data
    p_data=prompt_train_data(eval_data)
    #prepare model
    model.eval()
    #start eval
    total_batch_loss=0.0
    eval_mean_loss=0.0
    count=0   #steps
    for i in range(0,len(p_data),args.batch_size):
        data_batch=p_data[i:min(len(p_data),i+args.batch_size)]
        input_data=[item['prompt']+'[SEP]'+item['text'] for item in data_batch]
        label_data=[item['label'] for item in data_batch]
        inputs=tokenizer(input_data,padding='max_length',max_length=args.max_source_length,truncation=True,return_tensors='pt')
        with tokenizer.as_target_tokenizer():
            labels=tokenizer(label_data,padding='max_length',max_length=args.max_target_length,truncation=True,return_tensors='pt')
        with torch.no_grad():
            inputs.to(device)
            labels.to(device)
            outputs=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=labels['input_ids'])
            loss=outputs.loss.mean()
            total_batch_loss+=loss.mean().item()
            count+=1
    eval_mean_loss=total_batch_loss/count
    return eval_mean_loss
            
if args.do_train:
    try:
        #prepare device
        device=torch.device(f'cuda:{args.cuda_id}')
        #prepare train data
        if args.pretrain_mode:
            logger.info(f'*****training with dataset ske cndb hac*****')
            skefn = os.path.join(args.data_path,'ske2019/train.json')
            cndbfn = os.path.join(args.data_path,'cndbpedia_distsup/train.txt')
            hredfn = os.path.join(args.data_path,'HacRED/train.json')
            train_data = ljqpy.LoadJsons(cndbfn) + ljqpy.LoadJsons(skefn) + ljqpy.LoadJsons(hredfn)
            logger.info('load pretrain data successfully.')
        elif args.finetune_mode:
            logger.info(f'*****training with dataset {args.dataset}*****')
            train_data =ljqpy.LoadJsons(os.path.join(args.data_path,f'{args.dataset}/train.json'))
            logger.info('load finetune data successfully.')
        logger.info(f'len(train_data):{len(train_data)}')

        #deal data
        p_data=prompt_train_data(train_data)
        #print(p_data)

        #prepare model
        if args.use_new_model:  #pretrain_mode
            model=BartForConditionalGeneration.from_pretrained(model_name)
            logger.info(f'load moel {model_name} for train.')
        else:
            model=BartForConditionalGeneration.from_pretrained(f'/data/honglifeng/Rere/models/{args.use_model}')
            logger.info(f'load moel {args.use_model} for train.')
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        if args.multi_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
            device_ids=list(map(int,args.device_ids.split(',')))
            model=torch.nn.DataParallel(model,device_ids=device_ids)    
        model.to(device)

        #prepare optimizer  warmup
        optimizer=AdamW(model.parameters(),lr=args.lr)
        if args.do_schedule:
            step_nums=len(p_data)//args.batch_size*args.epochs
            scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=step_nums//10,num_training_steps=step_nums)
        
        #prepare loss record
        losssum_batchsize_100=0.0
        total_loss_list=[]
        best_loss=100
        best_evalloss_list=[]
        
        for epoch in range(args.epochs):
            for i in range(0,len(p_data),args.batch_size):
                model.train()
                optimizer.zero_grad()
                #prepare batch
                data_batch=p_data[i:min(len(p_data),i+args.batch_size)]
                input_data=[item['prompt']+'[SEP]'+item['text'] for item in data_batch]
                label_data=[item['label'] for item in data_batch]
                #encode
                inputs=tokenizer(input_data,max_length=args.max_source_length,padding='max_length',truncation=True,return_tensors='pt')
                with tokenizer.as_target_tokenizer():
                    labels=tokenizer(label_data,max_length=args.max_target_length,padding='max_length',truncation=True,return_tensors='pt')
                inputs.to(device)
                labels.to(device)
                
                #train
                outputs=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=labels['input_ids'])      
                loss=outputs.loss.mean()
                losssum_batchsize_100+=loss.item()
                loss.backward()
                optimizer.step()
                if args.do_schedule:scheduler.step()
                logger.info(f'epoch:{epoch+1}/{args.epochs}, batch:{i//args.batch_size}/{len(p_data)//args.batch_size}, loss:{loss.item()}')
                # save model
                if i and i//args.batch_size==args.save_interval:
                    mean_loss=losssum_batchsize_100/(args.batch_size*args.save_interval)
                    total_loss_list.append(mean_loss)
                    losssum_batchsize_100=0.0
                    if args.finetune_mode:continue  #微调模式没必要在一个周期内保存模型
                    #save model
                    if args.multi_gpu:
                        model.module.save_pretrained(f'../models/{args.save_model}')
                    else:model.save_pretrained(f'../models/{args.save_model}')
            
            #eval or not for save model
            logger.info(f'###########evaluate with dataset {args.dataset}###########')
            if args.do_eval:
                eval_loss=evaluate(model,tokenizer,args,device)
                logger.info(f'eval loss : {eval_loss}')
                if not best_evalloss_list:
                    best_evalloss_list.append(eval_loss)
                    if args.multi_gpu:
                        model.module.save_pretrained(f'../models/{args.save_model}')
                    else:model.save_pretrained(f'../models/{args.save_model}')
                    logger.info(f'save model {args.save_model} after this epoch.  eval_mode=True')
                elif eval_loss<min(best_evalloss_list):
                    best_evalloss_list.append(eval_loss)
                    if args.multi_gpu:
                        model.module.save_pretrained(f'../models/{args.save_model}')
                    else:model.save_pretrained(f'../models/{args.save_model}')    
                    logger.info(f'save model {args.save_model} after this epoch.  eval_mode=True')
            elif total_loss_list[-1]<best_loss:
                best_loss=total_loss_list[-1]
                if args.multi_gpu:
                        model.module.save_pretrained(f'../models/{args.save_model}')
                else:model.save_pretrained(f'../models/{args.save_model}')
                logger.info(f'save model {args.save_model} after this epoch.  eval_mode=False')
                total_loss_list=[]
            logger.info(f'#####################evaluate end#####################')
        logger.info("******************training finished******************")
    except Exception as e:
        logger.info(e)
 
 
def predict(p_data,model,args,tokenizer,obj='em1Text'):
     #start predict
    predict_list=[]
    for i in trange(0,len(p_data),args.batch_size):
        data_batch=p_data[i:min(len(p_data),i+args.batch_size)]
        input_data=[item['prompt']+'[SEP]'+item['text'] for item in data_batch]
        
        inputs=tokenizer(input_data,padding='max_length',max_length=args.max_source_length,truncation=True,return_tensors='pt')
        inputs.to(device)
        
        params = {"decoder_start_token_id":0,"early_stopping":False,"no_repeat_ngram_size":0,"length_penalty": 0,"num_beams":10,"use_cache":True}
        if args.multi_gpu:
            out_ids=model.module.generate(inputs['input_ids'],attention_mask=inputs['attention_mask'],max_length=args.max_target_length*2,**params)  #eg: tensor([[0,101,4992,21490,9050,102]], device='cuda:2')
        else:out_ids=model.generate(inputs['input_ids'],attention_mask=inputs['attention_mask'],max_length=args.max_target_length*2,**params)  #eg: tensor([[0,101,4992,21490,9050,102]], device='cuda:2')
        out_text=tokenizer.batch_decode(out_ids,clean_up_tokenization_spaces=True)
        out_text=[item.replace('[CLS]','').replace('[SEP]','').replace('[PAD]','').replace('[UNK]','').replace(' ','') for item in out_text]
        predict_list.extend(out_text)  
    if obj=='em1Text':  #更新p_data
        p_data=prompt_test_data(p_data,predict_list)
        return p_data        
    elif obj=='em2Text':#获取所有的预测结果
        predict_list=get_predict_triple(p_data,predict_list)
        return predict_list
                
if args.do_predict:
    try:
        logger.info(f'>>>>>>>>>>>>>>>predict with dataset {args.dataset}>>>>>>>>>>>>>>>')
        #prepare device
        device=torch.device(f'cuda:{args.cuda_id}')
        #prepare test data
        test_data=ljqpy.LoadJsons(os.path.join(args.data_path,args.dataset+'/test.json'))
        logger.info(f'len(test_data):{len(test_data)}')
        truth_triple=get_truth_triple(test_data)
        #prepare tqdm
        from tqdm import trange
        #prepare model and resultsfile path   预训练后单独运行预测:用use_model，微调后直接预测:用训练后保存的save_model
        if args.finetune_mode:
            model=BartForConditionalGeneration.from_pretrained(f'/data/honglifeng/Rere/models/{args.save_model}')
            result_file=f'/data/honglifeng/Rere/results/{args.save_model}'
            logger.info(f'load {args.save_model} for predict.')
        else:
            model=BartForConditionalGeneration.from_pretrained(f'/data/honglifeng/Rere/models/{args.use_model}')
            result_file=f'/data/honglifeng/Rere/results/{args.use_model}'
            logger.info(f'load {args.use_model} for predict.')
        if args.multi_gpu:
            model=torch.nn.DataParallel(model)
        model.to(device)
        model.eval()
        
        #deal and predict
        p_data=prompt_test_data(test_data)
        #print(p_data)
        p_data=predict(p_data,model,args,tokenizer,obj='em1Text')
        logger.info('first time predict finished.')
        predict_list=predict(p_data,model,args,tokenizer,obj='em2Text')
        logger.info('second time predict finished.')
        
        #calculate  and save results
        p,r,f1=pred_deal_evaluate(predict_list,truth_triple,args.save_model)
        logger.info(f'predict result | p {p} | r {r} | f1 {f1} |')
    except Exception as e:
        logger.info(e)

logger.info(f"-------------------------------------------log_over------------------------------------------")

      
    
