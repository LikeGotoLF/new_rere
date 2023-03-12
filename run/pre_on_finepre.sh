python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_predict  \
--use_mode  'fine_pretrain_hac' \
--save_model 'fine_pretrain_hac' \
--dataset 'HacRED' \
--batch_size 32 \
--describe 'pred HacRED on finetuned continue pretrained model.'