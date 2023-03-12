

python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_predict  \
--use_mode  'bart_pretrain' \
--save_model 'pretrain_pred_small' \
--dataset 'supersmall' \
--batch_size 32 \
--describe 'pred supersmall on continue pretrained model.'

