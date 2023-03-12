python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_predict  \
--use_mode  'bart_finetune_lic' \
--save_model 'pred_onlypred_finetune_lic' \
--dataset 'lic2020' \
--batch_size 32 \
--describe 'pred lic2020 on original bart model.'
