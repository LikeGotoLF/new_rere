

python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_train \
--do_predict  \
--finetune_mode \
--use_new_mode \
--save_model 'bart_finetune_lic' \
--dataset 'lic2020' \
--save_interval 100 \
--epochs 40 \
--do_schedule \
--describe 'finetune lic2020 on original bart, then predict.'