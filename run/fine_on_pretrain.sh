python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_train \
--do_predict  \
--finetune_mode \
--use_mode  'bart_pretrain' \
--save_model 'fine_pretrain_small' \
--dataset 'supersmall' \
--do_schedule \
--batch_size 32 \
--epochs 10 \
--describe 'use supersmall finetune continue pretrain model, then predict.'

python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_train \
--do_predict  \
--finetune_mode \
--use_mode  'bart_pretrain' \
--save_model 'fine_pretrain_lic' \
--dataset 'lic2020' \
--do_schedule \
--batch_size 32 \
--epochs 10 \
--describe 'use lic2020 finetune continue pretrain model, then predict.'

python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_train \
--do_predict  \
--finetune_mode \
--use_mode  'bart_pretrain' \
--save_model 'fine_pretrain_hac' \
--dataset 'HacRED' \
--do_schedule \
--batch_size 32 \
--epochs 10 \
--describe 'use HacRED finetune continue pretrain model, then predict.'

python ../src/ee_run.py \
--cuda_id 3 \
--multi_gpu \
--do_train \
--do_predict  \
--finetune_mode \
--use_mode  'bart_pretrain' \
--save_model 'fine_pretrain_ske' \
--dataset 'ske2019' \
--do_schedule \
--batch_size 32 \
--epochs 10 \
--describe 'use ske2019 finetune continue pretrain model, then predict.'