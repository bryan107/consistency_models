#####################################################
# Consistency distillation on 1D and MNIST datasets #
#####################################################

## L_CD^N (l2) on 1D
python cm_train.py \
--training_mode consistency_distillation \
--target_ema_mode fixed \
--start_ema 0.95 \
--scale_mode fixed \
--start_scales 40 \
--total_training_steps 600000 \
--loss_norm l2 \
--lr_anneal_steps 0 \
--teacher_model_path ../models/mnist/score_net_epoch20.pt \
--attention_resolutions 32,16,8 \
--class_cond True \
--use_scale_shift_norm True \
--dropout 0.0 \
--teacher_dropout 0.1 \
--ema_rate 0.999,0.9999,0.9999432189950708 \
--global_batch_size 2048 \
--image_size 64 \
--lr 0.000008 \
--num_channels 192 \
--num_head_channels 64 \
--num_res_blocks 3 \
--resblock_updown True \
--schedule_sampler uniform \
--use_fp16 True \
--weight_decay 0.0 \
--weight_schedule uniform \
--data_dir ../data/mnist
