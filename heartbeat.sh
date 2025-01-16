CUDA_VISIBLE_DEVICES='1' python src/main.py \
--output_dir experiments --comment "classification from Scratch" \
--name lra_heartbeat_fromScratch --records_file Classification_records.xls \
--data_dir ./dataset/Multivariate2018_ts/Multivariate_ts/Heartbeat/ \
--data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 \
--lr 0.001 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy \
--global_reg --l2_reg 1e-4 --d_model 64  --dim_feedforward 256 --num_heads 8



