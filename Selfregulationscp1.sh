CUDA_VISIBLE_DEVICES='0' python src/main.py \
--output_dir experiments --comment "classification from Scratch" \
--name lra_SelfRegulationSCP1_fromScratch --records_file Classification_records.xls \
--data_dir ./dataset/Multivariate2018_ts/Multivariate_ts/SelfRegulationSCP1 \
--data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 \
--lr 0.0005 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy \
--global_reg --l2_reg 0 --d_model 128  --dim_feedforward 256 --num_heads 8 --num_layers 3
