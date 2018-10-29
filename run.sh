#!/bin/bash

start=`date +%s`
now=$(date +"%Y.%m.%d_%H.%M.%S")
python script/experiment/train.py \
-d '(0,1)' \
--only_test false \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part trainval \
--exp_dir ~/logs/market1501Stride1Xqda$now \
--ids_per_batch 4 \
--steps_per_log 10 \
--epochs_per_val 5 \
--total_epochs 300 \
--hyper_parameter 0.01 \
--beta 1 \
--use_exp 1 \
>log.txt
end=`date +%s`
runtime=$(((end-start)/60))
echo $runtime minutes>>log.txt
uuencode log.txt log.txt | mail -s "pfa 0.01hp 1beta 300epochs 4id_per_batch exp used" anurags.it@nsit.net.in
start=`date +%s`
now=$(date +"%Y.%m.%d_%H.%M.%S")
python script/experiment/train.py \
-d '(0,1)' \
--only_test false \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part trainval \
--exp_dir ~/logs/market1501Stride1Xqda$now \
--ids_per_batch 8 \
--steps_per_log 10 \
--epochs_per_val 5 \
--total_epochs 300 \
--hyper_parameter 0.01 \
--beta 1 \
--usemean 1 \
>log.txt
end=`date +%s`
runtime=$(((end-start)/60))
echo $runtime minutes>>log.txt
uuencode log.txt log.txt | mail -s "pfa 0.01hp 1beta 300epochs 8id_per_batch mean used" anurags.it@nsit.net.in
start=`date +%s`
now=$(date +"%Y.%m.%d_%H.%M.%S")
python script/experiment/train.py \
-d '(0,1)' \
--only_test false \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part trainval \
--exp_dir ~/logs/market1501StrideXqda$now \
--ids_per_batch 4 \
--steps_per_log 10 \
--epochs_per_val 5 \
--total_epochs 300 \
--hyper_parameter 0.01 \
--beta 1 \
>log.txt
end=`date +%s`
runtime=$(((end-start)/60))
echo $runtime minutes>>log.txt 
uuencode log.txt log.txt | mail -s "pfa 0.01hp 1beta 300epochs 4id_per_batch" anurags.it@nsit.net.in
start=`date +%s`
now=$(date +"%Y.%m.%d_%H.%M.%S")
python script/experiment/train.py \
-d '(0,1)' \
--only_test false \
--last_conv_stride 1 \
--normalize_feature false \
--trainset_part trainval \
--exp_dir ~/logs/market1501StrideXqda$now \
--ids_per_batch 2 \
--steps_per_log 10 \
--epochs_per_val 5 \
--total_epochs 300 \
--hyper_parameter 0.01 \
--beta 1 \
>log.txt
end=`date +%s`
runtime=$(((end-start)/60))
echo $runtime minutes>>log.txt 
uuencode log.txt log.txt | mail -s "pfa 0.01hp 1beta 300 epochs 2ids_per_batch" anurags.it@nsit.net.in
