#!/bin/bash

# params of --dataset: ag_news, yahoo, imdb
count=0
lbd=0.5
for thr in 0.99; do
for mu in 10; do
    count=`expr $count+1`
    # if [[ $count -le 4 ]];then
    #     continue;
    # fi
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'yahoo' --num_labeled 20 --task semi --patience 10 --num_epoch 100 --lr_bert 5e-5 --mu $mu --thr $thr --lbd $lbd --aug_metric 'sim'
done
done

# count=0
# lbd=0.1
# for thr in 0.99; do
# for mu in 5; do
#     count=`expr $count+1`
#     # if [[ $count -le 4 ]];then
#     #     continue;
#     # fi
#     CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'yahoo' --num_labeled 20 --task semi --patience 10 --num_epoch 100 --lr_bert 5e-5 --mu $mu --thr $thr --lbd $lbd --aug_metric 'base'
# done
# done