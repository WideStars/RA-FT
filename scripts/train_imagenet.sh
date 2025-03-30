lr=1e-4
epoch=50
shots=16
seed=1
lambda1=0.1
lambda2=1.0
tau1=2.0
tau2=3.0
backbone="RN50" # RN50 RN101 ViT-B/32 ViT-B/16

gpuid=0

title=raft
log_file=${title}.log

CUDA_VISIBLE_DEVICES=${gpuid} python train_imagenet.py \
    --config ./configs/imagenet.yaml \
    --backbone ${backbone} \
    --shots ${shots} \
    --seed ${seed} \
    --lambda1 ${lambda1} \
    --lambda2 ${lambda2} \
    --tau1 ${tau1} \
    --tau2 ${tau2} \
    --train_epoch ${epoch} --lr ${lr} \
    --title ${title} --log_file ${log_file} \
    --desc "GPU${gpuid}, lambda1 = ${lambda1}, lambda2 = ${lambda2}, tau1 = ${tau1} tau2 = ${tau2}" 
