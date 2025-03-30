shots=16
dataset="imagenet"
backbone="RN50" # RN50 RN101 ViT-B/32 ViT-B/16

gpuid=0

title=raft
log_file=${title}_test.log
ckp_name=${title}_last.pth
checkpoint_dir=./checkpoint/s${shots}_${dataset}_${backbone}/
checkpoint=${checkpoint_dir}${ckp_name}

CUDA_VISIBLE_DEVICES=${gpuid} python test_imagenet.py \
    --config ./configs/${dataset}.yaml \
    --checkpoint ${checkpoint} \
    --batch_size 64 \
    --shots ${shots} \
    --title ${title} \
    --log_file ${log_file} \
    --desc "test ${dataset}, ${shots} shot with backbone ${backbone}." \
