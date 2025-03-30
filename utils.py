import os
import torch
import clip
import json
import random
import numpy as np
from torch import nn

from pathlib import Path


def setup_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)


def get_clip_feat_dim(clip_model, img=torch.ones((1, 3, 224, 224))):
    clip_model.eval()
    with torch.no_grad():
        output = clip_model.encode_image(img.cuda())
        print(f"{output.shape=}")
    return output.shape[1]


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as fp:
        return json.load(fp)


def log_init(logger, cfg):
    logger.info('**************************************************************')
    logger.info(f'Here are the args:')
    for arg in cfg.keys():
        logger.info(f'{arg} : {cfg[arg]}')


def make_dirs(*kargs):
    for dir in kargs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def make_dirs_from_file(*kargs):
    dirs = []
    for path in kargs:
        dirs.append(os.path.split(path)[0])
    make_dirs(*dirs)


def get_model_param_size(model):
    size = sum(param.numel() for param in model.parameters())
    return size


def save_model(save_dir, name, model, optimizer=None, epoch=0, lr_scheduler=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
    }
    torch.save(checkpoint, os.path.join(save_dir, name))


def load_model(checkpoint_path, model, 
               optimizer=None, lr_scheduler=None, key_filter=lambda key: True):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    net_state_dict = ckp['net']
    model_state_dict = model.state_dict()
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in model_state_dict and key_filter(k)}
    model_state_dict.update(net_state_dict)
    model.load_state_dict(model_state_dict)
    if optimizer and ckp['optimizer']:
        optimizer.load_state_dict(ckp['optimizer'])
    if lr_scheduler and ckp['lr_scheduler']:
        lr_scheduler.load_state_dict(ckp['lr_scheduler'])
    return model.cuda(), optimizer, lr_scheduler, ckp['epoch']


def cal_acc_mid(logits, labels):
    pred = torch.argmax(logits, -1)
    acc_num = (pred == labels.cuda()).sum().item()
    total = len(labels)
    return acc_num, total

def cal_acc(logits, labels):
    acc_num, total = cal_acc_mid(logits, labels)
    acc = 1.0 * acc_num / total
    return acc

class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
    
    def step(self, logits, labels):
        acc_num, total = cal_acc_mid(logits, labels)
        self.acc_num += acc_num
        self.total += total
    
    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total


@torch.no_grad()
def show_pred_error_info(logits: torch.Tensor, labels: torch.Tensor, logger):
    labels = labels.cuda()
    org_logits = logits
    logits = nn.Softmax(dim=-1)(logits)
    pred = torch.argmax(logits, -1)
    true_or_false = pred == labels
    print(f"ACC = {true_or_false.sum().item() * 100.0 / len(labels)} %")
    incorrect_indices = torch.nonzero(~true_or_false).squeeze(dim=-1)
    incorrect_labels = labels[incorrect_indices]
    
    topk = 3
    topk_probs, topk_predictions = torch.topk(logits, k=topk, dim=-1)
    topk_logits, _ = torch.topk(org_logits, k=topk, dim=-1)
    
    correct = torch.sum(torch.any(topk_predictions == labels.view(-1, 1), dim=1).float()).item()
    topk_accuracy = correct / len(labels) * 100.0
    print(f"{topk_accuracy = } %")
    logger.info(f"{topk_accuracy = } %")
    
    unique_values, counts = torch.unique(incorrect_labels, return_counts=True)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    sorted_values = unique_values[sorted_indices]
    
    logger.info("Incorrect labels sorted by frequency:")
    # for value, count in zip(sorted_values, sorted_counts):
    #     logger.info(f"{value}: {count}")
    logger.info(list(zip(sorted_values.detach().cpu().numpy(), sorted_counts.detach().cpu().numpy())))
    
    label_to_id = {}
    for idx, label_id in zip(incorrect_indices, incorrect_labels):
        idx, label_id = idx.item(), label_id.item()
        if label_id not in label_to_id:
            label_to_id[label_id] = []
        label_to_id[label_id].append(idx)
    
    def to_np(x):
        return x.detach().cpu().numpy()
    
    with open('log/s16_imagenet_RN50_test/top3_in.log', 'w') as f:
        logger.info("Incorrect predictions:")
        for label, ids in label_to_id.items():
            for i in ids:
                predicted_label = topk_predictions[i]
                prob = topk_probs[i]
                logger.info(f"ID: {i}, True: {label}, Pred: {to_np(predicted_label)}, Topk Prob: {to_np(prob)}, Topk Logits: {to_np(topk_logits[i])}, Include?: {label in predicted_label}")
                f.write(f"{label},{predicted_label.detach().cpu().numpy()},{prob.detach().cpu().numpy()},{label in predicted_label}\n")


class my_scheduler:
    
    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0) -> None:
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 0
        
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, self.schedule))
        self.id = 0
        assert len(self.schedule) == epochs * niter_per_ep
    
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.id]
        self.id += 1


# the following function is modified from Tip-Adapter
    
def ensemble(texts, clip_model):
    texts = clip.tokenize(texts).cuda()
    class_embeddings = clip_model.encode_text(texts)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    return class_embedding

def clip_classifier(feat_path, classnames, template, clip_model):
    if os.path.exists(feat_path):
        print(f"Loading texture features from {feat_path}")
        text_feats = torch.load(feat_path, map_location='cpu')
        return text_feats.cuda()
    
    with torch.no_grad():
        clip_weights = []
        bg_embedding = None
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            if isinstance(template, list):
                texts = [t.format(classname) for t in template]
            elif isinstance(template, dict):
                texts = template[classname]
            
            class_embedding = ensemble(texts, clip_model)
            clip_weights.append(class_embedding)
            
            bg_texts = [f"the background of {classname}."]
            bg_class = ensemble(bg_texts, clip_model)
            bg_embedding = (bg_embedding + bg_class) if bg_embedding is not None else bg_class

        bg_embedding /= len(classnames)
        clip_weights.append(bg_embedding)
        
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
        make_dirs_from_file(feat_path)
        torch.save(clip_weights, feat_path)
            
    return clip_weights
