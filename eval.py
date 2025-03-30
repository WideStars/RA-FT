import torch
from tqdm import tqdm

from utils import AvgACC, cal_acc


def fuse_logits(cls_logits, clip_logits, alpha=1.0):
    return alpha * cls_logits + (1 - alpha) * clip_logits


class Eval:
    
    def __init__(self, cfg, clip_model, my_pooling, val_loader, text_feats, logger) -> None:
        self.cfg = cfg
        self.clip_model = clip_model
        self.my_pooling = my_pooling
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.logger = logger
        self.batch_size = cfg['batch_size']
        
    def evaluate_epoch(self, images):
        image_featmaps, image_featvecs = self.clip_model.encode_image(images.cuda())
        cls_featvecs, _ = self.my_pooling(image_featmaps)
        image_featvecs = image_featvecs / image_featvecs.norm(dim=-1, keepdim=True)
        cls_featvecs = cls_featvecs / cls_featvecs.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_featvecs @ self.text_feats[:, :-1]
        cls_logits = 100. * cls_featvecs @ self.text_feats[:, :-1]
        return clip_logits, cls_logits

    def eval(self, use_alpha=None):
        ACC = AvgACC()
        self.clip_model.eval()
        self.my_pooling.eval()
        all_clip_logits = []
        all_cls_logits = []
        all_labels = []
        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    clip_logits, cls_logits = self.evaluate_epoch(images.cuda())
                    ACC.step(cls_logits, labels)
                    all_clip_logits.append(clip_logits)
                    all_cls_logits.append(cls_logits)
                    all_labels.append(labels)
        
        self.all_clip_logits = torch.cat(all_clip_logits, dim=0)
        self.all_cls_logits = torch.cat(all_cls_logits, dim=0)
        self.all_labels = torch.cat(all_labels, dim=0)
        
        best_alpha, zs_acc, last_acc, best_acc = self.search_hp()
        if use_alpha != None:
            logits = fuse_logits(self.all_cls_logits, self.all_clip_logits, use_alpha)
            acc = cal_acc(logits, self.all_labels) * 100.
            self.logger.info(f"{self.cfg['desc']} :*** valdation best alpha = {use_alpha:.4f} => {acc:.2f}% [{zs_acc=:.2f}, {last_acc=:.2f}, {best_acc=:.2f}]")
            print(f"{self.cfg['desc']} :*** valdation best alpha = {use_alpha:.4f} => {acc:.2f}% [{zs_acc=:.2f}, {last_acc=:.2f}, {best_acc=:.2f}]")
        return best_alpha
    
    # search on validation_set to get best β
    def search_hp(self):
        start = self.cfg['search_low']
        end = self.cfg['search_high']
        step = self.cfg['search_step']
        alpha_list = [i * (end - start) / step + start for i in range(step + 1)]
        best_alpha, best_acc = start, 0.
        accs = []
        best_logits = None
        for alpha in alpha_list:
            self.alpha = alpha
            logits = fuse_logits(self.all_cls_logits, self.all_clip_logits, alpha)
            acc = cal_acc(logits, self.all_labels) * 100.
            accs.append((alpha, acc))
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
                best_logits = logits
        
        self.logger.info(f"{self.cfg['desc']}:*** last acc => {accs[-1][-1]:.2f}%, best acc => {best_acc:.2f}% (alpha = {best_alpha:.4f}), accs => {accs}")
        print(f"{self.cfg['desc']}:*** last acc => {accs[-1][-1]:.2f}%, best acc => {best_acc:.2f}% (alpha = {best_alpha:.4f}), accs => {accs}")
        return best_alpha, accs[0][-1], accs[-1][-1], best_acc