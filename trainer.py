import sys
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F

from utils import my_scheduler, AvgACC, save_model
from eval import Eval

    
def clip_forward(clip_model, my_pooling, images, text_feats):
    image_featmaps, image_featvecs = clip_model.encode_image(images.cuda())
    cls_featvecs, bg_featvecs = my_pooling(image_featmaps)
    image_featvecs = image_featvecs / image_featvecs.norm(dim=-1, keepdim=True)
    cls_featvecs = cls_featvecs / cls_featvecs.norm(dim=-1, keepdim=True)
    bg_featvecs = bg_featvecs / bg_featvecs.norm(dim=-1, keepdim=True)
        
    clip_logits = 100. * image_featvecs @ text_feats[:, :-1]
    cls_logits = 100. * cls_featvecs @ text_feats[:, :-1]
    bg_logits = 100. * bg_featvecs @ text_feats
    return clip_logits, cls_logits, bg_logits, cls_featvecs, bg_featvecs


class Trainer:
    
    def __init__(self, cfg, clip_model, my_pooling, train_loader, test_loader, logger, text_feats, val_loader=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.my_pooling = my_pooling
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_dir = cfg['checkpoint_dir']
        self.save_interval = cfg['save_interval']
        self.epochs = cfg['train_epoch']
        self.log_dir = f"./log/{self.checkpoint_dir.split('/')[-1]}"
        
        self.optimizer = torch.optim.AdamW(self.my_pooling.parameters(), lr=cfg['lr'], eps=1e-4)
        self.scheduler = my_scheduler(self.optimizer, cfg['lr'], 1e-6, self.epochs, len(self.train_loader), 10)
        
        self.text_feats = text_feats
        self.eval = Eval(self.cfg, self.clip_model, self.my_pooling, test_loader, self.text_feats, self.logger)
                    
    def get_loss(self, labels, clip_logits, cls_logits, bg_logits, cls_featvecs, bg_featvecs):
        Con_loss = F.l1_loss(cls_logits, clip_logits)
        Cls_ce_loss = F.cross_entropy(cls_logits / self.cfg['tau1'], labels)
        Bg_ce_loss = F.cross_entropy(bg_logits / self.cfg['tau2'], torch.tensor([self.cfg['num_classes'] for _ in range(labels.shape[0])]).to(dtype=labels.dtype).cuda())
        Diff_loss = -F.cosine_embedding_loss(cls_featvecs, bg_featvecs, torch.tensor([1 for _ in range(cls_featvecs.shape[0])]).cuda())
        lambda1 = self.cfg['lambda1']
        lambda2 = self.cfg['lambda2']
        loss = lambda1 * Diff_loss +  lambda2 * Con_loss + Cls_ce_loss + Bg_ce_loss
        return loss, [Diff_loss, Con_loss, Cls_ce_loss, Bg_ce_loss]
    
    def train_mode(self):
        self.my_pooling.train()
        
    def train_epoch(self, epoch):
        self.train_mode()
        train_loss = 0.0
        ACC = AvgACC()
        loss_list = [0, 0, 0, 0]
        
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training epoch {epoch}") as tqdm_train:
            for _, (images, labels) in tqdm_train:
                images, labels = images.cuda(), labels.cuda()
                clip_logits, cls_logits, bg_logits, cls_featvecs, bg_featvecs = clip_forward(self.clip_model, self.my_pooling, images, self.text_feats)
                
                loss, losses = self.get_loss(labels, clip_logits, cls_logits, bg_logits, cls_featvecs, bg_featvecs)

                if torch.isnan(loss):
                    self.logger.info(f"{self.cfg['desc']}:!!! Loss is NaN. Program terminated.")
                    sys.exit()

                ACC.step(cls_logits, labels)
                train_loss += loss.item()
                for i, l in enumerate(losses):
                    loss_list[i] += l.item()
                tqdm_train.set_postfix(cur_loss=loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
            train_acc = ACC.cal()
            train_loss = train_loss / len(self.train_loader)
        
        train_acc *= 100.0
        print(f"{self.cfg['dataset']}_s{self.cfg['shots']}: {self.cfg['desc']}: {epoch=}, {train_acc=:.4f}, {loss_list=}")
        if epoch == self.epochs - 1:
            self.logger.info(f"[Diff_loss, Con_loss, Cls_ce_loss, Bg_ce_loss] => {loss_list}")
            
        return train_acc, train_loss
        
    def train(self):
        self.logger.info('-------------------- START TRAINING --------------------')
        train_name = self.logger.name
        train_st = time.time()
        # self.validate()
        
        for epoch in range(self.epochs):
            epoch_st = time.time()
            self.logger.info(f'====> Epoch: {epoch}')
            train_acc, train_loss = self.train_epoch(epoch)
            epoch_ed = time.time()
            self.logger.info(f"      train_acc: {train_acc:.4f} %    train_loss: {train_loss:.4f}    train_time: {(epoch_ed - epoch_st):.4f} s    lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
            
            # if (epoch % self.cfg['save_interval'] == 0 and epoch != 0):
                # save_model(self.checkpoint_dir, f'{train_name}_epoch_{epoch}.pth', self.my_pooling)
                # self.validate()
            if epoch == self.epochs - 1:
                save_model(self.checkpoint_dir, f'{train_name}_last.pth', self.my_pooling)
                self.validate()
        
        duration = int(time.time() - train_st)
        self.logger.info(f'Total time used for training: {duration // 3600} h {duration % 3600 // 60} min {duration % 60} sec')
        
    def validate(self):
        self.eval.my_pooling = self.my_pooling
        
        val_best_alpha = None
        if self.val_loader:
            self.eval.val_loader = self.val_loader
            val_best_alpha = self.eval.eval()
            
        self.eval.val_loader = self.test_loader
        self.eval.eval(use_alpha=val_best_alpha)
