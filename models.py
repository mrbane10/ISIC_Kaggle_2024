import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2,\
   efficientnet_b3, efficientnet_b4, efficientnet_b5
from torchvision.models.feature_extraction import create_feature_extractor
import lightning as L
import numpy as np
from metrics import *
import torch.optim as optim
from tqdm import tqdm

backbone = {'b0': efficientnet_b0, 'b1': efficientnet_b1, 'b2': efficientnet_b2,
           'b3': efficientnet_b3, 'b4':efficientnet_b4, 'b5': efficientnet_b5,}


weights ={'b0': './checkpoints/efficientnet_b0_rwightman-7f5810bc.pth',
          'b1': './checkpoints/efficientnet_b1_rwightman-bac287d4.pth',
          'b2': './checkpoints/efficientnet_b2_rwightman-c35c1473.pth',
          'b3': './checkpoints/efficientnet_b3_rwightman-b3899882.pth',
          'b4': './checkpoints/efficientnet_b4_rwightman-23ab8bcd.pth',
          'b5': './checkpoints/efficientnet_b5_lukemelas-1a07897c.pth'
         }


feat_dim = {
    'b0': 1280, 'b1': 1280, 'b2': 1408, 'b3': 1536, 'b4': 1792, 'b5': 2048,
}


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class MetaSVDD(nn.Module):
    def __init__(self, config):
        super().__init__()
        #backbone
        backbone_ = config['backbone']
        model_backbone = backbone[backbone_](weights = None)
        pretrained_weights = torch.load(weights[backbone_])
        model_backbone.load_state_dict(pretrained_weights)
        self.model_backbone = create_feature_extractor(
            model_backbone, return_nodes={'avgpool':'avgpool'})
        for param in self.model_backbone.parameters():
            param.requires_grad = False

        #meta
        self.meta_net = nn.Sequential(
                nn.Linear(32, 512),
                Swish_Module(),
                nn.Linear(512, 128),
                Swish_Module()
            )

        self.bottleneck = nn.Sequential(
                nn.Linear(feat_dim[backbone_] , 512),
                Swish_Module(),
                nn.Linear(512, 128),
                Swish_Module(),
            )

        #fuse
        self.fuse = nn.Linear(256, 256)

    def forward(self, x_img, x_meta):
        f_img = self.model_backbone(x_img)['avgpool']
        f_img = f_img.flatten(1)
        f_img = self.bottleneck(f_img)
        f_meta = self.meta_net(x_meta)
        f = torch.cat([f_img, f_meta], dim=-1)
        f = self.fuse(f)
        return f


def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


class Trainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.metasvdd= MetaSVDD(config)
        self.warm_up_n_epochs = 5
        self.R = config['R']
        self.nu = config['nu']
        self.c = torch.tensor(0.).cuda()
        self.epochs = 0

        #metrics
        self.auroc = AUROC()
        self.f1_adapt = F1AdaptiveThreshold()
        self.min_max = MinMax()
        self.thr = self.f1_adapt.value
        self.min, self.max = self.min_max.min, self.min_max.max
        self.save_hyperparameters()

    def score(self, x_img, x_meta):
        f = self.metasvdd(x_img, x_meta)
        dist = torch.sum((f - self.c) ** 2, dim=1)
        scores = dist - self.R ** 2
        scores = self.normalize(scores)
        return scores

    def normalize(self, score):
        return normalize(score, self.min, self.max, self.thr)

    def training_step(self, batch, batch_idx):
        self.metasvdd.train()
        x_img, x_meta = batch
        x_img = x_img.cuda()
        x_meta = x_meta.cuda()
        f = self.metasvdd(x_img, x_meta)
        dist = torch.sum((f - self.c) ** 2, dim=1)
        scores = dist - self.R ** 2
        loss = self.R ** 2 + (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(scores), scores))
        self.log_dict({'loss': loss.item()}, on_epoch=True, prog_bar=True, logger=True)

        #update R
        if self.epochs >= self.warm_up_n_epochs:
            self.R = torch.tensor(get_radius(dist.detach(), self.nu)).cuda()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        self.metasvdd.eval()
        x_img, x_meta, label = batch
        x_img = x_img.cuda()
        x_meta = x_meta.cuda()
        label = label.cuda()

        #score
        f = self.metasvdd(x_img, x_meta)
        dist = torch.sum((f - self.c) ** 2, dim=1)
        scores = dist - self.R ** 2

        #normaize
        self.min_max.update(scores)
        self.f1_adapt.update(scores, label)
        scores = self.normalize(scores)

        #metrics
        self.auroc.update(scores.cpu(), label.cpu())

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        self.log_dict({'val_auroc': auroc})
        print(f'val_auroc:{auroc}')
        self.thr = self.f1_adapt.compute()
        self.min, self.max = self.min_max.compute()
        self.f1_adapt.reset()
        self.min_max.reset()

    def backward(self, loss):
        loss.backward()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.metasvdd.parameters(),
            lr=0.001,
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 0.9, last_epoch=-1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader_(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def on_train_epoch_start(self):
        self.epochs +=1
        print(self.epochs)

    def on_train_start(self):
        n_samples = 0
        c = torch.zeros(256).cuda()
        self.metasvdd.eval()
        with torch.no_grad():
            for x_img, x_meta in tqdm(self.train_dataloader):
                x_img = x_img.cuda()
                x_meta = x_meta.cuda()
                f = self.metasvdd(x_img, x_meta)
                n_samples += f.shape[0]
                c += torch.sum(f, dim=0)
        c /= n_samples
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c

    def on_save_checkpoint(self, checkpoint):
        checkpoint['R'] = self.R
        checkpoint['C'] = self.c
        checkpoint['thr'] = self.thr
        checkpoint['min'] = self.min
        checkpoint['max'] = self.max

    def on_load_checkpoint(self, checkpoint):
        self.R = checkpoint['R']
        self.c = checkpoint['C']
        self.thr = checkpoint['thr']
        self.min = checkpoint['min']
        self.max = checkpoint['max']
        print(f'threshold: {self.thr}-- min:{self.min} -- max:{self.max}')
