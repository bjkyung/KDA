import numpy as np
import os
from termcolor import cprint
import math
from sklearn.cluster import KMeans
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from train import ssltrainer
from model import FeatMatch
from loss import common
from util import misc, metric
from util.command_interface import command_interface
from util.reporter import Reporter
from torch import nn
from torch.nn import functional as F


class FeatMatchTrainer(ssltrainer.SSLTrainer):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.fu, self.pu = [], []
        self.fp, self.yp, self.lp = None, None, None
        self.criterion = getattr(common, self.config['loss']['criterion'])

        self.attr_objs.extend(['fu', 'pu', 'fp', 'yp', 'lp'])
        self.load(args.mode)

    def init_model(self):
        model = FeatMatch(backbone=self.config['model']['backbone'],
                          num_classes=self.config['model']['classes'],
                          devices=self.args.devices,
                          num_heads=self.config['model']['num_heads'],
                          amp=self.args.amp)
        print(f'Use [{self.config["model"]["backbone"]}] model with [{misc.count_n_parameters(model):,}] parameters')
        return model

    def get_labeled_featrues(self):
        mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            labeled_dset = self.dataloader_train.labeled_dset.dataset
            xl = torch.stack([self.Tval(xi) for xi in labeled_dset.get_x()])
            bs = (self.config['train']['bsl'] + self.config['train']['bsu'])*self.config['transform']['data_augment']['K']
            fl = []
            for i in range(math.ceil(len(xl) / bs)):
                xli = self.Tnorm(xl[i*bs:min((i+1)*bs, len(xl))].to(self.default_device))
                fli = self.model.extract_feature(xli)
                fl.append(fli.detach().clone().float().cpu())
            fl = torch.cat(fl)
            yl = torch.tensor(labeled_dset.y).cpu()
        self.model.train(mode)

        return fl, yl

    def get_unlabeled_features(self, thres=0.5, max_iter=5):
        if len(self.fu) == 0:
            return None, None

        fu = torch.cat(self.fu).detach().clone()
        pu = torch.cat(self.pu).detach().clone()
        prob, yu = torch.max(pu, dim=1)

        flag = False
        for _ in range(max_iter):
            idx_thres = (prob > thres)
            yu_ = yu[idx_thres]
            class_distribution = torch.stack([torch.sum(yu_ == i) for i in range(self.config['model']['classes'])])

            if not (class_distribution > self.config['model']['pk']).all():
                thres = thres / 2.
            else:
                flag = True
                break

        del self.fu[:]
        del self.pu[:]

        if flag:
            return fu[idx_thres], yu[idx_thres]
        else:
            return None, None

    # def extract_fp(self):
    #     fl, yl = self.get_labeled_featrues()
    #     fu, yu = self.get_unlabeled_features()
    #     pk = self.config['model']['pk']
    #     rl = self.config['model']['l_ratio']

    #     fp, yp, lp = [], [], []
    #     for yi in torch.unique(yl, sorted=True):
    #         if fu is None:  # all prototypes extracted from labeled data
    #             fpi = self.extract_fp_per_class(fl[yl == yi], pk, record_mean=True)
    #             pkl = len(fpi)
    #             fp.append(fpi)
    #             yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
    #             lp.append(torch.ones_like(yp[-1]))
    #         else:
    #             if pk == 1:
    #                 fpi = self.extract_fp_per_class(torch.cat([fl[yl == yi], fu[yu == yi]]), 1, record_mean=True)
    #                 pkl = len(fpi)
    #                 fp.append(fpi)
    #                 yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
    #                 lp.append(torch.ones_like(yp[-1]))
    #             else:
    #                 # prototypes extracted from labeled data
    #                 fpi = self.extract_fp_per_class(fl[yl == yi], max(1, int(round(pk*rl))), record_mean=True)
    #                 pkl = len(fpi)
    #                 fp.append(fpi)
    #                 yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
    #                 lp.append(torch.ones_like(yp[-1]))
    #                 # prototypes extracted from unlabeled data
    #                 fpi = self.extract_fp_per_class(fu[yu == yi], max(1, pk - pkl), record_mean=True)
    #                 pku = len(fpi)
    #                 fp.append(fpi)
    #                 yp.append(torch.full((pku,), yi, device=self.default_device, dtype=torch.long))
    #                 lp.append(torch.zeros_like(yp[-1]))
    #     fp = [tensor.to(self.default_device) for tensor in fp]
    #     self.fp = torch.cat(fp).to(self.default_device)
    #     self.yp = torch.cat(yp).to(self.default_device)
    #     self.lp = torch.cat(lp).to(self.default_device)

    def extract_fp(self, topk_only=True):    #topK만 쓴다면 True로, 아니면 False
        fl, yl = self.get_labeled_featrues()
        fu, yu = self.get_unlabeled_features()
        pk = self.config['model']['pk']
        rl = self.config['model']['l_ratio']

        all_classes = torch.unique(yl, sorted=True)
        if fu is not None and topk_only:
            k = self.config['model']['classes'] // 2  # 클래스 수의 절반
            topk_classes = torch.topk(torch.bincount(yu), k=k, largest=True).indices
            other_classes = [cls for cls in all_classes if cls not in topk_classes]
        else:
            topk_classes = all_classes
            other_classes = []

        fp, yp, lp = [], [], []
        for yi in topk_classes:
            if fu is None:
                fpi = self.extract_fp_per_class(fl[yl == yi], pk, record_mean=True)
            else:
                if pk == 1:
                    fpi = self.extract_fp_per_class(torch.cat([fl[yl == yi], fu[yu == yi]]), 1, record_mean=True)
                else:
                    fpi = self.extract_fp_per_class(fl[yl == yi], max(1, int(round(pk*rl))), record_mean=True)
                    fp.append(fpi)
                    fpi = self.extract_fp_per_class(fu[yu == yi], max(1, pk - len(fpi)), record_mean=True)

            pkl = len(fpi)
            fp.append(fpi)
            yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
            lp.append(torch.ones_like(yp[-1]) if yi in topk_classes else torch.zeros_like(yp[-1]))

        for yi in other_classes:
            tiny_offset = 1e-6    # 랜덤하게 초기화된 텐서 생성
            random_init = torch.rand((1, fl.shape[1]), device=self.default_device) + tiny_offset #0으로 주면 forward시 문제생기므로 작은값 부여
            fp.append(random_init) # 그래서 topK가 아닌 것에도 torch.full로 작은값(tiny_initialize) 주기
            yp.append(torch.full((1,), yi, device=self.default_device, dtype=torch.long))
            lp.append(torch.zeros((1,), device=self.default_device, dtype=torch.float))
        
        fp = [tensor.to(self.default_device) for tensor in fp]
        self.fp = torch.cat(fp).to(self.default_device)
        self.yp = torch.cat(yp).to(self.default_device)
        self.lp = torch.cat(lp).to(self.default_device)   

    def extract_fp_per_class(self, fx, n, record_mean=True):
        if n == 1:
            fp = torch.mean(fx, dim=0, keepdim=True)
        elif record_mean:
            n = n-1
            fm = torch.mean(fx, dim=0, keepdim=True)
            if n >= len(fx):
                fp = fx                                                
            else:
                fp = self.kmeans(fx, n, 'cosine')
            fp = torch.cat([fm, fp], dim=0)
        else:
            if n >= len(fx):
                fp = fx
            else:
                fp = self.kmeans(fx, n, 'cosine')

        return fp
    

    @staticmethod
    def kmeans(fx, n, metric='cosine'):
        device = fx.device

        if metric == 'cosine':
            fn = fx / torch.clamp(torch.norm(fx, dim=1, keepdim=True), min=1e-20)
        elif metric == 'euclidean':
            fn = fx
        else:
            raise KeyError
        fn = fn.detach().cpu().numpy()
        fx = fx.detach().cpu().numpy()

        labels = KMeans(n_clusters=n).fit_predict(fn)
        fp = np.stack([np.mean(fx[labels == li], axis=0) for li in np.unique(labels)])
        fp = torch.FloatTensor(fp).to(device)

        return fp

    def data_mixup(self, xl, prob_xl, xu, prob_xu, alpha=0.75):
        Nl = len(xl)

        x = torch.cat([xl, xu], dim=0)
        prob = torch.cat([prob_xl, prob_xu], dim=0).detach()

        idx = torch.randperm(x.shape[0])
        x_, prob_ = x[idx], prob[idx]

        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)

        x = l * x + (1 - l) * x_
        prob = l * prob + (1 - l) * prob_
        prob = prob / prob.sum(dim=1, keepdim=True)

        xl, xu = x[:Nl], x[Nl:]
        probl, probu = prob[:Nl], prob[Nl:]

        return xl, probl, xu, probu

    def train1(self, xl, yl, xu):
        # Forward pass on original data
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_x = self.model(x)
        logits_x = logits_x.reshape(bsl + bsu, k, c)
        logits_xl, logits_xu = logits_x[:bsl], logits_x[bsl:]

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xu[:, 0].detach(), dim=1)
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1)

        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])
        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_x_mix = self.model(x_mix)
        logits_xl_mix, logits_xu_mix = logits_x_mix[:Nl], logits_x_mix[Nl:]

        # CLF loss
        loss_pred = self.criterion(None, probl_mix, logits_xl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xu_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_xl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    # Knowledge distillation을 한다면 perform_distillation=True, 그냥 featmatch만 한다면 False

    def train2(self, xl, yl, xu, perform_distillation=True):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, _, fx, _, _ = self.model(x, self.fp)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        fxu = fx.reshape(bsl + bsu, k, fx.size(1))[bsl:]
        self.fu.append(fxu[:, 0].detach().clone().float().cpu())

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        self.pu.append(prob_xu_fake.detach().clone().float().cpu())
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1)

        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])
        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)
        logits_xgl_mix, logits_xgu_mix = logits_xg_mix[:Nl], logits_xg_mix[Nl:]
        logits_xfl_mix, logits_xfu_mix = logits_xf_mix[:Nl], logits_xf_mix[Nl:]

        # CLF loss
        loss_pred = self.criterion(None, probl_mix, logits_xgl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xgu_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, probu_mix, logits_xfu_mix, None)

        #지식증류
        
        # 1) teacher model 구성 및 현재 상태 load
        if perform_distillation:
            teacher_model = FeatMatch(backbone=self.config['model']['backbone'],
                                      num_classes=self.config['model']['classes'],
                                      devices=self.args.devices,
                                      num_heads=self.config['model']['num_heads'],
                                      amp=self.args.amp)
            checkpoint = torch.load('weights/[cifar100][test][cnn13][4000]/run1/best_ckpt')
            teacher_model.load_state_dict(checkpoint['model'])
            teacher_model.to('cuda')
            teacher_model.eval()

            # KLDiv Loss
            self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')

            teacher_dim = teacher_model(x_mix).size(1)  # teacher 모델의 출력 차원
            student_dim = logits_xgu_mix.size(1)  # student 모델의 출력 차원
            adaptive_layer = nn.Linear(teacher_dim, student_dim)
            adaptive_layer.to('cuda')

            with torch.no_grad():
                logits_teacher_x = teacher_model(x_mix) # torch.Size([384, 128])
                logits_teacher_x = logits_teacher_x[:logits_xgu_mix.size(0), :]  # 첫 256개의 샘플만 선택(student와 차원맞추기)
                logits_teacher_adapted = adaptive_layer(logits_teacher_x)  # adaptive layer 적용
                teacher_probs = F.log_softmax(logits_teacher_adapted / 3, dim=1)
                student_probs = F.log_softmax(logits_xgu_mix / 3, dim=1)
        
            distillation_loss = self.distillation_criterion(student_probs, teacher_probs)
        else:
            distillation_loss = 0  # 지식 증류 과정이 실행되지 않을 경우 0으로 설정

        # Total loss 계산 (지식 증류 가중치 포함:config 내 loss에 kld 항목 생성 및 가중치 적용)
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)
        if perform_distillation:
            loss += coeff * self.config['loss']['kld'] * distillation_loss

        # Prediction
        pred_x = torch.softmax(logits_xgl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval1(self, x, y):
        logits_x = self.model(x)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_x.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_x_mix = self.model(x_mix)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_x_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_x.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval2(self, x, y):
        logits_xg, logits_xf, _, _, _ = self.model(x, self.fp)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_xg.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_xg_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, prob_mix, logits_xf_mix, None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*(self.config['loss']['mix']*loss_con + self.config['loss']['graph']*loss_graph)

        # Prediction
        pred_x = torch.softmax(logits_xg.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def forward_train(self, data):
        self.model.train()
        xl = data[0].reshape(-1, *data[0].shape[2:])
        xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)
        yl = data[1].to(self.default_device)
        xu = data[2].reshape(-1, *data[2].shape[2:])
        xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)

        if self.curr_iter < self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
        elif self.curr_iter == self.config['train']['pretrain_iters']:
            self.model.set_mode('train')
            self.extract_fp()
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
        else:
            self.model.set_mode('train')
            if self.curr_iter % self.config['train']['sample_interval'] == 0:
                self.extract_fp()
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)

        results = {
            'y_pred': torch.max(pred_x, dim=1)[1].detach().cpu().numpy(),
            'y_true': yl.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            }
        }

        return loss, results

    def forward_eval(self, data):
        self.model.eval()
        x = self.Tnorm(data[0].to(self.default_device))
        y = data[1].to(self.default_device)

        if self.curr_iter < self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.eval1(x, y)
        else:
            self.model.set_mode('train')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.eval2(x, y)

        results = {
            'y_pred': torch.max(pred_x, dim=1)[1].detach().cpu().numpy(),
            'y_true': y.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            }
        }

        return results


if __name__ == '__main__':
    args, config, save_root = command_interface()

    r = args.rand_seed
    reporter = Reporter(save_root, args)

    for i in range(args.iters):
        args.rand_seed = r + i
        cprint(f'Run iteration [{i+1}/{args.iters}] with random seed [{args.rand_seed}]', attrs=['bold'])

        setattr(args, 'save_root', save_root/f'run{i}')
        if args.mode == 'resume' and not args.save_root.exists():
            args.mode = 'new'
        args.save_root.mkdir(parents=True, exist_ok=True)

        trainer = FeatMatchTrainer(args, config)
        if args.mode != 'test':
            trainer.train()

        acc_val, acc_test = trainer.test()
        acc_median = metric.median_acc(os.path.join(args.save_root, 'results.txt'))
        reporter.record(acc_val, acc_test, acc_median)
        with open(args.save_root/'final_result.txt', 'w') as file:
            file.write(f'Val acc: {acc_val*100:.2f} %')
            file.write(f'Test acc: {acc_test*100:.2f} %')
            file.write(f'Median acc: {acc_median*100:.2f} %')

    reporter.report()
