import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from models.loss_utils.pearson_loss import pearson_loss

from utils.toolkit import accuracy_binary

class LUCIR_GAN(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(args['convnet_type'], False)
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_lr = args["init_lr"]
        self.init_milestones = args["init_milestones"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_epoch = args["init_epoch"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.milestones = args["milestones"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.T = args["T"]

        self.the_lambda = args["the_lambda"]
        self.K = args["K"]
        self.dist = args["dist"]
        self.lw_mr = args["lw_mr"]
        self.topk = 2

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, data_manager)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, data_manager):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones,
                                                       gamma=self.init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            if len(self._multiple_gpus) > 1:
                ignored_params = list(map(id, self._network.module.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                tg_params = [
                    {'params': base_params, 'lr': self.lrate, 'weight_decay': self.weight_decay},
                    {'params': self._network.module.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}, ]
            else:
                ignored_params = list(map(id, self._network.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                tg_params = [
                    {'params': base_params, 'lr': self.lrate, 'weight_decay': self.weight_decay},
                    {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}, ]

            optimizer = optim.SGD(tg_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
            anchor_dataset = data_manager.get_anchor_dataset(mode='train', appendent=self._get_memory())
            anchor_loader = DataLoader(anchor_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self._incremental_phase(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)


            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            self._cur_task, epoch+1, self.init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)


        logging.info(info)


    def _incremental_phase(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        num_old_classes = self._old_network.fc.out_features
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                cur_features = outputs['features']
                logits = outputs['logits']
                ref_outputs = self._old_network(inputs)
                ref_features = ref_outputs['features']
                ref_logits = ref_outputs['logits']

                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(self._device)) * self.the_lambda

                loss2 = nn.CrossEntropyLoss()(logits, targets)

                # outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                outputs_bs = logits
                assert (outputs_bs.size() == logits.size())
                gt_index = torch.zeros(outputs_bs.size()).to(self._device)
                gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                max_novel_scores = outputs_bs[:, num_old_classes:].topk(self.K, dim=1)[0]
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                if hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, self.K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    loss3 = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1),torch.ones(hard_num * self.K).view(-1, 1).to(self._device)) * self.lw_mr
                else:
                    loss3 = torch.zeros(1).to(self._device)
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            test_binary_acc = self._compute_accuracy_binary(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Test_binary {:.2f}'.format(
            self._cur_task, epoch+1, self.epochs, losses/len(train_loader), train_acc, test_acc, test_binary_acc)

            prog_bar.set_description(info)
        logging.info(info)



    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_binary(y_pred.T[0], y_true, self._known_classes)

        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top5'] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
                                decimals=2)
        # ret['top{}'.format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
        #                                            decimals=2)

        return ret

    def _compute_accuracy_binary(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts%2).cpu() == (targets%2)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)
