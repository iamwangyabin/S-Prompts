import os.path as osp
import threadpoolctl

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models.base import BaseLearner
import logging
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from utils.toolkit import  tensor2numpy
from utils.toolkit import accuracy_binary

import copy

from utils.inc_net import BaseNet, SimpleLinear



class InsPrompts(nn.Module):
    def __init__(self, pool_size, length, embed_dim, topk):
        super(InsPrompts, self).__init__()
        self.pool_size = pool_size
        self.length = length
        self.top_k = topk
        self.embedding_key="cls"
        self.batchwise_prompt=False

        self.prompt = nn.Parameter(torch.Tensor(self.pool_size, self.length, embed_dim).uniform_(0,1)*0.01, requires_grad=True)
        self.register_parameter('prompt',self.prompt)

        self.prompt_key = nn.Parameter(torch.Tensor(self.pool_size, embed_dim).uniform_(0,1)*0.01, requires_grad=True)
        self.register_parameter('prompt_key',self.prompt_key)

    def l2_normalize(self, x, axis=None, epsilon=1e-12):
        """l2 normalizes a tensor on an axis with numerical stability."""
        square_sum = torch.sum(torch.square(x), dim=axis, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
        return x * x_inv_norm

    def expand_to_batch(self, x, batch_size: int):
        """Expands unbatched `x` to the specified batch_size`."""
        return torch.tile(x, [batch_size] + [1 for _ in x.shape])

    def forward(self, x_embed, cls_features=None):
        if self.embedding_key == "mean":
            x_embed_mean = torch.mean(x_embed, dim=1)  # bs, emb
        elif self.embedding_key == "max":
            x_embed_mean = torch.max(x_embed, dim=1)
        elif self.embedding_key == "mean_max":
            x_embed_mean = torch.max(x_embed, dim=1) + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == "cls":
            if cls_features is None:  # just for init
                x_embed_mean = torch.max(x_embed, dim=1)
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError(
                "Not supported way of calculating embedding keys!")
        prompt_norm = self.l2_normalize(self.prompt_key, axis=1)
        x_embed_norm = self.l2_normalize(x_embed_mean, axis=1)

        sim = torch.matmul(x_embed_norm, (prompt_norm).t())  # bs, pool_size


        (_, idx) = torch.topk(sim, self.top_k)
        if self.batchwise_prompt:
            # prompt_id, id_counts = jax.unique(idx, return_counts=True, size=self.pool_size)
            prompt_id, id_counts = torch.unique(idx, return_counts=True) # 只能返回这么大的数据size=self.pool_size
            _, major_idx = torch.topk(id_counts, self.top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = self.expand_to_batch(major_prompt_id, x_embed.shape[0])

        # bs, allowed_size, prompt_len, embed_dim
        batched_prompt_raw = torch.index_select(self.prompt, 0, idx.view(-1)).view(idx.shape + self.prompt.shape[1:])
        bs, allowed_size, prompt_len, embed_dim = batched_prompt_raw.shape
        batched_prompt = torch.reshape(batched_prompt_raw,(bs, allowed_size * prompt_len, embed_dim))
        # bs, top_k, embed_dim
        batched_key_norm = torch.index_select(prompt_norm, 0, idx.view(-1)).view(idx.shape + prompt_norm.shape[1:])

        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim = batched_key_norm * x_embed_norm
        # reduce_sim = torch.sum(sim) / x_embed.shape[0]

        return batched_prompt, sim

class L2P(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)
        self.instance_prompt = InsPrompts(pool_size=10,
                                          length=5,
                                          embed_dim=self.feature_dim,
                                          topk=5
                                          )

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x_embed = self.convnet(x)['features']
        batched_prompt, reduce_sim = self.instance_prompt(x_embed, x_embed)

        x = self.convnet.patch_embed(x)
        x = torch.cat((self.convnet.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.convnet.pos_embed
        x = torch.cat([x[:, :1, :], batched_prompt, x[:, 1:, :]], dim=1)
        x = self.convnet.pos_drop(x)
        x = self.convnet.blocks(x)
        x = self.convnet.norm(x)
        if self.convnet.global_pool:
            x = x[:, 1:].mean(dim=1) if self.convnet.global_pool == 'avg' else x[:, 0]
        x = self.convnet.fc_norm(x)
        return x


    def forward(self, x, sim=False):
        x_embed = self.convnet(x)['features']
        batched_prompt, reduce_sim = self.instance_prompt(x_embed, x_embed)

        x = self.convnet.patch_embed(x)
        x = torch.cat((self.convnet.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.convnet.pos_embed
        x = torch.cat([x[:, :1, :], batched_prompt, x[:, 1:, :]], dim=1)
        x = self.convnet.pos_drop(x)
        x = self.convnet.blocks(x)
        x = self.convnet.norm(x)
        if self.convnet.global_pool:
            x = x[:, 1:].mean(dim=1) if self.convnet.global_pool == 'avg' else x[:, 0]
        x = self.convnet.fc_norm(x)
        out = self.fc(x)
        # import pdb;pdb.set_trace()
        if sim:
            return out, reduce_sim
        else:
            return out


class l2p_ganfake(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = L2P(args['convnet_type'], False)
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_milestones = args["init_milestones"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.milestones = args["milestones"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.T = args["T"]
        self.topk = 2 # origin is 5

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            if "instance_prompt" not in name:
                param.requires_grad_(False)
            if "fc.weight" in name:
                param.requires_grad_(True)
            elif "fc.bias" in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)


    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out, reduce_sim = self._network(inputs, True)
                logits = out['logits']

                loss = F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[1])[targets].to(targets.device)) \
                        - 0.01*torch.sum(reduce_sim)/inputs.shape[0]
                # import pdb;pdb.set_trace()
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
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            self._cur_task, epoch+1, self.init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # logits = self._network(inputs)['logits']
                out, reduce_sim = self._network(inputs, True)
                logits = out['logits']
                # import pdb;pdb.set_trace()
                loss_clf = F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[1])[targets].to(targets.device))


                loss=loss_clf - 0.01 * torch.sum(reduce_sim) / inputs.shape[0]

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
        ret['top5'] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)

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

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]