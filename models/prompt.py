import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.model_zoo.prompt_net import PromptNet
from utils.inc_net import DERNet
from utils.toolkit import target2onehot, tensor2numpy

from models.loss_utils.pearson_loss import pearson_loss
from models.loss_utils.arploss import ARPLoss


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

class PromptCIL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = PromptNet(args['convnet_type'], False)
        self.args = args
        self.EPSILON = args["EPSILON"]
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
        self.graph_lambda = args["graph_lambda"]
        self.finetune = args["finetune"]
        self.oodepoch = args["oodepoch"]
        self.init_epoch = args["init_epoch"]

    def reload_arg(self, args):
        self.args = args
        self.EPSILON = args["EPSILON"]
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
        self.graph_lambda = args["graph_lambda"]



    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))


    def incremental_resume(self, data_manager, resumedict):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(data_manager.get_task_size(self._cur_task), self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if self._cur_task > 0:
            anchor_dataset = data_manager.get_anchor_dataset(mode='train', appendent=self._get_memory())
            self.anchor_loader = DataLoader(anchor_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.load_state_dict(resumedict)
        self._network.to(self._device)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(data_manager.get_task_size(self._cur_task), self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if self._cur_task > 0:
            anchor_dataset = data_manager.get_anchor_dataset(mode='train', appendent=self._get_memory())
            self.anchor_loader = DataLoader(anchor_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)

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
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones,gamma=self.init_lr_decay)
            self._init_clean_train(train_loader, test_loader, optimizer, scheduler)
            for param in self._network.parameters():
                param.requires_grad = False
            for param in self._network.prompt_pool.parameters():
                param.requires_grad = True
            for param in self._network.ood_classifier.parameters():
                param.requires_grad = True
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            for param in self._network.parameters():
                param.requires_grad = True
            for param in self._network.fc.fc1.parameters():
                param.requires_grad = False
            for param in self._network.fc.fc2.parameters():
                param.requires_grad = True
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay) # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
            self._incremental_phase(train_loader, test_loader, self.anchor_loader, optimizer, scheduler)

            self._finetune_phase(data_manager, train_loader)


    def _init_clean_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network.forward_clean(inputs)['logits']
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
            info = 'Clean {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.args["init_epoch"], losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                feature = self._network.encoding_with_index(inputs, 0)
                # arplogit, loss = self._network.ood_classifier[0].arp_logit(feature, None, targets)
                arplogit = self._network.currentlogits(feature)
                loss = F.cross_entropy(arplogit, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(arplogit, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy_ood(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.args["init_epoch"], losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)



    def _compute_accuracy_ood(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                feature = self._network.encoding_with_index(inputs, 0)
                logit = self._network.currentlogits(feature)
                # logit, _ = self._network.ood_classifier[0].arp_logit(feature, None, None)
                # logit = self._network(inputs)['logits']
            predicts = torch.max(logit, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        # import pdb;pdb.set_trace()
        return np.around(tensor2numpy(correct)*100 / total, decimals=2)


    def _incremental_phase(self, train_loader, test_loader, anchorloader,optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        anchoriter = iter(anchorloader)
        # num_old_classes = sum(self._network.getclassifierID()[:-1])
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # only get new data
                # mask = (targets >= num_old_classes).nonzero().view(-1)
                # inputs = torch.index_select(inputs, 0, mask)
                # targets = torch.index_select(targets, 0, mask)-num_old_classes
                # output = self._network(inputs)
                try:
                    _, inputsANC, targetsANC = anchoriter.next()
                except:
                    anchoriter = iter(anchorloader)
                    _, inputsANC, targetsANC = anchoriter.next()
                anchor_inputs = inputsANC.to(self._device)
                anchor_targets = targetsANC.to(self._device)
                outputs = self._network(anchor_inputs)
                cur_features = outputs['features']
                ref_outputs = self._old_network(anchor_inputs)
                ref_features = ref_outputs['features']
                loss_kd = pearson_loss(ref_features, cur_features) * self.graph_lambda
                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets) + loss_kd
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
            # test_acc = self._compute_accuracy_for_pretask(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc, test_acc)

            prog_bar.set_description(info)
        logging.info(info)

    def _finetune_phase(self, data_manager, train_loader):
        # cnn_accy, nme_accy = self.eval_task()
        # print(cnn_accy)
        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.prompt_pool.parameters():
            param.requires_grad = True
        for param in self._network.ood_classifier.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=0.01, momentum=0.9, weight_decay=self.weight_decay) # 1e-5
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)

        prog_bar = tqdm(range(self.oodepoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._network.getclassifierID()[-2]).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._network.getclassifierID()[-2]
                feature = self._network.encoding_with_index(inputs, -1)
                # feature = self._network.encoding_clean(inputs)
                # arplogit, loss = self._network.ood_classifier[-1].arp_logit(feature, None, targets)

                arplogit = self._network.currentlogits(feature)
                loss = F.cross_entropy(arplogit, targets)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # import pdb;pdb.set_trace()
                losses += loss.item()
                _, preds = torch.max(arplogit, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = 'GenNewOOD {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        self._network.eval()


        classes = self._network.getclassifierID()
        for phase in range(len(classes)-2):
            ex_d, ex_t = self._get_exemplar_with_class_idxes(range(classes[phase], classes[phase + 1]))
            # import pdb;pdb.set_trace()
            ood_dataset = data_manager.get_dataset([], source='train', mode='train', appendent=(ex_d, ex_t))
            ood_loader = DataLoader(ood_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay) # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
            prog_bar = tqdm(range(self.finetune))
            # self.prine(self._network.named_parameters())

            for _, epoch in enumerate(prog_bar):
                self._network.eval()
                losses = 0.
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(ood_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    targets = targets - min(ex_t)
                    feature = self._network.encoding_with_index(inputs, phase)
                    # feature = self._network.encoding_clean(inputs)
                    # arplogit, loss = self._network.ood_classifier[phase].arp_logit(feature, None, targets)

                    arplogit = self._network.wholelogits(feature)[:,classes[phase]:classes[phase + 1]]
                    loss = F.cross_entropy(arplogit, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    _, preds = torch.max(arplogit, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)
                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                info = 'FinetuneOOD {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self.epochs, losses / len(ood_loader), train_acc)
                prog_bar.set_description(info)

            logging.info(info)

    # named_parameters
    def prine(self, paras):
        for name, para in enumerate(paras):
            if para[1].requires_grad:
                print('name:', para[0])
                # print(para[1])
                print('_____________________________')


    # def _eval_cnn(self, loader):
    #     self._network.eval()
    #     y_pred, y_true = [], []
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             logit = self._network(inputs)['logits']
    #         predicts = torch.topk(logit, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    #         y_pred.append(predicts.cpu().numpy())
    #         y_true.append(targets.cpu().numpy())
    #
    #     return np.concatenate(y_pred), np.concatenate(y_true)

    # def _compute_accuracy(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #     for i, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             logit = self._network(inputs)['logits']
    #         predicts = torch.max(logit, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)
    #     # import pdb;pdb.set_trace()
    #     return np.around(tensor2numpy(correct)*100 / total, decimals=2)


    # def _compute_accuracy_for_pretask(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #     for ii, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             out = []
    #             for i, p in enumerate(self._network.prompt_pool[:-1]):
    #                 xx = self._network.convnet(inputs + p.prompt.expand(inputs.shape))['features']
    #                 out.append(self._network.fc(xx)['logits'][:, sum(self._network.classifier_id[:i]):sum(self._network.classifier_id[:i + 1])])
    #             xx = self._network.convnet(inputs)['features']
    #             out.append(self._network.fc(xx)['logits'][:,sum(self._network.classifier_id[:i + 1]): ])
    #             # import pdb;pdb.set_trace()
    #             logit = torch.cat(out, dim=1)
    #         predicts = torch.max(logit, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)
    #
    #     return np.around(tensor2numpy(correct)*100 / total, decimals=2)


# class PromptDER(BaseLearner):
#
#     def __init__(self, args):
#         super().__init__(args)
#         self._network = DERNet(args['convnet_type'], False)
#
#     def after_task(self):
#         self._known_classes = self._total_classes
#         logging.info('Exemplar size: {}'.format(self.exemplar_size))
#
#     def incremental_train(self, data_manager):
#         self._cur_task += 1
#         self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
#         self._network.update_fc(self._total_classes)
#         logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
#
#         if self._cur_task > 0:
#             for i in range(self._cur_task):
#                 for p in self._network.convnets[i].parameters():
#                     p.requires_grad = False
#
#         logging.info('All params: {}'.format(count_parameters(self._network)))
#         logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
#
#         train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
#                                                  mode='train', appendent=self._get_memory())
#         self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#         test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
#         self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#         if len(self._multiple_gpus) > 1:
#             self._network = nn.DataParallel(self._network, self._multiple_gpus)
#         self._train(self.train_loader, self.test_loader)
#         self.build_rehearsal_memory(data_manager, self.samples_per_class)
#         if len(self._multiple_gpus) > 1:
#             self._network = self._network.module
#
#     def train(self):
#         self._network.train()
#         self._network.module.convnets[-1].train()
#         if self._cur_task >= 1:
#             for i in range(self._cur_task):
#                 self._network.module.convnets[i].eval()
#
#     def _train(self, train_loader, test_loader):
#         self._network.to(self._device)
#         if self._cur_task == 0:
#             optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,
#                                   lr=init_lr, weight_decay=init_weight_decay)
#             scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones,
#                                                        gamma=init_lr_decay)
#             self._init_train(train_loader, test_loader, optimizer, scheduler)
#         else:
#             optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, momentum=0.9,
#                                   weight_decay=weight_decay)
#             scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
#             self._update_representation(train_loader, test_loader, optimizer, scheduler)
#             if len(self._multiple_gpus) > 1:
#                 self._network.module.weight_align(self._total_classes - self._known_classes)
#             else:
#                 self._network.weight_align(self._total_classes - self._known_classes)
#
#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(init_epoch))
#         for _, epoch in enumerate(prog_bar):
#             self.train()
#             losses = 0.
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 logits = self._network(inputs)['logits']
#
#                 loss = F.cross_entropy(logits, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()
#
#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)
#
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
#
#             if epoch % 5 == 0:
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
#                     self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc)
#             else:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
#                     self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc)
#             prog_bar.set_description(info)
#
#         logging.info(info)
#
#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(epochs))
#         for _, epoch in enumerate(prog_bar):
#             self.train()
#             losses = 0.
#             losses_clf = 0.
#             losses_aux = 0.
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 outputs = self._network(inputs)
#                 logits, aux_logits = outputs["logits"], outputs["aux_logits"]
#                 loss_clf = F.cross_entropy(logits, targets)
#                 aux_targets = targets.clone()
#                 aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
#                                           aux_targets - self._known_classes + 1, 0)
#                 loss_aux = F.cross_entropy(aux_logits, aux_targets)
#                 loss = loss_clf + loss_aux
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()
#                 losses_aux += loss_aux.item()
#                 losses_clf += loss_clf.item()
#
#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)
#
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
#             if epoch % 5 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
#                     self._cur_task, epoch + 1, epochs, losses / len(train_loader), losses_clf / len(train_loader),
#                                     losses_aux / len(train_loader), train_acc, test_acc)
#             else:
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
#                     self._cur_task, epoch + 1, epochs, losses / len(train_loader), losses_clf / len(train_loader),
#                                     losses_aux / len(train_loader), train_acc)
#             prog_bar.set_description(info)
#         logging.info(info)
