"""
This is the domain incremental learning for our methods using CORe50 dataset.
"""


import os.path as osp
import threadpoolctl
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models.ganfake.clip import clip
from models.ganfake.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbonename
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        n_ctx = cfg.NCTX # number of context vectors
        ctx_init = cfg.CTXINIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = clip_imsize
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        device = clip_model.token_embedding.weight.device
        self.ctx = nn.Parameter(ctx_vectors).to(device)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(), but they should be ignored in load_model() as we want to use those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'

from utils.latencyTools.core50words import core50_classnames
from operator import itemgetter

class COOPNet(nn.Module):

    def __init__(self):
        super(COOPNet, self).__init__()
        cfg = cfgc()
        self.cfg = cfg
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_list = []

        # self.prompt_pool = nn.ModuleList([
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        #     PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        # ])
        self.prompt_pool = nn.ModuleList([
            PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model),
        ])

        # self.instance_prompt = nn.ModuleList([
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        #     nn.Linear(768, 10, bias=False),
        # ])

        self.instance_prompt = nn.ModuleList([
            nn.Linear(768, 10, bias=False),
        ])

        self.instance_keys = nn.Linear(768, 10, bias=False)

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        # image_features = self.image_encoder(image.type(self.dtype), self.instance_keys.weight)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
    #
    # def forward(self, image):
    #     logits = []
    #     text_features_list = []
    #     image_features_list = []
    #     image_features = self.image_encoder(image.type(self.dtype), self.instance_prompt[self.numtask-1].weight)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #
    #     for prompts in [self.prompt_pool[self.numtask-1]]:
    #         image_features_list.append(image_features)
    #         tokenized_prompts = prompts.tokenized_prompts
    #         text_features = self.text_encoder(prompts(), tokenized_prompts)
    #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #         logit_scale = self.logit_scale.exp()
    #         text_features_list.append(text_features)
    #         logits.append(logit_scale * image_features @ text_features.t())
    #
    #
    #     return {
    #         'logits': torch.cat(logits, dim=1),
    #         'features': image_features
    #     }

    def forward(self, image):
        logits = []
        text_features_list = []
        image_features_list = []
        image_features = self.image_encoder(image.type(self.dtype), self.instance_prompt[-1].weight)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_pool[-1]
        image_features_list.append(image_features)
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        text_features_list.append(text_features)
        logits.append(logit_scale * image_features @ text_features.t())


        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }



    def interface(self, image, selection):
        instance_batch = torch.stack([i.weight for i in self.instance_prompt], 0)[selection, :, :]
        image_features = self.image_encoder(image.type(self.dtype), instance_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for prompt in self.prompt_pool:
            tokenized_prompts = prompt.tokenized_prompts
            text_features = self.text_encoder(prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][50*ii:50*ii+50])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit


    def update_fc(self, nb_classes):
        self.numtask +=1
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

from models.base import BaseLearner
import logging
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from utils.toolkit import  tensor2numpy
from utils.toolkit import accuracy_binary, accuracy
from utils.latencyTools.CORe50_dataset.data_loader import COR50_Dataset, CORE50

class coop_core50(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = COOPNet()
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

        self.all_keys = []

        self.dataset_generator = CORE50(root='/home/wangyabin/workspace/datasets/core50/data/core50_128x128', scenario="ni")
        # self.dataset_list = []
        # for i, train_batch in enumerate(self.dataset_generator):
        #     self.dataset_list.append(train_batch)


    def after_task(self):
        self._old_network = self._network.copy().freeze()
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        del self.train_loader, self.test_loader


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}'.format(self._cur_task))


        train_x, train_y = self.dataset_generator.get_data_batchidx(self._cur_task)

        ################## joint training #####################
        # dataset_list = []
        # for i, train_batch in enumerate(self.dataset_generator):
        #     dataset_list.append(train_batch)
        # train_x = np.concatenate(np.array(dataset_list)[:, 0])
        # train_y = np.concatenate(np.array(dataset_list)[:, 1])
        #######################################################

        train_dataset = COR50_Dataset(train_x, train_y, mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_x, test_y = self.dataset_generator.get_test_set()
        test_dataset = COR50_Dataset(test_x, test_y, mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        del train_y, train_x, test_x, test_y
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        # self.clustering(self.train_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # if "instance_prompt"+"."+str(self._network.module.numtask-1) in name:
            if "instance_prompt" in name:
                param.requires_grad_(True)
            if "instance_keys" in name:
                param.requires_grad_(True)
            # if "prompt_pool"+"."+str(self._network.module.numtask-1) in name:
            if "prompt_pool" in name:
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
                logits = self._network(inputs)['logits']
                loss=F.cross_entropy(logits,targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            # test_acc = self._compute_accuracy(self._network, test_loader)
            test_acc = 0
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

                outputs = self._network(inputs)
                logits = outputs['logits']

                loss_clf=F.cross_entropy(logits,targets)
                loss = loss_clf #+ loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            # test_acc = self._compute_accuracy(self._network, test_loader)
            test_acc = 0

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                        self._cur_task, epoch+1, self.epochs, losses/len(train_loader), train_acc, test_acc)

            prog_bar.set_description(info)
        logging.info(info)


    def clustering(self, dataloader):
        from sklearn.cluster import KMeans

        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            # only get new data
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=50, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))
        del clustering, features






    #
    # def _eval_cnn(self, loader):
    #     self._network.eval()
    #     y_pred, y_true = [], []
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         targets = targets.to(self._device)
    #
    #         with torch.no_grad():
    #             if isinstance(self._network, nn.DataParallel):
    #                 feature = self._network.module.extract_vector(inputs)
    #             else:
    #                 feature = self._network.extract_vector(inputs)
    #             taskselection = []
    #             for task_centers in self.all_keys:
    #                 tmpcentersbatch = []
    #                 for center in task_centers:
    #                     tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
    #                 taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])
    #
    #             selection = torch.vstack(taskselection).min(0)[1]
    #
    #             if isinstance(self._network, nn.DataParallel):
    #                 outputs = self._network.module.interface(inputs, selection)
    #             else:
    #                 outputs = self._network.interface(inputs, selection)
    #
    #         predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
    #         # predicts = predicts + (selection * 10).unsqueeze(1)
    #         y_pred.append(predicts.cpu().numpy())
    #         y_true.append(targets.cpu().numpy())
    #
    #     return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    #




def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]