import os.path as osp
import threadpoolctl

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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
    NCTX = 8  #default16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'

import copy

class COOPNet(nn.Module):

    def __init__(self):
        super(COOPNet, self).__init__()
        cfg = cfgc()
        self.cfg = cfg
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(cfg, ['real', 'fake'], clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts


        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_list = []

        self.prompt_pool = nn.ModuleList([
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model),
            PromptLearner(self.cfg, ['real', 'fake'], self.clip_model)
        ])

        self.instance_prompt = nn.ModuleList([
            nn.Linear(768, 10, bias=False),
            nn.Linear(768, 10, bias=False),
            nn.Linear(768, 10, bias=False),
            nn.Linear(768, 10, bias=False),
            nn.Linear(768, 10, bias=False),
            nn.Linear(768, 10, bias=False),
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

    def forward(self, image, curlogits=False):
        logits = []
        text_features_list = []
        image_features_list = []
        # image_features = self.image_encoder(image.type(self.dtype), self.instance_keys.weight)# 这个是之前的sota
        image_features = self.image_encoder(image.type(self.dtype), self.instance_prompt[self.numtask-1].weight)
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # for prompts in self.prompt_pool[:self.numtask]: # 这个是之前的sota

        for prompts in [self.prompt_pool[self.numtask-1]]:
            image_features_list.append(image_features)
            tokenized_prompts = prompts.tokenized_prompts
            text_features = self.text_encoder(prompts(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features_list.append(text_features)
            logits.append(logit_scale * image_features @ text_features.t())


        return {
            'logits': torch.cat(logits, dim=1),
            # 'logits': logits,
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
            selectedlogit.append(logits[idx][2*ii:2*ii+2])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit


    def update_fc(self, nb_classes):
        # 对prompt进行增量  目前只需要对prompt增，增这个就代表增了分类头
        # prompter = PromptLearner(self.cfg, ['real', 'fake'], self.clip_model).to(self.clip_model.token_embedding.weight.device)
        # self.prompt_list.append(prompter)
        # del self.prompt_pool
        # self.prompt_pool = nn.ModuleList(self.prompt_list)
        self.numtask +=1
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

from methods.base import BaseLearner
import logging
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from utils.toolkit import  tensor2numpy
from utils.toolkit import accuracy_binary, accuracy


class coop_ganfake(BaseLearner):

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
        self.topk = 2 # origin is 5

        self.all_keys = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        # self._network.numtask=1 # 这个是为了zeroshot

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

        self.clustering(self.train_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            # if "prompt_pool"+"."+str(self._network.module.numtask-1) not in name:
            #     param.requires_grad_(False)
            if "instance_prompt"+"."+str(self._network.module.numtask-1) in name:
                param.requires_grad_(True)
            if "instance_keys" in name:
                param.requires_grad_(True)
            if "prompt_pool"+"."+str(self._network.module.numtask-1) in name:
                param.requires_grad_(True)


        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        # param = [{'params': self._network.module.instance_prompt.parameters()}, {'params': self._network.module.prompt_pool.parameters()}]

        # [name for name, param in self._network.named_parameters()]
        if self._cur_task==0:
            # optimizer = optim.SGD(self._network.prompt_pool[self._cur_task:].parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            # optimizer = optim.SGD(param, momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay)
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
                logits = self._network(inputs,True)['logits']
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

                # only get new data
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                outputs = self._network(inputs)
                logits = outputs['logits']
                # import pdb;pdb.set_trace()
                #icarl

                # targets=targets-self._known_classes
                loss_clf=F.cross_entropy(logits,targets)

                # lucir
                # outputs_bs = logits
                # assert (outputs_bs.size() == logits.size())
                # gt_index = torch.zeros(outputs_bs.size()).to(self._device)
                # gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
                # gt_scores = outputs_bs.masked_select(gt_index)
                # max_novel_scores = outputs_bs[:, 2*(self._network.module.numtask-1):].topk(2, dim=1)[0]
                # hard_index = targets.lt(2*(self._network.module.numtask-1))
                # hard_num = torch.nonzero(hard_index).size(0)
                # gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, 2)
                # max_novel_scores = max_novel_scores[hard_index]
                # assert (gt_scores.size() == max_novel_scores.size())
                # assert (gt_scores.size(0) == hard_num)
                # loss3 = nn.MarginRankingLoss(margin=0.1)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1),torch.ones(hard_num * 2).view(-1, 1).to(self._device))

                # import pdb;pdb.set_trace()
                # loss_clf = F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[1])[targets].to(targets.device))
                # loss_kd=_KD_loss(logits[:,:self._known_classes],self._old_network(inputs)["logits"],self.T) # 之前的sota
                loss = loss_clf #+ loss_kd

                #lwf
                # fake_targets=targets-self._known_classes
                # loss_clf = F.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[1])[fake_targets].to(targets.device))

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
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))


    def _evaluate(self, y_pred, y_true):
        ret = {}

        grouped = accuracy_binary(y_pred.T[0], y_true, self._known_classes)

        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top5'] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)

        return ret

    # def _eval_nme(self, loader, class_means):
    #     self._network.eval()
    #     vectors, y_true = self._extract_vectors(loader)
    #     vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #
    #     dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
    #     scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
    #
    #     return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
                taskselection = []
                for task_centers in self.all_keys:
                    tmpcentersbatch = []
                    for center in task_centers:
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])


                selection = torch.vstack(taskselection).min(0)[1]

                print("Kmeans分类正确率：{:.3f}".format(sum(selection == ((targets/2).int()))/len(targets)))

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    # def _evaluate(self, y_pred, y_true):
    #     ret = {}
    #     grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
    #     ret['grouped'] = grouped
    #     ret['top1'] = grouped['total']
    #     ret['top5'] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true), decimals=2)
    #
    #     return ret

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