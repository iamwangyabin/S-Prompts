import torch
import torch.nn as nn
import copy

from models.clip.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames


class SliNet(nn.Module):

    def __init__(self, args):
        super(SliNet, self).__init__()
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.class_num = 1
        if args["dataset"] == "cddb":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(cddb_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(domainnet_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.prompt_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
            for i in range(args["total_sessions"])
        ])

        # self.instance_keys = nn.Linear(768, 10, bias=False)


        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def forward(self, image):
        logits = []
        image_features = self.image_encoder(image.type(self.dtype), self.prompt_pool[self.numtask-1].weight)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.classifier_pool[self.numtask-1]
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(logit_scale * image_features @ text_features.t())
        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        image_features = self.image_encoder(image.type(self.dtype), instance_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for prompt in self.classifier_pool:
            tokenized_prompts = prompt.tokenized_prompts
            text_features = self.text_encoder(prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit


    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
