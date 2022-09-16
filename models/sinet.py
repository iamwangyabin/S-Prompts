import torch
import torch.nn as nn
import copy

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn

class ViT_Prompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)


    def forward(self, x, instance_tokens=None, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)
        if instance_tokens is not None:
            x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_Prompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

        self.class_num = 1
        if args["dataset"] == "cddb":
            self.class_num = 2
            self.classifier_pool = nn.ModuleList([
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
                for i in range(args["total_sessions"])
            ])
        elif args["dataset"] == "domainnet":
            self.class_num = 345
            self.classifier_pool = nn.ModuleList([
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
                for i in range(args["total_sessions"])
            ])
        elif args["dataset"] == "core50":
            self.class_num = 50
            self.classifier_pool = nn.ModuleList([
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
                for i in range(args["total_sessions"])
            ])

        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.prompt_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
            for i in range(args["total_sessions"])
        ])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image):
        logits = []
        image_features = self.image_encoder(image, self.prompt_pool[self.numtask-1].weight)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        image_features = self.image_encoder(image, instance_batch)

        logits = []
        for prompt in self.classifier_pool:
            logits.append(prompt(image_features))

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
