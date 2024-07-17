import torch.nn as nn
import torch

from . import _blocks
import torch.nn.functional as F
Norm = nn.LayerNorm


class AttentionMoudle(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(AttentionMoudle, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(
            dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = dim ** (-0.5)
        self.act = nn.ReLU()
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v):
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None):
        q, k, v = self.get_qkv(q, k, v)

        attn_ori = torch.matmul(q, k.transpose(2, 1))
        attn_confusion = F.softmax(attn_ori, dim=-1)
        attn = self.scale * attn_confusion
        attn_mask = attn
        out = torch.matmul(attn_mask, v.float())
        return attn_confusion, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(
            self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(
            self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Network(nn.Module):

    def __init__(self, **params):
        super(Network, self).__init__()
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self.classes = params.get('classes', 10)
        self.channels = params.get('channels', 1)
        self.out_chann = 128

        _w_init = params.get(
            'w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.))

        self.layer_img = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True, batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[3, 3, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True, batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[3, 3, 64, 128], stride=1, padding='valid', activation='relu', batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(
                shape=[3, 3, 128, self.classes], stride=1, padding='valid', activation='relu', batch_norm=True,
                w_init=_w_init, b_init=nn.init.zeros_
            ),
            nn.Flatten(start_dim=2, end_dim=-1)
        )
        self.layer_asc = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True, batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[3, 3, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True, batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[3, 3, 64, 128], stride=1, padding='valid', activation='relu', batch_norm=True,
                w_init=_w_init, b_init=_b_init
            ),
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(
                shape=[3, 3, 128, self.classes], stride=1, padding='valid', activation='relu', batch_norm=True,
                w_init=_w_init, b_init=nn.init.zeros_
            ),
            nn.Flatten(start_dim=2, end_dim=-1)
        )

        self.channel_num = 16
        self.asc2img_att = AttentionMoudle(self.channel_num)
        self.img2asc_att = AttentionMoudle(self.channel_num)
        self.proj1 = Mlp(in_features=self.channel_num*2,
                         hidden_features=self.classes*self.channel_num,
                         out_features=1)
        self.proj2 = Mlp(in_features=self.channel_num*2,
                         hidden_features=self.classes*self.channel_num,
                         out_features=1)
        self.proj3 = Mlp(in_features=self.channel_num,
                         hidden_features=self.classes*self.channel_num,
                         out_features=1)
        self.proj4 = Mlp(in_features=self.channel_num,
                         hidden_features=self.classes*self.channel_num,
                         out_features=1)
        self.weight_asc = nn.Linear(self.classes, 1)
        self.weight_img = nn.Linear(self.classes, 1)
        self.flat = nn.Flatten()
        self.eyemat = 1-torch.eye(self.classes, requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, x, asc):
        img_feature = self.layer_img(x)
        asc_feature = self.layer_asc(asc)
        # print(img_feature.shape[-1])
        # confusion_mat = torch.matmul(asc_feature, img_feature.transpose(1, 2))
        # confusion_mat_T = torch.matmul(
        #     img_feature, asc_feature.transpose(1, 2))
        confusion_mat, asc2img = self.asc2img_att(
            asc_feature, img_feature,   img_feature)
        confusion_mat2, img2asc = self.img2asc_att(
            img_feature, asc_feature, asc_feature)
        fuse_img = torch.concat([asc2img, img_feature], dim=-1)
        fuse_asc = torch.concat([img2asc, asc_feature], dim=-1)

        logits_fuse_img = self.proj1(fuse_img)
        logits_fuse_asc = self.proj2(fuse_asc)
        logits_asc = self.proj4(asc_feature)
        logits_img = self.proj3(img_feature)
        return logits_fuse_img.squeeze(-1), logits_fuse_asc.squeeze(-1), logits_img.squeeze(-1), logits_asc.squeeze(-1), confusion_mat, confusion_mat2
