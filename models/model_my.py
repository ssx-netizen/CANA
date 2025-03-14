from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from reformer_pytorch import LSHSelfAttention
from nystrom_attention import NystromAttention

from torch.nn.init import xavier_normal



###########################
### MY Implementation ###
###########################
class Multimodal(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(Multimodal, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Set
        self.bucket_size = 128

        ### Path Transformer + Attention Head
        self.path_selfattention = LSHSelfAttention(dim=size[1], heads=4, bucket_size=self.bucket_size, n_hashes=2, causal=False)
        self.path_self_gate = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_self_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        self.path_cross = nn.MultiheadAttention(embed_dim=size[1], num_heads=4, batch_first=True)
        self.path_cross_gate = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_cross_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        ### Omic Transformer + Attention Head
        self.omic_selfattention = nn.MultiheadAttention(embed_dim=hidden[-1], num_heads=4, batch_first=True)
        self.omic_self_gate = Attn_Net_Gated(L=hidden[-1], D=hidden[-1], dropout=dropout, n_classes=1)
        self.omic_self_rho = nn.Sequential(*[nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Dropout(dropout)])
        self.omic_cross = nn.MultiheadAttention(embed_dim=hidden[-1], num_heads=4, batch_first=True)
        self.omic_cross_gate = Attn_Net_Gated(L=hidden[-1], D=hidden[-1], dropout=dropout, n_classes=1)
        self.omic_cross_rho = nn.Sequential(*[nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Dropout(dropout)])
        ### Fusion Layer
        # if self.fusion == 'concat':
        #     self.mm = nn.Sequential(*[nn.Linear(256*6, 256*6), nn.ReLU(), nn.Linear(256*6, size[2]), nn.ReLU()])
        # elif self.fusion == 'bilinear':
        #     self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        # elif self.fusion == 'lmf':
        #     self.mm = LMF(input_dim=256, output_dim=256, rank=8, modal=4)
        # else:
        #     self.mm = None
        self.fusion = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.fusion_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ##############
        # Norm
        self.norm_path_self = nn.LayerNorm(size[2])
        self.norm_path_cross = nn.LayerNorm(size[2])
        self.norm_omic_self = nn.LayerNorm(hidden[-1])
        self.norm_omic_cross = nn.LayerNorm(hidden[-1])
        
        ##############################
        # attention
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier_ps = nn.Linear(size[2], n_classes)
        self.classifier_pc = nn.Linear(size[2], n_classes)
        self.classifier_os = nn.Linear(hidden[-1], n_classes)
        self.classifier_oc = nn.Linear(hidden[-1], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # pdb.set_trace()
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        ### Path
        h_path_self = h_path_bag.unsqueeze(0)
        b, t, d = h_path_self.shape
        paddings = torch.zeros(b, 2*self.bucket_size - (t % (2*self.bucket_size)), d, device=h_path_self.device)
        # print(h.shape)
        h_path_self = torch.cat((h_path_self, paddings), dim=1)
        # print(h.shape)
        input_mask = torch.ones(b, t, device=h_path_self.device).bool()
        paddings_mask = torch.zeros(b, 2*self.bucket_size - (t % (2*self.bucket_size)), device=h_path_self.device).bool()
        input_mask = torch.cat((input_mask, paddings_mask), dim=1)
        h_path_self = self.path_selfattention(h_path_self, input_mask=input_mask)
        h_path_self = h_path_self.squeeze(0)
        h_path_self = h_path_self[:t]
        h_path_self = self.norm_path_self(h_path_self+h_path_bag)
        A_path_self, h_path_self = self.path_self_gate(h_path_self.squeeze(0))
        A_path_self = torch.transpose(A_path_self, 1, 0)
        h_path_self = torch.mm(F.softmax(A_path_self, dim=1) , h_path_self)
        h_path_self = self.path_self_rho(h_path_self)

        h_path_cross, _ = self.path_cross(h_path_bag.unsqueeze(0), h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0))
        h_path_cross = self.norm_path_cross(h_path_cross+h_path_bag)
        A_path_cross, h_path_cross = self.path_cross_gate(h_path_cross.squeeze(0))
        A_path_cross = torch.transpose(A_path_cross, 1, 0)
        h_path_cross = torch.mm(F.softmax(A_path_cross, dim=1) , h_path_cross)
        h_path_cross = self.path_cross_rho(h_path_cross)
        
        ### Omic
        h_omic_self, _ = self.omic_selfattention(h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0))
        h_omic_self = self.norm_omic_self(h_omic_self+h_omic_bag)
        A_omic_self, h_omic_self = self.omic_self_gate(h_omic_self.squeeze(0))
        A_omic_self = torch.transpose(A_omic_self, 1, 0)
        h_omic_self = torch.mm(F.softmax(A_omic_self, dim=1) , h_omic_self)
        h_omic_self = self.omic_self_rho(h_omic_self)

        h_omic_cross, _ = self.omic_cross(h_omic_bag.unsqueeze(0), h_path_bag.unsqueeze(0), h_path_bag.unsqueeze(0))
        h_omic_cross = self.norm_omic_cross(h_omic_cross+h_omic_bag)
        A_omic_cross, h_omic_cross = self.omic_cross_gate(h_omic_cross.squeeze())
        A_omic_cross = torch.transpose(A_omic_cross, 1, 0)
        h_omic_cross = torch.mm(F.softmax(A_omic_cross, dim=1) , h_omic_cross)
        h_omic_cross = self.omic_cross_rho(h_omic_cross)
        # h_omic_trans = self.omic_transformer(h_omic_bag)
        # A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        # A_omic = torch.transpose(A_omic, 1, 0)F
        # h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        # h_omic = self.omic_rho(h_omic).squeeze()
        
        # if self.fusion == 'bilinear':
        #     pass
        #     # h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        # elif self.fusion == 'concat':
        #     h = self.mm(h_omic_bag.view(-1))
        #     pass
        #     # h = self.mm(torch.cat([h_path, h_omic], axis=0))
        # elif self.fusion == "lmf":
        #     h = self.mm([h_path_self, h_path_cross, h_omic_self, h_omic_cross])

        # MMGF
        # h = torch.cat([h_path_self, h_path_cross, h_omic_self, h_omic_cross], axis=0)
        # h = torch.cat([h_omic_self, h_omic_cross], axis=0)
        # A, h = self.fusion(h)
        # A = torch.transpose(A, 1, 0)
        # h = torch.mm(F.softmax(A, dim=1) , h)
        # h = self.fusion_rho(h).squeeze()
        
        ################################
        # h = h_omic_self.squeeze()
        # print(h_path_self.shape)
        ################################
        # attention
        h = torch.cat([h_path_self, h_path_cross, h_omic_self, h_omic_cross], axis=0)
        queries = self.query(h)  # [batch_size, num_vectors, feature_dim]
        keys = self.key(h)       # [batch_size, num_vectors, feature_dim]
        
        # 计算点积注意力分数
        scores = torch.matmul(queries, keys.transpose(0, 1))  # [batch_size, num_vectors, num_vectors]
        # 归一化注意力分数
        attn_weights = F.softmax(scores, dim=1)  # [batch_size, num_vectors, num_vectors]
        
        # 加权求和
        h = torch.matmul(attn_weights, h).sum(dim=0).squeeze()  # [batch_size, num_vectors, feature_dim]   

        
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        ps = self.classifier_ps(h_path_self)
        pc = self.classifier_pc(h_path_cross)
        os = self.classifier_os(h_omic_self)
        oc = self.classifier_oc(h_omic_cross)
        
        # print(logits)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        
        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        # attention_scores = {'path_self': A_path_self, 'path_cross': A_path_cross, 'omic_self': A_omic_self, 'omic_cross': A_omic_cross}
        
        return logits, ps, pc, os, oc


    def captum(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        #x_path = torch.randn((10, 500, 1024))
        #x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6 = [torch.randn(10, size) for size in omic_sizes]
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]
        h_path_bag = self.wsi_net(x_path)#.unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag = torch.reshape(h_path_bag, (500, 10, 256))
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans = torch.reshape(h_path_trans, (10, 6, 256))
        A_path, h_path = self.path_attention_head(h_path_trans)
        A_path = F.softmax(A_path.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_path = torch.bmm(A_path, h_path).squeeze(dim=1)

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans = torch.reshape(h_omic_trans, (10, 6, 256))
        A_omic, h_omic = self.omic_attention_head(h_omic_trans)
        A_omic = F.softmax(A_omic.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_omic = torch.bmm(A_omic, h_omic).squeeze(dim=1)

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=1))

        logits  = self.classifier(h)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk
        
class LMF(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.1, rank=8, modal=4):
        super(LMF, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.rank = rank
        self.modal = modal
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.post_fusion_dropout = nn.Dropout(p=self.dropout)
        self.factors = []
        for i in range(self.modal):
            self.factors.append(nn.Parameter(torch.Tensor(self.rank, self.input_dim + 1, self.output_dim).to(device)))

        for i in range(self.modal):
            xavier_normal(self.factors[i])
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.modal):
            self.factors[i] = self.factors[i].to(device)

    def forward(self, h_list):
        batch_size = h_list[0].shape[0]
        for i in range(self.modal):
            h_list[i] = torch.cat((h_list[i], torch.ones(batch_size, 1, requires_grad=False).to(h_list[i].device)), dim=1)
            h_list[i] = torch.matmul(h_list[i], self.factors[i])
            h_list[i] = torch.sum(h_list[i], dim=0).squeeze()
        output = None
        for i in range(self.modal):
            if output is None:
                output = h_list[i]
            else:
                output = output * h_list[i]
        return output
    
class SNNSIG(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(SNNSIG, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        # size = self.size_dict_WSI[model_size_wsi]
        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # fc.append(nn.Dropout(0.25))
        # self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Set
        self.bucket_size = 128

        ### Path Transformer + Attention Head
        # self.path_selfattention = LSHSelfAttention(dim=size[1], heads=4, bucket_size=self.bucket_size, n_hashes=2, causal=False)
        # self.path_self_gate = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        # self.path_self_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        # self.path_cross = nn.MultiheadAttention(embed_dim=size[1], num_heads=4, batch_first=True)
        # self.path_cross_gate = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        # self.path_cross_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        ### Omic Transformer + Attention Head
        self.omic_selfattention = nn.MultiheadAttention(embed_dim=hidden[-1], num_heads=4, batch_first=True)
        # self.omic_self_gate = Attn_Net_Gated(L=hidden[-1], D=hidden[-1], dropout=dropout, n_classes=1)
        # self.omic_self_rho = nn.Sequential(*[nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Dropout(dropout)])
        # self.omic_cross = nn.MultiheadAttention(embed_dim=hidden[-1], num_heads=4, batch_first=True)
        # self.omic_cross_gate = Attn_Net_Gated(L=hidden[-1], D=hidden[-1], dropout=dropout, n_classes=1)
        # self.omic_cross_rho = nn.Sequential(*[nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(), nn.Dropout(dropout)])
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*6, 256*6), nn.ReLU(), nn.Linear(256*6, 256), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        elif self.fusion == 'lmf':
            self.mm = LMF(input_dim=256, output_dim=256, rank=8, modal=4)
        else:
            self.mm = None
        # self.fusion = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        # self.fusion_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ##############
        # Norm
        # self.norm_path_self = nn.LayerNorm(size[2])
        # self.norm_path_cross = nn.LayerNorm(size[2])
        self.norm_omic_self = nn.LayerNorm(hidden[-1])
        # self.norm_omic_cross = nn.LayerNorm(hidden[-1])
        
        ### Classifier
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, **kwargs):
        # x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # pdb.set_trace()
        # h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        ### Path
        # h_path_self = h_path_bag.unsqueeze(0)
        # b, t, d = h_path_self.shape
        # paddings = torch.zeros(b, 2*self.bucket_size - (t % (2*self.bucket_size)), d, device=h_path_self.device)
        # # print(h.shape)
        # h_path_self = torch.cat((h_path_self, paddings), dim=1)
        # # print(h.shape)
        # input_mask = torch.ones(b, t, device=h_path_self.device).bool()
        # paddings_mask = torch.zeros(b, 2*self.bucket_size - (t % (2*self.bucket_size)), device=h_path_self.device).bool()
        # input_mask = torch.cat((input_mask, paddings_mask), dim=1)
        # h_path_self = self.path_selfattention(h_path_self, input_mask=input_mask)
        # h_path_self = h_path_self.squeeze(0)
        # h_path_self = h_path_self[:t]
        # h_path_self = self.norm_path_self(h_path_self+h_path_bag)
        # A_path_self, h_path_self = self.path_self_gate(h_path_self.squeeze(0))
        # A_path_self = torch.transpose(A_path_self, 1, 0)
        # h_path_self = torch.mm(F.softmax(A_path_self, dim=1) , h_path_self)
        # h_path_self = self.path_self_rho(h_path_self)

        # h_path_cross, _ = self.path_cross(h_path_bag.unsqueeze(0), h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0))
        # h_path_cross = self.norm_path_cross(h_path_cross+h_path_bag)
        # A_path_cross, h_path_cross = self.path_cross_gate(h_path_cross.squeeze(0))
        # A_path_cross = torch.transpose(A_path_cross, 1, 0)
        # h_path_cross = torch.mm(F.softmax(A_path_cross, dim=1) , h_path_cross)
        # h_path_cross = self.path_cross_rho(h_path_cross)
        
        # ### Omic
        h_omic_self, _ = self.omic_selfattention(h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0), h_omic_bag.unsqueeze(0))
        h_omic_self = self.norm_omic_self(h_omic_self+h_omic_bag)
        # A_omic_self, h_omic_self = self.omic_self_gate(h_omic_self.squeeze(0))
        # A_omic_self = torch.transpose(A_omic_self, 1, 0)
        # h_omic_self = torch.mm(F.softmax(A_omic_self, dim=1) , h_omic_self)
        # h_omic_self = self.omic_self_rho(h_omic_self)

        # h_omic_cross, _ = self.omic_cross(h_omic_bag.unsqueeze(0), h_path_bag.unsqueeze(0), h_path_bag.unsqueeze(0))
        # h_omic_cross = self.norm_omic_cross(h_omic_cross+h_omic_bag)
        # A_omic_cross, h_omic_cross = self.omic_cross_gate(h_omic_cross.squeeze())
        # A_omic_cross = torch.transpose(A_omic_cross, 1, 0)
        # h_omic_cross = torch.mm(F.softmax(A_omic_cross, dim=1) , h_omic_cross)
        # h_omic_cross = self.omic_cross_rho(h_omic_cross)
        # h_omic_trans = self.omic_transformer(h_omic_bag)
        # A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        # A_omic = torch.transpose(A_omic, 1, 0)
        # h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        # h_omic = self.omic_rho(h_omic).squeeze()
        
        if self.fusion == 'bilinear':
            pass
            # h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = h_omic_self
            h = self.mm(h.view(-1))
            # h = self.mm(torch.cat([h_path, h_omic], axis=0))
        elif self.fusion == "lmf":
            h = self.mm([h_path_self, h_path_cross, h_omic_self, h_omic_cross])

        # MMGF
        # h = torch.cat([h_path_self, h_path_cross, h_omic_self, h_omic_cross], axis=0)
        # A, h = self.fusion(h)
        # A = torch.transpose(A, 1, 0)
        # h = torch.mm(F.softmax(A, dim=1) , h)
        # h = self.fusion_rho(h).squeeze()
                
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        # print(logits)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        
        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        # attention_scores = {'path_self': A_path_self, 'path_cross': A_path_cross, 'omic_self': A_omic_self, 'omic_cross': A_omic_cross}
        # attention_scores = {}
        
        return logits
    

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, **kwargs):

        h = kwargs['x_path'].unsqueeze(0) #[B, n, 1024]
        # print(h.shape)
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits