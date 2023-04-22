import torch
import torch.nn as nn
import numpy as np
import random
import copy
from sparsemax import Sparsemax

class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv = nn.Conv1d(dim, dim, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(dim)
        # self.bn = nn.LayerNorm(dim, eps=1e-5)
        self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        out = self.activation(out)
        return out

class CNN1D(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(CNN1D, self).__init__()

        layers = [ResidualBlock(dim) for i in range(num_layers)]
        

        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.conv_layers(x)
        out = self.linear(x)
        return out


class FFN(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(FFN, self).__init__()

        layers = []
        for i in range(num_layers):
                layers += [nn.Linear(dim, dim)]
            layers += [nn.LeakyReLU(inplace=True)]
            
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=225):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HRE(nn.Module):
    def __init__(self, args = None):
        super(HRE, self).__init__()
        self.d_model = args.d_model
        self.maxsize = args.truncation_length
        self.position = args.position_embedding
        self.class_num = args.class_num
        self.superclass_num = args.superclass_num
        self.fineclass_num = args.fineclass_num
        
        self.Encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=1024, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.Encoder_layer, num_layers=6)
        self.cnn1d = CNN1D(self.d_model)
        self.ffn = FFN(self.d_model)
        self.sparsemax = Sparsemax()
        self.emb = nn.Embedding(num_embeddings=1, embedding_dim=self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model,max_len = self.maxsize+1)
        self.linear = nn.Linear(in_features=1, out_features=self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model, eps=1e-5)
        
        self.fc_k_coarse = nn.Linear(self.d_model, 32)
        self.fc_q_coarse = nn.Linear(self.d_model, 32)
        
        self.fc_k_fine = nn.Linear(self.d_model, 32)
        self.fc_q_fine = nn.Linear(self.d_model, 32)
        
        
        
        self.pred = nn.Sequential(
                nn.Linear(self.d_model, self.d_model//4),
                nn.BatchNorm1d(self.d_model//4),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.d_model//4, self.d_model)
            )
        self.coarse_classifier = nn.Linear(self.d_model, self.superclass_num)
        self.global_classifier = nn.Linear(self.d_model, self.class_num)
        self.global_classifier2 = nn.Linear(self.d_model*2, self.class_num)
        self.fine_classifier = nn.Linear(self.d_model, self.fineclass_num*self.superclass_num)

        
    def move_chain(self, text, alpha, tau):
        start_val = 17
        end_val = 18
        start_indices = (text == start_val).nonzero(as_tuple=True)[0]
        end_indices = (text == end_val).nonzero(as_tuple=True)[0]

        pairs = [(start, end) for start in start_indices for end in end_indices if start < end]
        valid_pairs = []
        for pair in pairs:
            pair_fragment = text[pair[0]:pair[1]+1]
            if (pair_fragment == start_val).sum() == (pair_fragment == end_val).sum():
                valid_pairs.append(pair)

        if valid_pairs:
            start_idx, end_idx = random.choice(valid_pairs)
        else:
            start_idx, end_idx = -1, -1

        if start_idx == -1 and end_idx == -1:
            fragment_len = random.randint(1, len(text))
            start_idx = random.randint(0, len(text) - fragment_len)
            end_idx = start_idx + fragment_len - 1

        fragment = text[start_idx:end_idx+1]
        uniform = torch.rand_like(alpha)
        alpha = torch.log(F.softmax(alpha))
        gumbel = -torch.log(-torch.log(uniform))
        pi = (logits + gumbel) / tau
        insert_idx = pi.argmax()+1
        
        text = torch.cat((text[:insert_idx], fragment, text[insert_idx:]))
        
        return text
    
    def data_augmentation(self, text, alpha, granularity, tau, mask, superclass=None):
        device = torch.device(torch.get_device(text))
        if granularity == 0:
            unique, counts = torch.unique(superclass, return_counts=True)
            duplicate_indices = torch.where(counts > 1)[0]
            for idx in duplicate_indices:
                idx_list = (superclass == superclass[0]).nonzero(as_tuple=True)[0].tolist()
                 if random.random() < 0.1:
                    org_idx_list = copy.copy(idx_list)
                    random.shuffle(idx_list)
                    text[org_idx_list] = text[idx_list]
                    mask[org_idx_list] = mask[idx_list]
                    # alpha[org_idx_list] = alpha[idx_list]

        text_list = []
        for idx in range(text.size()[0]):
            mask_num = self.maxsize - torch.sum(mask[idx])
            pad_text = text[mask_num:]
            text_temp = move_chain(text[:mask_num], alpha[:mask_num], tau)
            text_temp = torch.cat([text_temp,pad_text],dim=0)
            text_list.append(text_temp.unsqueeze(0))
        text = torch.cat(text_list,dim=0)
        return text, mask

        
    def forward(
            self,
            text,
            superclass=None,
            alpha = None,
            tau = 1
    ):
        device = torch.device(torch.get_device(text))
        bs = text.size()[0]
        eps = 1e-8
        
        padding = torch.all(text.eq(0).to(device), dim=-1)
        mask_count = torch.sum(1-1.*padding,1)
        src_key_padding_mask = torch.cat((torch.zeros(padding.shape[0], 1, dtype=torch.bool).to(device), padding), dim=-1)
        
        if alpha:
            # bs = bs*2
            text_coarse,src_key_padding_mask = data_augmentation(text, 1-alpha[0], 0, src_key_padding_mask_coarse, superclass = superclass) 
            text_fine,src_key_padding_mask_fine = data_augmentation(text, 1-alpha[1]-alpha[0], 1, src_key_padding_mask) 
            text = torch.cat([text_coarse,text_fine],dim=0)
            src_key_padding_mask = torch.cat([src_key_padding_mask_coarse,src_key_padding_mask_fine], dim=0)
        clsnode = self.emb(torch.zeros(1, dtype=int).to(device)) 
        mnode = self.linear(text)
        mnode = torch.cat((clsnode.unsqueeze(0).expand(mnode.shape[0], 1, self.d_model), mnode), dim=1) 
        if self.position:
            mnode = self.pos_emb(mnode.permute(1,0,2))
            mnode = mnode.permute(1,0,2)
        mnode = self.transformer_encoder(src=mnode,
                                         src_key_padding_mask=src_key_padding_mask)
        mnode =  self.norm1(mnode)
        clsnode = mnode[:, 0]
        mnode = self.cnn1d(mnode[:, 1:].permute(0,2,1))
        mnode = mnode.permute(0,2,1) # bs, t, dim
        mnode = mnode*(1.-1.*src_key_padding_mask.unsqueeze(1).repeat(1,self.d_model,1))
        
        if alpha:
            mnode_k = self.fc_k_coarse(mnode[:bs])
            clsnode = clsnode[:bs]
            src_key_padding_mask = src_key_padding_mask[:bs]
        else:
            mnode_k = self.fc_k_coarse(mnode)
        clsnode = clsnode.unsqueeze(2)
        clsnode_q = self.fc_q_coarse(clsnode) # bs, dim, 1
        pi = torch.bmm(mnode_k, clsnode_q).squeeze() # bs, t
        pi = pi-1.*src_key_padding_mask*1e8
        alpha_coarse = self.sparsemax(pi) # bs, t
        
        if alpha:
            mnode_k = self.fc_k_fine(mnode[bs:])
            clsnode = clsnode[bs:]
            src_key_padding_mask = src_key_padding_mask[bs:]
        else:
            mnode_k = self.fc_k_fine(mnode)
        clsnode_q = self.fc_q_fine(clsnode) # bs, dim, 1
        pi = torch.bmm(mnode_k, clsnode_q).squeeze()
        pi = pi * (1.-1.*alpha_coarse.bool())
        pi = pi * (1.-1.*src_key_padding_mask)
        pi[pi==0] = -1e8
        alpha_fine = F.softmax(pi)
        
        alpha = [alpha_coarse, alpha_fine]
        if alpha:
          z_coarse = torch.bmm(alpha_coarse.unsqueeze(1),mnode[:bs]).squeeze()
          z_fine = torch.bmm(alpha_fine.unsqueeze(1),mnode[bs:]).squeeze()
        
        z = [z_coarse, z_fine]
        
        p_coarse = self.pred(z_coarse)
        p_fine = self.pred(z_fine)
        p = [p_coarse, p_fine]
        
        
        cls_coarse = self.coarse_classifier(z_coarse)
        cls_fine = self.fine_classifier(z_fine).reshape(bs,self.superclass_num,self.fineclass_num)
        cls_fine = torch.bmm(F.softmax(cls_coarse).unsqueeze(1),cls_fine).squeeze()
        if alpha:
            cls_global = self.global_classifier(clsnode.squeeze()[bs:])
        else:
            cls_global = self.global_classifier(clsnode.squeeze())
        cls_global2 = self.global_classifier2(torch.cat([z_coarse,z_fine],dim=-1))
        
        cls = [cls_global,cls_coarse,cls_fine, cls_global2]
        
        return cls, z, p, alpha
