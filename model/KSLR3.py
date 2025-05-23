import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings

class KGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations, llm_emb, A_in=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.llm_dim = llm_emb.shape[1]

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.register_buffer('llm_emb', llm_emb.float())

        checkpoint = torch.load('trained_model/KGAT/amazon-book/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain_model1/model_epoch74.pth', map_location='cpu')
        
        self.adapter = nn.Sequential(
            nn.Linear(self.llm_dim, min(256,(self.llm_dim + self.embed_dim) // 2)),
            nn.LeakyReLU(),
            nn.Linear(min(256,(self.llm_dim + self.embed_dim) // 2) , self.embed_dim)
        )
        self.user_embed = nn.Embedding(n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.user_embed.weight)   
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        adapter_state = {k.replace('adapter.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('adapter.')}
        self.adapter.load_state_dict(adapter_state)
        # self.user_embed.load_state_dict(user_embed_state)
        # self.relation_embed.load_state_dict(relation_embed_state)
        # self.trans_M.data.copy_(trans_M_state)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))
        # for k in range(self.n_layers):
        #     layer_state = {
        #         key.replace(f'aggregator_layers.{k}.', ''): value
        #         for key, value in checkpoint['model_state_dict'].items()
        #         if key.startswith(f'aggregator_layers.{k}.')
        #     }
        #     self.aggregator_layers[k].load_state_dict(layer_state)

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        # A_in.data = checkpoint['model_state_dict']['A_in']

        # 复制一份
        self.llm_user_embed = nn.Embedding(n_users, self.embed_dim)
        self.llm_relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.llm_trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        user_embed_state = {k.replace('user_embed.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('user_embed.')}
        relation_embed_state = {k.replace('relation_embed.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('relation_embed.')
        }
        trans_M_state = checkpoint['model_state_dict']['trans_M']

        self.llm_user_embed.load_state_dict(user_embed_state)  # 复制user_embed参数
        self.llm_relation_embed.load_state_dict(relation_embed_state)  # 复制relation_embed参数
        self.llm_trans_M.data.copy_(trans_M_state)

        self.llm_aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.llm_aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type)
            )
        for k in range(self.n_layers):
            # 原始模块加载
            layer_state = {
                key.replace(f'aggregator_layers.{k}.', ''): value
                for key, value in checkpoint['model_state_dict'].items()
                if key.startswith(f'aggregator_layers.{k}.')
            }
            self.llm_aggregator_layers[k].load_state_dict(layer_state)

        self.llm_A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        self.llm_A_in.requires_grad = False  # 保持与原始一致
        self.llm_A_in.data = checkpoint['model_state_dict']['A_in']

        # self.llm_entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        # with torch.no_grad():
        #     llm_entity_embed = self.adapter(self.llm_emb)
        # llm_user_emb = self.llm_user_embed.weight
        # llm_entity_user_embed =  torch.cat([llm_entity_embed, llm_user_emb], dim=0)
        # self.llm_entity_user_embed.weight = nn.Parameter(llm_entity_user_embed)
    
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
    
    def _init_adapter_weights(self):
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def get_llm_entity_embeddings(self):
        return self.adapter(self.llm_emb)
    
    def llm_entity_user_embed(self):
        llm_entity_emb = self.get_llm_entity_embeddings()  # (n_entities, embed_dim)
        llm_user_emb = self.llm_user_embed.weight          # (n_users, embed_dim)
        return torch.cat([llm_entity_emb, llm_user_emb], dim=0)

    def calc_llm_embeddings(self):
        ego_embed = self.llm_entity_user_embed()
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.llm_A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed

    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed

    def info_nce_loss(self, embed_1, embed_2, temperature=0.1):
        """
        计算 InfoNCE 对比损失（NT-Xent loss）
        
        Args:
            embed_1: (batch_size, embed_dim)  第一组嵌入（如 ID 嵌入）
            embed_2: (batch_size, embed_dim)  第二组嵌入（如 LLM 嵌入）
            temperature: float, 温度系数（默认 0.1）
            
        Returns:
            nce_loss: 对比损失
        """
        batch_size = embed_1.size(0)
        
        # 计算余弦相似度（归一化后点积）
        embed_1 = F.normalize(embed_1, dim=1)  # (batch_size, embed_dim)
        embed_2 = F.normalize(embed_2, dim=1)  # (batch_size, embed_dim)
        
        # 正样本相似度（对角线元素）
        sim_pos = torch.sum(embed_1 * embed_2, dim=1) / temperature  # (batch_size)
        
        # 负样本相似度（矩阵乘法）
        sim_neg = torch.mm(embed_1, embed_2.T) / temperature  # (batch_size, batch_size)
        
        # 排除自身对比（对角线置为 -inf）
        mask = torch.eye(batch_size, dtype=torch.bool, device=embed_1.device)
        sim_neg = sim_neg.masked_fill(mask, -float('inf'))
        
        # 计算 InfoNCE loss
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (batch_size, 1 + batch_size)
        labels = torch.zeros(batch_size, dtype=torch.long, device=embed_1.device)  # 正样本在 0 位置
        nce_loss = F.cross_entropy(logits, labels)
        
        return nce_loss

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()                       # (n_users + n_entities, concat_dim)
        llm_all_embed = self.calc_llm_embeddings()  

        user_embed = all_embed[user_ids]                            # (cf_batch_size, concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, concat_dim)

        llm_user_embed = llm_all_embed[user_ids]                            # (cf_batch_size, concat_dim)
        llm_item_pos_embed = llm_all_embed[item_pos_ids]                    # (cf_batch_size, concat_dim)
        llm_item_neg_embed = llm_all_embed[item_neg_ids]                    # (cf_batch_size, concat_dim)

        ave_user_embed = (user_embed + llm_user_embed) / 2
        ave_item_pos_embed = (item_pos_embed + llm_item_pos_embed) / 2
        ave_item_neg_embed = (item_neg_embed + llm_item_neg_embed) / 2

        # Equation (12)
        pos_score = torch.sum(ave_user_embed * ave_item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(ave_user_embed * ave_item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        user_contrastive_loss = self.info_nce_loss(user_embed, llm_user_embed)
        item_pos_contrastive_loss = self.info_nce_loss(item_pos_embed, llm_item_pos_embed)
        item_neg_contrastive_loss = self.info_nce_loss(item_neg_embed, llm_item_neg_embed)
        contrastive_loss = (user_contrastive_loss + item_pos_contrastive_loss + item_neg_contrastive_loss) / 3

        l2_loss = _L2_loss_mean(ave_user_embed) + _L2_loss_mean(ave_item_pos_embed) + _L2_loss_mean(ave_item_neg_embed)

        loss = cf_loss + self.cf_l2loss_lambda * l2_loss + 0.02 * contrastive_loss
        return cf_loss, self.cf_l2loss_lambda * l2_loss, 0.02 * contrastive_loss, loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def llm_calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.llm_relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.llm_trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.llm_entity_user_embed()[h]                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.llm_entity_user_embed()[pos_t]                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.llm_entity_user_embed()[neg_t]                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss



    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed(h_list)
        t_embed = self.entity_user_embed(t_list)

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def llm_update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.llm_relation_embed.weight[r_idx]
        W_r = self.llm_trans_M[r_idx]

        h_embed = self.llm_entity_user_embed()[h_list]
        t_embed = self.llm_entity_user_embed()[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def llm_update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.llm_update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        llm_A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        llm_A_in = torch.sparse.softmax(llm_A_in.cpu(), dim=1)
        self.llm_A_in.data = llm_A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        llm_all_embed = self.calc_llm_embeddings()  
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)
        llm_user_embed = llm_all_embed[user_ids]        # (n_users, concat_dim)
        llm_item_embed = llm_all_embed[item_ids]        # (n_items, concat_dim)
        ave_user_embed = (user_embed + llm_user_embed) / 2
        ave_item_embed = (item_embed + llm_item_embed) / 2

        # Equation (12)
        cf_score = torch.matmul(ave_user_embed, ave_item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return (self.calc_kg_loss(*input) + self.llm_calc_kg_loss(*input))/2
        if mode == 'update_att':
            self.update_attention(*input)
            self.llm_update_attention(*input)
            return None
        if mode == 'predict':
            return self.calc_score(*input)

