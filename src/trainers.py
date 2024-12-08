import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from modules import NCELoss, PCLoss
from utils import recall_at_k, ndcg_k, nCr
from datasets import RecWithContrastiveLearningDataset
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@20": "{:.4f}".format(recall[1]),
            "NDCG@20": "{:.4f}".format(ndcg[1]),
        }
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  
        pos_logits = torch.sum(pos * seq_emb, -1)  
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss
    


class ELCRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, args):
        super(ELCRecTrainer, self).__init__(
            model, train_dataloader, args
        )

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
            cl_batch = torch.cat(inputs, dim=0)
            cl_batch = cl_batch.to(self.device)
            cl_sequence_output = self.model(cl_batch)
            bz = cl_sequence_output.shape[0]
            seq = cl_sequence_output.shape[1]
            clu_num = self.model.cluster_centers.shape[0]
            xx = (cl_sequence_output * cl_sequence_output).sum(-1).reshape(bz, seq, 1).repeat(1, 1, clu_num)
            cc = (self.model.cluster_centers * self.model.cluster_centers).sum(-1).reshape(1, 1, clu_num).repeat(bz, seq, 1)
            xc = torch.matmul(cl_sequence_output, self.model.cluster_centers.T)
            dis = xx + cc - 2 * xc
            index = torch.argmin(dis, dim=-1)
            shift = self.model.cluster_centers[index]

            if self.args.prototype == "shift":
                cl_sequence_output += shift

            elif self.args.prototype == "concat":
                cl_sequence_output = torch.concat([cl_sequence_output, shift], dim=-1)

            if self.args.seq_representation_instancecl_type == "mean":
                cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
            cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
            batch_size = cl_batch.shape[0] // 2
            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

            if self.args.de_noise:
                cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
            else:
                cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
            return cl_loss
    
    @ staticmethod
    def distance(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        dis = xx_cc - 2 * xc
        return dis
    
    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)

        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)

        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss


    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        if train:

            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            rec_cf_data_iter = enumerate(dataloader)

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                sequence_output = self.model(input_ids)

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                cl_losses = []
                sample_distance_losses = []
                for cl_batch in cl_batches:
                    if self.args.contrast_type == "InstanceCL":
                        cl_loss = self._instance_cl_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches
                        )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":

                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()


                            for cluster in self.model.cluster_centers:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss)
                        else:
                            continue
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)

                        else:

                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )

                            cl_losses.append(self.args.cf_weight * cl_loss1)

                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = F.normalize(sequence_output, p=2, dim=1)

                            seq2intents = []
                            intent_ids = []

                            self.model.cluster_centers.data = F.normalize(self.model.cluster_centers.data, p=2, dim=1)
                            center = self.model.cluster_centers

                            sample_center_distance = self.distance(sequence_output, center)
                            index = torch.argmin(sample_center_distance, dim=-1)
                            sample_distance_loss = sample_center_distance.mean()

                            center_center_distance = self.distance(center, center)
                            center_center_distance.flatten()[:-1].view(center.shape[0] - 1, center.shape[0] + 1)[:, 1:].flatten()
                            center_distance_loss = -center_center_distance.mean()

                            sample_distance_losses.append(self.args.trade_off*(sample_distance_loss+center_distance_loss))

                            seq2intent = self.model.cluster_centers[index]

                            intent_ids.append(index)
                            seq2intents.append(seq2intent)

                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )

                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += cl_loss

                for dis_loss in sample_distance_losses:
                    joint_loss += dis_loss


                self.optim.zero_grad()
                joint_loss.backward(retain_graph=True)
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()


                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(dataloader)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(dataloader)),
            }

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

            return rec_avg_loss / len(dataloader), joint_avg_loss / len(dataloader)

        else:

            rec_data_iter = enumerate(dataloader)
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
