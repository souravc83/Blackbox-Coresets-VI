# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
import torch.nn.functional as F
from psvi.models.neural_net import VILinear, categorical_fn
from psvi.submodular import FacilityLocation, submodular_optimizer
from psvi.submodular.euclidean import euclidean_dist_pair_np
from psvi.submodular.cossim import cossim_pair_np

from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import defaultdict
import numpy as np
import time
from typing import List
import random
import os
import json 
import re
from psvi.experiments.experiments_utils import set_up_model
from tqdm import tqdm
import faiss
import pickle
import pandas as pd


def pseudo_subsample_init(x, y, num_pseudo=20, nc=2, seed=0):
    r"""
    Initialize on random subsets from each class with approximately equal
    """
    torch.manual_seed(seed)
    N, _ = x.shape
    cnt = 0
    u, z = torch.Tensor([]), torch.Tensor([])
    for c in range(nc):
        idx_c, pts_with_c = (
            torch.arange(N)[y == c],
            num_pseudo // nc if c < nc - 1 else num_pseudo - cnt,
        )
        u, z = torch.cat(
            (u, x[idx_c[torch.randperm(len(idx_c))[:pts_with_c]]])
        ), torch.cat((z, c * torch.ones(pts_with_c)))
        cnt += num_pseudo // nc
    return u.requires_grad_(True), z


def pseudo_rand_init(x, y, num_pseudo=20, nc=2, seed=0, variance=0.1):
    r"""
    Initialize on noisy means of the observed datapoints and random labels equally split among classes
    """
    torch.manual_seed(seed)
    _, D = x.shape
    u = (
        (x[:, :].mean() + variance * torch.randn(num_pseudo, D))
        .clone()
        .requires_grad_(True)
    )
    z = torch.Tensor([])
    for c in range(nc):
        z = torch.cat(
            (
                z,
                c
                * torch.ones(
                    num_pseudo // nc
                    if c < nc - 1
                    else num_pseudo - (nc - 1) * (num_pseudo // nc)
                ),
            )
        )
    return u, z


r"""
Model specific computations for psvi variational objective used to estimate the coreset posterior over black-box sparsevi construction
"""


def elbo(net, u, z, w):
    r"""
    ELBO computed on (u,z): variational objective for posterior approximation using only the coreset datapoints
    """
    pseudo_nll = -dist.Bernoulli(logits=net(u).squeeze(-1)).log_prob(z).matmul(w)
    sampled_nkl = sum(m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear))
    return (pseudo_nll.sum() - sampled_nkl).sum()


def sparsevi_psvi_elbo(net, x, u, y, z, w, N):  # variational objective for
    r"""
    PSVI-ELBO: variational objective for true data conditioned on coreset data (called in outer optimization of the sparse-bbvi construction)
    """
    Nu, Nx = u.shape[0], x.shape[0]
    all_data, all_labels = torch.cat((u, x)), torch.cat((z, y))
    all_nlls = -dist.Bernoulli(logits=net(all_data).squeeze(-1)).log_prob(all_labels)
    pseudo_nll, data_nll = N / Nu * all_nlls[:, :Nu].matmul(w), all_nlls[:, Nu:].sum(-1)
    sampled_nkl = sum(m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear))
    log_weights = -pseudo_nll + sampled_nkl
    weights = log_weights.softmax(-1).squeeze()
    return weights.mul(N / Nx * data_nll - pseudo_nll).sum() - log_weights.mean()


def forward_through_coreset(net, u, x, z, y, w):
    r"""
    Likelihood computations for coreset next datapoint selection step
    """
    Nu = u.shape[0]
    with torch.no_grad():
        all_data, all_labels = torch.cat((u, x)), torch.cat((z, y))
        all_lls = dist.Bernoulli(logits=net(all_data).squeeze(-1)).log_prob(all_labels)
        core_ll, data_ll = all_lls[:, :Nu], all_lls[:, Nu:]
        sampled_nkl = sum(
            m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear)
        )
        log_weights = core_ll.matmul(w) + sampled_nkl
        weights = log_weights.softmax(-1).squeeze()
        return core_ll.T, data_ll.T, weights


def predict_through_coreset(net, xt, x, y, w=None):
    r"""
    Importance-weight correction for predictions using the coreset posterior
    """
    Ntest = xt.shape[0]
    with torch.no_grad():
        all_data = torch.cat((xt, x))
        all_logits = net(all_data).squeeze(-1)
        pnlls = -dist.Bernoulli(logits=all_logits[:, Ntest:]).log_prob(y)
        pseudo_nll = pnlls.matmul(w) if w is not None else pnlls.sum(-1)
        test_data_logits = all_logits[:, :Ntest]
        sampled_nkl = sum(
            m.sampled_nkl() for m in net.modules() if isinstance(m, VILinear)
        )
        log_weights = -pseudo_nll + sampled_nkl
        weights = log_weights.softmax(-1).squeeze()
        return test_data_logits, weights


def make_dataloader(data, minibatch, shuffle=True):
    r"""
    Create pytorch dataloader from given dataset and minibatch size
    """
    return DataLoader(data, batch_size=minibatch, pin_memory=True, shuffle=shuffle)


def compute_empirical_mean(dloader):
    r"""
    Compute the mean of the observed data distribution
    """
    trainsum, nb_samples = 0., 0. # compute statistics of the training data
    for data, _ in dloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        trainsum += data.mean(2).sum(0) # use with caution: might raise overflow for large datasets
        nb_samples += batch_samples
    return trainsum / nb_samples 


def pred_on_grid(
    model,
    n_test_per_dim=250,
    device=None,
    **kwargs,
):
    r"""
    Predictifons over a 2-d grid for visualization of predictive posterior on 2-d synthetic datasets
    """
    _x0_test = torch.linspace(-3, 4, n_test_per_dim)
    _x1_test = torch.linspace(-2, 3, n_test_per_dim)
    x_test = torch.stack(torch.meshgrid(_x0_test, _x1_test), dim=-1).to(device)

    with torch.no_grad():
        return model(x_test.view(-1, 2)).squeeze(-1).softmax(-1).mean(0)
    
def process_wt_index(log_core_idcs, log_core_wts):
    # these two lists should have the same length
    log_core_idcs = [[int(x) for x in y] for y in log_core_idcs]
    
    idc_wt_list = []
    
    for counter, idc_list in enumerate(log_core_idcs):
        this_wt = log_core_wts[counter]
        this_wt_dict = {}
        for index in idc_list:
            this_wt_dict[index] = float(this_wt[index])
        idc_wt_list.append(this_wt_dict)
    return idc_wt_list 


def get_wt_index(results_folder, dnm, mfvi_selection_method):
    #dnm = dnm.lower()
    
    contents = os.listdir(results_folder)
    search_str = f'mfvi_selection_{mfvi_selection_method}_{dnm}_2023_04_05'
    item_path = None 
    for item in contents:
        if search_str in item:
            item_path = os.path.join(results_folder, item, 'results.json')
            break
    
    if not item_path:
        raise ValueError(f" Folder {search_str} does not exist")
    
    with open(item_path, 'r') as f:
        this_dict = json.load(f)
    
    dict_1 = this_dict[dnm]['mfvi_selection']
    coreset_size = list(dict_1.keys())[0]
    wt_index = dict_1[coreset_size]['0']['wt_index']
    return wt_index

    
            
            
    
class MeanFieldVI():
    """
    same as run_mfvi, but puts it inside a class
    """
    def __init__(
        self,
        xt=None,
        yt=None,
        mc_samples=4,
        data_minibatch=128,
        num_epochs=100,
        log_every=10,
        N=None,
        D=None,
        lr0net=1e-3,  # initial learning rate for optimizer
        mul_fact=2,  # multiplicative factor for total number of gradient iterations in classical vi methods
        seed=0,
        distr_fn=categorical_fn,
        architecture=None,
        n_hidden=None,
        nc=2,
        log_pseudodata=False,
        train_dataset=None,
        test_dataset=None,
        init_sd=None,
        forgetting_score_flag=False,
        data_path=None,
        load_from_saved=False,
        dnm=None,
        **kwargs,
    ):
        self.mc_samples = mc_samples
        self.data_minibatch = data_minibatch
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.N = N
        self.D = D
        self.lr0net = lr0net
        self.seed = seed
        self.distr_fn = distr_fn
        self.architecture = architecture
        self.n_hidden = n_hidden
        self.nc = nc
        self.log_pseudodata = log_pseudodata
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.init_sd = init_sd
        self.mul_fact = mul_fact
        self.forgetting_score_flag = forgetting_score_flag
        self.data_path = data_path
        self.net_state_dict_fname = f'net_state_dict_{seed}.pt'
        self.forgetting_fname = f'forgetting_{seed}.pt'
        self.load_from_saved = load_from_saved
        self.dnm = dnm
        
    
    def train_an_epoch(self):
        for xbatch, ybatch in self.train_loader:
            xbatch, ybatch = xbatch.to(self.device, non_blocking=True), ybatch.to(
                self.device, non_blocking=True
            )
            
            self.optim_vi.zero_grad()
            scaling_factor = self.n_train / xbatch.shape[0]
            data_nll = -(
               scaling_factor
                * self.distr_fn(logits=self.net(xbatch).squeeze(-1)).log_prob(ybatch).sum()
            )
            kl = sum(m.kl() for m in self.net.modules() if isinstance(m, VILinear))
            mfvi_loss = data_nll + kl
            mfvi_loss.backward()
            self.optim_vi.step()
            
            self.elbos_mfvi.append(-mfvi_loss.item())


        
    def before_train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(self.seed), np.random.seed(self.seed), torch.manual_seed(self.seed)
        
        self.net = set_up_model(
            architecture=self.architecture, 
            D=self.D, 
            n_hidden=self.n_hidden, 
            nc=self.nc, 
            mc_samples=self.mc_samples, 
            init_sd=self.init_sd, 
        ).to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=False,
        )
        self.n_train = len(self.train_loader.dataset)
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.data_minibatch,
            pin_memory=True,
            shuffle=True,
        )

        self.optim_vi = torch.optim.Adam(self.net.parameters(), self.lr0net)
        self.total_iterations = self.mul_fact * self.num_epochs
        
        self.nlls_mfvi = []
        self.accs_mfvi = []
        self.times_mfvi = [0]
        self.elbos_mfvi = []
        self.t_start = time.time()
        
        # for forgetting scores
        self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(self.device)
        self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(self.device)
        self.never_learnt_events = torch.ones(self.n_train, requires_grad=False).to(self.device)


    
    def test(self):
        total, test_nll, corrects = 0, 0, 0
        for xt, yt in self.test_loader:
            xt, yt = xt.to(self.device, non_blocking=True), yt.to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                test_logits = self.net(xt).squeeze(-1).mean(0)
                corrects += test_logits.argmax(-1).float().eq(yt).float().sum()
                total += yt.size(0)
                test_nll += -self.distr_fn(logits=test_logits).log_prob(yt).sum()
        self.times_mfvi.append(self.times_mfvi[-1] + time.time() - self.t_start)
        self.nlls_mfvi.append((test_nll / float(total)).item())
        self.accs_mfvi.append((corrects / float(total)).item())
        print(f"predictive accuracy: {(100*self.accs_mfvi[-1]):.2f}%")
        
    
    def after_epoch(self):
        # if we are not calculating forgetting scores
        # then just return, do not do any work
        if not self.forgetting_score_flag:
            return 
        
        for i, (xbatch, ybatch) in enumerate(self.train_loader):
            xbatch, ybatch = xbatch.to(self.device, non_blocking=True), ybatch.to(
                self.device, non_blocking=True
            )
            
            batch_ind = list(range(
                             (i * self.data_minibatch), (min((i + 1) * self.data_minibatch, self.n_train))
                            ))
            
            
            with torch.no_grad():
                train_logits = self.net(xbatch).squeeze(-1).mean(0)
                curr_acc = train_logits.argmax(-1).float().eq(ybatch).float()
                forget_ind = torch.tensor(batch_ind)[self.last_acc[batch_ind] > curr_acc]
                self.forgetting_events[forget_ind] += 1
                self.last_acc[batch_ind] = curr_acc
                
                # not learnt, curr_acc = 0, 1-curr_acc = 1, 
                # learnt: curr_acc = 1, 1- curr_acc = 0
                self.never_learnt_events[batch_ind] = torch.min(
                    self.never_learnt_events[batch_ind],
                    1.0 - curr_acc
                )

    
    def run(self):
        
        self.before_train()
        
        if self.load_from_saved:
            load_succeeded = self.load() 
            if load_succeeded:
                return 

        
        for i in tqdm(range(self.total_iterations)):
            self.train_an_epoch()
            self.after_epoch()
            
            if i % self.log_every == 0 or i == self.total_iterations -1:
                self.test()
        
        # if forgetting combine never learnt and forgetting:
        if self.forgetting_score_flag:
            self.forgetting_events = torch.max(
                self.total_iterations * self.never_learnt_events,
                self.forgetting_events
            )
        
        self.save()
    
    def _get_net_fname(self):
        net_state_dict_fname = \
            f'net_state_dict_{self.dnm}_{self.architecture}_{self.num_epochs}_{self.seed}.pt'
        net_full_fname = os.path.join(self.data_path, net_state_dict_fname)
        return net_full_fname
    
    def _get_forgetting_fname(self):
        forgetting_fname = \
            f'forgetting_{self.dnm}_{self.architecture}_{self.num_epochs}_{self.seed}.pt'
        forgetting_full_fname = os.path.join(self.data_path, forgetting_fname)
        return forgetting_full_fname
        
    
    def save(self):
        net_full_fname = self._get_net_fname()
        forgetting_full_fname = self._get_forgetting_fname()
        
        torch.save(self.net.state_dict(), net_full_fname)
        torch.save(self.forgetting_events, forgetting_full_fname)
    
    def load(self):
        net_full_fname = self._get_net_fname()
        forgetting_full_fname = self._get_forgetting_fname()
        if not (os.path.exists(net_full_fname) and os.path.exists(forgetting_full_fname)):
            print(f"Net file name {net_full_fname} does not exist, running pretrain")
            return False
        
        # map to device, could have been saved by GPU, but get loaded from CPU
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        self.forgetting_events = torch.load(forgetting_full_fname,  map_location=map_location)
        self.net.load_state_dict(torch.load(net_full_fname, map_location=map_location))
        return True

                
                
    
class KmeansCluster():
    def __init__(self, x, y, num_classes=2, 
                 balance=True,  seed=None, dist="euclidean"):
        self.x = x 
        self.y = y 
        self.balance = balance
        self.num_classes = num_classes
        valid_dist = ["cosine", "euclidean"]
        
        if dist not in valid_dist:
            raise ValueError(f"{dist} is not one ov valid dist: {valid_dist}")
        self.dist = dist 
        
        self.kmeans_dict = []
        self.cluster_centers = []
        if seed is None:
            self.seed = time.time() 
        else:
            self.seed = seed
    
    def set_num_clusters(self, num_clusters):
        self.num_clusters = num_clusters
        self.pts_per_class = int(np.floor(num_clusters / self.num_classes))
        if self.pts_per_class < 2:
            self.pts_per_class = 2
    
    def run_kmeans(self):
        if self.balance:
            self.run_kmeans_balanced()
        else:
            self.run_kmeans_unbalanced()
        
    
    def run_kmeans_balanced(self):
        # reset
        self.kmeans_dict = []
        self.cluster_centers = []
        
        for class_index in range(self.num_classes):
            indices = np.where(self.y.int().numpy() == class_index)[0]
            this_x = self.x.numpy()[indices]
            # reshape MNIST 
            if this_x.ndim == 3:
                this_x = this_x.reshape(-1, this_x.shape[1] * this_x.shape[2])
            
            if self.dist == "cosine":
                this_x = preprocessing.normalize(this_x)
            
            kmeans = KMeans(
                n_clusters=self.pts_per_class, 
                random_state=self.seed, n_init="auto").fit(this_x)
            this_kmeans_dict = defaultdict(list)
            for counter, label in enumerate(kmeans.labels_):
                this_kmeans_dict[label].append(indices[counter])
            
            self.kmeans_dict.append(this_kmeans_dict)
            self.cluster_centers.append(kmeans.cluster_centers_)
            
    def run_kmeans_unbalanced(self):
        self.kmeans_dict = []
        self.cluster_centers = []
        this_x = self.x.numpy()
        # reshape MNIST 
        if this_x.ndim == 3:
            this_x = this_x.reshape(-1, this_x.shape[1] * this_x.shape[2])

        if self.dist == "cosine":
            this_x = preprocessing.normalize(this_x)
        
        kmeans = KMeans(
            n_clusters=self.num_clusters , 
            random_state=self.seed, n_init="auto").fit(this_x)
        
        
        this_kmeans_dict = defaultdict(list)
        for counter, label in enumerate(kmeans.labels_):
            this_kmeans_dict[label].append(counter)
        
        self.kmeans_dict.append(this_kmeans_dict)
        self.cluster_centers.append(kmeans.cluster_centers_)        

                
    def get_arbitrary_pts(self, total_pts):
        pts_per_cluster = [total_pts // self.num_clusters] * self.num_clusters
        pts_per_cluster[-1] = total_pts - sum(pts_per_cluster[:-1])
        core_idcs = []
        counter = 0
        for this_kmeans_dict in self.kmeans_dict:
            for k in this_kmeans_dict.keys():
                if len(this_kmeans_dict[k]) > 0: 
                    num_pts = pts_per_cluster[counter]
                    counter += 1
                    pts_this_cluster = np.random.choice(this_kmeans_dict[k], num_pts, replace=False)
                    print(f"Points from this cluster: {pts_this_cluster}")
                    
                    core_idcs = core_idcs + list(pts_this_cluster)
        
        return core_idcs
    
    
    
class KmeansFaiss(KmeansCluster):
    def __init__(self, x, y, num_classes=2, 
                 balance=True,  seed=None, dist="euclidean"):
        
        super().__init__(x,y,num_classes, balance, seed, dist)
    
    def run_kmeans_balanced(self):
        self.kmeans_dict = []
        self.cluster_centers = []
        
        for class_index in range(self.num_classes):
            indices = np.where(self.y.int().numpy() == class_index)[0]
            this_x = self.x.numpy()[indices]
            # reshape MNIST 
            if this_x.ndim == 3:
                this_x = this_x.reshape(-1, this_x.shape[1] * this_x.shape[2])
            
            if self.dist == "cosine":
                this_x = preprocessing.normalize(this_x)
            data_dim = this_x.shape[1]
            kmeans = faiss.Kmeans(
                data_dim, self.pts_per_class, niter=20, verbose=False, gpu=False
            )
            kmeans.train(this_x)
            this_index = faiss.IndexFlatL2 (data_dim)
            this_index.add(this_x)
            
            D, I = this_index.search (kmeans.centroids, 1)  
            del D
            center_list = [x[0] for x in I.tolist()]
            
            self.cluster_centers += center_list
        print(self.cluster_centers) 
    
    def run_kmeans_unbalanced(self):
        self.cluster_centers = []
        this_x = self.x.numpy()
        if this_x.ndim == 3:
            this_x = this_x.reshape(-1, this_x.shape[1] * this_x.shape[2])

        if self.dist == "cosine":
            this_x = preprocessing.normalize(this_x)
        data_dim = this_x.shape[1]
        kmeans = faiss.Kmeans(
            data_dim, self.pts_per_class, niter=20, verbose=False, gpu=False
        )
        kmeans.train(this_x)
        this_index = faiss.IndexFlatL2 (data_dim)
        this_index.add(this_x)

        D, I = index.search (kmeans.centroids, 1)  
        del D
        center_list = [x[0] for x in I.tolist()]
        self.cluster_centers = center_list
    
    def get_arbitrary_pts(self):
        return self.cluster_centers


    
class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        return idx, data, target, self.weights[idx]


    
class Selection():
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False):
        self.train_dataset = train_dataset
        self.num_pseudo = num_pseudo
        self.nc = nc
        self.seed = seed
        self.forgetting_flag = forgetting_flag
        self.core_idc = []
        self.wt_vec = None
        self.chosen_dataset = None
        
        # required for El2N score calculation 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select(self) -> List[int]:
        """
        get the selected indices
        """
        raise NotImplementedError("Child class must implement selection")
    
    def get_subset(self): 
        """
        get the subset of the torch dataset
        """
        
        self.core_idc = self.select()
        self.chosen_dataset = Subset(self.train_dataset, self.core_idc)
        return self.chosen_dataset 
    
    def get_weighted_subset(self, wt_vec=None):

        # if we have not run select yet, then run select
        if len(self.core_idc) == 0:
            self.core_idc = self.select()
            self.core_idc = list(
                np.random.permutation(self.core_idc)
            )
                
        # define the weights
        if not self.wt_vec:
            n_coreset = len(self.core_idc)
            n_train = len(self.train_dataset)
            scaling_factor = n_train / n_coreset
            self.wt_vec = scaling_factor * torch.ones(n_coreset)
        
        
        self.chosen_dataset = WeightedSubset(
            dataset=self.train_dataset,
            indices=self.core_idc,
            weights=self.wt_vec
        )
        
        return self.chosen_dataset

    def pretrain(
        self,
        test_dataset,
        architecture,
        D,
        n_hidden,
        distr_fn,
        mc_samples,
        init_sd,
        data_minibatch,
        pretrain_epochs,
        lr0net, 
        log_every, 
        data_folder,
        load_from_saved,
        dnm):
        """
        A number of models require pretraining the net.
        This uses MFVI to pretrain the net on the training dataset
        """
        self.pretrained_vi = MeanFieldVI(

            mc_samples=mc_samples,
            data_minibatch=data_minibatch,
            num_epochs=pretrain_epochs,
            log_every=log_every,
            N=len(self.train_dataset),
            D=D,
            lr0net=lr0net,  # initial learning rate for optimizer
            mul_fact=2,  # multiplicative factor for total number of gradient iterations in classical vi methods
            seed=self.seed,
            distr_fn=categorical_fn,
            architecture=architecture,
            n_hidden=n_hidden,
            nc=self.nc,
            train_dataset=self.train_dataset,
            test_dataset=test_dataset,
            init_sd=init_sd,
            forgetting_flag=self.forgetting_flag,
            data_path=data_folder,
            load_from_saved=load_from_saved,
            dnm=dnm
        )
        
        self.pretrained_vi.run()
        
        self.pretrained_net = self.pretrained_vi.net


def sample_multinomial(pval, k):
    if type(pval) == list:
        pval = np.array(pval)
        
    N = pval.shape[0]
    samples = np.random.multinomial(2*N, pval, size=1)[0]
    idx_sorted = np.argsort(samples)
    print(idx_sorted[-k:])
    
    return idx_sorted[-k:]
    

class RandomSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed):
        super().__init__(train_dataset, num_pseudo,  nc, seed)

    def select(self):
        n_train = len(self.train_dataset)
        pts_per_class = self.num_pseudo//self.nc
        core_idc = []
        
        pts_in_last_class = self.num_pseudo - (self.nc - 1) * pts_per_class
        for c in range(self.nc):
            idx_c = np.arange(n_train)[self.train_dataset.targets == c]
            if c == (self.nc - 1):
                num_pts = pts_in_last_class
            else:
                num_pts = pts_per_class
            chosen_idx = np.random.choice(idx_c, num_pts, replace=False)
            core_idc = core_idc + chosen_idx.tolist()
        
        return core_idc
    
    def pretrain(
        self,
        test_dataset,
        architecture,
        D,
        n_hidden,
        distr_fn,
        mc_samples,
        init_sd,
        data_minibatch,
        pretrain_epochs,
        lr0net, 
        log_every,
        data_folder,
        load_from_saved,
        dnm):
            pass 
    
            
class KmeansSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, embedding_flag=False, dist="euclidean", data_folder=None, dnm='MNIST'):
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag)
        self.embedding_flag = embedding_flag
        self.dist = dist 
        self.data_folder = data_folder
        self.dnm = dnm
    
    def select(self):
        return self._run_kmeans_loaded()
    
    def _run_kmeans(self):
        
        train_y = self.train_dataset.targets
        n_train = len(self.train_dataset)
        
        
        if self.embedding_flag:
            # for now this is hardcoded to lenet value
            # later, we might need to provide this with architecture
            embedding_size = 84
            embeddings = torch.zeros(n_train, embedding_size, requires_grad=False).to(self.device)
            data_minibatch = 128
            kmeans_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
            last_layer = None
            
            pretrained_net = self.pretrained_net
            pretrained_net.eval()
            
        
            with torch.no_grad():
                for i, (data, target) in enumerate(kmeans_dataloader):
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    batch_ind = list(range(
                        (i * data_minibatch), (min((i + 1) * data_minibatch, n_train))
                    ))

                    x = data
                    for layer in pretrained_net:
                        last_layer = x
                        x = layer(x)

                    embeddings[batch_ind] = last_layer.sum(0).squeeze(0)
        
            train_x = embeddings.detach().cpu()
        else:
            train_x = self.train_dataset.data
        
        kmeans_cluster = KmeansCluster(
            x=train_x, y=train_y, num_classes=self.nc, 
            seed=self.seed, dist=self.dist
        )
        num_clusters = self._set_num_clusters()
        
        kmeans_cluster.set_num_clusters(num_clusters)
        kmeans_cluster.run_kmeans()
        core_idc = kmeans_cluster.get_arbitrary_pts(self.num_pseudo)
                
        return core_idc
    
    def _run_kmeans_loaded(self):
        embedding_fname = f'embedding_{self.dnm}_{self.seed}.csv'

        full_fname = os.path.join(self.data_folder, embedding_fname)
        emb_df = pd.read_csv(full_fname, sep=',', header=None)
        train_x = torch.from_numpy(emb_df.values)
        
        train_y = self.train_dataset.targets
        n_train = len(self.train_dataset)

        kmeans_cluster = KmeansCluster(
            x=train_x, y=train_y, num_classes=self.nc, seed=self.seed, dist=self.dist
        )
        
        num_clusters = self._set_num_clusters()
        
        kmeans_cluster.set_num_clusters(num_clusters)
        kmeans_cluster.run_kmeans()
        core_idc = kmeans_cluster.get_arbitrary_pts(self.num_pseudo)
                
        return core_idc
    
    
    def _set_num_clusters(self):
        if self.num_pseudo == 30:
            return 30
        elif self.num_pseudo == 50:
            return 50
        elif self.num_pseudo == 80:
            return 20
        elif self.num_pseudo == 100:
            return 20


    def pretrain(
        self,
        test_dataset,
        architecture,
        D,
        n_hidden,
        distr_fn,
        mc_samples,
        init_sd,
        data_minibatch,
        pretrain_epochs,
        lr0net, 
        log_every,
        data_folder,
        load_from_saved,
        dnm):
            # if we are not using embeddings and just the original data points
            # we do not need pretraining.
            # if we are using embeddings, we run the same pretraining procedure as 
            # 
            if not self.embedding_flag:
                pass 
            else:
                # for now the embeddings are only for lenet
                if architecture != 'lenet':
                    raise ValueError("embeddings are calculated only for lenet")
                    
                super().pretrain(
                    test_dataset,
                    architecture,
                    D,
                    n_hidden,
                    distr_fn,
                    mc_samples,
                    init_sd,
                    data_minibatch,
                    pretrain_epochs,
                    lr0net, 
                    log_every,
                    data_folder,
                    load_from_saved,
                    dnm
                )
    

class ScoreSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, 
                 forgetting_flag=False, score_type='least_confidence',
                 data_folder=None, dnm='MNIST'):
        
        self.data_folder = data_folder
        self.dnm = dnm
        
        if score_type == "forgetting":
            forgetting_flag = True
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag)
        allowed_scores = ['least_confidence', 'entropy', "el2n", "forgetting"]

        if score_type not in allowed_scores:
            raise ValueError(f"{score_type} not in {allowed_scores}")
        
        self.score_type = score_type

    def select(self):
        # make sure this is pretrained
        #score_arr = self._get_uncertainty_score()
        score_arr = self._get_uncertainty_score_loaded()
        # select the top scores, for each class 
        n_train = len(self.train_dataset)
        pts_per_class = self.num_pseudo//self.nc
        core_idc = []
        
        pts_in_last_class = self.num_pseudo - (self.nc - 1) * pts_per_class
        for c in range(self.nc):
            idx_c = np.arange(n_train)[self.train_dataset.targets == c]
            if c == (self.nc - 1):
                num_pts = pts_in_last_class
            else:
                num_pts = pts_per_class
            score_arr_sub = score_arr[idx_c]
            top_k = torch.topk(score_arr_sub, num_pts).indices
            top_k_arr = top_k.detach().numpy().tolist()
            core_idc = core_idc + idx_c[top_k_arr].tolist()

        return core_idc

    
    def _get_uncertainty_score(self):
        softmax_fn = torch.nn.Softmax()
        n_train = len(self.train_dataset)
        data_minibatch = 128
        
        score_arr = torch.zeros(n_train, requires_grad=False)
        # shuffle=False is very important for this
        score_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(score_dataloader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.pretrained_net(data).squeeze(-1).mean(0)
                outputs_prob = softmax_fn(output)
                if self.score_type == "least_confidence":
                    score = self._least_confidence_score(outputs_prob)
                elif self.score_type == "entropy":
                    score = self._entropy_score(outputs_prob)
                elif self.score_type == "el2n":
                    score = self._el2n_score(outputs_prob, target)
                elif self.score_type == "forgetting":
                    score = self._forgetting_score(i, data_minibatch, n_train)
                
                
                score_arr[i * data_minibatch: min((i + 1) * data_minibatch, n_train)] = score
        
        return score_arr
    
    def _get_uncertainty_score_loaded(self):
        #self.data_folder = "/home/studio-lab-user/all_data/vi_data"
        #self.dnm = 'MNIST'
        #self.seed = 0
        
        score_fname = os.path.join(self.data_folder, f'score_psvi_{self.dnm}_{self.seed}.csv')
        score_df = pd.read_csv(score_fname) 
        score_arr = torch.from_numpy(score_df[self.score_type].values)
        
        return score_arr
    
    
    # from eqn 2 https://arxiv.org/pdf/2204.08499.pdf
    def _least_confidence_score(self, outputs_prob):
        score = 1.0 -  outputs_prob.max(1).values
        return score
    
    def _entropy_score(self, outputs_prob):
        p_eps = outputs_prob + 1e-20
        p_log_p = outputs_prob.mul(torch.log(p_eps))
        entropy_score = p_log_p.sum(1)
        return entropy_score
    
    def _el2n_score(self, outputs_prob, target):
        targets_onehot = F.one_hot(target.long(), num_classes=self.nc)
        el2n_score = torch.linalg.vector_norm(
            x=(outputs_prob - targets_onehot),
            ord=2,
            dim=1
        )
                        
        return el2n_score
    
    def _forgetting_score(self, i, data_minibatch, n_train):
        return self.pretrained_vi.forgetting_events[i * data_minibatch: min((i + 1) * data_minibatch, n_train)]
    

    
class ScoreCalculator:
    def __init__(self, outputs_prob, target, nc=10):
        self.outputs_prob = outputs_prob
        self.target = target
        self.nc = nc
    
    def least_confidence_score(self):
        score = 1.0 -  self.outputs_prob.max(1).values
        return score
    
    def entropy_score(self):
        p_eps = self.outputs_prob + 1e-20
        p_log_p = self.outputs_prob.mul(torch.log(p_eps))
        entropy_score = - p_log_p.sum(1)
        
        return entropy_score
    
    def el2n_score(self):
        targets_onehot = F.one_hot(self.target.long(), num_classes=self.nc)
        el2n_score = torch.linalg.vector_norm(
            x=(self.outputs_prob - targets_onehot),
            ord=2,
            dim=1
        )
                        
        return el2n_score




class KmeansScoreSelection(ScoreSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, 
                 score_type="least_confidence", embedding_flag=False, dist="euclidean", data_folder=None,
                dnm='MNIST'):
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag, score_type)
        self.embedding_flag = embedding_flag 
        self.dist = dist 
        self.data_folder = data_folder
        self.dnm = dnm
        
    def select(self):
        # make sure this is pretrained
        self.score_arr = self._get_uncertainty_score()
        self._run_kmeans_loaded()
        
        return self._combine_kmeans_score()
    
    def _run_kmeans(self):
        train_y = self.train_dataset.targets
        n_train = len(self.train_dataset)
        
        if self.embedding_flag:
            # for now this is hardcoded to lenet value
            # later, we might need to provide this with architecture
            embedding_size = 84
            embeddings = torch.zeros(n_train, embedding_size, requires_grad=False).to(self.device)
            data_minibatch = 128
            kmeans_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
            last_layer = None
            
            pretrained_net = self.pretrained_net
            pretrained_net.eval()
            
            with torch.no_grad():
                for i, (data, target) in enumerate(kmeans_dataloader):
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    batch_ind = list(range(
                        (i * data_minibatch), (min((i + 1) * data_minibatch, n_train))
                    ))

                    x = data
                    for layer in pretrained_net:
                        last_layer = x
                        x = layer(x)

                    embeddings[batch_ind] = last_layer.sum(0).squeeze(0)
        
            train_x = embeddings.detach().cpu()
        else:
            train_x = self.train_dataset.data


        self.kmeans_cluster = KmeansCluster(x=train_x, y=train_y, num_classes=self.nc, seed=self.seed, dist=self.dist)
        self.kmeans_cluster.set_num_clusters(20)
        self.kmeans_cluster.run_kmeans()
    
    
    def _run_kmeans_loaded(self):
        embedding_fname = f'embedding_{self.dnm}_{self.seed}.csv'

        full_fname = os.path.join(self.data_folder, embedding_fname)
        emb_df = pd.read_csv(full_fname, sep=',', header=None)
        train_x = torch.from_numpy(emb_df.values)
        
        train_y = self.train_dataset.targets
        n_train = len(self.train_dataset)

        self.kmeans_cluster = KmeansCluster(
            x=train_x, y=train_y, num_classes=self.nc, seed=self.seed, dist=self.dist
        )
        
        num_clusters = self._set_num_clusters()
        
        self.kmeans_cluster.set_num_clusters(num_clusters)
        self.kmeans_cluster.run_kmeans()

    
    def _set_num_clusters(self):
        if self.num_pseudo == 30:
            return 30
        elif self.num_pseudo == 50:
            return 50
        elif self.num_pseudo == 80:
            return 20
        elif self.num_pseudo == 100:
            return 20

    
    def _combine_kmeans_score(self):
        alpha = 2. 
        
        num_clusters = self._set_num_clusters()
        pts_per_cluster = int(self.num_pseudo / num_clusters)
        
        
        core_idcs = []
        for this_kmeans_dict in self.kmeans_cluster.kmeans_dict:
            for k in this_kmeans_dict.keys():
                if len(this_kmeans_dict[k]) > 0: 
                    indices = this_kmeans_dict[k]
                    score_arr_sub = [x + alpha for x in self.score_arr[indices]]
                    score_sum = sum(score_arr_sub)
                    pvals = [x/score_sum for x in score_arr_sub]
                    chosen_indices = list(sample_multinomial(pval=pvals, k=pts_per_cluster))
                    
                    #max_score_index = torch.argmax(score_arr_sub).detach().numpy()
                    #core_idcs.append(indices[max_score_index])
                    core_idcs = core_idcs + [indices[x] for x in chosen_indices]
        
        return core_idcs


class RandomScoreSelection(ScoreSelection):
    """
    mix random values with scores
    """
    def __init__(self, train_dataset, num_pseudo,  nc, seed, 
                 forgetting_flag=False, score_type='least_confidence'):
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag, score_type)

    def select(self):
        # make sure this is pretrained
        random_core_idc = self._make_random_selection()
        num_scored_pts = self.num_pseudo - len(random_core_idc)
        
        scored_core_idc = self._make_scored_selection(num_scored_pts)
        core_idc = random_core_idc + scored_core_idc
        
        return core_idc
    
    def _make_random_selection(self):
        n_train = len(self.train_dataset)
        pts_per_class = self.num_pseudo//(2 * self.nc)
        if pts_per_class < 1:
            pts_per_class = 1
            
        core_idc = []
        
        pts_in_last_class = self.num_pseudo//2 - (self.nc - 1) * pts_per_class
        if pts_in_last_class < 1:
            pts_in_last_class = 1
            
        for c in range(self.nc):
            idx_c = np.arange(n_train)[self.train_dataset.targets == c]
            if c == (self.nc - 1):
                num_pts = pts_in_last_class
            else:
                num_pts = pts_per_class
            chosen_idx = np.random.choice(idx_c, num_pts, replace=False)
            core_idc = core_idc + chosen_idx.tolist()
        
        return core_idc
    
    def _make_scored_selection(self, num_scored_pts):
        # make sure this is pretrained
        score_arr = self._get_uncertainty_score()
        # select the top scores, for each class 
        n_train = len(self.train_dataset)
        pts_per_class = num_scored_pts//self.nc
        scored_core_idc = []
        
        pts_in_last_class = num_scored_pts - (self.nc - 1) * pts_per_class
        for c in range(self.nc):
            idx_c = np.arange(n_train)[self.train_dataset.targets == c]
            if c == (self.nc - 1):
                num_pts = pts_in_last_class
            else:
                num_pts = pts_per_class
            score_arr_sub = score_arr[idx_c]
            top_k = torch.topk(score_arr_sub, num_pts).indices
            top_k_arr = top_k.detach().numpy().tolist()
            scored_core_idc = scored_core_idc + idx_c[top_k_arr].tolist()

        return scored_core_idc

class RandomIncrementalSelection(ScoreSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, 
                 forgetting_flag=False, score_type='entropy'):
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag, score_type)
        self.pretrained_net = None
    
    def update_current_state(self, current_core_idc, current_net):
        self.pretrained_net = current_net
        self.current_core_idc = current_core_idc
        
    def select(self):
        score_arr = self._get_uncertainty_score()
        curr_len = len(self.current_core_idc)
        top_k = torch.topk(score_arr, (self.num_pseudo + curr_len)).indices
        
        top_k_arr = top_k.detach().numpy().tolist()
        
        for new_index in top_k_arr:
            if new_index not in self.current_core_idc:
                core_idc = self.current_core_idc + [new_index]
               
                break
                
        return core_idc
    def get_weighted_subset(self, wt_vec=None):

        # for incremental, we keep calling select at each step
        # so commenting this out
        #if len(self.core_idc) == 0:
        #    self.core_idc = self.select()
        self.core_idc = self.select()
        
        # define the weights
        if not self.wt_vec:
            n_coreset = len(self.core_idc)
            n_train = len(self.train_dataset)
            scaling_factor = n_train / n_coreset
            self.wt_vec = scaling_factor * torch.ones(n_coreset)
        
        
        self.chosen_dataset = WeightedSubset(
            dataset=self.train_dataset,
            indices=self.core_idc,
            weights=self.wt_vec
        )
        
        return self.chosen_dataset

class WeightedKmeansSelection(KmeansScoreSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, score_type="entropy", embedding_flag=False, dist="euclidean"):
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag, score_type, embedding_flag, dist)

    def select(self):
        # make sure this is pretrained
        self._run_kmeans()
        core_idc = self.kmeans_cluster.get_arbitrary_pts()
        
        return core_idc
    

    
    def get_weighted_subset(self, wt_vec=None):

        if len(self.core_idc) == 0:
            self.core_idc = self.select()
        
        n_coreset = len(self.core_idc)
        n_train = len(self.train_dataset)
        scaling_factor = n_train / n_coreset
            
        score_arr = self._get_uncertainty_score()
        wt_vec_init = score_arr[self.core_idc]
        self.wt_vec = (scaling_factor/wt_vec_init.sum()) * wt_vec_init
        
        #print(wt_vec)
        
        self.chosen_dataset = WeightedSubset(
            dataset=self.train_dataset,
            indices=self.p,
            weights=self.wt_vec
        )
        
        return self.chosen_dataset

    
class CoresetSelect():
    def __init__(self, 
                 train_dataset=None, 
                 test_dataset=None, 
                 data_minibatch=128,
                 num_pseudo=100, 
                 nc=2, 
                 architecture='logistic_regression', 
                 D=None, 
                 n_hidden=100, 
                 mc_samples=4, 
                 init_sd = None,
                 lr0net=1e-3, 
                 log_every=10, 
                 distr_fn=categorical_fn, 
                 seed=0,
                 score_method="kmeans",
                 pretrain_epochs=5,
                 data_folder=None,
                 load_from_saved=False,
                 dnm=None,
                 distance_fn="euclidean",
                 last_layer_only=False
                ):
        self.architecture = architecture
        self.score_method = score_method
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_pseudo = num_pseudo
        self.seed = seed
        self.nc = nc
        self.D = D
        self.n_hidden = n_hidden
        self.distr_fn = distr_fn
        self.mc_samples = mc_samples
        self.init_sd = init_sd
        self.data_minibatch = data_minibatch
        self.pretrain_epochs = pretrain_epochs
        self.lr0net = lr0net
        self.data_folder = data_folder
        
        self.load_from_saved = load_from_saved
        self.dnm = dnm
        self.distance_fn = distance_fn 
        self.last_layer_only = last_layer_only 
        
        
    
    def select_data(self):
        if self.architecture == "lenet":
            embedding_flag = True
        else:
            embedding_flag = False 
            
        if self.score_method == "kmeans":
            select_method = KmeansSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                embedding_flag=embedding_flag,
                dist=self.distance_fn,
                data_folder=self.data_folder,
                dnm=self.dnm
            )
        elif self.score_method == "kmeans_gradient":
            select_method = KmeansGradientSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                embedding_flag=embedding_flag,
                dist=self.distance_fn,
                last_layer_only=self.last_layer_only
            )
        elif self.score_method == "submodular":
            select_method = SubmodularSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                embedding_flag=embedding_flag,
                dist=self.distance_fn,
                last_layer_only=self.last_layer_only
            )

        elif self.score_method == "random":
            select_method = RandomSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed
            )
        elif self.score_method in ["el2n", "least_confidence", "entropy", "forgetting"]:
            select_method = ScoreSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                score_type=self.score_method,
                data_folder=self.data_folder,
                dnm=self.dnm
            )
        elif self.score_method in [
            "scored_kmeans_el2n", "scored_kmeans_forgetting", 
            "scored_kmeans_entropy", "scored_kmeans_least_confidence"]:
            m = re.search(r'scored_kmeans_(.*)', self.score_method)

            scoring_method = m.group(1)
            
            select_method = KmeansScoreSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                score_type=scoring_method,
                embedding_flag=embedding_flag,
                dist=self.distance_fn,
                data_folder=self.data_folder,
                dnm=self.dnm
            )
            
        elif self.score_method in [
            "scored_random_el2n", "scored_random_forgetting", 
            "scored_random_entropy", "scored_random_least_confidence"]:
            m = re.search(r'scored_random_(.*)', self.score_method)

            scoring_method = m.group(1)
            
            select_method = RandomScoreSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                score_type=scoring_method
            )
        elif self.score_method in ["weighted_kmeans"]:
            select_method = WeightedKmeansSelection(
                train_dataset=self.train_dataset,
                num_pseudo=self.num_pseudo,
                nc=self.nc,
                seed=self.seed,
                score_type="entropy",
                embedding_flag=embedding_flag,
                dist=self.distance_fn
            )

        else:
            raise ValueError(f"{self.score_method} is not implemented")
            

        select_method.pretrain(
            test_dataset=self.test_dataset,
            architecture=self.architecture,
            D=self.D,
            n_hidden=self.n_hidden,
            distr_fn=self.distr_fn,
            mc_samples=self.mc_samples,
            init_sd=self.init_sd,
            data_minibatch=self.data_minibatch,
            pretrain_epochs=self.pretrain_epochs,
            lr0net=self.lr0net, 
            log_every=10,
            data_folder=self.data_folder,
            load_from_saved=self.load_from_saved,
            dnm=self.dnm
        )

        self.chosen_dataset = select_method.get_weighted_subset()
        log_core_idcs = select_method.core_idc
        log_core_wts = select_method.wt_vec.detach().numpy().tolist()
        self.wt_index = {str(k): v for k, v in zip(log_core_idcs, log_core_wts)} 
            

class KmeansGradientSelection(KmeansSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, embedding_flag=True, dist="euclidean", last_layer_only=False):
        super().__init__(train_dataset, num_pseudo,  nc, seed, 
                         forgetting_flag, embedding_flag, dist)
        if not embedding_flag:
            raise ValueError("Embedding flag must be true, to call kmeans_gradient")
        
        self.last_layer_only = last_layer_only

        
    def select(self):
        train_y = self.train_dataset.targets
        n_train = len(self.train_dataset)

        # for now this is hardcoded to lenet value
        # later, we might need to provide this with architecture
        gradients = self._get_embeddings()
        gradients_torch = torch.from_numpy(gradients)
        kmeans_cluster = KmeansCluster(x=gradients_torch, y=train_y, num_classes=self.nc, seed=self.seed, dist=self.dist)
        kmeans_cluster.set_num_clusters(self.num_pseudo)
        kmeans_cluster.run_kmeans()
        core_idc = kmeans_cluster.get_arbitrary_pts()
        
        return core_idc

 

    def _get_embeddings(self):
        # for now this is hardcoded to lenet value
        # later, we might need to provide this with architecture
        embedding_size = 84
        n_train = len(self.train_dataset)

        self.distr_fn = categorical_fn

        #embeddings = torch.zeros(n_train, embedding_size, requires_grad=False).to(self.device)
        data_minibatch = 128
        kmeans_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
        last_layer = None
            
        pretrained_net = self.pretrained_net
        pretrained_net.eval()
            
        gradients = []
        
        for i, (data, target) in enumerate(kmeans_dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            batch_num = target.shape[0]
            xbatch = data
            
            with torch.no_grad():
                for layer in pretrained_net:
                    last_layer = xbatch
                    xbatch = layer(xbatch)
            
                embeddings = last_layer.sum(0).squeeze(0)
            output = pretrained_net(data)
            output_mean = output.mean(dim=0)
            data_nll = -(
               self.distr_fn(logits=output_mean.squeeze(-1)).log_prob(target).sum()
            )
            kl = sum(m.kl() for m in pretrained_net.modules() if isinstance(m, VILinear))
            mfvi_loss = data_nll + kl
            
            with torch.no_grad():
                
                bias_parameters_grads = torch.autograd.grad(mfvi_loss, output_mean)[0]
                
                # if last_layer_only is true, we consider only the last layer gradients
                # and not the penultimate layer
                if self.last_layer_only: 
                    gradients.append(bias_parameters_grads.cpu().numpy())
                else:
                    weight_parameters_grads = embeddings.view(
                        batch_num, 1, embedding_size).repeat(
                        1, self.nc, 1) * bias_parameters_grads.view(
                        batch_num, self.nc, 1).repeat(
                        1, 1, embedding_size
                    )

                    gradients.append(
                        torch.cat(
                            [bias_parameters_grads, weight_parameters_grads.flatten(1)],
                            dim=1
                        ).cpu().numpy()
                    )
                
        gradients = np.concatenate(gradients, axis=0)
        print(f"Gradients: {gradients.shape}")
                    
        return gradients

class SubmodularSelection(KmeansGradientSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, embedding_flag=True, dist="euclidean", last_layer_only=False):
        super().__init__(train_dataset, num_pseudo,  nc, seed, 
                         forgetting_flag, embedding_flag, dist, last_layer_only)


    def select(self):
        n_train = len(self.train_dataset)
        pts_per_class = self.num_pseudo//self.nc
        core_idc = []
        
        pts_in_last_class = self.num_pseudo - (self.nc - 1) * pts_per_class
        
        self._greedy = "LazyGreedy"
        gradients = self._get_embeddings()
        
        for c in range(self.nc):
            idx_c = np.arange(n_train)[self.train_dataset.targets == c]
            if c == (self.nc - 1):
                num_pts = pts_in_last_class
            else:
                num_pts = pts_per_class
            
            sel_gradients = gradients[idx_c, :]
            
            if self.dist == "euclidean": 
                matrix = -1. * euclidean_dist_pair_np(sel_gradients)
            else:
                matrix = -1. * cossim_pair_np(sel_gradients)
            #matrix = -1. * cossim_pair_np(sel_gradients)
            
            matrix -= np.min(matrix) - 1e-3
            submod_function = FacilityLocation(index=idx_c, similarity_matrix=matrix)
            submod_optimizer = submodular_optimizer.__dict__[self._greedy](
                args=None, 
                index=idx_c,
                budget=num_pts
            )
            class_result = submod_optimizer.select(
                gain_function=submod_function.calc_gain,
                update_state=submod_function.update_state
            )
                
            #chosen_idx = np.random.choice(idx_c, num_pts, replace=False)
            print(class_result)
            core_idc = core_idc + class_result.tolist()
        
        return core_idc
    
class LogResource:
    def __init__(self):
        self.curr_time = time.time()
        self.prev_time = time.time()
        self.time_per_epoch = []
        self.memory_per_epoch = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
    def update(self):
        self.prev_time = self.curr_time
        self.curr_time = time.time()
        self.time_per_epoch.append( 
            self.curr_time - self.prev_time
        )
        
        gb_conv_factor = 1024**3
        
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(0)/gb_conv_factor
        else:
            gpu_memory = 0
        
        self.memory_per_epoch.append(gpu_memory) 
    
    def get_resources(self):
        avg_epoch_time = np.mean(self.time_per_epoch) 
        avg_gpu_memory = np.mean(self.memory_per_epoch)
        
        return {'time': avg_epoch_time, 'memory': avg_gpu_memory}
    

def rec_dd():
    return defaultdict(rec_dd)

def get_results(subfolder=None):
    folder_name = '/home/studio-lab-user/all_data/vi_result/'
    #subfolder = None
    if not subfolder:
        fname = os.path.join(folder_name, 'results.pk')
    else:
        fname = os.path.join(folder_name, subfolder, 'results.pk')

    with open(fname, 'rb') as f:
           results = pickle.load(f)
    
    return results 

def retrieve_results(subfolder_name=None, method='psvi_evaluate', 
                     dataset='MNIST', coreset_size=30):
    results = get_results(subfolder_name)
    final_dict = results[dataset][method][coreset_size][0]
    
    
    weights = final_dict['vs'][-1]
    labels = final_dict['zs'][-1]
    chosen_indices = final_dict['chosen_indices']
    
    if 'alpha' in final_dict.keys():
        alpha = final_dict['alpha']
    else:
        alpha = np.zeros(1)

    
    return_dict = {
        'chosen_indices': chosen_indices,
        'weights': weights,
        'labels': labels,
        'alpha': alpha
    }
    
    return return_dict
