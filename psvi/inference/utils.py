# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
import torch.nn.functional as F
from psvi.models.neural_net import VILinear, categorical_fn
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import time
from typing import List
import random
from psvi.experiments.experiments_utils import set_up_model
from tqdm import tqdm



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
        forgetting_score_flag=False
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
        self.forgetting_score_flag
    
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

    
    def run(self):
        self.before_train()
        
        for i in tqdm(range(self.total_iterations)):
            self.train_an_epoch()
            self.after_epoch()
            
            if i % self.log_every == 0 or i == self.total_iterations -1:
                self.test() 

                
                
    
class KmeansCluster():
    def __init__(self, x, y, num_classes=2, 
                 balance=True, seed=None):
        self.x = x 
        self.y = y 
        self.balance = balance
        self.num_classes = num_classes
        
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
        # reset
        self.kmeans_dict = []
        self.cluster_centers = []
        
        for class_index in range(self.num_classes):
            indices = np.where(self.y.int().numpy() == class_index)[0]
            this_x = self.x.numpy()[indices]
            kmeans = KMeans(
                n_clusters=self.pts_per_class, 
                random_state=self.seed, n_init="auto").fit(this_x)
            this_kmeans_dict = defaultdict(list)
            for counter, label in enumerate(kmeans.labels_):
                this_kmeans_dict[label].append(indices[counter])
            
            self.kmeans_dict.append(this_kmeans_dict)
            self.cluster_centers.append(kmeans.cluster_centers_)
                
                
    def get_arbitrary_pts(self):
        core_idcs = []
        for this_kmeans_dict in self.kmeans_dict:
            for k in this_kmeans_dict.keys():
                if len(this_kmeans_dict[k]) > 0: 
                    core_idcs.append(this_kmeans_dict[k][0])
        
        return core_idcs

class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        return data, target, self.weights[idx]


    
class Selection():
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False):
        self.train_dataset = train_dataset
        self.num_pseudo = num_pseudo
        self.nc = nc
        self.seed = seed
        self.forgetting_flag = forgetting_flag
        self.core_idc = []
        self.chosen_dataset = None
    
    def select(self) -> List[int]:
        """
        get the selected indices
        """
        raise NotImplementedError("Child class must implement selection")
    
    def get_subset(self): 
        """
        get the subset of the torch dataset
        """
        # TODO: implement weighted subset from 
        # https://github.com/souravc83/deepcore_coresets/blob/b6438c5ce4517df31590d38ea1c480be574a534a/utils.py#L10
        
        self.core_idc = self.select()
        self.chosen_dataset = Subset(self.train_dataset, self.core_idc)
        return self.chosen_dataset 
    
    def get_weighted_subset(self):
        # initialize the weights
        n_train = len(self.train_dataset)
        scaling_factor = n_train / self.num_pseudo
        wt_vec = scaling_factor * torch.ones(self.num_pseudo)
        
        self.core_idc = self.select()
        
        self.chosen_dataset = WeightedSubset(
            dataset=self.train_dataset,
            indices=self.core_idc,
            weights = wt_vec
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
        log_every):
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
            forgetting_flag=self.forgetting_flag
        )
        
        self.pretrained_vi.run()

    

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
    
            
class KmeansSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False):
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag)
    
    def select(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        kmeans_cluster = KmeansCluster(x=train_x, y=train_y, num_classes=self.nc, seed=self.seed)
        kmeans_cluster.set_num_clusters(self.num_pseudo)
        kmeans_cluster.run_kmeans()
        core_idc = kmeans_cluster.get_arbitrary_pts()
        return core_idc


class EL2NSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed):
        super().__init__(train_dataset, num_pseudo,  nc, seed)

    def select(self):
        # make sure this is pretrained
        el2n_arr = self._get_el2n_score()
        top_k = torch.topk(el2n_arr, self.num_pseudo).indices
        top_k_arr = top_k.detach().numpy().tolist()
        
        return top_k_arr

    
    def _get_el2n_score(self):
        softmax_fn = torch.nn.Softmax()
        n_train = len(self.train_dataset)
        data_minibatch = 128
        
        el2n_arr = torch.zeros(n_train, requires_grad=False)
        # shuffle=False is very important for this
        el2n_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(el2n_dataloader):
                output = self.pretrained_vi.net(data).squeeze(-1).mean(0)
                outputs_prob = softmax_fn(output)
                targets_onehot = F.one_hot(target.long(), num_classes=self.nc)
                el2n_score = torch.linalg.vector_norm(
                    x=(outputs_prob - targets_onehot),
                    ord=2,
                    dim=1
                )
                
                el2n_arr[i * data_minibatch: min((i + 1) * data_minibatch, n_train)] = el2n_score
        
        return el2n_arr


# TODO merge EL2NSelection with UncertaintySelection
class ScoreSelection(Selection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, 
                 forgetting_flag=False, score_type='least_confidence'):
        
        if score_type == "forgetting":
            forgetting_flag = True
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag)
        allowed_scores = ['least_confidence', 'entropy', "el2n", "forgetting"]

        if score_type not in allowed_scores:
            raise ValueError(f"{score_type} not in {allowed_scores}")
        
        self.score_type = score_type

    def select(self):
        # make sure this is pretrained
        score_arr = self._get_uncertainty_score()
        top_k = torch.topk(score_arr, self.num_pseudo).indices
        top_k_arr = top_k.detach().numpy().tolist()
        
        return top_k_arr

    
    def _get_uncertainty_score(self):
        softmax_fn = torch.nn.Softmax()
        n_train = len(self.train_dataset)
        data_minibatch = 128
        
        score_arr = torch.zeros(n_train, requires_grad=False)
        # shuffle=False is very important for this
        score_dataloader = DataLoader(self.train_dataset, batch_size=data_minibatch, shuffle=False)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(score_dataloader):
                output = self.pretrained_vi.net(data).squeeze(-1).mean(0)
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
    
    def _forgetting_score(self, i, minibatch):
        return self.forgetting_events[i * data_minibatch: min((i + 1) * data_minibatch, n_train)]
        


class KmeansScoreSelection(ScoreSelection):
    def __init__(self, train_dataset, num_pseudo,  nc, seed, forgetting_flag=False, score_type="least_confidence"):
        
        super().__init__(train_dataset, num_pseudo,  nc, seed, forgetting_flag, score_type)

    def select(self):
        # make sure this is pretrained
        self.score_arr = self._get_uncertainty_score()
        self._run_kmeans()
        
        return self._combine_kmeans_score()
    
    def _run_kmeans(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        self.kmeans_cluster = KmeansCluster(x=train_x, y=train_y, num_classes=self.nc, seed=self.seed)
        self.kmeans_cluster.set_num_clusters(self.num_pseudo)
        self.kmeans_cluster.run_kmeans()
    
    def _combine_kmeans_score(self):
        core_idcs = []
        for this_kmeans_dict in self.kmeans_cluster.kmeans_dict:
            for k in this_kmeans_dict.keys():
                if len(this_kmeans_dict[k]) > 0: 
                    indices = this_kmeans_dict[k]
                    score_arr_sub = self.score_arr[indices]
                    max_score_index = torch.argmax(score_arr_sub).detach().numpy()[0]
                    core_idcs.append(indices[max_score_index])
        
        return core_idcs
