import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F


class FreqOneLayer(torch.nn.Module):
    """
    This network is close to logistic regression
    but has more weights since we add one for each class.
    But this layer is closer to a traditional neural network.
    """
    def __init__(self, input_dim, num_classes):
        super(FreqOneLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    
class FreqLogReg(torch.nn.Module):
    """
    This is the exact logistic regression network
    """
    def __init__(self, input_dim, num_classes):
        super(FreqLogReg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes-1)
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    

class RunFrequentistModel:
    def __init__(self, train_dataset, test_dataset, model, is_logreg = True, 
                 num_classes=2, data_minibatch=128, num_epochs=100, learning_rate= 1e-3, 
                 test_interval=10):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.is_logreg = is_logreg
    
        self.data_minibatch = data_minibatch
        self.num_epochs = num_epochs
        self.test_interval = test_interval
        self.num_classes = num_classes
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=data_minibatch, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=data_minibatch, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        if self.is_logreg:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_one_epoch_logreg(self):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            #if args['cuda']:
                #data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            loss.backward()
            self.optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            #if batch_idx % 200 == 0:    # print every 2000 mini-batches
            #    print(f'batch: {batch_idx} loss: {running_loss / 200 }')
            #    running_loss = 0.0
        
    
    def train_one_epoch(self):
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            #if args['cuda']:
                #data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target.long())
            loss.backward()
            self.optimizer.step()
    
    def test_current_model(self):
        test_acc, test_nll, total_eg = 0, 0, 0
        softmax_fn = torch.nn.Softmax()
        with torch.no_grad():
            for data, target in self.test_dataloader:
                output = self.model(data)
                outputs_prob = softmax_fn(output)
                _, predicted = torch.max(output.data, 1)
                test_acc += (predicted == target.long()).sum().item()
                test_nll += -dist.Bernoulli(probs=outputs_prob[:,1]).log_prob(target).sum()
                total_eg += target.size(0)
        
        test_acc = test_acc / total_eg
        test_nll = test_nll / total_eg 
        
        print(f"Accuracy: {test_acc}, NLL: {test_nll}")
    
    def test_logreg(self):
        test_acc, test_nll, total_eg = 0, 0, 0
        sigmoid_fn = torch.nn.Sigmoid()
        with torch.no_grad():
            for data, target in self.test_dataloader:
                output = self.model(data)
                outputs_prob = sigmoid_fn(output.squeeze())
                test_acc += outputs_prob.gt(0.5).float().eq(target).sum()
                test_nll += -dist.Bernoulli(probs=outputs_prob).log_prob(target).sum()
                total_eg += target.size(0)
        test_acc = test_acc / total_eg
        test_nll = test_nll / total_eg 
        
        print(f"Accuracy: {test_acc}, NLL: {test_nll}")


                
    def train(self):
        if self.is_logreg:
            train_func = self.train_one_epoch_logreg
            test_func = self.test_logreg 
        else:
            train_func = self.train_one_epoch
            test_func = self.test_current_model
            
        for epoch in range(self.num_epochs):
            train_func()
            
            if epoch % self.test_interval == 0:
                print(f"Epoch: {epoch}")
                
                test_func()
    
    def get_largest_el2n_indices(self, coreset_size):
        """
        returns a list of indices with the largest EL2N scores
        """
        el2n_arr = self.get_el2n_scores()
        top_k = torch.topk(el2n_arr, coreset_size).indices
        top_k_arr = top_k.detach().numpy().tolist()
        return top_k_arr
        
        
                
    
    def get_el2n_scores(self):
        sigmoid_fn = torch.nn.Sigmoid()
        softmax_fn = torch.nn.Softmax()
        n_train = len(self.train_dataset)
        
        el2n_arr = torch.zeros(n_train, requires_grad=False)
        
        # shuffle=False is very important for this
        el2n_dataloader = DataLoader(self.train_dataset, batch_size=self.data_minibatch, shuffle=False)
        
        with torch.no_grad():
            for i, (data, target) in enumerate(el2n_dataloader):
                output = self.model(data)
                if self.is_logreg: 
                    outputs_prob_1 = sigmoid_fn(output)
                    outputs_prob_0 = 1. - outputs_prob_1 
                    outputs_prob = torch.hstack((
                        outputs_prob_0.unsqueeze(1),
                        outputs_prob_1.unsqueeze(1)
                    )).squeeze()
                    
                else:
                    outputs_prob = softmax_fn(output)
                targets_onehot = F.one_hot(target.long(), num_classes=self.num_classes)
                                      
                el2n_score = torch.linalg.vector_norm(
                    x=(outputs_prob - targets_onehot),
                    ord=2,
                    dim=1
                )
                
                el2n_arr[i * self.data_minibatch: min((i + 1) * self.data_minibatch, n_train)] = el2n_score
        
        return el2n_arr
                
