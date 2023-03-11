import torch

class FreqLogReg(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FreqLogReg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes-1)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class RunFrequentistModel:
    def __init__(self, train_dataset, test_dataset,model, 
                 data_minibatch=128, num_epochs=100, learning_rate= 1e-3, 
                 test_interval=10):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
    
        self.data_minibatch = data_minibatch
        self.num_epochs = num_epochs
        self.test_interval = test_interval
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=data_minibatch, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=data_minibatch, shuffle=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.BCELoss()
        
    
    def train_one_epoch(self):
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            #if args['cuda']:
                #data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.T, target.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
    
    def test_current_model(self):
        test_acc, test_nll, total_eg = 0, 0, 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                output = self.model(data)
                test_acc += output.squeeze().gt(0.5).float().eq(target).float().sum()
                test_nll += -dist.Bernoulli(probs=output.squeeze()).log_prob(target).sum()
                total_eg += target.size(0)
        
        test_acc = test_acc / total_eg
        test_nll = test_nll / total_eg 
        
        print(f"Accuracy: {test_acc}, NLL: {test_nll}")
                
    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
            
            if epoch % self.test_interval == 0:
                print(f"Epoch: {epoch}")
                
                self.test_current_model()
    