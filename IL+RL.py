import torch
import torchvision
import torch.optim as optim
from torchvision.transforms import v2
# import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
# from utilities import sigmoid_d
# from utilities import tanh_d
import math
import time


class IL(nn.Module):
    def __init__(self, lr=.001, type=0, n_iter=25, beta=100, gamma=.05, alpha=0, lr_min=.001, r=.000001, batch_norm = 0, weight_decay = 1e-5, scheduler = 15, dropout = 0, num_classes = 10):
        super().__init__()
        self.conv = True
        self.n_iter = n_iter
        self.l_rate = lr
        self.N = 0
        self.type = type   #BP=0, BP-Adam=1, IL=2, IL-MQ=3
        self.alpha = alpha
        self.beta = beta
        self.mod_prob = 1 / (1 + self.beta)
        self.gamma = gamma
        self.lr_min = lr_min
        self.r = r
        self.batch_norm = batch_norm
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.dropout = dropout
        self.num_classes = num_classes

        self.wts = nn.Sequential(
                nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(5,5), stride=(2,2), bias=True),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(5,5), stride=(2,2), bias=True),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), bias=True),
                nn.ReLU(),
                nn.Flatten()
            ),

            nn.Sequential(
                nn.Linear(1024, self.num_classes, bias=True)
            )
        )
        
        self.num_layers = len(self.wts) + 1
        self.wt_var = [alpha for x in range(self.num_layers - 1)] 

        if self.type == 1:
            self.bp_optim = torch.optim.Adam(self.wts.parameters(), lr=self.l_rate, weight_decay=self.weight_decay)
        else:
            self.bp_optim = torch.optim.SGD(self.wts.parameters(), lr=self.l_rate, weight_decay=self.weight_decay)

        self.optims = self.create_optims()

    ############################## COMPUTE FORWARD VALUES ##################################
    def initialize_values(self, x):
        with torch.no_grad():
            h = [torch.randn(1, 1) for i in range(self.num_layers)]

            #First h is the input
            h[0] = x.clone()

            #Compute FF values
            for i in range(1, self.num_layers):
                h[i] = self.wts[i-1](h[i-1].detach())
            return h


    def create_optims(self):
        if self.type == 4:
            optims = []
            for l in range(0, self.num_layers - 1):
                optims.append(torch.optim.Adam(self.wts[l].parameters(), lr=self.alpha))
            return optims
        else:
            return None


    ############################## Minimize F w.r.t. Neuron Activities ##################################
    def compute_targets(self, h, global_target):
        mse = torch.nn.MSELoss(reduction='sum')
        bce = torch.nn.BCELoss(reduction='sum')
        NLL = nn.NLLLoss(reduction='sum')
        selected = None # Selected neurons/nodes
        # softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            targ = [h[i].clone() for i in range(self.num_layers)]
            targ[-1] = (1 - self.mod_prob) * global_target.clone() + self.mod_prob * targ[-1]

        # Iterative updates
        for i in range(self.n_iter):
            decay = (1 / (1 + i))
            with torch.no_grad():
                p = self.wts[-1](targ[-2])
            for layer in reversed(range(1, self.num_layers - 1)):
                with torch.no_grad():
                    self.bp_optim.zero_grad()
                    # Compute error
                    if layer < self.num_layers - 2:
                        err = targ[layer + 1] - p  # MSE gradient   # e(l+1)
                    else:
                        err, selected = BrainProp_err(p, label = global_target.clone(), fake_target = True, picked = selected) 
                        correct_target = torch.as_tensor(global_target >= p, dtype = torch.float32) * selected               

                _, dfdt = torch.autograd.functional.vjp(self.wts[layer], targ[layer], err)
                with torch.no_grad():
                    p = self.wts[layer - 1](targ[layer - 1])
                    e_top = targ[layer] - p     # e(l)
                    dt = decay * self.gamma * (dfdt - e_top)
                    targ[layer] = targ[layer] + dt
        return targ, selected, correct_target



    ############################## TRAIN ##################################
    def train_wts(self, x, global_target, y, ep):

        with torch.no_grad():
            # Get feedforward and target values
            h = self.initialize_values(x)

            # Get targets
            if self.type > 1:
                h_hat, selected, correct_target = self.compute_targets(h, global_target)

        # Count datapoint
        self.N += 1

        # Update weights
        if self.type < 2:
            self.BP_update(x, y,ep, correct_target = correct_target, picked = selected)
        elif self.type == 2:
            self.LMS_update(h_hat, y, ep, correct_target = correct_target, picked = selected)
        elif self.type == 3:
            self.MQ_update(h_hat, y, ep, correct_target = correct_target, picked = selected)
        elif self.type == 4:
            self.Adam_update(h_hat, y)

        '''# Count datapoint
        self.N += 1'''

        return False, h[-1]

    def BP_update(self, x, y,ep, correct_target, picked = None):
        # NLL = nn.NLLLoss(reduction='sum')
        mse = torch.nn.MSELoss(reduction='sum')
        conv_weight_bs = [[] for i in range(self.num_layers-1)]
        conv_weight_as = [[] for i in range(self.num_layers-1)]
        bn_weight_bs = [[] for i in range(self.num_layers-3)]
        bn_weight_as = [[] for i in range(self.num_layers-3)]
        
        ## Get BP Gradients
        z = x.clone().detach()
        for i in range(0, self.num_layers - 1):
            z = self.wts[i](z)

        #Get loss
        BP, selected_classes = BrainProp_output(z)
        target = F.one_hot(y.detach(), num_classes=self.num_classes).to(torch.float32).to(dev) * selected_classes
        loss = .5 * mse(BP,   target.detach()) / z.size(0)
        #Update
        self.bp_optim.zero_grad()
        loss.backward()

        # Record relevant parameters
        # loss_track[i].append(np.array(cur_loss))
        for i in range(self.num_layers - 1):
            with torch.no_grad():
                conv_weight_bs[i] = self.wts[i][0].weight.data.clone()
                grads = self.wts[i][0].weight.grad.clone().abs().mean().cpu()
                conv_grads[i].append(np.array(grads))
                if i < self.batch_norm:
                    grads = self.wts[i][2].weight.grad.clone().abs().mean().cpu()
                    bn_grads[i].append(np.array(grads))
                        
        self.bp_optim.step()
        
        for i in range(self.num_layers - 1):
            with torch.no_grad():
                # Record the weight data after step
                conv_weight_as[i] = self.wts[i][0].weight.data.clone()
                conv_step_size[i].append((conv_weight_bs[i] - conv_weight_as[i]).abs().mean().cpu())
                if i < self.batch_norm:
                    bn_weight_as[i] = self.wts[i][2].weight.data.clone()
                    bn_step_size[i].append((bn_weight_bs[i] - bn_weight_as[i]).abs().mean().cpu())

        # del conv_weight_bs, conv_weight_as, bn_weight_bs, bn_weight_as


    def LMS_update(self, targ, y, ep, correct_target, picked = None):
        # Use **kwarg then no need to specify mse ... below
        mse = torch.nn.MSELoss(reduction='sum')
        # bce = torch.nn.BCELoss(reduction='sum')
        # NLL = nn.NLLLoss(reduction='sum')
        # softmax = torch.nn.Softmax(dim=1)
        ## Update each weight matrix
        for i in range(self.num_layers-1):
            #Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i+1].detach()) / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                target = F.one_hot(y.detach(), num_classes=self.num_classes).to(torch.float32).to(dev)
                loss = .5 * mse(p,  target) / p.size(0)

            #Compute weight gradients
            self.bp_optim.zero_grad()
            loss.backward()
            with torch.no_grad():
                # Update weights with normalized step size and precision weighting
                self.wts[i][0].weight.data -= self.wts[i][0].weight.grad * self.alpha
                self.wts[i][0].bias.data -= self.wts[i][0].bias.grad * self.alpha
                if i < self.num_layers - 3:
                  self.wts[i][2].weight.data -= self.wts[i][2].weight.grad * self.alpha
                  self.wts[i][2].bias.data -= self.wts[i][2].bias.grad * self.alpha


    def MQ_update(self, targ, y, ep, correct_target, picked = None):
        mse = torch.nn.MSELoss(reduction='none')
        conv_weight_bs = [[] for i in range(self.num_layers-1)] # Weights before updating
        conv_weight_as = [[] for i in range(self.num_layers-1)] # After updating
        bn_weight_bs = [[] for i in range(self.batch_norm)]
        bn_weight_as = [[] for i in range(self.batch_norm)]
        decay = int(ep/self.scheduler)+1
        # NLL = nn.NLLLoss(reduction='sum')
        # softmax = torch.nn.Softmax(dim=1)

        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i + 1].detach()) / p.size(0)
                for j in range(loss.size(dim = 0)):
                    loss[j] = loss[j] * corrected_factor[j]
                loss = loss.sum() / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                # target = F.one_hot(y.detach(), num_classes=self.num_classes).to(torch.float32).to(dev)
                loss, corrected_factor = BrainProp_loss(output = p, decay = decay, picked = picked, correct_target = correct_target, label = y.detach(), fake_target = False)
                loss = loss.sum() / p.size(0)

            # Compute weight gradients
            self.bp_optim.zero_grad()
            loss.backward()
            with torch.no_grad():
                # Record relevant parameters
                cur_loss = loss.clone().abs().mean().cpu()
                loss_track[i].append(np.array(cur_loss))
                conv_weight_bs[i] = self.wts[i][0].weight.data.clone()
                grads = self.wts[i][0].weight.grad.clone().abs().mean().cpu()
                conv_grads[i].append(np.array(grads))
                if i < self.batch_norm:
                    grads = self.wts[i][2].weight.grad.clone().abs().mean().cpu()
                    bn_grads[i].append(np.array(grads))
                    bn_weight_bs[i] = self.wts[i][2].weight.data.clone()

                # Update weights
                wtfrac = self.alpha / ((self.wt_var[i]) + self.r) + self.lr_min
                lr_track[i].append(float(wtfrac))
                # wtfrac /= decay
                # wtfrac *= LR_Enlarge[i]        #Enlarge lr if needed
                self.wts[i][0].weight.data -= wtfrac * self.wts[i][0].weight.grad
                self.wts[i][0].bias.data -= wtfrac * self.wts[i][0].bias.grad 
                if i < self.batch_norm:
                  self.wts[i][2].weight.data -= wtfrac * self.wts[i][2].weight.grad 
                  self.wts[i][2].bias.data -= wtfrac * self.wts[i][2].bias.grad
                # Update variances as moving average of absolute gradient
                self.avgRt = min(self.N / (self.N + 1), .999)
                params = torch.cat((self.wts[i][0].weight.grad.view(-1).clone(), self.wts[i][0].bias.grad.view(-1).clone()), dim=0)
                self.wt_var[i] = self.avgRt * self.wt_var[i] + (1 - self.avgRt) * torch.mean(torch.abs(params))

                # Record relevant parameters after step
                conv_weight_as[i] = self.wts[i][0].weight.data.clone()
                conv_step_size[i].append((conv_weight_bs[i] - conv_weight_as[i]).abs().mean().cpu())
                if i < self.batch_norm:
                    bn_weight_as[i] = self.wts[i][2].weight.data.clone()
                    bn_step_size[i].append((bn_weight_bs[i] - bn_weight_as[i]).abs().mean().cpu())

        # del conv_weight_bs, conv_weight_as, bn_weight_bs, bn_weight_as

    def Adam_update(self, targ, y):
        mse = torch.nn.MSELoss(reduction='sum')
        NLL = nn.NLLLoss(reduction='sum')
        softmax = torch.nn.Softmax(dim=1)
        for i in range(self.num_layers - 1):
            # Compute local losses, sum neuron-wise and avg batch-wise
            if i < (self.num_layers - 2):
                p = self.wts[i](targ[i].detach())
                loss = .5 * mse(p, targ[i + 1].detach()) / p.size(0)
            else:
                p = self.wts[i](targ[i].detach())
                target = F.one_hot(y.detach(), num_classes=self.num_classes).to(torch.float32).to(dev)
                loss = .5 * mse(p,  target) / p.size(0)

            # Compute weight gradients, update with adam
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()


# Load Data
def get_data(batch_size=64, data=0):
  if data == 0:
    d_name = 'CIFAR100'
    num_train = 50000
    transform = v2.Compose([v2.ToTensor(),          
                            v2.RandomHorizontalFlip(),
                            v2.RandomRotation(degrees=(-5,5)),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='/scratch-local/hchen', train=True,
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, drop_last = True)

    testset = torchvision.datasets.CIFAR100(root='/scratch-local/hchen', train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                              shuffle=False, drop_last = True)

  else:
    d_name = 'CIFAR10'
    num_train = 50000

    transform = v2.Compose([v2.ToTensor(),          
                            v2.RandomHorizontalFlip(),
                            v2.RandomRotation(degrees=(-5,5)),
                            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/scratch-local/hchen', train=True,
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, drop_last = True)

    testset = torchvision.datasets.CIFAR10(root='/scratch-local/hchen', train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                              shuffle=False, drop_last = True)


  print(d_name)



  return train_loader, test_loader, d_name, num_train


def compute_num_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct


def test(test_losses, test_accuracies, model, test_loader, seed, lr, dev):
  bce = torch.nn.BCELoss(reduction='none')
  mse = torch.nn.MSELoss(reduction='none')
  # softmax = torch.nn.Softmax(dim=1)
  with torch.no_grad():
    test_accuracies[lr][seed].append(0)
    test_losses[lr][seed].append(0)
    testn = 0
    for batch_idx, (images, y) in enumerate(test_loader):
      images = images.to(dev)
      y = y.to(dev)
      target = F.one_hot(y, num_classes=model.num_classes).to(dev)

      # Test and record losses and accuracy over whole test set
      h = model.initialize_values(images)
      global_loss = torch.mean(mse(h[-1], target).sum(1))
      test_accuracies[lr][seed][-1] += compute_num_correct(h[-1], y)
      test_losses[lr][seed][-1] += global_loss.item()
      testn += images.size(0)

    test_accuracies[lr][seed][-1] /= testn
    test_losses[lr][seed][-1] /= testn


def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_accuracies, epochs, dev, b_size):
    test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)

    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=model.num_classes)
            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), target.detach(), y)

        test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)
        print(ep+1, 'Acc:', test_accuracies[lr][seed][-1] * 100)



def train(models, batch_size, data, dev, epochs, test_losses, test_accuracies):

    for l in range(len(models)):
        print(f'Training Alpha:{models[l][0].alpha}')
        for m in range(len(models[0])):

            train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)
            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_accuracies, epochs, dev, batch_size)

            print(f'Seed:{m}', f'MaxAcc:{max(test_accuracies[l][m])}',
                  f'LastAcc:{test_accuracies[l][m][-1]}')
            

############################### Return only the outputs of selected node, and zeros others ######################################
def BrainProp_output(x, epsilon=0.02):  # Remove some unnecessary ".to(dev)"

    eps = torch.full([x.size(dim=0)], epsilon).to(dev)

    selected_classes_exploitation = torch.argmax(x, dim=-1).to(dev)
    try:
      selected_classes_exploration = torch.squeeze(torch.multinomial(F.softmax(x, dim=-1), 1), -1).to(dev)
    except Exception as e:
      print(e)
      print(x)
      # print(F.softmax(x, dim=-1))

    selected_classes = torch.where(torch.rand(x.size(dim=0)).to(dev)>eps, selected_classes_exploitation, selected_classes_exploration)

    selected_classes = F.one_hot(selected_classes, x.size(dim=-1))

    return selected_classes*x, selected_classes.detach()
    # return selected_classes * F.softmax(x, dim = -1)


############################### Compute the loss between targets and predictions in weight update #########################################
def BrainProp_loss(output, decay, picked, correct_target, label = None, fake_target = False, one_hot = False, **kwargs):
    threshold = 1.0
    small_step = 0.8 / decay
    selected_classes = picked

    if fake_target: #Generate fake targets to compute the loss
        selected_output = output * selected_classes
        new_label = nudge(selected_output.detach(), label = label)
    else:
        if one_hot:
          new_label = label.to(torch.float32) * selected_classes
        else:
          new_label = F.one_hot(label, output.size(dim=-1)).to(torch.float32)*selected_classes

    new_output = output * selected_classes

    # above_threshold = torch.as_tensor(new_output >= threshold, dtype=torch.float32)
    # between_threshold = torch.as_tensor(((1- threshold) < new_output)  & (new_output < threshold), dtype=torch.float32)
    # thresholded_output = between_threshold * new_output + above_threshold
    thresholded_output = new_output

    corrected_factor = [1 if i == output.size(dim=-1) else -1 for i in (correct_target == selected_classes).sum(dim=-1)]
    # corrected_factor = [small_step for i in (correct_target == selected_classes).sum(dim=-1)]
    # print("corrected_factor: ", "\n", corrected_factor)

    selected_loss = 0.1 * output.size(dim=-1) * F.mse_loss(new_label, thresholded_output, reduction = 'none', **kwargs)

    last_layer = [small_step if i == output.size(dim=-1) else small_step for i in (correct_target == selected_classes).sum(dim=-1)]
    # last_layer = [small_step for i in (correct_target == selected_classes).sum(dim=-1)]
    for j in range(selected_loss.size(dim = 0)):
      selected_loss[j] = selected_loss[j] * last_layer[j]

    # print("Modified Loss: ",selected_loss[:5])
    return selected_loss, corrected_factor
    # return  output.size(dim=-1)* NLL(log_softmax(output), label)


################################### Compute errors in activations update
def BrainProp_err(output, label = None, fake_target = False, picked = None, **kwargs):# Maybe remove output.size(dim=-1) in the return command
    threshold = 1.0

    if picked == None:
      selected_output, selected_classes = BrainProp_output(output)
    else:
      selected_output = output * picked
      selected_classes = picked

    if fake_target:
        new_label = nudge(selected_output.detach(), label = label)*selected_classes
    else:
        new_label = label.to(torch.float32)*selected_classes

    new_output = output * selected_classes

    # above_threshold = torch.as_tensor(new_output >= threshold, dtype=torch.float32)
    # between_threshold = torch.as_tensor(((1- threshold) < new_output)  & (new_output < threshold), dtype=torch.float32)
    # thresholded_output = between_threshold * new_output + above_threshold
    thresholded_output = new_output

    selected_err = 0.1 * output.size(dim=-1) * (new_label - thresholded_output)
    return selected_err, selected_classes
    # return  output.size(dim=-1)* NLL(log_softmax(output), label)


###################################### Determine how to nudge in activations update
def nudge(output, label = None, **kwargs):
  proportion = 0.05  
  min_nudge = 5.0   

  nudge_size = (output.detach() * proportion).abs()
  nudge_size = torch.as_tensor(nudge_size >= min_nudge, dtype = torch.float32) * nudge_size \
                + torch.as_tensor( (0 < nudge_size) & (nudge_size < min_nudge), dtype = torch.float32) * min_nudge
  nudged_output = output.detach() + nudge_size
  
  # nudged_output = torch.as_tensor(output.detach() != 0, dtype = torch.float32)

  # nudged_output = torch.as_tensor(new_output <= 1, dtype = torch.float32) * new_output + torch.as_tensor(new_output > 1, dtype = torch.float32)
  # nudged_output = label

  return nudged_output



dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {dev}")
torch.cuda.empty_cache()
# torch.manual_seed(0)
# torch.set_default_dtype(torch.float64)
models = []
epochs=100  #50
batch_size= 64   #64
data=0   # 0 for CIFAR100, 1 for CIFAR10
num_seeds=1   #5
alpha=[2e-05] 
model_type=3
beta=100
gamma= 0.04 
Batch_Norm = 0 # Amount of Batch_Norm() applied in the model
LR_Enlarge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])


for l in range(len(alpha)):
  #Add list of seeds at this learning rate
  models.append([])
  for m in range(num_seeds):
    # BP-SGD
    if model_type == 0:
        models[-1].append(IL(type=0, lr=alpha[l]))

    # BP-Adam
    elif model_type == 1:
        models[-1].append(IL(type=1, lr=alpha[l], weight_decay=5e-3, num_classes=100))

    # IL n_iter = 7, gamma=.03
    elif model_type == 2:
        models[-1].append(IL(n_iter=3, gamma=gamma, beta=beta, type=2, alpha=alpha[l]))

    # IL-MQ
    elif model_type == 3:
        models[-1].append(IL(n_iter=3, gamma=gamma,  beta=beta, type=3, alpha=alpha[l], lr_min=.001, batch_norm = Batch_Norm, weight_decay=1e-5, scheduler=999, num_classes=100))

    # IL-Adam
    elif model_type == 4:
        models[-1].append(IL(n_iter=3, gamma=gamma, beta=beta, type=4, alpha=alpha[l]))

  # To Device
  for i in range(len(models[-1])):
    models[-1][i].to(dev)


# grads = {}
conv_grads = {}
conv_step_size = {}
bn_grads = {}
bn_step_size = {}
loss_track = {}
lr_track = {}
train_acc = []
test_acc = []
for i in range(models[-1][-1].num_layers-1):
    # grads[i] = []
    conv_grads[i] = []
    conv_step_size[i] = []
    loss_track[i] = []
    lr_track[i] = []
    if i < models[-1][-1].num_layers-3:
        bn_grads[i] = []
        bn_step_size[i] = []
    

#################################################
# Create Containers
test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
test_accs = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]


#################################################
# Get data
train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)
log_interval = len(train_loader)//5


#################################################
# Have a first test
test(test_losses, test_accs, models[0][0], test_loader, seed = 0, lr = 0, dev = dev)


#################################################
# Train
print(f'\nTRAINING MODEL TYPE {model_type}')
print(f"Number of Classes: {models[0][0].num_classes}")
print(f'\nNumber of layers {models[0][0].num_layers-1}')
if Batch_Norm == True:
    print("Applied BatchNorm")
else:
    print("BatchNorm not applied")
print(f"Dropout Rate: {models[0][0].dropout}")
print(f'Training LR: {models[0][0].l_rate}')
print(f'Training Alpha: {models[0][0].alpha}')
print(f'Training lr_min: {models[0][0].lr_min}')
print(f'Training n_iter: {models[0][0].n_iter}')
print(f'Training gamma: {models[0][0].gamma}')
print(f'Training batch_size: {batch_size}')
print(f'Regularization: {models[0][0].weight_decay}')
print(f'Learning rate scheduler: Decay every {models[0][0].scheduler} epochs\n')



softmax = torch.nn.Softmax(dim=1)
start_time = time.time()
for ep in range(epochs):
    train_accuracies = 0.0
    for batch_idx, (images, y) in enumerate(train_loader):
        images = images.to(dev)
        y = y.to(dev)
        target = F.one_hot(y, num_classes=models[0][0].num_classes)
        if images.size(0) == batch_size:
            _, after_train_p = models[0][0].train_wts(images.detach(), target.detach(), y, ep)

        # Print out train accuracies
        # print(f"Y size: {y.size}")
        train_accuracies += compute_num_correct(softmax(after_train_p), y) / len(images)
        # print((batch_idx+1) % log_interval)

        if (batch_idx+1) % log_interval == 0 or batch_idx == len(train_loader)-1:
          print('Train Epoch: {} [{}/{} ({:.1f}%)]\tAccuracy: {:.2f}%'.format(

            ep+1,

            (batch_idx+1) * len(images),

            len(train_loader.dataset),

            100. * (batch_idx+1) / len(train_loader),

            train_accuracies*100./(batch_idx+1)))
          train_acc.append(train_accuracies*100./(batch_idx+1))

    test(test_losses, test_accs, models[0][0], test_loader, seed = 0, lr = 0, dev = dev)
    print("\n", ep+1, 'Acc:', test_accs[0][0][-1] * 100, "\n")
    test_acc.append(test_accs[0][0][-1] * 100)

print("Time:--- %s seconds ---" % (time.time() - start_time))
print(f'Seed:{0}', f'0axAcc:{max(test_accs[0][0])}',
      f'LastAcc:{test_accs[0][0][-1]}')


# Save the grads changes
for i in range(models[-1][-1].num_layers-1):
    conv_grads[i] = np.array(conv_grads[i])
    conv_step_size[i] = np.array(conv_step_size[i])
    loss_track[i] = np.array(loss_track[i])
    lr_track[i] = np.array(lr_track[i])
    if i < models[-1][-1].num_layers-3:
        bn_grads[i] = np.array(bn_grads[i])
        bn_step_size[i] = np.array(bn_step_size[i])

combined_info = {"conv_grads" : conv_grads,
                 "conv_step_size" : conv_step_size,
                 "bn_grads" : bn_grads,
                 "bn_step_size" : bn_step_size,
                 "LR_Enlarge" : LR_Enlarge,
                 "Loss": loss_track,
                 "Lr_Track": lr_track,
                 "Train_Acc": train_acc,
                 "Test_Acc": test_acc
                 }


with open('*******.pkl', 'wb+') as fp:
    pickle.dump(combined_info, fp)
    print('\nDictionary saved successfully to file:*******.pkl')