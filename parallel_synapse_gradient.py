import numpy as np 
import torch 
import datetime
import time
import matplotlib.pyplot as plt 
import pickle 
import argparse
from queue import deque
from torch.utils.data import DataLoader, Dataset
from utils_parallel_syn_gradient import *
import os
    
class ParallelSyn(torch.nn.Module):
    '''
    Self-defined parallel synapse model class
    
    Attributes:
		N: int
			input dimension
		M: int
			number of parallel synapses
		seed: int
			random seed
		device: torch.device
			cpu or gpu
		ampli: torch.nn.Parameter of shape (N, M)
			amplitude of the parallel synapses, initialized as random numbers between 0 and 1
		slope: torch.nn.Parameter of shape (N, M)
			slope of the parallel synapses, initialized as random numbers between 0 and 500
		thres: torch.nn.Parameter of shape (N, M)
			threshold of the parallel synapses, initialized as random numbers between 0 and 1
		theta: torch.nn.Parameter of shape (1)
			global threshold, initialized as a random number between 0 and 1
		actv: torch tensor of shape (nSample, 1)
			activation of the neuron, computed by the forward function
    '''
    def __init__(self, params):
        super().__init__()
        for k in params:
            setattr(self, k, params[k])
        torch.manual_seed(self.seed)
        self.ampli = torch.nn.Parameter(torch.rand( self.N, self.M,device = params['device']))  
        self.slope = torch.nn.Parameter(torch.rand(self.N, self.M, device = params['device'])*500) 
        self.thres = torch.nn.Parameter(torch.rand(self.N, self.M, device = params['device'])) 
        self.theta = torch.nn.Parameter(torch.rand(1, device = params['device']))
        
    def forward(self, data):  
        '''
		Compute the activation of the neuron given the input data
		Inputs:
			data: torch tensor of shape (nSample, N)
				input data
		Outputs:
			actv: torch tensor of shape (nSample, 1)
        '''
        self.actv = (data.unsqueeze(2) - self.thres.unsqueeze(0))    
        self.actv = self.actv * self.slope.unsqueeze(0)          
        self.actv = torch.tanh(self.actv)                         
        self.actv = self.ampli.pow(2).unsqueeze(0) * self.actv    
        self.actv = self.actv.mean(dim=(1, 2))             

class TrainParallelSyn():
    def __init__(self, params):
        for k in params:
            setattr(self, k, params[k])
        self.loss_history = deque()
        self.acc_history = deque()
        self.time = deque()
    def lossFunc(self, model, label): 
        self.loss = hinge_loss(model.actv,  label, model.theta,self.margin) 
    def accu(self, model, label):
        self.acc = (torch.sign(model.actv - model.theta) == label).sum()/self.P
    def train(self, model, label, inputX, t1):
        self.optim = torch.optim.Adam([
            {"params": model.ampli},
            {"params": model.slope},
            {"params": model.theta},
            {"params": model.thres, 'lr': self.threslr}
            ], lr = self.adamlr) 
        self.thresPool = torch.tensor(inverse_cdf(np.random.uniform(size= (self.NthresPool,1))), device = model.device).float()
        for k in range(self.Nepoch): 
            self.shuffle_invalid(model)
            with torch.no_grad():
                model.slope.clamp_min_(0)
            model.forward(inputX)
            self.lossFunc(model, label)
            self.loss.backward()     
            self.optim.step() 
            self.optim.zero_grad()
                    
                    
            model.forward(inputX) 
            self.accu(model, label)

            if (k % self.downSample) == 5: 
                if len(self.acc_history) > self.maxRecord * self.downSample:
                    self.acc_history.popleft()
                    self.loss_history.popleft()
                    self.time.popleft()
                self.acc_history.append(self.acc.detach())
                self.loss_history.append(self.loss.detach()) 
                self.time.append(time.time() - t1) 
                
            if self.acc > 0.9999999:
                break
        
    def shuffle_invalid(self, model):
        with torch.no_grad():
            mask = model.ampli < self.minAmpli
            model.thres[mask] = self.thresPool[torch.randint(self.NthresPool, (mask.sum(),))].ravel()
            model.ampli[mask] = self.minAmpli

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("N", type=int, help="N")
	parser.add_argument("M", type=int, help="M")
	parser.add_argument("P", type=int, help="P")
	parser.add_argument("seed", type=int, help="seed")

	args = parser.parse_args()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	model_params = {
		'N': args.N, # input dimension
		'M': args.M,# parallel synapse number 
		'seed': args.seed,
		'device': device
		}     
	train_params = {
		      'margin': 0.1, # only applied when 'loss' is hinge
		      'threslr': 1e-6,
		      'adamlr': 0.003,
		'minAmpli': 1e-1,
		'Nepoch': 160000,
		'P': args.P,  
		'maxRecord': 400,
		'downSample': 100,
		'NthresPool': int(args.P/2), 
	}   
	path = ''
	folder = './N_'+str(model_params['N'])
	path += 'N_'+str(model_params['N']) + '_M_'+str(model_params['M'])\
		+'_P_' +str(train_params['P'])\
		+ '_seed_'+str(model_params['seed'])  
	if os.path.isfile(folder + '/' + path+'_data') and os.path.isfile(folder + '/' + path):
		print('loading existing model')        
		data_ = load_model(folder + '/' + path+'_data')
		inputX, label = data_[:,:-1].to(model_params['device']), data_[:,-1].to(model_params['device'])
		model = ParallelSyn(model_params)
		model.to(model_params['device'])
		state_dict = torch.load(folder + '/' + path, map_location=model_params['device'])
		model.load_state_dict(state_dict) 
	else:
		print('creating new model')
		inputX, label = generate_data_torch(nDimension = model_params['N'], \
						nSample = train_params['P'], \
						randomSeed = model_params['seed'],
						device = model_params['device']) 
		path = ''
		folder = './N_'+str(model_params['N'])
		path += 'N_'+str(model_params['N']) + '_M_'+str(model_params['M'])\
			+'_P_' +str(train_params['P'])\
			+ '_seed_'+str(model_params['seed'])

		data_ = torch.hstack((inputX.cpu(),label.reshape(-1,1).cpu())) 
		save_model(data_, folder + '/' + path+'_data') 

		model = ParallelSyn(model_params)
		model.to(model_params['device'])
	
	trial = TrainParallelSyn(train_params) 
	t1 = time.time()
	count = 0
	for repeat in range(800):
		trial.train(model,  label, inputX,  t1)
		if trial.acc > 0.999999:
			plot_trial(trial, model, folder + '_png' + '/'+ path + '_true', repeat,time.time()-t1)
			torch.save(model.state_dict(), folder + '/' + path )
			save_model(trial, folder + '/' + path+'_trial') 
			break
		plot_trial(trial, model, folder + '_png' + '/'+ path , repeat,time.time()-t1)
		torch.save(model.state_dict(), folder + '/' + path ) 
		save_model(trial, folder + '/' + path+'_trial') 


