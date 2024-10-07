import numpy as np 
import torch 
import datetime
import time
import matplotlib.pyplot as plt 
import pickle 
import argparse
from queue import deque
from torch.utils.data import DataLoader, Dataset

def generate_data_torch(nDimension = 10, nSample = 100, randomSeed = 2, device =torch.device('cpu')):
	torch.manual_seed(randomSeed)
	data = torch.rand(nDimension, nSample, dtype=torch.float16, device = device)
	label = torch.ones((nSample, 1), dtype=torch.int8,device = device).ravel()
	label[:int(nSample/2)] = -1
	return data, label

def hingeLoss(actv, theta, label, margin):
	return (torch.maximum(torch.zeros_like(actv), margin - (actv - theta) * label)).sum()

def custom_pdf(x, a = 0., b = .5):
	return b * np.abs(x-.5) + a
 
def inverse_cdf(p):
	x = np.linspace(0, 1, 1000) 
	cdf = np.cumsum(custom_pdf(x))/np.sum(custom_pdf(x))   
	return np.interp(p, cdf, x)

def save_model(obj, file):
	with open(file, 'wb') as outp:
		pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def plot_trial(trial, model, path, repeat, t):
    accList= [acc.cpu() for acc in trial.acc_history]
    lossList= [loss.cpu() for loss in trial.loss_history]
    fig = plt.figure(figsize = (8, 5))  
    plt.subplot(2,3,1)
    plt.plot(trial.time,accList)
    plt.title('acc. ={:.6f}, '.format(trial.acc.detach().cpu())+ str(repeat) + ' * '+str(trial.Nepoch) + ' epoch')
    plt.xlabel('time cost (s)')
    plt.grid() 
    plt.subplot(2,3,2)
    plt.plot(trial.time,lossList) 
    plt.title('loss\n '+'time past: '+str(datetime.timedelta(seconds=t)))
    plt.xlabel('time cost (s)')
    plt.grid() 
    plt.subplot(2,3,4)
    plt.hist(model.thres.detach().cpu().numpy().ravel(), bins = 30)
    plt.title('thres hist')
    plt.subplot(2,3,5)
    plt.hist(model.slope.detach().cpu().numpy().ravel(), bins = 30)
    plt.title('slope hist')
    plt.subplot(2,3,6)
    plt.hist(model.ampli.detach().cpu().numpy().ravel(), bins = 30)
    plt.title('ampli hist, {:.3f} of ampli < {:.3f}'.format((model.ampli.detach().cpu().numpy().ravel() < trial.minAmpli).mean(), trial.minAmpli))
    plt.tight_layout()
    plt.savefig(path + '.png')
    plt.close(fig)
    # plt.show() 

class ParallelSyn(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        for k in params:
            setattr(self, k, params[k])
        torch.manual_seed(self.seed)
        self.ampli = torch.nn.Parameter(torch.rand( self.N, self.M,device = params['device']))  
        self.slope = torch.nn.Parameter(torch.rand(self.N, self.M, device = params['device'])*500) 
        self.thres = torch.nn.Parameter(torch.rand(self.N, self.M, device = params['device'])) 
    def forward(self, data):  
        
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
        self.loss = hingeLoss(model.actv,  label, self.margin) 
    def accu(self, model, label):
        self.acc = (torch.sign(model.actv ) == label).sum()/self.P
    def train(self, model, label, inputX, batchSize,t1):
        self.optim = torch.optim.Adam([
            {"params": model.ampli},
            {"params": model.slope},
            {"params": model.thres, 'lr': self.threslr}
            ], lr = self.adamlr) 
        self.thresPool = torch.tensor(inverse_cdf(np.random.uniform(size= (self.NthresPool,1))), device = model.device).float()#.half()
        train_dataset = MyDataset(inputX, label)
        train_loader = DataLoader(train_dataset, batch_size=train_params['batchSize'])
        for k in range(self.Nepoch): 
            for batch_idx, (data, target) in enumerate(train_loader):
                self.shuffle_invalid(model)
                with torch.no_grad():
                    model.slope.clamp_min_(0)
                model.forward(data) 
                self.lossFunc(model, target)
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

	model_params = {
		 'N': args.N, # input dimension
		 'M': args.M,# parallel synapse number 
		 'seed': args.seed,
		 'device': torch.device('cuda:0')
		 }   
	train_params = {
			  'margin': 0.1, # only applied when 'loss' is hinge
			  'threslr': 1e-6,
			  'adamlr': 0.003,
		'minAmpli': 1e-1,
		'Nepoch': 8000,
		'P': args.P,  
		'maxRecord': 4000,
		'downSample': 50,
		'NthresPool': int(args.P/2)
	}  

	inputX, label = generate_data_torch(nDimension = model_params['N'], \
									nSample = train_params['P'], \
									randomSeed = model_params['seed'],
								   device = model_params['device']) 
	path = ''
	folder = './N_'+str(model_params['N'])
	path += 'N_'+str(model_params['N']) + '_M_'+str(model_params['M'])\
	     +'_P_' +str(train_params['P']) + '_seed_'+str(model_params['seed'])

	data_ = torch.vstack((inputX.cpu(),label.cpu())) 
	save_model(data_, folder + '/' + path+'_data') 
	
	model = ParallelSyn(model_params)
	model.to(model_params['device'])
	
	trial = TrainParallelSyn(train_params)
	t1 = time.time()

	for repeat in range(4000):
		trial.train(model,  label, inputX)
		if trial.acc > 0.999999:
			plot_trial(trial, model, folder + '_png' + '/'+ path + '_true', repeat,t2-t1 )
			save_model(model.state_dict(), folder + '/' + path)
			break
		
		t2 = time.time()
		plot_trial(trial, model, folder + '_png' + '/'+ path, repeat,t2-t1)
		torch.save(model.state_dict(), folder + '/' + path) 



