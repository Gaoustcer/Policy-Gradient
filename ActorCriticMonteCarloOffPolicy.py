import torch
import torch.nn as nn
import gym
import numpy as np
from random import random
from random import randint
from datetime import datetime
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
def make_env():
    return gym.make('CartPole-v0')
class Net(nn.Module):
    def __init__(self,device,input_num=4,output_num=2) -> None:
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_num,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,out_features=output_num)
        ).to(device)
        self.device = device
    
    def forward(self,input):
        if isinstance(input,np.ndarray):
            input = torch.from_numpy(input).to(self.device).to(torch.float32)
        return self.net(input)

class Actor(nn.Module):
    def __init__(self,device) -> None:
        super(Actor,self).__init__()
        self.base = Net(device)
        self.device = device
        self.softmax = nn.Softmax().to(self.device)
    
    def forward(self,input):
        # input = torch.from_numpy(input).to(self.device).to(torch.float32)
        return self.softmax(self.base(input))
MAX_SIZE = 1024
BATCH_SIZE = 64
NUM_TEST = 64
class ActorCritic(object):
    def __init__(self,train_epoch = 400,collect_eposide = 32,device = torch.device('cuda:0'),epsilon = 0.1,gamma = 0.9) -> None:
        self.trainepoch = train_epoch
        self.trainenv = make_env()
        self.testenv = make_env()
        # self.memory = np.random.random((MAX_SIZE,10))
        self.rewardlist = []
        self.actionlist = []
        self.currentstatelist = []
        self.nextstatelist = []
        self.probabilitylist = []
        self.memoryindex = 0
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.collecteposide = collect_eposide
        self.targetactor = Actor(self.device)
        self.actor = Actor(self.device)
        self.critic = Net(self.device)
        self.targetcritic = Net(self.device)
        self.criticoptimizer = torch.optim.Adam(self.critic.parameters(),lr=0.01)
        self.actoroptimizer = torch.optim.Adam(self.actor.parameters(),lr=0.01)
        self.batchsize = BATCH_SIZE
        self.TDloss = nn.MSELoss()
        self.numtest = NUM_TEST
        self.testtime = 1
        self.writer = SummaryWriter('logs/offlinePG/'+datetime.now().strftime('%m%d_%H%M%S'))

    
    def getaction(self,state,mode = 'collect'):
        '''
        mode is collect means use targetnet to determine what action todo
        model is evaluate means use net to evaluate the performance of net
        '''
        if random() < self.epsilon:
            return randint(0,1),0.5
        else:
            # print("state is",state)
            # print("type of state is",type(state))
            if mode == 'collect':
                prob = self.targetactor(state)
            elif mode == 'evaluate':
                prob = self.actor(state)
            else:
                raise NotImplementedError
            # prob = self.actor(state)
            action = torch.distributions.Categorical(prob).sample().item()
            # print("call the getaction",mode,action,prob[action])
            return action,prob[action]
            # return torch.distributions.Categorical(prob).sample().item()
            # return int(torch.argmax(state))
    def collectoneepisode(self):
        done = False
        state = self.trainenv.reset()
        self.actionlist = []
        self.rewardlist = []
        self.currentstatelist = []
        self.nextstatelist = []
        self.probabilitylist = []
        while done == False:
            # print(self.getaction(state,mode='collect'))
            action,prob = self.getaction(state,mode='collect')
            next_state, reward, done, info = self.trainenv.step(action)
            reward = self.getrealreward(next_state)
            self.actionlist.append(action)
            self.currentstatelist.append(state)
            self.rewardlist.append(reward)
            self.nextstatelist.append(next_state)
            self.probabilitylist.append(prob)
            state = next_state
    def getrealreward(self,state):
        x,x_dot,theta,theta_dot = state
        r1 = (self.trainenv.x_threshold - abs(x)) / self.trainenv.x_threshold - 0.8
        r2 = (self.trainenv.theta_threshold_radians - abs(theta)) / self.trainenv.theta_threshold_radians - 0.5
        return r1 + r2
    # def collect(self):
    #     collecttime = 0
    #     while collecttime < self.collecteposide:
    #         current_state = self.trainenv.reset()
    #         # action = self.getaction(current_state)
    #         done = False
    #         while done == False:
    #             action = self.getaction(current_state)
    #             next_state,reward,done,info = self.trainenv.step(action)
    #             reward = self.getrealreward(next_state)
    #             self.memory[self.memoryindex] = np.hstack((current_state,(action,reward),next_state))
    #             self.memoryindex += 1
    #             self.memoryindex %= MAX_SIZE
    #             current_state = next_state
    #         collecttime += 1
    
    def CriticUpdate(self):
        # update the Critic Net for Action Value Evaluation
        # sample from memory replay buffer s,a,r,s_
        # TD error is $Q(s,a)-(r+gamma*\max_{a_}Q(s_,a_))$
        self.criticoptimizer.zero_grad()
        # index = np.random.choice(MAX_SIZE,BATCH_SIZE)
        # sampledata = self.memory[index,:]
        # currentstate = sampledata[:,:4]
        action = torch.tensor(self.actionlist).to(self.device).to(torch.int64)
        reward = torch.tensor(self.rewardlist).to(self.device).to(torch.float32)
        currentstate = torch.from_numpy(np.array(self.currentstatelist)).to(self.device).to(torch.float32)
        nextstate = torch.from_numpy(np.array(self.nextstatelist)).to(self.device).to(torch.float32)
        # action = torch.from_numpy(sampledata[:,4]).to(self.device).to(torch.int64)
        # reward = torch.from_numpy(sampledata[:,5]).to(self.device).to(torch.float32)
        # nextstate = sampledata[:,6:]
        TDreward = reward + self.gamma * torch.max(self.targetcritic(nextstate),dim=-1)[0].detach()
        currentreward = torch.gather(self.critic(currentstate),-1,action.unsqueeze(-1)).squeeze()
        loss = self.TDloss(TDreward,currentreward)
        loss.backward()
        self.criticoptimizer.step()
        pass
    def testresult(self):
        totalresult = 0
        for _ in range(self.numtest):
            state = self.testenv.reset()
            done = False
            
            while done == False:
                action,_ = self.getaction(state,mode='evaluate')
                state,reward,done,info = self.testenv.step(action)
                totalresult += reward
        self.writer.add_scalar('reward',totalresult/self.numtest,self.testtime)
        self.testtime += 1


    def ActorUpdate(self):
        # sample a batch data from memory s,a,r,s_
        # update the Actor based on optimization J(theta)
        # loss function is Q(s,a)\log \pi(a|s)
        # You need to seed Learning rate < 0 to gradient incredement
        # pi(a|s)/beta(a|s) Q(s,a) is the weight
        self.actoroptimizer.zero_grad()
        action = torch.tensor(self.actionlist).to(self.device).to(torch.int64)
        # reward = torch.tensor(self.rewardlist).to(self.device).to(torch.float32)
        currentstate = torch.from_numpy(np.array(self.currentstatelist)).to(self.device).to(torch.float32)
        currentprob =  self.actor(currentstate)
        currentprob = torch.gather(currentprob,-1,action.unsqueeze(0)).squeeze().detach().cuda()
        oldprob = torch.tensor(self.probabilitylist).detach().cuda()
        current_policy = self.actor(currentstate)
        policy = torch.gather(current_policy,-1,action.unsqueeze(-1)).squeeze()
        value = [sum(self.rewardlist[i:]) for i in range(len(self.rewardlist))]
        value = torch.tensor(value).to(self.device).to(torch.float32)
        weight = currentprob/oldprob * value.cuda().detach()
        # value = torch.gather(current_value,-1,action.unsqueeze(-1)).squeeze().detach()
        self.BCELoss = BCELoss(weight=weight)
        loss = self.BCELoss(policy,torch.ones(len(self.rewardlist)).to(self.device))
        loss.backward()
        self.actoroptimizer.step()
        pass
    def learn(self):
        # while True:
        #     done = False
        #     state = self.trainenv.reset()
        #     while done == False:
        #         action = self.getaction(state)
        #         next_state,reward,done,info = self.trainenv.step(action)
        #         reward = self.getrealreward(next_state)
        #         self.memory[self.memoryindex] = np.hstack((state,(action,reward),next_state))
        #         self.memoryindex += 1
        #         if self.memoryindex == MAX_SIZE:
        #             break
        #         state = next_state
        #     # print("Initial the buffer",self.memoryindex)
        #     if self.memoryindex == MAX_SIZE:
        #         break
        print("start to learn")
        self.memoryindex = 0
        from tqdm import tqdm
        for index in tqdm(range(self.trainepoch)):
            # self.collect()
            self.collectoneepisode()
            self.ActorUpdate()
            # self.CriticUpdate()
            self.testresult()
            # self.testenv()
            if index % 10 == 0:
                self.targetcritic.load_state_dict(self.critic.state_dict())
            if index % 100 == 0:
                self.targetactor.load_state_dict(self.actor.state_dict())
        




def funcall()->int:
    return random()
if __name__ == "__main__":
    Agent = ActorCritic()
    Agent.learn()
