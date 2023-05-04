import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import time
import  matplotlib.pyplot as plt
import gym
import  random
import math
import numpy as np

class env:
    
    
    def __init__(self):
        super(env, self).__init__()
        self.state=np.array([50,0,30,0.5,0.5,0.5,0.5,0.5]).astype(np.float32)
        self.θ=np.zeros([50,50],dtype='complex')
        for i in range(50):
            # t=np.random.random()
            t=0.5
            self.θ[i][i]=math.cos(2*math.pi*t)+1j*math.cos(2*math.pi*t)
            
        M=8
        N=50
        P=10**(5/10)/1000  #5dBm 功率 
        self.w=np.ones([M,1],dtype=complex)
        for i in range(M):
            self.w[i]=(0.019881768219173 + 0.000000011251913j)
        self.Hr_H=np.ones([1,N])
        self.Hd_H=np.ones([1,M])
        k=10**(1000/10); #1000db
        self.G=np.zeros([N,M])
        self.g=np.sqrt(k/(k+1))+np.multiply(np.sqrt(1/(k+1)),(np.random.randn(1,M)+1j*np.random.randn(1,M)))/np.sqrt(2);
        self.users=[(48,-3),(49,5),(46,1)]
        # self.users=[(50,0)]
    def reset(self): # 获取state
        # 30 --> 5
        self.state=np.array([50,0,5,0.5,0.5,0.5,0.5,0.5]).astype(np.float32)
        self.θ=np.zeros([50,50],dtype='complex')
        for i in range(50):
            # t=np.random.random()
            t=0.5
            self.θ[i][i]=math.cos(2*math.pi*t)+1j*math.cos(2*math.pi*t)
        return self.state    
    
    def step(self, action):
        
        # b=[]
        # for aa in action:
        #     aa = aa *2-1
        #     b.append(aa)
        # action=b   # 0-1 -> -1,1
        # action =[ ]
        reward = 0
        # (-pi,pi)
        # 水平
        a=action[0]*0.1#*math.pi
        # 竖直
        b=action[1]*0.1#*math.pi 
        
        v_max=1
        v=(action[2])*0.1#*0.5+0.5)*v_max
        # v=0.5
        
        detla_x=a#v*math.cos(b)*math.cos(a)
        detla_y=b#v*math.cos(b)*math.sin(a)
        detla_z=v#v*math.sin(b)
        
        if (self.state[0]+detla_x)>=25 and (self.state[0]+detla_x)<=75:
            self.state[0]+=detla_x
        else:
            pass
            # return self.state, 0, False,None,None
        
        if self.state[1]+detla_y>=-25 and self.state[1]+detla_y<=25:
            self.state[1]+=detla_y
        else:
            pass
            # return self.state, 0, False,None,None
        
        if self.state[2]+detla_z>=2 and self.state[2]+detla_z<=10:
            self.state[2]+=detla_z
        else:
            pass
            # return self.state, 0, False,None,None
        
        
        for i in range(5):
            angle=action[i+3]*0.1
            self.state[i+3]+=angle        
            if self.state[i+3]<0:
                self.state[i+3]+=1
            if self.state[i+3]>=1:
                self.state[i+3]-=1

        for i in range(5):
            # t=np.random.random()
            t=(self.state[i+3])
            for j in range(10):
                self.θ[10*i+j][10*i+j]=math.cos(2*math.pi*t)+1j*math.cos(2*math.pi*t)
        

        reward = self.output(self.state[0],self.state[1],self.state[2])
        # reward=reward[0][0]
                    
        s_ = self.state
        done  = False
        
        
        # s_ 为 array([ , , ,])   
        # return s_, np.array([reward]).astype(np.float32), done,None,None
        return s_, reward, done,None#,None


    def one_object_output(self,x,y,z,x_user,y_user):
        
        distance_irs_user=np.sqrt((x-x_user)**2+(y-y_user)**2+z**2)
        distance_user_ap=np.sqrt((x_user)**2+(y_user)**2+2**2)
        distance_ap_irs=np.sqrt((0-x)**2+(0-y)**2+(2-z)**2)
        g2=np.multiply(self.g,np.sqrt(10**(-3)*10**(5/10)*distance_ap_irs**(-2.2)))
        for i in range(50):
            self.G[i]=g2

        d_hr=distance_irs_user
        d_hd=distance_user_ap
        
        hr_H=np.multiply(self.Hr_H,np.sqrt(10**(-3)*10**(-10/10)*10**(5/10)*d_hr**(-3)))
        hr=np.conj(hr_H).T
        hd_H=np.multiply(self.Hd_H,np.sqrt(10**(-3)*10**(-10/10)*d_hd**(-3)))#;
        hd=np.conj(hd_H).T
        
        r = abs(np.dot(np.dot(np.dot(hr_H,self.θ),self.G)+hd_H,self.w))**2
        return r[0][0]
    
    # def output(self,d0,dr):
    def output(self,x ,y ,z):
        total_r=0
        arr=[]
        for p in self.users: 
            one_r=self.one_object_output(x,y,z,p[0],p[1])
            total_r+=one_r      
            arr.append(one_r) 
        return total_r/len(self.users)#-(np.var(arr)*(10**9.5))
    