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
        # 增加用户移动性
        self.steps=0
        
        super(env, self).__init__()
        self.state=np.array([50,0,5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]).astype(np.float32)
        self.θ=np.zeros([50,50],dtype='complex')
        for i in range(50):
            # t=np.random.random()
            # t=0.5
            t=i%10
            self.θ[i][i]=math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)+1j*math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)
            # self.θ[i][i]=math.cos(2*math.pi*t)+1j*math.cos(2*math.pi*t)
            
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
        # self.users=[(48,-3),(49,5),(46,1),(30,10),(70,-20),(39,-18),(55,19),(70,11),(37,8),(26,-12.5)]
        # self.users=[(48,-3),(49,5),(46,1)]
        # self.users=[(48,-3),(48,3)]
        self.users=[(48,20)]#,(48,-20)
        self.users_angle=[]
        for u in self.users:
            user_x=u[0]
            user_y=u[1]
            user_z=0
            irs_x =self.state.tolist()[0]
            irs_y =self.state.tolist()[1]
            irs_z =self.state.tolist()[2]
            o = np.arctan(( irs_x-user_x ) / (irs_z-user_z))/(math.pi*2)
            p = np.arctan(( irs_y-user_y ) / (irs_z-user_z))/(math.pi*2)
            self.users_angle.append((o,p))
        # 10 users
        # self.users=[(50,0)]
    def reset(self): # 获取state
        # 30 --> 5
        # self.state=np.array([50,0,5,0.5,0.5,0.5,0.5,0.5]).astype(np.float32)
        self.state=np.array([50,0,5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]).astype(np.float32)
        self.steps=0
        # self.users=[(48,-3),(49,5),(46,1)]
        self.users=[(48,20)]#,(48,-20)
        # self.θ=np.zeros([50,50],dtype='complex')
        # for i in range(50):
        #     t=i%10
        #     self.θ[i][i]=math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)+1j*math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)
           
        # self.θ=np.zeros([50,50],dtype='complex')
        # for i in range(50):
        #     # t=np.random.random()
        #     t=0.5
        #     self.θ[i][i]=math.cos(2*math.pi*t)+1j*math.cos(2*math.pi*t)
        return self.state    
    
    def step(self, action):
        #********************mobile********************#
        # self.steps+=1
        # temp=[]
        # for u in self.users:
        #     x=u[0]
        #     y=u[1]
        #     x=x+5/10000
        #     temp.append((x,y))
        # self.users=temp          
        #********************mobile********************#
          
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
        
        # for i in range(5):        
        for i in range(10):
            angle=action[i+3]*0.1
            self.state[i+3]+=angle        
            if self.state[i+3]<0:
                self.state[i+3]+=1
            if self.state[i+3]>=1:
                self.state[i+3]-=1

        # for i in range(5):
        #     # t=np.random.random()
        #     t=(self.state[i+3])
        #     t2=(self.state[i+3+5])
        #     for j in range(10):
        #         self.θ[10*i+j][10*i+j]=math.cos(2*math.pi*t)*math.cos(2*math.pi*t2) +1j*math.cos(2*math.pi*t)*math.cos(2*math.pi*t2)
        
        ###
        # self.θ=np.zeros([50,50],dtype='complex')
        # for i in range(50):
        #     t=i%10
        #     self.θ[i][i]=math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)+1j*math.cos(2*math.pi*t)*math.cos(2*math.pi*t+5)  
        ###

        reward = self.output()#self.state[0],self.state[1],self.state[2]
        # reward=reward[0][0]
                    
        s_ = self.state
        done  = False
        
        
        # s_ 为 array([ , , ,])   
        # return s_, np.array([reward]).astype(np.float32), done,None,None
        return s_, reward, done,None#,None


    def one_object_output(self,x,y,z,x_user,y_user,f):
        
        distance_irs_user=np.sqrt((x-x_user)**2+(y-y_user)**2+z**2)
        distance_user_ap=np.sqrt((x_user)**2+(y_user)**2+2**2)
        distance_ap_irs=np.sqrt((0-x)**2+(0-y)**2+(2-z)**2)
        g2=np.multiply(self.g,np.sqrt(10**(-3)*10**(5/10)*distance_ap_irs**(-2)))#-2.2
        for i in range(50):
            self.G[i]=g2

        d_hr=distance_irs_user
        d_hd=distance_user_ap
        
        hr_H=np.multiply(self.Hr_H,np.sqrt((10**(-3))*(10**(-10/10))*(10**(5/10))*(d_hr**(-1))))#-3 # -1
        hr=np.conj(hr_H).T
        hd_H=np.multiply(self.Hd_H,np.sqrt((10**(-3))*(10**(-10/10))*(d_hd**(-1))))#;-3  #-0.8
        hd=np.conj(hd_H).T
        
        # r = abs(np.dot(np.dot(np.dot(hr_H,self.θ),self.G)+hd_H,self.w))**2
        
        θ=np.zeros([50,50],dtype='complex')
        for i in range(5):
            # t=np.random.random()
            t=(self.state[i+3])-f[0]
            if t<0:
                t+=1
            t2=(self.state[i+3+5])-f[1]
            if t2<0:
                t2+=1
            for j in range(10):
                θ[10*i+j][10*i+j]=math.cos(2*math.pi*t)*math.cos(2*math.pi*t2) +1j*math.cos(2*math.pi*t)*math.cos(2*math.pi*t2)
        
        r = abs(np.dot(np.dot(np.dot(hr_H,θ),self.G)+hd_H,self.w))**2
        return r[0][0]
    
    # def output(self,d0,dr):
    def output(self):
        total_r=0
        arr=[]
        ###
        self.users_angle=[]
        for u in self.users:
            user_x=u[0]
            user_y=u[1]
            user_z=0
            irs_x =self.state.tolist()[0]
            irs_y =self.state.tolist()[1]
            irs_z =self.state.tolist()[2]
            o = np.arctan(( irs_x-user_x ) / (irs_z-user_z))/(math.pi*2)
            p = np.arctan(( irs_y-user_y ) / (irs_z-user_z))/(math.pi*2)
            self.users_angle.append((o,p))
        ###
        for i in range(len(self.users)): 
            p= self.users[i]
            f= self.users_angle[i]
            one_r=self.one_object_output(self.state[0],self.state[1],self.state[2],p[0],p[1],f)
            total_r+=one_r      
            arr.append(one_r) 
        return total_r/len(self.users)#-(np.var(arr)*(10**9.5))
    
    
    '''
    total_steps 3000000 reward [0.02254932] s [29.925344  9.97646   2.00241 ]
    ans_max 
            9.771327027181558e-10 
    
    ans_state 
            [2.9915308e+01 1.0039918e+01 2.0000691e+00 1.7728329e-02 1.3621092e-02 9.8945028e-01 1.2707591e-02 8.6168144e-03]
    
    
    
    
    ans_max 3.0841003052306504e-06 ans_state [25.003265   23.018454    2.0010896   0.26124963  0.25815082  0.26471037
  0.26354408  0.26185796  0.7028972   0.70030254  0.7032219   0.7030627
  0.6925693 ]
    
    
    ans_max 6.248416411575145e-08 ans_state [ 25.005531   -22.44181      2.0071726    0.767906     0.27152106
   0.26432893   0.25604406   0.26406842   0.81904733   0.3174646
   0.31303293   0.32379565   0.3240798 ]
    '''
    