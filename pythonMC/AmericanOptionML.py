
# coding: utf-8

# In[3]:


import proj_helpers.FinObj as fin
import numpy as np
import proj_helpers.riskEngine as risk


# In[4]:


#Map for units of time
t_map = {'2y':0.5,'year':1,'semester':2,'quarter':4,'month':12,'week':52,'2day':180,'day':360}


# In[5]:


#MC Parameters
paths = 1000000
t_del = 1

t_unit = 200 #Unit of time: default = 1 for years, 12 for months, 360 for days etc...
time_s = 200 #Total number of time steps in t_units


# In[6]:


#Stock Parameters
spot = 100
vol = 0.11 #vol per year
rate = 0.01 #Rate in years
div = 0.0
expr = time_s/t_unit

#Share object
share1 = fin.Stock(spot,vol,t_unit,rate)


# In[7]:


strike=100
payoff=fin.amCallVec #Define option payoff

#Option Parameters
myOption = fin.Option(share1,payoff,time_s,strike,t_unit)

callbs=risk.BSCall(spot, strike, div, rate, vol, expr) #Compute black scholes price


# In[ ]:


#Test against BS Call Price for paths
import time
call_p = {} #Call prices as a function of path size
path_array = [10,100,250,500,1000,2000,5000,7500,10000,25000,50000,100000,500000,1000000]
for paths in path_array:
    t0=time.time()
    test=fin.bridgePayoffs(share1,myOption,paths,time_s,t_unit,scheme=fin.mc_Mil,anti=True,num_basis=4)
    call_p[paths]=np.mean(test)
    print("Time : "+str(time.time()-t0)+", Paths : "+str(paths)         +" LSM Price : "+str(call_p[paths])+" BS Price : "+str(callbs))


# In[ ]:


#Test against BS Call Price for time steps
paths = 75000
call_pst = {} #Call prices as a function of path size
step_array = [2,10,25,50,75,100,150,200]
for t in step_array:
    t0=time.time()
    test=fin.bridgePayoffs(share1,myOption,paths,t,t_unit,scheme=fin.mc_Mil,anti=True,num_basis=4)
    call_pst[t]=np.mean(test)
    print("Time : "+str(time.time()-t0)+", Time Steps : "+str(t)         +" LSM Price : "+str(call_pst[t])+" BS Price : "+str(callbs))

