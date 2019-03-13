
# coding: utf-8

# In[11]:


import proj_helpers.FinObj as fin
import numpy as np
import proj_helpers.riskEngine as risk


# In[12]:


#Map for units of time
t_map = {'2y':0.5,'year':1,'semester':2,'quarter':4,'month':12,'week':52,'2day':180,'day':360}


# In[13]:


#MC Parameters
paths = 50000
t_del = 1

t_unit = 200 #Unit of time: default = 1 for years, 12 for months, 360 for days etc...
time_s = 200 #Total number of time steps in t_units


# In[14]:


#Stock Parameters
spot = 100
vol = 0.20 #vol per year
rate = 0.05 #Rate in years
div = 0.0
expr = time_s/t_unit

#Share object
share1 = fin.Stock(spot,vol,t_unit,rate)


# In[15]:


strike=100
payoff=fin.amCallVec #Define option payoff

#Option Parameters
myOption = fin.Option(share1,payoff,time_s,strike,t_unit)

callbs=risk.BSPut(spot, strike, div, rate, vol, expr) #Compute black scholes price


# In[ ]:


#Plot BS call vs. LSM call against spot
import matplotlib.pyplot as pplot
spots=np.linspace(80,120,20)
call_bs = {}
call_lsm = {}
for s in spots:
    call_bs[s] = risk.BSCall(s, strike, div, rate, vol, expr)
    share1.setSpot(s) #Change spot price for LSM
    call_lsm[s] = fin.priceLSM(share1,myOption,paths,time_s,t_unit,anti=True,num_basis=4)
    print("Spot : " + str(s) + " BS Price : " + str(call_bs[s]) + " LSM Price " + str(call_lsm[s]))


# In[ ]:


pplot.plot(spots,call_bs.values(),label='Black Scholes')
pplot.plot(spots,call_lsm.values(),label='My LSM')
pplot.legend()
pplot.xlabel('Spot')
pplot.ylabel('Price')
pplot.show()


# In[ ]:


# #Test against BS Call Price for paths
# import time
# call_p = {} #Call prices as a function of path size
# path_array = [10,100,250,500,1000,2000,5000,7500,10000,25000,50000,100000,500000,1000000]
# for paths in path_array:
#     t0=time.time()
#     test=fin.bridgePayoffs(share1,myOption,paths,time_s,t_unit,scheme=fin.mc_Mil,anti=True,num_basis=4)
#     call_p[paths]=np.mean(test)
#     print("Time : "+str(time.time()-t0)+", Paths : "+str(paths)\
#          +" LSM Price : "+str(call_p[paths])+" BS Price : "+str(callbs))


# In[ ]:


# #Test against BS Call Price for time steps
# paths = 75000
# call_pst = {} #Call prices as a function of path size
# step_array = [2,10,25,50,75,100,150,200]
# for t in step_array:
#     t0=time.time()
#     test=fin.bridgePayoffs(share1,myOption,paths,t,t_unit,scheme=fin.mc_Mil,anti=True,num_basis=4)
#     call_pst[t]=np.mean(test)
#     print("Time : "+str(time.time()-t0)+", Time Steps : "+str(t)\
#          +" LSM Price : "+str(call_pst[t])+" BS Price : "+str(callbs))

