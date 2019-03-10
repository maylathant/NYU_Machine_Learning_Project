#Anthony Maylath 2/18/2019 ~ Define smoothing functions for nosiy valuation
import numpy as np

#Log functions
def genLog(x, a, b, c, d):
    return a * np.log(b * (x - c))+d

def derLog(x, a, b, c, d): #First log derivative
    return a/(x-c)

def gammaLog(x, a, b, c, d): #Second log derivative
    return -1*a/((x-c)*(x-c))

#General polynomial order 3
def genPol(x, a, b, c, d):
	return a*x*x*x + b*x*x + c*x + d

def derPol(x, a, b, c, d):
	return 3*a*x*x + 2*b*x + c

def gammaPol(x, a, b, c, d):
	return 6*a*x + 2*b

#Sigmoid function
def genSig(x, a, b, c , d):
	return c/(1+np.exp(-a*(x-b))) + d

def derSig(x, a, b, c , d):
	temp = a*c*np.exp(a*(x-b))/(np.exp(a*(x-b)) +1)
	return temp/(np.exp(a*(x-b)) +1)

def gammaSig(x, a, b, c , d):
	num = -a*a*c*(np.exp(a*(x-b)) - 1)*np.exp(a*(x-b))
	den = np.exp(a*(x-b)) + 1
	return num/den/den/den

#error function
def genErr(x, a, b, c , d):
	return a*np.exp(-b*(x-c)*(x-c)) + d

#exponential function
def genExp(x,a,b,c,d):
	return a*np.exp(b*(x-c)) + d
