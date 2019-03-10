#/*Define classes and functions for multi-underlying exotic products Monte Carlo*/
#Anthony Maylath 2/21/2019
import numpy as np
import sys

def iscorrel(corr):
    '''
    Check if a square matrix is a correlation matrix
    '''
    rows, cols = np.shape(corr)
    for i in range(0,rows):
        for j in range(0,cols):
            if abs(corr[i][j]) > 1:
                print("Error: did not recieve a valid correlation matrix" \
                    + " row " + str(i) + " column " + str(j) + " has value " + str(corr[i][j]))
                return False
            if i == j and corr[i][j] != 1:
                print("Error: did not recieve a valid correlation matrix. Diagonals are not all 1")
                return False
            
    return True

class MultiDes:
	'''
	MultiUnderlying Exotic Product
	'''
	def makeSchedule(self, call_dates,t_unit):
		'''
		Create autocall schedule from units and call_dates
		Note: call_dates must divide t_unit
		'''
		if t_unit%call_dates != 0:
			print("Error: t_unit must be divisible by number of calldates")
			sys.exit()

		self.schedule = [x/call_dates*t_unit for x in range(1,call_dates+1)]

	def setCorrelation(self, corr):
		'''
		Perform checks to verify correlation matrix
		'''
		n = len(self.stock)
		rows, cols = np.shape(corr)
		if n != rows or n != cols:
			print("Error: expecting square matrix of size " + str(n)\
				+" but got matrix of size " + str(rows) + "x" + str(cols))
			sys.exit()

		#Ensure the values in the correlation matrix are bt -1,1
		assert(iscorrel(corr))

		self.corr = corr


	def __init__(self, stock, C, payoff, ssize, downin, exp, strikes=[100],\
				call_dates=1, t_unit=1, corr=np.array([])):
		'''
		stock : an array of stock objects
		C : coupon
		payoff : payoff function for the product
		corr : correlation ~ square matrix with correlations of each of the shares
		'''
		self.stock = stock
		self.C = C/t_unit #Coupon
		self.payoff = payoff #Payoff function
		self.ssize = ssize #Notional
		self.downin = downin #Down and in Barrier
		self.exp = exp #Time to expiration
		self.strikes = strikes #Fixed strike prices (list)
		self.t_unit = t_unit

		#Initiate schedule
		self.makeSchedule(call_dates,t_unit)

		#Set correlation
		self.setCorrelation(corr)

	def rescale(self, t_unit):
		'''
		Change scaling of coupon
		1 = years, 360 = days, 12 = months, etc...
		'''
		self.C = self.C*self.t_unit/t_unit
		self.t_unit = t_unit

	def setStrike(self,strikes):
		'''
		Set the strike vector of the note
		'''
		self.strikes = strikes

	def getChol(self):
		'''
		Compute Cholesky Decomposition of correlation matrix
		'''
		self.chol = np.linalg.cholesky(self.corr)

def fixMultiCall(spots, note, t):
	'''
	Autocall with fixed strikes
	spots : list of current spot prices
	note : exotic note object
	t : current time
	'''
	if t in note.schedule:
		#Check if all spots are above barrier
		if all(sp > sk for sp, sk in zip(spots,note.strikes)): return note.ssize*(1+t*note.C)
		#Check if the note matures
		if t == note.exp:
			#If any of the performances are negative, give poor performance
			if any(sp < sk*note.downin for sp, sk in zip(spots,note.strikes)):
				return min(sp/sk for sp, sk in zip(spots,note.strikes))
			#Otherwise, normal call event
			else:
				return note.ssize*(1+t*note.C)
	#If not in note schedule, return 0
	return 0


