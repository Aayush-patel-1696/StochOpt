import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint          # not needed here but included for illustration
from scipy.optimize import Bounds

global Assets,Scenarios,Gamma

def utility_val(x):             # calculates the value and the gradient of the expected utility function
    
    global Realizations         # utility function realizations, will be used for Hessian calculations
    
    Realizations = np.exp(-Gamma*x[Assets:])
    val = sum(Realizations)/Scenarios
    gradient = (-Gamma/Scenarios) * Realizations
    gradient = np.append(np.zeros(Assets),gradient,axis=0)
    return (val,gradient)
         
def utility_Hp(x,p):            # returns Hessian of the expected utility function multiplied by a vector p
                                # this is easy because the Hessian is diagonal
    
    Hp =  Realizations * p[Assets:] * (Gamma**2/Scenarios)
    Hp = np.append(np.zeros(Assets),Hp,axis=0)
    return Hp
        
# read the data
  
with open(r'C:\Users\aayus\Documents\GitHub\StochOpt\disater-relief\returns_500.dat','r') as f:
    Assets, Scenarios = [int(x) for x in next(f).split()] 
    print("Assets",Assets,", Scenarios",Scenarios)
    Return = np.array([float(x) for line in f for x in line.split()]).reshape(Assets,Scenarios)
              
Prob = np.ones(Scenarios)/Scenarios
IndexReturn = Return[0]         # row 0 return is the index

Gamma = 20.0    # parameter of the exponential utility
    
# decision variables are Portfolio = x[0:Assets] and PortfolioReturn = x[Assets:Assets+Scenarios] 

# set the lower and upper bounds for the variables

LowBox = np.append(np.zeros(Assets) , - np.ones(Scenarios) , axis=0)
UppBox = np.append(0.1 * np.ones(Assets), np.ones(Scenarios), axis=0)

Box = Bounds(LowBox, UppBox)

# the constraints are Portfolio @ Return - PortfolioReturn = 0

ConsMat = np.append(Return.T, -np.identity(Scenarios), axis=1)

# add the constraint sum(Portfolio) = 1

NewRow = np.append(np.ones(Assets),np.zeros(Scenarios), axis=0)
ConsMat = np.append(ConsMat,[NewRow],axis=0)

Lower = np.append(np.zeros(Scenarios), 1)
Upper = Lower
constraint_vec = LinearConstraint(ConsMat,lb=Lower,ub=Upper)

# starting point

x0 = np.zeros(Assets+Scenarios)

# optimize

SolutionVec = minimize(utility_val, x0, method='trust-constr', jac=True, hessp=utility_Hp,
               constraints = constraint_vec, options={'verbose': 1,'maxiter': 1000}, bounds=Box)
               
Card = 0
Portfolio = SolutionVec.x[:Assets]
for i in range(Assets):
    if (Portfolio[i] > 1.0e-4):
        print(i,'  ', round(Portfolio[i],4))
        Card += 1  
 
print('Portfolio Cardinality', Card)

PortfolioReturn = SolutionVec.x[Assets:]

Time = np.linspace(0, Scenarios, Scenarios+1)
  
IndexWealth = np.empty(Scenarios+1)
PortfolioWealth = np.empty(Scenarios+1)
IndexWealth[0] = 1.0
PortfolioWealth[0] = 1.0
for k in range(Scenarios):
    IndexWealth[k+1] = IndexWealth[k] * (1.0 + IndexReturn[k])
    PortfolioWealth[k+1] = PortfolioWealth[k] * (1.0 + PortfolioReturn[k])
        
fig, Wealth = plt.subplots(1,1,figsize = (24, 8))

Wealth.plot(Time, IndexWealth, color="blue", label="Index")
Wealth.plot(Time, PortfolioWealth, color="green", label="Portfolio")
Wealth.set(xlabel="Time", ylabel="Cumulative Wealth", title="")
Wealth.legend()

plt.show()

