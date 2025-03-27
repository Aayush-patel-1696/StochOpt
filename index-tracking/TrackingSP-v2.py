import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

with open(r'C:\Users\aayus\Documents\GitHub\StochOpt\index-tracking\returns_500.dat','r') as f:
    Assets, Scenarios = [int(x) for x in next(f).split()] 
    print("Assets",Assets,", Scenarios",Scenarios)
    Return = np.array([float(x) for line in f for x in line.split()]).reshape(Assets,Scenarios)

N_assets = 502
Return = Return[:N_assets+1,:Scenarios]         
Prob = np.ones(Scenarios)/Scenarios
IndexReturn = Return[0]         # row 0 return is the index to track

     
# create a Gurobi model

MM = gp.Model()

# decision variables

Upper = np.ones(Assets)
Upper[0]=0.0                   # do not use the index to track the index


Portfolio = MM.addVars(Assets,ub = Upper)
Portfolio[Assets-1].lb = -1.0  # the last row are just zeros
Total = MM.addVar(ub = 1.0)
Error = MM.addVars(Scenarios, lb=float('-inf')) # in this version we allow negative errors
PortfolioReturn = MM.addVars(Scenarios, lb=float('-inf'))

# objective

Lambda = 0.003                   # lasso regularization

MM.setObjective( sum(Error[k]*Error[k] for k in range(Scenarios)) + Lambda*Total,GRB.MINIMIZE)

# constraints

MM.addConstr(sum(Portfolio[i] for i in range(Assets-1)) == Total)
MM.addConstrs(PortfolioReturn[k] == sum(Portfolio[i] * Return[i,k] for i in range(Assets)) for k in range(Scenarios))
MM.addConstrs(Error[k] == PortfolioReturn[k] - IndexReturn[k] for k in range(Scenarios))
MM.addConstr(sum(Error[k] for k in range(Scenarios)) >= 0)  
    
# Solve

MM.setParam("Method", 1) 
MM.setParam("OptimalityTol", 1.0e-8) 
MM.setParam("Presolve", 0)

MM.optimize()

Final = np.empty(Assets)
Card = 0
for i in range(Assets-1):
    if (Portfolio[i].x > 1.0e-5):
        #print(i,'  ', round(Portfolio[i].x,4))
        Card += 1
    else:
        Portfolio[i].ub = 0.0       # force very small entries to zero, fix existing zeros
 
print('Portfolio Cardinality', Card)

Total.Obj = 0.0                     # remove the lasso regularization term

MM.optimize()

Card = 0

for i in range(Assets-1):
    if (Portfolio[i].x > 1.0e-5):
        print(i,'  ', round(Portfolio[i].x,4))   
        Card += 1
 
print('Portfolio Cardinality', Card)

ErrorSum = 0.0
for k in range(Scenarios):
    ErrorSum += Error[k].x
print('PureProfit', ErrorSum) 

CorrectedReturn = np.empty(Scenarios)
for k in range(Scenarios):
    CorrectedReturn[k] = PortfolioReturn[k].x  - max(0,ErrorSum)/Scenarios      # correct the bias by withdrawing cash

Time = np.linspace(0, Scenarios, Scenarios+1)
  
IndexWealth = np.empty(Scenarios+1)
PortfolioWealth = np.empty(Scenarios+1)
IndexWealth[0] = 1.0
PortfolioWealth[0] = 1.0
for k in range(Scenarios):
    IndexWealth[k+1] = IndexWealth[k] * (1.0 + IndexReturn[k])
    PortfolioWealth[k+1] = PortfolioWealth[k] * (1.0 + CorrectedReturn[k])
        
fig, Wealth = plt.subplots(1,1,figsize = (24, 8))

Wealth.plot(Time, IndexWealth, color="blue", label="Index")
Wealth.plot(Time, PortfolioWealth, color="green", label="Portfolio")
Wealth.set(xlabel="Time", ylabel="Cumulative Wealth", title="")
Wealth.legend()

plt.show()