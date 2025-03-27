import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time

start = time.time()

with open(r'C:\Users\aayus\Documents\GitHub\StochOpt\index-tracking\returns_500.dat','r') as f:
    Assets, TotalScenarios = [int(x) for x in next(f).split()] 
    print("Assets",Assets,", Scenarios",TotalScenarios)
    Raw_Return = np.array([float(x) for line in f for x in line.split()]).reshape(Assets,TotalScenarios)

Scenarios = 70
N_Assets = 100
Prob = np.ones(Scenarios)/Scenarios
IndexReturn = Raw_Return[0,:Scenarios]         # row 0 return is the index to track

Return = Raw_Return[:N_Assets+1,:Scenarios]
Assets = len(Return)-1

max_assets = 20
              
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

# Lambda = 0.003                   # lasso regularization

MM.setObjective( sum(Error[k]*Error[k] for k in range(Scenarios)))


# constraints

MM.addConstr(sum(Portfolio[i] for i in range(Assets-1)) == Total)
MM.addConstrs(PortfolioReturn[k] == sum(Portfolio[i] * Return[i+1,k] for i in range(Assets-1)) for k in range(Scenarios))
MM.addConstrs(Error[k] == PortfolioReturn[k] - IndexReturn[k] for k in range(Scenarios))
MM.addConstr(sum(Error[k] for k in range(Scenarios)) >= 0)  

z = pd.Series(MM.addVars(Assets,
                        vtype = gp.GRB.BINARY))
for i in range(Assets-1):
    MM.addConstr(Portfolio[i] <= z[i], 
                f'dummy_restriction_{i}')

# [NEW] sum(z_i) <= max_assets: number of assets constraint
MM.addConstr(z.sum() <= max_assets, 'max_assets_restriction')
# Solve

MM.setParam("Method", 1) 
MM.setParam('OutputFlag', 0)
MM.setParam('TimeLimit', 60*5) # in secs
MM.setParam("OptimalityTol", 1.0e-8) 
MM.setParam("Presolve", 0)

MM.optimize()

end = time.time()
print("Time taken: ", end-start)

# Final = np.empty(Assets)
# Card = 0
# for i in range(Assets-1):
#     if (Portfolio[i].x > 1.0e-5):
#         #print(i,'  ', round(Portfolio[i].x,4))
#         Card += 1
#     else:
#         Portfolio[i].ub = 0.0       # force very small entries to zero, fix existing zeros
 
# print('Portfolio Cardinality', Card)



# z = pd.Series(MM.addVars(Assets,
#                         vtype = gp.GRB.BINARY))
# for i in range(Assets-1):
#     MM.addConstr(Portfolio[i] <= z[i], 
#                 f'dummy_restriction_{i}')

# # [NEW] sum(z_i) <= max_assets: number of assets constraint
# MM.addConstr(z.sum() <= max_assets, 'max_assets_restriction')

# Total.Obj = 0.0                     # remove the lasso regularization term

# MM.optimize()

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
print("Time taken:", end-start)

CorrectedReturn = np.empty(Scenarios)
Testreturn = Raw_Return[:N_Assets+1,Scenarios:]
IndexTestReturn = Raw_Return[0,Scenarios:]
test_scenarios = TotalScenarios-Scenarios
PortfolioTestReturn = np.empty(test_scenarios)
for k in range(test_scenarios):
    PortfolioTestReturn[k] = sum(Portfolio[i].x * Testreturn[i+1,k] for i in range(Assets-1))       # correct the bias by withdrawing cash


# for k in range(Scenarios):
#     CorrectedReturn[k] = PortfolioReturn[k].x  - max(0,ErrorSum)/Scenarios      # correct the bias by withdrawing cash

Time = np.linspace(0, test_scenarios, test_scenarios+1)
  
IndexWealth = np.empty(test_scenarios+1)
PortfolioWealth = np.empty(test_scenarios+1)
IndexWealth[0] = 1.0
PortfolioWealth[0] = 1.0
for k in range(test_scenarios):
    IndexWealth[k+1] = IndexWealth[k] * (1.0 + IndexTestReturn[k])
    PortfolioWealth[k+1] = PortfolioWealth[k] * (1.0 + PortfolioTestReturn[k])
        
fig, Wealth = plt.subplots(1,1,figsize = (24, 8))

Wealth.plot(Time, IndexWealth, color="blue", label="Index")
Wealth.plot(Time, PortfolioWealth, color="green", label="Portfolio")
Wealth.set(xlabel="Time", ylabel="Cumulative Wealth", title="")
Wealth.legend()

plt.show()

plt.savefig('index_tracking_MIP.png')