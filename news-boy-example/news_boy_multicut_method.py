# %%
import cvxpy as cp
import numpy as np

# %%
# Data of the problem
cost = np.array([3]).reshape(-1,1)
all_demand = np.array([[1.5],[2],[3.5],[4],[5.5]]).reshape(-1,1)
selling_price = np.array([5]).reshape(-1,1)
salvage_price= np.array([1]).reshape(-1,1)
senario = len(all_demand)
senario_prob = (1/senario)*(np.ones_like(all_demand))

# %%
def solve_second_stage_subproblem(selling_price,salvage_price,supply,demand):

    sell = cp.Variable(shape=(1,1),name="sell")
    salvage = cp.Variable(shape=(1,1),name="salvage")

    objective = cp.Minimize(-1*(selling_price.T@sell)-1*(salvage_price.T@salvage))   # Objective function for second stage problem
    constraints =  [sell+salvage<=supply,sell<=demand,salvage>=0,sell>=0]  # Constriants for second stage problem

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem

# %%
def solve_first_stage_problem(produce,g_ks,alpha_ks,senario_prob,all_demand):
    
    # Intiatel variables for the problem
    produce  = cp.Variable(shape=(1,1),name="produce")
    v = cp.Variable(shape=(len(senario_prob),1),name="value")

    # Contraints for first stage problem
    constraints = []
    for i in range(0,len(g_ks)):
        for j in range(0,len(g_ks[i])):
            constraints.append(np.array([g_ks[i][j]]).T@produce+np.array([alpha_ks[i][j]])<=v[j])

    constraints.extend([produce>=0,v>=-1000000,produce<=max(all_demand)])

    objective = cp.Minimize((cost.T@produce)+(senario_prob.T@v))

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem

# %%
produce = np.array([3])            # Initial Guess
 
# Intitalize g and alpha for storing gs and alphas for each cut 
g_ks = []
alpha_ks = []

iter=0
objctive_values  = [np.nan]
epsilon = 10**(-4)
while True:

    # Solve Second stage problem for each demand and store its duals and objective values
    duals = []
    objs= []
    for demand in all_demand:

        second_stage_sol = solve_second_stage_subproblem(selling_price,salvage_price,produce,demand)

        temp_dual = second_stage_sol.constraints[0].dual_value   # Take the duals of 1st contraint
        temp_obj = second_stage_sol.value                        # Take the objective value of second stage problem

        # Store duals and objective values for each senario
        duals.append(temp_dual) 
        objs.append(temp_obj)

    # Reshaping the values 
    duals = np.array(duals).reshape(-1,1)
    objs = np.array(objs).reshape(-1,1)

    gks_batch = []
    alpha_ks_batch = []

    for i in range(0,senario):
        gks_batch.append(-duals[i])
        alpha_ks_batch.append(objs[i]+duals[i]@produce)
  
    g_ks.append(gks_batch)
    alpha_ks.append(alpha_ks_batch)

    # Solve the first stage problem
    first_stage_sol = solve_first_stage_problem(produce,g_ks,alpha_ks,senario_prob,all_demand)

    obj_value = first_stage_sol.value
    new_produce = first_stage_sol.var_dict["produce"].value
    new_limit = first_stage_sol.var_dict["value"].value

    produce,limit = new_produce,new_limit # swap the values

    if np.abs(obj_value - objctive_values[-1])<= epsilon:
        print("Terminating condition satisfied !")
        break
    else:
        pass

    objctive_values.append(obj_value)

    produce,limit = new_produce,new_limit # swap the values
    iter = iter +1
    
    print(f"\n----------Iteration no.  {iter}--------------------")
    print(f"\nproduction is {produce[0][0]}")
    print(f"\nobjctive value is {first_stage_sol.value}\n")

# %%



