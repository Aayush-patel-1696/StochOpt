# %%
import cvxpy as cp
import numpy as np
from itertools import product

# %%
shipping_cost = np.array([[1,3,2],[3,2,2]]).reshape(2,3)
given_demand = np.array([[80,120,150],[120,180,200],[100,150,180]]).reshape(3,3)   #np.array([[100,150,120],[120,180,150],[150,200,180]]).reshape(3,3)
selling_price = np.array([[16,16]]).reshape(2,1)
produce = np.array([0,0]).reshape(2,1)
production_cost = np.array([11,10]).reshape(2,1)
produce_limit = np.array([300,400]).reshape(2,1)
senario = 27
senario_prob = (1/senario)*(np.ones((senario,1)))

# %%
senario_comb = []
index = np.arange(0,3)
index_permute = product(index,repeat=3)
for i in index_permute:
    temp_array =[]
    for j in range(0,3):
        temp_array.append(float(given_demand[j,:][i[j]]))
    senario_comb.append(temp_array)
all_demand = np.array(senario_comb).reshape(27,3)

# %%
all_demand

# %%
def solve_second_stage_problem(produce,demand,shipping_cost):
 
    sell = cp.Variable(shape=(2,3))
    salvage = cp.Variable(shape=(2,1))

    objective = cp.Minimize(-16*cp.sum(sell,keepdims=True) + cp.trace(((shipping_cost@(sell.T)))))
    constraints = [cp.sum(sell,axis=1,keepdims=True) +salvage <= produce,cp.sum(sell,axis=0,keepdims=True) <= demand.reshape(1,-1), sell>= np.zeros_like(sell),salvage>= np.zeros_like(salvage)]

    problem = cp.Problem(objective, constraints)
    problem.solve()
 
    return problem

# %%
def solve_first_stage_problem(produce,g_ks,alpha_ks,senario_prob,production_cost,produce_limit): 

    chi = 0.5
    
    produce  = cp.Variable(shape=(2,1),name="produce")
    v = cp.Variable(shape=(len(senario_prob),1),name="value")
    mean = cp.Variable(shape = (1,1),name='mean')
    w = cp.Variable(shape=(len(senario_prob),1),name="dummy_value")


    constraints = []
    for i in range(0,len(g_ks)):
        for j in range(0,len(g_ks[i])):
            constraints.append(np.array([g_ks[i][j]])@produce+np.array([alpha_ks[i][j]])<=v[j])

    for i in range(0,len(senario_prob)):
        constraints.append(w[i]>=(v[i]-mean))

    constraints.extend([produce>=0,produce<=produce_limit,mean==senario_prob.T@v,w>=0,v>=-100000])

    objective = cp.Minimize(production_cost.T@produce + mean + chi*senario_prob.T@w)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem

# %%
# produce = np.array([3,3,3]).reshape(-1,1)
limit = -100000

g_ks = []
alpha_ks = []

objctive_values  = [np.nan]
epsilon = 10**(-4)
iter = 0
while True:

    # Solve Second stage problem for each demand and store its duals and objective values
    duals = []
    objs= []
    for demand in all_demand:

        second_stage_sol = solve_second_stage_problem(produce,demand,shipping_cost)

        temp_dual = second_stage_sol.constraints[0].dual_value   # Take the duals of 1st contraint
        temp_obj = second_stage_sol.value                        # Take the objective value of second stage problem

        # Store duals and objective values for each senario
        duals.append(temp_dual) 
        objs.append(temp_obj)

     # Reshaping the values 
    duals = np.array(duals).reshape(-1,2)
    objs = np.array(objs).reshape(-1,1)

    gks_batch = []
    alpha_ks_batch = []

    for i in range(0,senario):
        gks_batch.append(-duals[i])
        alpha_ks_batch.append(objs[i]+duals[i].T@produce)
  
    g_ks.append(gks_batch)
    alpha_ks.append(alpha_ks_batch)

    first_stage_sol = solve_first_stage_problem(produce,g_ks,alpha_ks,senario_prob,production_cost,produce_limit)
    obj_value = first_stage_sol.value
    new_produce = first_stage_sol.var_dict["produce"].value
    new_limit = first_stage_sol.var_dict["value"].value

    if np.abs(obj_value - objctive_values[-1])<= epsilon:
        print("Terminating condition satisfied !")
        break
       
    else:
        pass

    objctive_values.append(obj_value)
    produce,limit = new_produce,new_limit # swap the values
    iter = iter+1
    
    print(f"\n----------Iteration no.  {iter}--------------------")
    print(f"\nproduction is {produce}")
    print(f"\nobjctive value is {first_stage_sol.value}\n")


# %%
produce

# %%



