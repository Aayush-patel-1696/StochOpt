""""
Example to solve Optimization Problem using Scipy

Maximize : 3x1 + 2x2

Constraints: 
1. 2x1 + x2 <= 3
2. x1 + 2x2 <= 4


Bounds:
4. x1>=0 
5. x2>=0

Above problem will be solved by Optimize library of scipy
"""


from scipy.optimize import linprog

# Coefficients of Objective Function
obj = [-3, -2]

# Coefficients of Inequality Constraints
lhs_eq = [[2, 1], [1, 2]]
rhs_eq = [3, 4]

# Bounds
bnd = [(0, float("inf")), (0, float("inf"))]

# Solve the Linear Program
opt = linprog(c=obj, A_ub=lhs_eq, b_ub=rhs_eq, bounds=bnd, method="highs")

print(opt)



