# %% [markdown]
# # Index Tracking with Gurobi

# %% [markdown]
# This Python notebook is part of the webinar [Proven Techniques for Solving Financial Problems with Gurobi](https://www.gurobi.com/events/proven-techniques-for-solving-financial-problems-with-gurobi/).
# 
# The sequence of python code will:
# 1. Import stock data from yahoo finance
# 2. Clean up the data and change format
# 3. Perform an index tracking experiment

# %% [markdown]
# ## Importing Data from YFinance
# 
# - Adjusted Stock price data for SP100 constitutents 
# - Data from 2010 to 2022

# %%
import pandas as pd
from utils.data_import import get_mkt_constitution, get_yf_data
import os
from datetime import datetime

# Options
FIRST_DATE  = "2015-01-01"
LAST_DATE   = "2022-01-01"
N_PROCESSES = 10
MKT_INDEX   = "^SP100" # ^GSPC for SP500 or ^SP100 
#MKT_INDEX   = "^GSPC"

if not os.path.exists("data"):
    os.mkdir("data")
    
# get mkt constitutents    
tickers = get_mkt_constitution(MKT_INDEX)

today = datetime.today().strftime('%Y-%m-%d')
print(f"Available Tickers for {MKT_INDEX} at {today}")
print(tickers)
print(" ")

df_prices = get_yf_data(tickers, 
                        FIRST_DATE,
                        LAST_DATE,
                        N_PROCESSES)

print("\n\nOriginal price data")
print(df_prices.head())

# %% [markdown]
# ## Cleaning and Splitting the Data

# %%
from sklearn.model_selection import train_test_split
import numpy  as np
import matplotlib.pyplot as plt
from utils.data_clean import clean_data

# %load_ext autoreload
# %autoreload 2

THRESH_VALID_DATA = 0.95 # defines where to cut stocks with missing data
PERC_SIZE_TRAIN = 0.75   # defines the size of train dataset (in %)

df_ret, df_train, df_test  = clean_data(
    df_prices, 
    MKT_INDEX,
    thresh_valid_data = THRESH_VALID_DATA,
    size_train = PERC_SIZE_TRAIN
)

df_train.to_parquet("data/ret-data-cleaned-TRAIN.parquet")
df_test.to_parquet("data/ret-data-cleaned-TEST.parquet")

# %% [markdown]
# ## Unconstrained Index Tracking
# 
# $
# \begin{array}{llll}
#   & \min              & \frac{1}{T} \; \sum_{t = 1}^{T} \left(\sum_{i = 1}^{I} \; w_{i} \: \times \: r_{i,t} - R_{t}\right)^2 \\
#   & \text{subject to} &   \sum_{i = 1}^{I} w_{i}  = 1  \\
#   &                   & w_i \geq 0 \\
# \end{array}
# $
# 
# 
# 
# $
# \begin{array}{lll}
# & where: \\
# & \\
# & w_i  &: \text{Weight of asset i in index} \\
# & R_{t} &: \text{Returns of tracked index (e.g. SP500) at time t} \\
# & r_{i,j} &: \text{Return of asset i at time t}
# \end{array}
# $

# %%
import gurobipy as gp
import pandas as pd
import numpy as np
from random import sample, seed

seed(20220209) # reproducibility

mkt_index = "^SP100"
n_assets = 20

# data from main notebook
r_it = pd.read_parquet("data/ret-data-cleaned-TRAIN.parquet")

r_mkt = r_it[mkt_index]

r_it = r_it.drop(mkt_index, axis = 1)

tickers = list(r_it.columns)

sampled_tickers = sample(tickers, n_assets)

r_it = r_it[sampled_tickers]

print(r_it.head())

# %% [markdown]
# # Setup opt problem and solve

# %%
# Create an empty model
m = gp.Model('gurobi_index_tracking')

# PARAMETERS 

# w_i: the i_th stock gets a weight w_i
w = pd.Series(m.addVars(sampled_tickers, 
                         lb = 0,
                         ub = 1,
                         vtype = gp.GRB.CONTINUOUS), 
               index=sampled_tickers)

# CONSTRAINTS

# sum(w_i) = 1: portfolio budget constrain (long only)
m.addConstr(w.sum() == 1, 'port_budget')

m.update()

# eps_t = R_{i,t}*w - R_{M,t}
my_error = r_it.dot(w) - r_mkt

# set objective function
m.setObjective(
    gp.quicksum(my_error.pow(2)), 
    gp.GRB.MINIMIZE)     

# Optimize model
m.setParam('OutputFlag', 0)
m.optimize()

w_hat  = [i.X for i in m.getVars()]

print(f"Solution:") 

for i, i_ticker in enumerate(sampled_tickers):
    print(f"{i_ticker}:\t {w_hat[i]*100:.2f}%")

# check constraints
print(f"\nchecking constraints:")
print(f"sum(w) = {np.sum(w_hat)}")

# %%
# check out of sample plot
import matplotlib.pyplot as plt

df_test = pd.read_parquet("data/ret-data-cleaned-TEST.parquet")

print(df_test.columns)
print(sampled_tickers)
df_test_mkt = df_test[mkt_index]

r_hat = df_test[sampled_tickers].dot(w_hat)

cumret_r = np.cumprod(1+ r_hat)
cumret_mkt = np.cumprod(1+ df_test_mkt)

fig, ax = plt.subplots()
ax.plot(cumret_mkt.index,
        cumret_mkt, 
       label = mkt_index)

ax.plot(cumret_r.index,
        cumret_r,
       label = f"ETF ({n_assets} assets)")

ax.legend()
ax.set_title(f'ETF and {mkt_index}')
ax.set_xlabel('')
ax.set_ylabel('Cumulative Returns')

plt.xticks(rotation = 90)

plt.show()

# %% [markdown]
# ## Constrained Index Tracking
# 
# $
# \begin{array}{llll}
#   & \min              & \frac{1}{T} \; \sum_{t = 1}^{T} \left(\sum_{i = 1}^{I} \; w_{i} \: \times \: r_{i,t} - R_{t}\right)^2 \\
#   & \text{subject to} &   \sum_{i = 1}^{I} w_{i}  = 1  \\
#   &                   &   \sum_{i = 1}^{I} z_{i} \leq K \\
#   &                   & w_i \geq 0 \\
#   &                   & z_i \in {0, 1}
# \end{array}
# $
# 
#   
# 
# $
# \begin{array}{lllll}
# & where: \\
# & \\
# & w_i  &: \text{Weight of asset i in index} \\
# & z_i &: \text{Binary variable (0, 1) that decides wheter asset i is in portfolio} \\
# & R_{t} &: \text{Returns of tracked index (e.g. SP500) at time t} \\
# & r_{i,j} &: \text{Return of asset i at time t}
# \end{array}
# $

# %%
# Create an empty model
m = gp.Model('gurobi_index_tracking')

# PARAMETERS 

max_assets = 10

# w_i: the i_th stock gets a weight w_i
w = pd.Series(m.addVars(sampled_tickers, 
                         lb = 0,
                         ub = 0.2,
                         vtype = gp.GRB.CONTINUOUS), 
               index=sampled_tickers)

# [NEW] z_i: the i_th stock gets a binary z_i
z = pd.Series(m.addVars(sampled_tickers,
                        vtype = gp.GRB.BINARY),
                index=sampled_tickers)

# CONSTRAINTS

# sum(w_i) = 1: portfolio budget constrain (long only)
m.addConstr(w.sum() == 1, 'port_budget')

# [NEW]  w_i <= z_i: restrictions of values of w_i so take it chose particular tickers
for i_ticker in sampled_tickers:
    m.addConstr(w[i_ticker] <= z[i_ticker], 
                f'dummy_restriction_{i_ticker}')

# [NEW] sum(z_i) <= max_assets: number of assets constraint
m.addConstr(z.sum() <= max_assets, 'max_assets_restriction')

m.update()

# eps_t = R_{i,t}*w - R_{M,t}
my_error = r_it.dot(w) - r_mkt

# set objective function
m.setObjective(
    gp.quicksum(my_error.pow(2)), 
    gp.GRB.MINIMIZE)     

# Optimize model
m.setParam('OutputFlag', 0)
m.setParam('TimeLimit', 60*5) # in secs
#m.setParam('MIPGap', 0.05) # in secs
m.optimize()

params = [i.X for i in m.getVars()]

n_assets = len(sampled_tickers)
w_hat = params[0:n_assets]
z_hat = params[n_assets:]
MIPGap = m.getAttr('MIPGap')
status = m.getAttr("Status")

print(f"Solution for w:") 

for i, i_ticker in enumerate(sampled_tickers):
    print(f"{i_ticker}:\t {w_hat[i]*100:.2f}%")

# check constraints
print(f"\nchecking constraints:")
print(f"sum(w) = {np.sum(w_hat)}")
print(f"sum(z) = {np.sum(z_hat)}")
print(f"w <= z = {w_hat <= z_hat}")
print(f"MIPGap={MIPGap}")
print(f"Status={status}")

# %%
# check out of sample plot
import matplotlib.pyplot as plt

df_test = pd.read_parquet("data/ret-data-cleaned-TEST.parquet")

print(df_test.columns)
print(sampled_tickers)
df_test_mkt = df_test[mkt_index]

r_hat = df_test[sampled_tickers].dot(w_hat)

cumret_r = np.cumprod(1+ r_hat)
cumret_mkt = np.cumprod(1+ df_test_mkt)

fig, ax = plt.subplots()
ax.plot(cumret_mkt.index,
        cumret_mkt, 
       label = mkt_index)

ax.plot(cumret_r.index,
        cumret_r,
       label = f"ETF ({n_assets} assets)")

ax.legend()
ax.set_title(f'ETF and {mkt_index}')
ax.set_xlabel('')
ax.set_ylabel('Cumulative Returns')

plt.xticks(rotation = 90)

plt.show()


