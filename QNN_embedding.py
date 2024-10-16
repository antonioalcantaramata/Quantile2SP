### This is an example on how to embed an QNN model within a 2SP ###
### Check the 'problems' folder for a more comprehensive example ###

import gurobipy as gp
from utils import *
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('...', index_col=0)

y = data['obj']
X = data.drop(['obj'], axis=1, inplace=False)

seed = 0
np.random.seed(seed)
msk = np.random.rand(len(X)) < 0.8
X_train = X[msk]
y_train = y[msk]
X_val = X[~msk]
y_val = y[~msk]

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = scaler.feature_names_in_)
X_mean = scaler.mean_
X_std = scaler.scale_
X_val = pd.DataFrame(scaler.transform(X_val), columns = scaler.feature_names_in_)

y_train = pd.DataFrame(scaler.fit_transform(y_train.values.reshape(-1, 1)), columns = [y.name])
y_mean = scaler.mean_
y_std = scaler.scale_
y_val = pd.DataFrame(scaler.transform(y_val.values.reshape(-1, 1)), columns = [y.name])


q = np.linspace(0.01, 0.99, num=50, endpoint=True)
n_hidden = 1 # Select the NN structure of the fitted model
n_nodes = 64

model = QuantFCNN(input_size=X_train.shape[1], quantiles=q, hidden_layers=n_hidden, hidden_size=n_nodes, drop=0)
model.load_state_dict(torch.load('...')) # Fitted model path


SP = gp.Model('2SP')
lambda_tradeoff = 0.1 # Risk-aversion parameter
### Add first-stage variables and constraints
first_stage_vars = {} # Save first-stage decision variables
objective_fs = {} # Save first-stage objective

### Network formulation for second stage
x = {}
z = {}

x[0] = {}
z[0] = {}

x_maxs = {}
x_mins = {}
x_maxs[0] = {}
x_mins[0] = {}


tol = 10 # Set the tolerance for quantile crossing
layers = getattr(model, 'layers')

l_0 = layers[0]
for i in range(l_0.in_features):
    x[0][i] = (first_stage_vars[i] - X_mean[i])/X_std[i] # Standardize fs inputs for the network model
    x_maxs[0][i] = (1 - X_mean[i])/X_std[i] # Set max value of the inputs (1 as example)
    x_mins[0][i] = (0 - X_mean[i])/X_std[i] # Set min value of the inputs (0 as example)
SP.update()

for ind, layer in enumerate(layers):
    l = layer
    
    #x[ind+1][i] tracks the output of node i of layer ind
    #z[ind+1][i] is the binary defining whether node i of layer ind is active
    x[ind+1] = {}
    z[ind+1] = {}
    x_maxs[ind+1] = {}
    x_mins[ind+1] = {}
    
    for i in range(l.out_features):
        # Get weights and biases
        m = l.weight.detach().numpy()[i]
        b = l.bias.detach().numpy()[i]
        
        if ind < len(layers) - 1: 
            # Compute upper and lower bounds
            ub = sum(x_maxs[ind][j] * max(0,m[j]) + x_mins[ind][j] * min(0,m[j]) for j in range(l.in_features)) + b
            lb = sum(x_mins[ind][j] * max(0,m[j]) + x_maxs[ind][j] * min(0,m[j]) for j in range(l.in_features)) + b
            x_maxs[ind+1][i] = ub
            x_mins[ind+1][i] = lb
            
            # Define vars
            x[ind+1][i] = SP.addVar(0, max(0,ub), name='x_' + str(ind+1) + '_' + str(i))
            z[ind+1][i] = SP.addVar(0, 1, vtype=gp.GRB.BINARY, name='z_' + str(ind+1) + '_' + str(i))
            # Big-M representation of node i
            SP.addConstr(x[ind+1][i] >= sum(x[ind][j] * m[j] for j in range(l.in_features)) + b)
            SP.addConstr(x[ind+1][i] <= sum(x[ind][j] * m[j] for j in range(l.in_features)) + b - lb*(1-z[ind+1][i]))
            SP.addConstr(x[ind+1][i] <= ub * z[ind+1][i])
        else:
            x[ind+1][i] = SP.addVar(lb=-gp.GRB.INFINITY, name='x_' + str(ind+1) + '_' + str(i))
            # Final quantiles
            SP.addConstr(x[ind+1][i] == (sum(x[ind][j] * m[j] for j in range(l.in_features)) + b)*y_std[0] + y_mean[0])
    
    SP.update()

objective_cvar = 0
objective_mean = 0
for i in range(l.out_features):
    if i < 45: # This index indicates the quantile of the distribution. 45th quantile represents VaR 90
        objective_mean += x[ind+1][i]
    else:
        objective_mean += x[ind+1][i]
        objective_cvar += x[ind+1][i]

if tol != 'No':
    for i in range(l.out_features - 1):
        SP.addConstr(x[ind+1][i] <= x[ind+1][i+1] + tol)
    SP.update()


SP.setObjective((1+lambda_tradeoff)*objective_fs + (1/50)*objective_mean + (lambda_tradeoff/5)*objective_cvar, gp.GRB.MINIMIZE)
SP.update()
