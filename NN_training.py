from utils import *
from sklearn.preprocessing import StandardScaler

print('\n')
print('***** QNN and IQNN training script ******')
print('-- Do not forget to add your data path --')
print('-- Default hyper-parameters are given  --')
print('\n')


### Data loading ###
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


### Hyper-parameter selection ###
q = np.linspace(0.01, 0.99, num=50, endpoint=True)

incr = False
bs = 256
opti = 'Adam'
n_hidden = 1
n_nodes = 64
lr = 5e-4
l1_reg = 0.0001
l2_reg = 0.0001
drop = 0.05
iters = 2000


print('Hyper-parameters:')
print('Incremental output layer:', incr)
print('Batch size:', bs)
print('Optimizer', opti)
print('Hidden layers:', n_hidden)
print('Nodes per hidden layer:', n_nodes)
print('Learning rate:', lr)
print('L1 reg penalty:', l1_reg)
print('L2 reg penalty:', l2_reg)
print('Dropout rate:', drop)
print('Number of epochs;', iters)


network_obj = QNN_obj(X_train, X_val, y_train, y_val, incr, n_hidden, n_nodes, drop, iters, lr, bs, l1_reg, l2_reg, opti, q)
best_loss, fitted_model = network_obj.train()
print('-- Trained (I)QNN is saved --')