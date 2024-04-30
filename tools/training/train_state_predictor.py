import sys
sys.path.append('./')
import torch
from nn.state_predictor import StatePredictor
from dtsd.envs.src import transformations
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
argparser = argparse.ArgumentParser()
argparser.add_argument('--path2logs', type=str, default='data')
args = argparser.parse_args()

def read_all_logs(path2logs):
    loglist = os.listdir(path2logs)
    loglist = [file for file in loglist if '.npz' in file]
    data_dict = {}

    # collect all metrics
    for i,log_name in enumerate(loglist):
        worker_log = np.load(os.path.join(path2logs,log_name),allow_pickle=True)
        for file in worker_log.files:
            if file not in []:
                print(worker_log[file].shape)
                if file not in data_dict.keys():
                    data_dict[file] = worker_log[file]
                else:
                    data_dict[file] = np.vstack((data_dict[file],worker_log[file]))

    return data_dict

data_dict = read_all_logs(args.path2logs)

for key in data_dict.keys():
    print(key,data_dict[key].shape)


# for debug: make a figure for each data, with subplots for each dimension
'''
for key in data_dict.keys():
    nrows = 2
    ncols = data_dict[key].shape[-1]//nrows
    print('nrows:',nrows,'ncols:',ncols)
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,5))
    
    for traj in data_dict[key]:
        for i in range(traj.shape[1]):
            row = i//ncols
            col = i%ncols
            if ncols == 1:
                ax[row].plot(traj[:,i])
            else:
                ax[row,col].plot(traj[:,i])
    for a in ax.flatten():
        a.grid()
    fig.suptitle(key)
    fig.tight_layout()

    plt.show()
    plt.close()
'''
n_epochs = 500
device = 'cuda'
oos = 'pure_intg'    
exp_name = 'oa2x_'+oos
log_path = './logs/state_predictors/'+exp_name+ '/'
log_every = 10  # log every n epochs
os.makedirs(log_path,exist_ok=True)
# data dict to dataset


base_pos = data_dict['qposs'][:,:,0:3]
base_quat = data_dict['qposs'][:,:,3:7]
base_rpys = np.zeros((base_quat.shape[0],base_quat.shape[1],3))
for i in range(base_quat.shape[0]):
    for j in range(base_quat.shape[1]):
        base_rpys[i,j] = transformations.quat_to_euler(base_quat[i,j])
base_pose = np.concatenate([base_pos,base_rpys],axis=-1)
base_twist = data_dict['qvels'][:,:,0:6]

train_set_y = np.concatenate([base_pose,base_twist,data_dict['toe_contact_states']],axis=-1)


#make dataset
train_set_obs = torch.tensor(data_dict['observations']).float().to(device)
train_set_act = torch.tensor(data_dict['actions']).float().to(device)
train_set_y = torch.tensor(train_set_y).float().to(device)

# make model
obs_dim = data_dict['observations'].shape[-1]
action_dim = data_dict['actions'].shape[-1]
x0 = torch.tensor([
                    0,0,0.5,
                    0,0,0,
                    0,0,0,
                    0,0,0,
                    1,1,
                    ]).float().to(device)
model = StatePredictor(
                        x0=x0,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        oos=oos,
                        dt = 0.03
                        ).to(device)

# make optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# make loss
loss_fn = torch.nn.MSELoss()

# train
training_loss = []
for epoch in range(n_epochs):
    # forward pass multiple trajs in parallel
    model.reset_x_minus()
    pred = model.predict(train_set_obs,train_set_act)
    # compute loss
    loss = loss_fn(pred,train_set_y)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % log_every == 0:
        print('epoch:',epoch,'loss:', round(loss.item(),4))
        training_loss.append(loss.item())
        # save model
        torch.save(model,log_path+'model.pt')

with torch.no_grad():
    # plot and save training loss
    plt.plot(
                np.arange(len(training_loss))*log_every,
                training_loss,
            )
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.tight_layout()
    plt.savefig(log_path+'training_loss.png')
    plt.show()

    # save model
    torch.save(model,log_path+'final_model.pt')
    del model


    # reload the model
    model = torch.load(log_path+'final_model.pt')
    print(model)

    test_path = log_path+'/test/'
    os.makedirs(test_path,exist_ok=True)
    # test 10 random samples from train set
    for k in range(10):
        print('testing sample:',k)
        idx = np.random.randint(0,train_set_obs.shape[0])
        obs = train_set_obs[idx]
        act = train_set_act[idx]
        y = train_set_y[idx]
        model.reset_x_minus()
        pred = model.predict(obs.unsqueeze(0),act.unsqueeze(0))
        # print('obs:',obs)
        # print('act:',act)
        # print('y:',y)

        # plot pred vs y
        nrows = 2
        ncols = train_set_y.shape[-1]//nrows    
        fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,5))
        for i in range(train_set_y.shape[-1]):
            row = i//ncols
            col = i%ncols
            
            if i in [12,13]:
                y[:,i] = torch.round(y[:,i])
                pred[0,:,i] = torch.round(pred[0,:,i])

            ax[row,col].plot(y[:,i].cpu().detach().numpy(),label='true')
            ax[row,col].plot(pred[0,:,i].cpu().detach().numpy(),label='pred',linestyle='--')
            ax[row,col].grid()
            ax[row,col].legend()
            ax[row,col].set_ylim(-1.5,1.5)
        fig.tight_layout()
        plt.savefig(test_path+'pred_vs_y_'+str(k)+'.png')
        # plt.show()