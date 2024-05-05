import sys, argparse, time, os

# description: main file for running experiments.
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  option = sys.argv[1]
  sys.argv.remove(sys.argv[1])
  
  if option == 'sb3':
    """to run sb3 algos"""
    import torch
    import yaml
    import torch.nn as nn
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    import stable_baselines3
    
    parser.add_argument("--timesteps", default=600000000)
    parser.add_argument("--logdir",             default="./logs/ppo/", type=str)   # where to store log information
    # env params to play with 
    parser.add_argument("--exp_conf_path",  default="./exp_confs/default.yaml", type=str)  # path to econf file of experiment parameters
    args = parser.parse_args()
    torch.set_num_threads(1)
    from src.util import env_factory
    env_fn     = env_factory(
                              args.exp_conf_path
                            )
    conf_file = open(args.exp_conf_path)
    exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)
    policy_kwargs= {
                    'log_std_init': -2,
                    'ortho_init': False,
                    'activation_fn': nn.Tanh,
                    'net_arch': {
                                'pi': [128,128],
                                'vf': [128,128]
                              }
                    }
    exp_conf['algo']['params']['policy_kwargs'] = policy_kwargs
    train_env = make_vec_env(env_fn,n_envs= exp_conf['algo']['n_envs'])
    eval_env = make_vec_env(env_fn,n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.logdir,
                             log_path=args.logdir, eval_freq=50000,
                             deterministic=True, render=False)
    algo_module = getattr(stable_baselines3,exp_conf['algo']['name'])
    algo = algo_module("MlpPolicy",train_env,verbose=0,tensorboard_log=args.logdir,**exp_conf['algo']['params'])
    algo.learn(total_timesteps=args.timesteps,progress_bar=True,callback=eval_callback)
  
  elif option == "sb3_contrib":
    """to run sb3 algos"""
    import torch
    import yaml
    import torch.nn as nn
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    import sb3_contrib
        
    parser.add_argument("--timesteps", default=600000000)
    parser.add_argument("--logdir",             default="./logs/ppo/", type=str)   # where to store log information
    # env params to play with 
    parser.add_argument("--exp_conf_path",  default="./exp_confs/default.yaml", type=str)  # path to econf file of experiment parameters
    args = parser.parse_args()
    torch.set_num_threads(1)
    from src.util import env_factory
    env_fn     = env_factory(
                              args.exp_conf_path
                            )
    conf_file = open(args.exp_conf_path)
    exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)
    policy_kwargs= {
                    'log_std_init': -2,
                    'ortho_init': False,
                    'activation_fn': nn.Tanh,
                    'net_arch': {
                                'pi': [128,128],
                                'vf': [128,128]
                              }
                    }
    exp_conf['algo']['params']['policy_kwargs'] = policy_kwargs
    train_env = make_vec_env(env_fn,n_envs= exp_conf['algo']['n_envs'])
    eval_env = make_vec_env(env_fn,n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.logdir,
                             log_path=args.logdir, eval_freq=50000,
                             deterministic=True, render=False)
    algo_module = getattr(sb3_contrib,exp_conf['algo']['name'])
    algo = algo_module("MlpLstmPolicy",train_env,verbose=0,tensorboard_log=args.logdir,**exp_conf['algo']['params'])
    algo.learn(total_timesteps=args.timesteps,progress_bar=True,callback=eval_callback)

    
  else:
    "to run custom ppo"
    from algos.ppo import run_experiment
    parser.add_argument("--timesteps",          default=600000000,           type=float) # timesteps to run experiment for
    parser.add_argument('--discount',           default=0.95,          type=float) # the discount factor
    parser.add_argument('--std',                default=0.13,          type=float) # the fixed exploration std
    parser.add_argument("--a_lr",               default=1e-4,          type=float) # adam learning rate for actor
    parser.add_argument("--c_lr",               default=1e-4,          type=float) # adam learning rate for critic
    parser.add_argument("--eps",                default=1e-6,          type=float) # adam eps
    parser.add_argument("--kl",                 default=0.02,          type=float) # kl abort threshold
    parser.add_argument("--grad_clip",          default=0.05,          type=float) # gradient norm clip


    parser.add_argument("--max_itr",            default=2000,          type=int)   # maximum policy updates
    parser.add_argument("--batch_size",         default=64,            type=int)   # batch size for policy update
    parser.add_argument("--epochs",             default=8,             type=int)   # number of updates per iter
    parser.add_argument("--workers",            default=30,             type=int)   # how many workers to use for exploring in parallel
    parser.add_argument("--seed",               default=0,             type=int)   # random seed for reproducibility
    parser.add_argument("--traj_len",           default=500,          type=int)   # max trajectory length for environment
    parser.add_argument("--prenormalize_steps", default=10000,         type=int)   # number of samples to get normalization stats 
    parser.add_argument("--sample",             default=50000,          type=int)   # how many samples to do every iteration

    parser.add_argument("--layers",             default="128,128",     type=str)   # hidden layer sizes in policy
    parser.add_argument("--save_actor",         default=None,          type=str)   # where to save the actor (default=logdir)
    parser.add_argument("--save_critic",        default=None,          type=str)   # where to save the critic (default=logdir)
    parser.add_argument("--logdir",             default="./logs/ppo/", type=str)   # where to store log information
    parser.add_argument("--nolog",              action='store_true')               # store log data or not.
    parser.add_argument("--recurrent",          action='store_true')               # recurrent policy or not
    parser.add_argument("--randomize",          action='store_true')               # randomize dynamics or not
    
    # env params to play with 
    parser.add_argument("--exp_conf_path",  default="./exp_confs/default.yaml", type=str)  # path to econf file of experiment parameters
    parser.add_argument("--load_initial_agent_from", default=None, type=str)              # path to intital policy if to be loaded

    args = parser.parse_args()
    

    run_experiment(args)
                                        
  

