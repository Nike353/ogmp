import os,sys
sys.path.append('./')

from nn.lstm_vae import LSTM_VAE
from nn.lstm_ae import LSTM_AE
from src.ae_trainer import quick_train
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.misc_funcs import *


exists_not_none = lambda gkey,gdict: gkey in gdict.keys() and  not(gdict[gkey] is None)

cwd = os.getcwd()

save_reconstruction_plots = False
mode_groups = ['flat', 'gap', 'block', 'pitch_flip','roll_flip','yaw_flip']


trng_conf =     {   

                    'prev_rtm': {

                                'entry': 'preview_cmmp.prev_ad.oracle',
                                # 'entry': 'preview_cmmp.preview_v3.prev_rtm',

                                'scan_xlen_infront': 1.0,
                                # 'prediction_horizon': 30,
                                'state_feedback': [
                                                    # 'roll', 'pitch', 'yaw', 
                                                    # 'x', 
                                                    # 'y', 
                                                    'z',
                                                    # 'yaw',
                                                    # 'roll_dot', 'pitch_dot', 'yaw_dot',
                                                    # 'x_dot', 'y_dot', 'z_dot',
                                                    
                                                ],
                                'terrain_map_resolution': [0.04,0.04],
                                },
                    'exp_path': './logs/encoders/flips_s\u03C1_ae32_lp_yf/',
                    'input_type': 'base_only',
                    'n_epochs': 2000,
                    "tasks_for_trng": {


                                    # "pitch_flip":
                                    #     {

                                    #         "param_names":  ['h0', 'goal_theta'],
                                    #         'param_0':
                                    #             {
                                    #                 'start': 0.75,
                                    #                 'stop': 3.0,
                                    #                 'num': 10,
                                    #             },
                                    #         'param_1':
                                    #             {
                                    #                 'start': -2*np.pi,
                                    #                 'stop':  2*np.pi,
                                    #                 'num': 2,
                                    #             },


                                    #     },

                                    # "roll_flip":
                                    #     {

                                    #         "param_names":  ['h0', 'goal_theta'],
                                    #         'param_0':
                                    #             {
                                    #                 'start': 0.75,
                                    #                 'stop': 3.0,
                                    #                 'num': 10,
                                    #             },
                                    #         'param_1':
                                    #             {
                                    #                 'start': -2*np.pi,
                                    #                 'stop':  2*np.pi,
                                    #                 'num': 2,
                                    #             },


                                    #     },
                                    
                                    "yaw_flip":
                                        {
                                            "param_names":  ['h0', 'goal_theta'],
                                            'param_0':
                                                {
                                                    'start': 0.0,
                                                    'stop': 0.0,
                                                    'num': 1,
                                                },
                                            'param_1':
                                                {
                                                    'start': -2*np.pi,
                                                    'stop':  2*np.pi,
                                                    'num': 20,
                                                },
                                        }

                                    # "flat":
                                    #         {
                                            
                                    #         "vary_x0":{
                                    #                     "dofs": [
                                    #                                 'x',
                                    #                                 'z',
                                    #                                 'x_dot',
                                    #                                 'z_dot',
                                    #                             ],
                                    #                     "n_samples": [
                                    #                                     1,
                                    #                                     1,
                                    #                                     4,
                                    #                                     4,
                                    #                                 ],
                                    #                     "val_lims": [
                                    #                                     [0.0,0.0],
                                    #                                     [0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                 ]
                                    #                 },
                    
                                    #         "param_names": ["goal_x"],
                                    #         'param_0':
                                    #             {
                                    #                 # 'start': 0.25,
                                    #                 'start': -1.0,
                                    #                 'stop': 1.5,
                                    #                 'num': 10,
                                    #             },

                                    #         },

                                    # "gap":
                                    #         {
                                    #         "vary_x0":{
                                    #                     "dofs": [
                                    #                                 'x',
                                    #                                 'z',
                                    #                                 'x_dot',
                                    #                                 'z_dot',
                                    #                             ],
                                    #                     "n_samples": [
                                    #                                     1,
                                    #                                     1,
                                    #                                     3,
                                    #                                     3,
                                    #                                 ],
                                    #                     "val_lims": [
                                    #                                     [0.0,0.0],
                                    #                                     [0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                 ]
                                    #                 },
                    
                                    #         "param_names":  ['start','length','height'],
 
                                            
                                    #         'param_0':
                                    #             {
                                    #                 'start': 0.2,
                                    #                 'stop': 0.6,
                                    #                 'num': 3,
                                    #             },
                                    #         'param_1':
                                    #             {
                                    #                 'start': 0.1,
                                    #                 'stop': 0.4,
                                    #                 'num': 3,
                                    #             },
                                    #         'param_2':
                                    #             {

                                    #                 'start': -0.5,
                                    #                 'stop': -0.5,
                                    #                 'num': 1,
                                    #             },
                                            
                                            
                                    #         },

                                    # "block":
                                    #         {
                                    #         "vary_x0":{
                                    #                     "dofs": [
                                    #                                 'x',
                                    #                                 'z',
                                    #                                 'x_dot',
                                    #                                 'z_dot',
                                    #                             ],
                                    #                     "n_samples": [
                                    #                                     1,
                                    #                                     1,
                                    #                                     3,
                                    #                                     3,
                                    #                                 ],
                                    #                     "val_lims": [
                                    #                                     [0.0,0.0],
                                    #                                     [0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                     [-0.5,0.5],
                                    #                                 ]
                                    #                 },
                                    #         "param_names":  ['start','length','height'],
                                    #         # "param_values": [
                                    #         #                     [0.2,0.3,+0.2],
                                    #         #                     [0.2,0.3,+0.3],
                                    #         #                     [0.2,0.3,+0.4],
                                    #         #                 ]
                                    #         'param_0':
                                    #             {
                                    #                 'start': 0.45,
                                    #                 'stop': 0.45,
                                    #                 'num': 1,
                                    #             },
                                    #         'param_1':
                                    #             {
                                    #                 'start': 0.3,
                                    #                 'stop': 0.5,
                                    #                 'num': 3,
                                    #             },
                                    #         'param_2':
                                    #             {

                                    #                 'start': 0.2,
                                    #                 'stop':  0.4,
                                    #                 'num': 3,
                                    #             },
                                            
                                    #         }
                                            
                                    },
            
                    'tasks_for_test':  {
                                        

                                            # "gap":
                                            #         {
                                            #         "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, G],
                                            #         "param_names":  ['start','end','height'],
                                            #         "param_values": [
                                            #                             [0.25,0.4,-0.5],
                                            #                             [0.4,0.73,-0.5],
                                            #                             [0.36,0.76,-0.5],
                                            #                         ]
                                            #         },

                                            # "block":
                                            #         {
                                            #         "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, G],
                                            #         "param_names":  ['start','end','height'],
                                            #         "param_values": [
                                            #                             [0.3,0.55,+0.22],
                                            #                             [0.3,0.55,+0.36],
                                            #                             [0.3,0.55,+0.42],
                                            #                         ]
                                            #         }
                                            
                                            
                                        },
            
                    
                    'enc_arch': LSTM_AE, #LSTM_AE, LSTM_VAE
                    'enc_dim':2,
                    'enc_conf': {


                                    # for LSTM_AE
                                    'h_dims': [32],
                                    'h_activ': torch.nn.ReLU(),
                                    'out_activ':None,

                                    # for LSTM_VAE
                                    # 'h_dim': 32,
            
                                },
                    #  'training_style': {
                                        
                    #                     'scheme' : 'full_dataset', #iterative_sampling
                    
                    #                     # 'scheme' : 'iterative_sampling', # full_dataset, iterative_sampling
                    #                     # 'n_samples': 10,
                                        
                    #                     }
                    # 'train_frm_dataset_at':  cwd+"/dtsd/analysis_results/aug_dataset2.npz",
                }


def run_trng(
                    exp_folderpath,
                    train_set,
                    traj_names,
                    encd_dim, 
                    iter_i
                ):

    trng_conf.update({'latent_dim':encd_dim})
    trng_conf_file =  open(exp_folderpath+'/trng_conf.yaml','w')
    yaml.dump(
                trng_conf ,
                trng_conf_file,
                default_flow_style=False,
                sort_keys=False
            )

    if iter_i != 0:
        trng_conf['enc_conf']['load_model_frm'] = exp_folderpath+'/enc_'+str(iter_i-1)+'.pt'



    model, _ , losses = quick_train(
                                            trng_conf['enc_arch'], 
                                            train_set,
                                            # input_dim=5,
                                            encoding_dim=encd_dim,
                                            epochs=trng_conf['n_epochs'],
                                            pbar_conf={'id':iter_i},
                                            **trng_conf['enc_conf']
                                            )

    np.savez_compressed( 
                        exp_folderpath+'/trng_log.npz',
                        loss=losses
                        )
    
    # save model
    torch.save(model,exp_folderpath +'/model_'+str(iter_i)+'.pt')

    # save trianing plot
    plt.plot(losses)
    plt.title('traing curve ')
    plt.grid()
    plt.xlabel('epochs')
    plt.xlabel('loss')
    plt.savefig(exp_folderpath+'/training_curve'+str(iter_i)+'.png')
    plt.close()

    return model


if __name__ == '__main__':


        
    exp_folderpath =  trng_conf['exp_path']+'/dim_'+str(trng_conf['enc_dim'])
    os.makedirs(exp_folderpath,exist_ok=True)


    # load train data
    train_set, traj_names = load_trajs_frm_preview(
                                        conf= trng_conf,
                                        verbose=False,
                            )


    model = run_trng(
                        exp_folderpath,
                        train_set,
                        traj_names,
                        encd_dim=trng_conf['enc_dim'],
                        iter_i=0
                    )


    # move model and traning data to cpu for plotting
    model = model.to('cpu')
    # print(model.device)
    for i,sample in enumerate(train_set):
        train_set[i] =sample.to('cpu')

    # group bins
    zs_groups = {}
    groups_variant_names = {}
    for mode_group in mode_groups:
        zs_groups[mode_group] ={'mean':[],'std':[] }
                            
        groups_variant_names[mode_group] = []


    is_vae = True if trng_conf['enc_arch'].__name__ == "LSTM_VAE" else False


    
    if is_vae:
        # get train_set encodings
        zs_train_stats = get_vae_encodings(model, train_set)
    
    else:
        zs_train_stats = get_encodings(model, train_set)


    # group train_set encodings
    for traj_name,z_mean,z_std in zip(
                                        traj_names, 
                                        zs_train_stats['mean'],
                                        zs_train_stats['std']
                                    ):
        
        for mode_group in mode_groups:
            if mode_group in traj_name:
                
                groups_variant_names[mode_group].append(traj_name.replace(mode_group+'_',''))
                
                zs_groups[mode_group]['mean'].append(z_mean)
                zs_groups[mode_group]['std'].append(z_std)
                break
    
    # plot train_set latents and reconstructions
    plot_latents(
                    zs_train_stats['mean'],
                    savepath=exp_folderpath,
                    sample_names=traj_names

                )
    # plot train_set logits
    plot_latent_logits( 
                        zs_mean=zs_train_stats['mean'],
                        zs_std=zs_train_stats['std'],
                        sample_names=traj_names,
                        savepath=exp_folderpath


                        )
    
    # exit()
    get_and_plot_reconstructions(   
                                    model=model,
                                    samples=train_set,
                                    savepath=exp_folderpath,
                                    sample_names=traj_names,
                                    is_vae=is_vae,
                                )
    

    # load test data if exists
    if trng_conf['tasks_for_test']:

        trng_conf['tasks_for_trng'] = trng_conf['tasks_for_test']
        test_set, traj_names_test = load_trajs_frm_preview(
                                            conf= trng_conf,
                                )
        if is_vae:
            # get train_set encodings
            zs_test_stats = get_vae_encodings(model, test_set)
        else:
            zs_test_stats = get_encodings(model, test_set)

        for mode_group in mode_groups:
            zs_groups[mode_group+'_test'] = {'mean':[],'std':[] }
            groups_variant_names[mode_group+'_test'] = []

        # group test_set encodings
        for traj_name,z_mean,z_std  in zip(
                                            traj_names_test,
                                            zs_test_stats['mean'],
                                            zs_test_stats['std']
                                        
                                        ):
            
            for mode_group in mode_groups:
                if mode_group in traj_name:
                    
                    groups_variant_names[mode_group+'_test'].append(traj_name.replace(mode_group+'_',''))
                    
                    # zs_groups[mode_group+'_test'].append(z)
                    zs_groups[mode_group+'_test']['mean'].append(z_mean)
                    zs_groups[mode_group+'_test']['std'].append(z_std)

                    break
    
    
    # plot 2d mode space
    plot_2d_mode_space_stats(
                                zs_groups,
                                groups_variant_names,
                                savepath=exp_folderpath,
                                annotate_variant_name=False,
                                # vary_variants_alpha=True,
                                show_std=False,
                                export_mode_space=True,
                                )        


    print("exported mode space:") 
    data = np.load(exp_folderpath+'/mode_space.npz')
    for file in data.files:
        print('\t:', file,data[file].shape)
    exit()

    exit()
    # fig = plt.figure(2)


    # N    = 200
    # X    = np.linspace( -2, 2, N)
    # Y    = np.linspace( -2, 2, N)
    # X, Y = np.meshgrid(X, Y)
    # pos  = np.dstack((X, Y))

    # # rv2  = multivariate_normal([0, -1], [[2, 0], [0, 1]])
    # # Z2   = rv2.pdf(pos)
    # # plt.contour(X, Y, Z2)
    # # plt.show()

    # for i,z in enumerate(zs):
    #     # plt.text(z[0],z[1],all_traj_names[i],rotation=45)
    #     mean = z_means[i]
    #     cov = np.diag(np.square(z_stds[i]))

    #     # plt.text(mean[0],mean[1],all_traj_names[i],rotation=45)
    #     rv   = multivariate_normal(
    #                                 # z_means[i], 
    #                                 # [
    #                                 #     [z_std[i,0], 0.8], [0.8, 1]],
    #                                 mean=mean,
    #                                 cov=cov,
    #                             )
    #     Z    = rv.pdf(pos)

    #     if 'walk' in all_traj_names[i]:
    #         color_i = 0
    #         alpha_i = 0.125*int(all_traj_names[i].split('fs')[1])
    #     else:
    #         color_i = 1
    #         alpha_i = 0.125*int(all_traj_names[i].split('uf_h')[1])
    
    #     plt.scatter(
    #                     mean[0],
    #                     mean[1],
    #                     color='C'+str(color_i),
    #                     alpha=alpha_i
    #                 )

    #     plt.contour(X, Y, Z,

    #                 levels=1,
    #                 colors='C'+str(color_i),
    #                 alpha=alpha_i
    #                 )
    #     # if i > 0:
    #     #     break

    # # plt.grid()
    # plt.legend(all_traj_names[0:i+1])
    # plt.show()

    # exit()
