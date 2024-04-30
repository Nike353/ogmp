import yaml
import os


conf_file_paths = []
for exp_log in os.listdir("./exp_confs"):

    # ./logs
    # if exp_log != 'cassie_results':

    #     if exp_log == 'sub_optimal_policies':
    #         for sub_exp_log in os.listdir("./logs/sub_optimal_policies"):
    #             conf_file_paths.append("./logs/sub_optimal_policies/"+sub_exp_log+"/exp_conf.yaml")

    #     else:
    #         conf_file_paths.append("./logs/"+exp_log+"/exp_conf.yaml")
    
    # ./exp_confs
    exp_name = exp_log.replace('.yaml','')
    if exp_name not in ['default','tstng_exp_conf','tstng_conf','trng_parms_to_vary']:
        conf_file_paths.append("./exp_confs/"+exp_log)


# print(conf_file_paths)



for file_path in conf_file_paths:
    # file_path = "./logs/"+exp_log+"/exp_conf.yaml"
    trng_exp_conf_file = open(file_path) # remove
    trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
    trng_exp_conf_file.close()


    if 'oce' in file_path:
        print(file_path)
        trng_exp_conf['env_entry'] = 'dtsd.envs.biped_oce.biped_env' 
        trng_exp_conf['observations'].pop('obs_clock_osu')
        trng_exp_conf['observations'].update({'clock_osu':None})
    # if 'initialisations' in trng_exp_conf.keys():
    #     if 'set_skateboard_pos_z' in trng_exp_conf['initialisations'].keys():
    #         if trng_exp_conf['initialisations']['set_skateboard_pos_z'] == 'under_robot_on_terrain':
    #             print(file_path)
    #             trng_exp_conf['initialisations'].update({'set_skateboard_h_on_terrain_under_robot':[0.0420]})

    #             trng_exp_conf['initialisations'].pop('set_skateboard_pos_z')



    # trng_exp_conf.update({'actions':
    #                         {
    #                             'pd_targets': [0,10]
    #                         }
    #                     })


    # if 'follow_target' in file_path:
    #     trng_exp_conf['rewards'].update({'target_robot_pos_diff_exp_weight':0.5})
    #     print(file_path)
        
    
    # if 'skate_board' in file_path:


    #     if 'set_robot_nominal' in trng_exp_conf['initialisations'].keys():
    #         trng_exp_conf['initialisations'].update({'set_robot_one_leg_on_skateboard':'L_toe'})
    #         trng_exp_conf['initialisations'].pop('set_robot_nominal')


    #     if 'set_robot_on_skateboard_pose' in trng_exp_conf['initialisations'].keys():
    #         trng_exp_conf['initialisations'].update({'set_robot_one_leg_on_skateboard':'L_toe'})
    #         trng_exp_conf['initialisations'].pop('set_robot_on_skateboard_pose')
    #     if 'set_skateboard_below_ltoe' in trng_exp_conf['initialisations'].keys():
    #         trng_exp_conf['initialisations'].update({'set_skateboard_pos_x':'robot_base_x'})
    #         trng_exp_conf['initialisations'].update({'set_skateboard_pos_y':[0.098]})
    #         trng_exp_conf['initialisations'].pop('set_skateboard_below_ltoe')

    #     if 'set_skateboard_pos_x' in trng_exp_conf['initialisations'].keys():
    #         trng_exp_conf['initialisations'].update({'set_skateboard_pos_x':'robot_base_pos_x'})

    '''
    if 'commands' in trng_exp_conf.keys():
        temp_dict = {}
        for key in trng_exp_conf['commands'].keys():
            temp_dict.update({key+'_lim':trng_exp_conf['commands'][key]})
        keys_2_remove = list(trng_exp_conf['commands'].keys())
        for key in keys_2_remove:
            trng_exp_conf['commands'].pop(key)
        trng_exp_conf['commands'].update(temp_dict)

    
    # gen params
    trng_exp_conf['visualize_reference'] = False
    trng_exp_conf.pop('dynamics_randomization')

    # sim_params
    if 'target_marker' in trng_exp_conf['sim_params']['render'].keys():
        trng_exp_conf['sim_params'].update({'target_marker': trng_exp_conf['sim_params']['render']['target_marker']})            
    trng_exp_conf['sim_params'].pop('render')
    trng_exp_conf['sim_params'].update(
                                        {
                                            'render':False,
                                            'init_viewer_every_reset':False,

                                        })
    # terrain randomisation
    if 'terrain' in trng_exp_conf['sim_params'].keys():
        for key in trng_exp_conf['sim_params']['terrain'].keys():
            
            temp_dict = {}
            for prop in trng_exp_conf['sim_params']['terrain'][key].keys():

                if 'xlim' in prop:
                    temp_dict.update({
                                        prop.replace('xlim','x')+'_lim':
                                        trng_exp_conf['sim_params']['terrain'][key][prop]
                                        })
                elif 'ylim' in prop:
                    temp_dict.update({
                                    prop.replace('ylim','y')+'_lim':
                                    trng_exp_conf['sim_params']['terrain'][key][prop]
                                    })                        
                else:
                    temp_dict.update({
                                    prop+'_lim':
                                    trng_exp_conf['sim_params']['terrain'][key][prop]
                                    })
            
            all_keys = list(trng_exp_conf['sim_params']['terrain'][key].keys())
            for prop in all_keys:
                trng_exp_conf['sim_params']['terrain'][key].pop(prop)
            
            trng_exp_conf['sim_params']['terrain'][key].update(temp_dict)       
        
    # update trajectory
    if 'traj_slice_t' in trng_exp_conf.keys():
        trng_exp_conf.update({"traj_slice_t_lim":trng_exp_conf.pop('traj_slice_t') })
    

    '''
    upd_trng_exp_conf_file = open(file_path,'w') # remove
    yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)


# for file_name in os.listdir('./exp_confs'):
    
    
#     file_path = os.path.join('./exp_confs',file_name)
#     if 'skate_wheels'in file_name:
#         trng_exp_conf_file = open(file_path) # remove
#         trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
#         trng_exp_conf_file.close()
#         trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')

#         os.remove(file_path)
#         file_path = file_path.replace('wheels','board')
#         upd_trng_exp_conf_file = open(file_path,'w') # remove
#         yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
    
#     if 'skate_board' in file_name:
#         trng_exp_conf_file = open(file_path) # remove
#         trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
#         trng_exp_conf_file.close()
#         # trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')
#         # try:
#         #     trng_exp_conf['observations'].pop('robot_wheel_vel')
        
#         if 'set_robot_on_traj' in trng_exp_conf['initialisations'].keys():
#             trng_exp_conf['initialisations'].pop('set_robot_on_traj')
#             trng_exp_conf['initialisations'].update({'set_robot_on_skateboard_pose':None})
#         # except:
#         #     print(file_name,trng_exp_conf['observations'])
#         # trng_exp_conf['observations'].update({'skateboard_tvel':None})
#         # trng_exp_conf['observations'].update({'skateboard_avel':None})

#         # os.remove(file_path)
#         # file_path = file_path.replace('wheels','board')
#         upd_trng_exp_conf_file = open(file_path,'w') # remove
#         yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
