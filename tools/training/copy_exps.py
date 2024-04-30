
import os
import csv
from datetime import datetime


if __name__ == '__main__':
    
    exp_names = []
    remote_log_path = ' lkrajan@discovery.usc.edu:/home1/lkrajan/drcl_projects/cpptf_v2/logs/'
    exp_confs = os.listdir('./exp_confs')
    command = 'rsync -av'

    # copy based on the confs
    # for exp_name in exp_confs:
    #     exp_name = exp_name.replace('.yaml','')
        
    #     if exp_name not in ['default','tstng_exp_conf','tstng_conf','trng_parms_to_vary']:
    #         print('to be copied:',exp_name)
    #         command += remote_log_path+exp_name


    # copy in list
    for exp_name in [

                        # 'mr_f9_nm'
                        # 'mr_f10_nm',
                        # 'gfwg1_ni',
                        'gfwg3_lh',

                        ]:
        # exp_name = exp_name.replace('.yaml','')
        
        if exp_name != 'default' and ('tstng' not in exp_name):
            command += remote_log_path+exp_name


    # name trend
    # for exp_no in range(45):
    #     exp_name = 'oce_rt_'+str(exp_no) #exp_name.replace('.yaml','')
        
    #     if exp_name != 'default' and ('tstng' not in exp_name):
    #         command += remote_log_path+exp_name    


    command += ' ./logs/'
    print(command)

    os.system(command)

