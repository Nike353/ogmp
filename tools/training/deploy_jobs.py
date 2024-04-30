# squeue -u lkrajan -h -t pending,running -r -O "state" | uniq -c

# #!/bin/bash
# #SBATCH --partition=epyc-64
# #SBATCH --ntasks=1
# #SBATCH	--cpus-per-task=64
# #SBATCH	--mem=128GB
# #SBATCH	--time=48:00:00

# module load python
# source /home1/lkrajan/python_venvs/osudrl/bin/activate

# python3 main.py ppo --workers 64 \
#                     --batch_size 64 \
#                     --sample 50000 \
#                     --epochs 8 \
#                     --traj_len 500 \
#                     --timesteps 600000000 \
#                     --discount 0.95 \
#                     --layers 128,128 \
#                     --std 0.13 \
#                     --logdir ${logdir} \
#                     --exp_conf_path ${exp_conf_path} \
#                     --recurrent

import os
import csv
from datetime import datetime
import argparse
import shutil
import subprocess
import random

cwd = os.getcwd()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--logdir",  default='/logs/', type=str)  
    parser.add_argument("--local",  action='store_true')

    args = parser.parse_args()

    in_exp_confs = os.listdir('./exp_confs')
    now = datetime.now()

    date_str = now.strftime("%d-%m-%Y")

    with open('./exp_deploy_logs/'+date_str+'.csv','a+') as trng_log:

        csv_writer = csv.writer(trng_log)
        csv_writer.writerow(['subtime','job_id','exp_name'])
            
        for folder_name in in_exp_confs:
            if os.path.isdir('./exp_confs/'+folder_name) and folder_name not in ['bg_b_pr3_2_obs2']:
                
                print('\n',folder_name)                
                
                if not os.path.exists(cwd +args.logdir+'/'+folder_name):
                    os.mkdir(cwd +args.logdir+'/'+folder_name)

                logdir = cwd +args.logdir+folder_name+'/'
                if not os.path.exists(logdir):
                    os.mkdir(logdir)
                
                

                


                for exp_name in os.listdir('./exp_confs/'+folder_name):
                    
                    if '.yaml' in exp_name:

                        exp_conf_path = cwd +'/exp_confs/'+folder_name+'/'+exp_name                            
                        exp_name = exp_name.replace('.yaml','')
                        
                        if args.local:

                            src_cmnd = 'source ~/igym/bin/activate'
                            command = "python3 train_sb3.py ppo --timesteps 6000 --logdir "+logdir+" --exp_conf_path "+exp_conf_path

                            command = src_cmnd
                            print(command)

                            process = subprocess.Popen(
                                                        "gnome-terminal -e 'bash -c \""+command+";bash\"'",
                                                        stdout=subprocess.PIPE,
                                                        stderr=None,
                                                        shell=True
                                                        )
                            
                    
                        else:
                            command = "sbatch --export=logdir="+logdir+",exp_conf_path="+exp_conf_path+" template.job"
                            job_id = os.popen(command).read().replace('Submitted batch job ','')         

                            now = datetime.now()
                            time_str = now.strftime("%H:%M:%S")
                            # write the log
                            csv_writer.writerow([time_str, job_id, folder_name+'/'+exp_name])

                            print("\tdeployed exp: ",exp_name,"job id:",job_id)
                    else:
                        shutil.copy(    
                                        cwd +'/exp_confs/'+folder_name+'/'+exp_name,# src
                                        cwd +args.logdir+folder_name+'/'+exp_name # dst
                                    )
