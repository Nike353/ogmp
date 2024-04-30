# a file to monitor the jobs of an experiment
from datetime import datetime
import argparse
import csv
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='mr_f7_nm')
    parser.add_argument('--exp_date', type=str, help='ddmmyyyy', default='07-03-2024')
    args = parser.parse_args()
    
    # read and list the job ids from the csv file
    deploy_log_path = './exp_deploy_logs/'+args.exp_date+'.csv'
    

    rows = []
    with open(deploy_log_path, newline='') as csvfile:
        log_reader = csv.reader(csvfile)
        for row in log_reader:
            if args.exp_name in row[2]:
                rows.append(row)

    # find unique elements for coloumn 2
    unique_elements = set()
    for row in rows:
        unique_elements.add(row[2])
    
    # find duplicates for each unique element
    duplicates_per_ue = {}
    for ue in unique_elements:
        duplicates_per_ue[ue] = []
        for row in rows:
            if ue in row[2]:
                duplicates_per_ue[ue].append(row)
    
    # find the max by time
    latest_jobs = []
    for ue in unique_elements:
        duplicates_per_ue[ue].sort(key=lambda x: datetime.strptime(x[0], '%H:%M:%S'))
        # log the latest job
        latest_jobs.append(duplicates_per_ue[ue][-1])

    # print the latest jobs
    print('latest jobs:')
    for lj in latest_jobs:
        print(lj)

    for lj in latest_jobs:
        # get the status of the job as as a string
        job_id = lj[1]
        job_status = os.popen('squeue -j '+job_id).read().split('\n')[1].split()[4]
        if job_status == 'R':
            # check if the output file is created
            output_file = './logs/'+lj[2]+'/'+'actor.pt'

            if os.path.exists(output_file):
                pass
                
        
        
