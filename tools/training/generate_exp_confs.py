import yaml
import copy
import os
import csv
from itertools import product
import argparse

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))
cwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_confpath",  default='./exp_confs/pw_2_stilts.yaml', type=str)  
    parser.add_argument("--vary_paramspath", default='./exp_confs/trng_parms_to_vary.yaml', type=str)             
    parser.add_argument("--remove_base_conf",  action='store_true')               
    args = parser.parse_args()
    trng_base_name = args.base_confpath.split('/')[-1].replace('.yaml','')

    print('trng_base_name:',trng_base_name)

    base_exp_conf_file = open(args.base_confpath) 
    base_exp_conf = yaml.load(base_exp_conf_file, Loader=yaml.FullLoader)
    vary_trng_params_file = open(args.vary_paramspath) 
    params_to_vary =  yaml.load(vary_trng_params_file, Loader=yaml.FullLoader)


    variant_list = list(product_dict(**params_to_vary))


    print("n. variants:",len(variant_list))

    exp_id = 0

    os.mkdir( './exp_confs/'+trng_base_name)

    with open('./exp_confs/'+trng_base_name+'/param_vary_list.csv','w') as pvl:


        csv_writer = csv.writer(pvl)
        for variant_id,variant in enumerate(variant_list):
            this_variant = copy.deepcopy(base_exp_conf)

            if variant_id == 0:
                header_names = ['exp_names']+[pn for pn in variant.keys()]
                print(header_names)
                csv_writer.writerow(header_names)

            csv_writer.writerow([str(exp_id)]+ [pv for pv in variant.values()])          

            for param_name in variant.keys():
                val = this_variant
                param_path = param_name.split('/')
                param_val = variant[param_name]

                for i in range(len(param_path)-1):
                    # print(val.keys(),param_path[i])
                    val = val[param_path[i]]
                # print(val)
                # if param_path[-1] in val.keys():
                #     val.pop(param_path[-1])    
                val.update({param_path[-1]:param_val})
            
            
            logdir = cwd +'/logs/'+trng_base_name+'/'+str(exp_id)+'/'

            this_variant.update({'logdir':logdir})
            dumpfile_at = './exp_confs/'+trng_base_name+'/'+str(exp_id)+'.yaml'
            
            
            trng_exp_conf_file =  open(dumpfile_at,'w')
            yaml.dump(this_variant,trng_exp_conf_file,default_flow_style=False,sort_keys=False)
            print('dumped file:',dumpfile_at)
            exp_id+=1
        if args.remove_base_conf:
            os.remove(args.base_confpath)
            print('deleted file:',args.base_confpath)
        

