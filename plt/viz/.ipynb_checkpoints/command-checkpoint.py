import os
import subprocess
os.system("python test.py")
#!python test.py
#os.popen("python test.py")
print(1)
subprocess.call("python test.py", shell=True)




def main():  
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seeds', help='number of seeds', type=int, default=0)
    parser.add_argument('--num_timesteps', type=str, default="1e3")
    parser.add_argument('--play', default=False, action='store_true')
    args = parser.parse_args()
    
    for i in range(args.seeds):
        os.system("python algos/ppo2_sil_online/run.py --alg=ppo2_sil_online --num_timestep="+args.num+timesteps+" --seed=0 --log_path=~/Desktop/logs/nosil --env=Swimmer-vn ")
        
    

