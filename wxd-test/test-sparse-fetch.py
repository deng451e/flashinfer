import torch 
import time 
import argparse
import numpy as np 
import random
def main(args,log):
    if args.dtype == 'fp16': type_ = torch.float16
    if args.dtype == 'fp32': type_ = torch.float32
    if args.device == 'cpu': device = 'cpu'
    if args.device == 'gpu': device = 'cuda'
    

    

     
     
   
    log += "==================================\n"
    log += f"device:{device}, dtype:{type_}\n"
      # warmup
    for seq_len in range(1000,10001,1000):
        
        for dim in [64,80,128]:
            for ratio in range(5,10):
                ratio /= 10 
            
                cache = torch.randn(seq_len,dim,device=device,dtype=type_)
                indices = list(range(seq_len))
                
                selected_indices = torch.tensor(random.sample(indices,int(len(indices)*ratio))).to(device)


    for seq_len in range(1000,10001,1000):
        
        for dim in [64,80,128]:
            for ratio in range(5,10):
                ratio /= 10 
                times  = []
                for _ in range(args.repeat):
                    cache = torch.randn(seq_len,dim,device=device,dtype=type_)
                    indices = list(range(seq_len))
                    selected_indices = torch.tensor(random.sample(indices,int(len(indices)*ratio))).to(device)
                    st = time.time()
                    _ = torch.index_select(cache, dim=0, index=selected_indices)  # Using index_select
                    times.append(time.time()-st)
                log += f"seq_len:{seq_len}, dim:{dim}, ratio:{ratio}, time:{np.mean(times)} sec \n"

    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="fp16") 
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    
    main(args,"")