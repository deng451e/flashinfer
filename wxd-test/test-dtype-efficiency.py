import torch 
import time 
import argparse
import numpy as np 

def main(args,log):
    if args.dtype == 'fp16': type_ = torch.float16
    if args.dtype == 'fp32': type_ = torch.float32
    if args.device == 'cpu': device = 'cpu'
    if args.device == 'gpu': device = 'cuda'
    
    # warmup
    log += "==================================\n"
    log += f"device:{device}, dtype:{type_}\n"
    
    for power in range(1,16):
        dim = 2**power
        x = torch.randn(dim,dim,dtype=type_) 
        y = torch.randn(dim,dim,dtype=type_) 
        if device=='gpu':
            x = x.to(device)
            y = y.to(device)

        _ = x@y
        
    for power in range(1,16):
        dim = 2**power  
        compute_times = []
        transfer_times = []
        x = torch.randn(dim,dim,dtype=type_) 
        y = torch.randn(dim,dim,dtype=type_) 
        for _ in range(args.repeat):
            
            
            
            if device=='cuda':
                st = time.time()
                x_ = x.to(device)
                y_ = y.to(device)
                transfer_times.append(time.time()-st)
                
            else:
                x_ = x.to(device)
                y_ = y.to(device)
            st = time.time()
            _ = x_@y_
            compute_times.append(time.time()-st)
        if device=='cpu': log += f"dim:{dim}, compute time:{np.mean(compute_times)} sec \n"
        if device=='cuda': log += f"dim:{dim}, compute time:{np.mean(compute_times)} sec, transfer time:{np.mean(transfer_times)} sec \n"

    print(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="fp16") 
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    
    main(args,"")