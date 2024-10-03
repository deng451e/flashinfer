 
for arch_name in "opt-1.3b" "opt-2.7b" "opt-6.7b" "opt-13b"
do 
  for seq_len in 1000 5000 10000 20000
  do
       
        for q_len in 1 10 50 100 
        do  
            for ratio in  0 0.01 0.1 0.2 0.5 1
            do 
            CMD="test-mix-attention-single.py --arch_name $arch_name --seq_len $seq_len --q_len $q_len --ratio $ratio"
            
            output=$(python   $CMD   2>&1 )  
            echo "$outpt" | tee -a mix-attention-single.log
            done
        done
  done
done


 

