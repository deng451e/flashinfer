rm dtype-efficienct.log
for device in "cpu"  "gpu"  
do 
  for dtype in "fp16" "fp32"
  do
       
        
            CMD="test-dtype-efficiency.py --device $device --dtype $dtype"
            output=$(python   $CMD   2>&1 )  
            echo "$output" | tee -a mix-efficienct-single.log
           
  done
done


 

