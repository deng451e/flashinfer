rm sparse-fetch-efficienct.log

for dtype in "fp16" "fp32"
do
      
      
          CMD="test-sparse-fetch.py --dtype $dtype"
          output=$(python   $CMD   2>&1 )  
          echo "$output" | tee -a  sparse-fetch-efficienct.log
          
done



 

