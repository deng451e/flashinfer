rm ${PWD}/bench_mergeState_zeroCopy_a6000.log
rm ${PWD}/bench_mergeState_a6000.log
export CUDA_VISIBLE_DEVICES=1
${PWD}/../build/bench_all_mergeState_ZeroCopy_methods > ${PWD}/bench_mergeState_zeroCopy_a6000.log
${PWD}/../build/bench_all_mergeState_methods > ${PWD}/bench_mergeState_a6000.log