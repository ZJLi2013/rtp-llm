

```sh
# 1. Prepare ROCM runtime Docker 
drun -v $HOME/hip-tests:/workspace -w /workspace rocm/dev-ubuntu-22.04:6.1-complete 
apt update && apte install cmake 
# commit as rocm-dev:base 
# 2. launch RTP-LLM with AMD-implement
drun -v $HOME/rtp-llm:/workspace/ -w /workspace rocm-dev:base
cd rtp-llm 
# 3. build amd_impl
mkdir -p build && cd build 
cmake ..
make all
# 4, test amd_impl  
./rocm_test 

```