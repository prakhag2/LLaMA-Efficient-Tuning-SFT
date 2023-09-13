# LLaMA-Efficient-Tuning-SFT
Fork of https://github.com/hiyouga/LLaMA-Efficient-Tuning. Filtered only for SFT related code.

# Steps to follow.

1. Create VMs with L4s (or any other accelerator)
   
2. Install Python, CUDA driver. Rough steps:

Python:  
sudo apt update && sudo apt upgrade -y   
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev wget git liblzma-dev  
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz  
tar -xf Python-3.10.*.tgz  
cd Python-3.10.*/  
./configure --prefix=/usr/local --enable-optimizations --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"  
make -j $(nproc)  
sudo make altinstall  

CUDA:  
sudo apt-get update -y  
sudo apt-get upgrade  
sudo apt-get dist-upgrade   
sudo reboot  

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py  
sudo python3.10 install_gpu_driver.py  

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb  
sudo dpkg -i cuda-keyring_1.0-1_all.deb  
sudo add-apt-repository contrib  
sudo apt-get update  
sudo apt-get -y install cuda  

export CUDA_HOME=/usr/local/cuda-12.2  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/extras/CUPTI/lib64  
export PATH=$PATH:$CUDA_HOME/bin  

3. Install this repo  
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning-SFT  
cd LLaMA-Efficient-Tuning  
sudo pip3.10 install -r requirements.txt  
pip3.10 install deepspeed bitsandbytes  

4. Ensure that *all the VMs* created can ssh into each other. Rough steps:  
   
On VM1, execute:  
ssh-keygen -t rsa  
cat ~/.ssh/id_rsa.pub -> copy the public key  

On VM2, execute:  
Copy the public key copied above in ~/.ssh/authorized_keys  
chmod 600 ~/.ssh/authorized_keys  
chmod 700 ~/.ssh  

You may want to ensure that /etc/ssh/sshd_config file is appropriate:  
Port 2300  
ListenAddress 0.0.0.0  
ListenAddress ::  
PermitRootLogin yes  
StrictModes no  
PubkeyAuthentication yes  
AuthorizedKeysFile      .ssh/authorized_keys .ssh/authorized_keys2  
ChallengeResponseAuthentication no  
UsePAM no  

You may also have to run the following to reflect the changes  
service sshd restart  
sudo reboot  

5. On every VM, run  
accelerate config  

Pick the following options:  
In which compute environment are you running?  This machine                                                                                                       
Which type of machine are you using?           multi-GPU                                                                                                        
How many different machines will you use (use more than 1 for multi-node training)? [1]: 4                       
What is the rank of this machine?               0 (rank == machine-index)                                                                                                                
What is the IP address of the machine that will host the main process? 1.1.1.1                                   
What is the port you will use to communicate with the main process? 9001                                         
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no                                                                                            
What rendezvous backend will you use? ('static', 'c10d', ...): static  
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes  
Do you wish to optimize your script with torch dynamo?[yes/NO]:no  
Do you want to use DeepSpeed? [yes/NO]: yes  
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no  
What should be your DeepSpeed's ZeRO optimization stage? 3                                                                                                                
Where to offload optimizer states? none                                                                                                             
Where to offload parameters?       none                                                                                                             
How many gradient accumulation steps you're passing in your script? [1]: 4                                       
Do you want to use gradient clipping? [yes/NO]: no                                                               
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: no                                   
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes  
Which Type of launcher do you want to use? standard                                                                                                         
How many GPU(s) should be used for distributed training? [1]:32                                                  
Do you wish to use FP16 or BF16 (mixed precision)? BF16                                                             

6. On every VM, run  
huggingface-cli login -> meta-llama/Llama-2-70b-chat-hf is a gated model  

7. On every VM, run  
accelerate launch src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-70b-chat-hf --do_train --dataset alpaca_gpt4_en --template llama2 --finetuning_type lora --lora_target q_proj,v_proj --output_dir path_to_sft_checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5 --num_train_epochs 1.0 --plot_loss --bf16 --overwrite_output_dir  
