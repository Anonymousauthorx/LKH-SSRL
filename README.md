# Learning Lin-Kernighan-Helsgaun Heuristic for Routing Optimization with Self-Supervised Reinforcement Learning



### Generate the training dataset

```bash
python data_generate.py -train
```
### Train the Model 

```bash
CUDA_VISIBLE_DEVICES="0" python train.py --file_path train --eval_file_path val --eval_batch_size=100 --save_dir=saved/tsp_lkhssrl --learning_rate=0.0001
```
### Finetune the node decoder for large sizes
The finetuning process takes less than 1 minute for each size.
```bash
CUDA_VISIBLE_DEVICES="0" python finetune_node.py
```
### Testing
Test with the pretrained model on TSP with 500 nodes:
```bash
python test.py --dataset test/500.pkl --model_path pretrained/lkhssrl.pt --n_samples 1000 --lkh_trials 1000 --lkhssrl_trials 1000
```
We test on the TSPLIB instances 
```bash
python tsplib_test.py
```

## Other Routing Problems (CVRP, CVRPTW)
### Testing with pretrained models
test for CVRP with 100 customers, PDP and CVRPTW with 40 customers
```bash
# Capacitated Vehicle Routing Problem (CVRP)
python CVRPdata_generate.py -test
python CVRP_test.py --dataset CVRP_test/cvrp_100.pkl --model_path pretrained/cvrp_lkhssrl.pt --n_samples 1000 --lkh_trials 10000 --lkhssrl_trials 10000
# CVRP with Time Windows (CVRPTW)
python CVRPTWdata_generate.py -test
python CVRPTw_test.py --dataset CVRPTW_test/cvrptw_40.pkl --model_path pretrained/cvrptw_lkhssrl.pt --n_samples 1000 --lkh_trials 10000 --lkhssrl_trials 10000
```
### Training
train for CVRP with 100-500 customers, CVRPTW with 40-200 customers
```bash
# Capacitated Vehicle Routing Problem (CVRP)
python CVRPdata_generate.py -train
CUDA_VISIBLE_DEVICES="0" python CVRP_train.py --save_dir=saved/cvrp_neurolkh
# CVRP with Time Windows (CVRPTW)
python CVRPTWdata_generate.py -train
CUDA_VISIBLE_DEVICES="0" python CVRPTW_train.py --save_dir=saved/cvrptw_lkhssrl
```

## Dependencies
* Python >= 3.6
* Pytorch
* sklearn
* Numpy
* tqdm


## Acknowledgements
* The LKH code is the 3.0.6 version (http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz)

