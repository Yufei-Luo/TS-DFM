# LearnTS 
This model targets to predict transition state structures from reactant and product structures. 
The reference paper is now submitted in Nature Comm. Please, pray for successful publication.
More details are described in [reference](https://www.nature.com/articles/s41467-023-36823-3)
## Install 
pip install -r requirements.txt
python setup.py install 


### Additional requirements 
When you use db.py, you need to download [ard\_gsm](https://github.com/cgrambow/ard_gsm) package

## Example
### Training new model 
please run training script (example/train.py) after dumping db by running db.py 
(Note: current address of db foler is ../trajectory5\_4/train-db-distance.pkl, please change this information before running training script)

To reproduce the same reult of reference paper, please use below options.

```
python train.py --batch_size 10 --learning_rate 2e-4 
                --gru_dropout 0.4 --accelerator ddp   --num_layers 2 --num_pair_embedding 128 --num_gru_layers 2
                --max_epoch 3000  --gradient_clip_val 1.0 
                --log_every_n_steps 10 --gpus 8 --num_nodes 1 --precision 64 
                --factor1 1000.0 --factor2 0.0 --factor3 0.0 --factor4 0.0 --factor5 0.0
```

Here, factor2 and factor3 correspond to c' in the reference paper. If you want different values, please change them.


### Inference 

example/inference.py 


### Reproducability 
db\_indices.txt include series of indices, which are required to reproduce the [publication](https://www.nature.com/articles/s41467-023-36823-3)
First 80% of integers represent indices of training reactions and the following 10% of integers correspond to the validation reactions.
The rest of numbers represent test reactions

In Python script,

    test_ratio = 0.1
    valid_ratio = 0.1

    indices =list( range(0, 11961) )
    length = len(indices)
    test_size = int(length*test_ratio)
    valid_size = int(length*valid_ratio)
    train_size = length - test_size - valid_size

    print(length, test_size, valid_size, train_size)

    np.random.shuffle(indices)

    train_filenames = filenames[:train_size]
    valid_filenames = filenames[train_size:train_size+valid_size]
    test_filenames = filenames[train_size+valid_size:]
