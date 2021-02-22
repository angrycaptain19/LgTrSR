# [LgTrSR] Self-adaptive Local-global aware Transformer for Session-based Recommendation 
## Introdution
Here is my pytorch implementation of LgTrSR described in the competition report **Self-adaptive Local-global aware Transformer for Session-based Recommendation.**

```
LgTrSR
├── data-ml                    
├── data-zf                    
├── network                   // proposed four models
├── dataset                   // if you don't want to transform dataset to pickle format, you can use this.
├── train.py                  // training models, and you can import different models from network folder
├── inference.py              // inference models
├── util.py                   
└── README                    
└── tools                 
    ├── data_vis.py            //visualize datasets
    ├── dataload.py            //transform dataset to pickle format and load datasets
    ├── get_seq_len.py         //get the accurate sequence length from the datasets
    └── get_user_item_id.py    //get the accurate user id and item id from the datasets

```

## Requirements:
* **python 3.6**
* **numpy**
* **pytorch 1.1.0**


## Datasets

The benchmark datasets I used for training and validation could be download from [Link](https://drive.google.com/drive/folders/1dGgnAT42HZ4O21hncR8YKIgetwvLvwk8?usp=sharing "Link"). Additionally, I use word2vec pre-trained models, taken from GLOVE, which you could download from [Link](https://nlp.stanford.edu/projects/glove/ "Link"). 

## Transform dataset to Pickle
In order to speed training, I transform the dataset to the pickle format.

```
python tools/dataload.py
```
## Training

```
python train.py  
        [--max_epochs 2]  
        [--batch_size 32]  
        [--num_layers 3]  
        [--lr 0.001]  
        [--sequence_length 64]  
        [--lstm_size 256]  
        [--embedding_dm 50]
```

## Inference

```
python inference.py
```
## Visualization

```
python tools/data_vis.py 
```
## Citations
Please consider citing original paper in your publications if the project helps your research. 
```
@article{jiang2021lgtrsr,
  title   =  {{LgTrSR}: Self-adaptive Local-global aware Transformer for Session-based Recommendation},
  author  =  {Jiang, Juyong},
  year    =  {2021}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

