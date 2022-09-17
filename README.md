# S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning

This is the official implementation of our NIPS 2022 paper "S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning".
In this paper, we propose one simple paradigm (named as S-Prompting) and two concrete approaches to highly reduce the forgetting degree in one of the most typical continual learning scenarios, i.e., domain increment learning (DIL).

**S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning** <br>
Yabin Wang, Zhiwu Huang, Xiaopeng Hong <br>
[[Project Page]](https://arxiv.org/abs/2205.05467.pdf) [[Paper]](https://arxiv.org/abs/2205.05467.pdf) [[Video]](https://www.youtube.com/watch?v=bszy34vY-2o)

## Dependencies

```python
conda create -n sp python=3.8
conda activate sp
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```



Run `pip instalasdfdsafdasfdsafl -r requirements.txt` to install required dependencies.

## Datasets
Please refer to the following links to download and prepare data. 
```
DomainNet:
http://ai.bu.edu/M3SDA/
CoRE50:
https://vlomonaco.github.io/core50/index.html#dataset
CDDB (need to ask the authors for the deepfake data):
https://arxiv.org/abs/2205.05467
```

Organize the dataset folder as:

```
domainnet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ├── angel
│   ├── animal_migration
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
│   ├── alarm_clock
│   ├── ambulance
│   ├── angel
│   ... ...
... ...
```

```
core50
└── core50_128x128
    ├── labels.pkl
    ├── LUP.pkl
    ├── paths.pkl
    ├── s1
    ├── s10
    ├── s11
    ├── s2
    ├── s3
    ├── s4
    ├── s5
    ├── s6
    ├── s7
    ├── s8
    └── s9

```
```
CDDB
├── biggan
│   ├── test
│   ├── train
│   └── val
├── gaugan
│   ├── test
│   ├── train
│   └── val
├── san
│   ├── test
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── test
│   ├── train
│   └── val
├── wild
│   ├── test
│   ├── train
│   └── val
... ...
```



## Training:


### DomainNet:
```
python main.py --config configs/cddb_hard_slip.json
```

### CoRE50:
```
python main.py --config configs/cddb_hard_slip.json
```

### CDDB:
```
python main.py --config configs/cddb_hard_slip.json
```

## Evaluation:

Please refer to 
[[Evaluation Code]](https://github.com/iamwangyabin/SPrompts_eval).

## License

Please check the MIT  [license](./LICENSE) that is listed in this repository.


## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [PyCIL](https://github.com/G-U-N/PyCIL)
- [Best Incremental Learning](https://github.com/Vision-Intelligence-and-Robots-Group/Best-Incremental-Learning)


## Citation

If you use any content of this repo for your work, please cite the following bib entry:
```
@article{wang2022s,
  title={S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning},
  author={Wang, Yabin and Huang, Zhiwu and Hong, Xiaopeng},
  journal={arXiv preprint arXiv:2207.12819},
  year={2022}
}
```
