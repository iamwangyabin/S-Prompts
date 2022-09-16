# S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning

Pytorch implementation for NIPS 2022 paper "S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning".
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
Download the following datasets to `datasets/`
* [CDDB benchmark Dataset]()

Organize the dataset folder as:
```
Project
├── datasets
|   ├── CDDB
|   ├── checkpoints
```



## Training:
### DyTox
Using DyTox, train the model with binary/multi labels, with different sequences:
```
cd DyTox
bash train.sh 0,1 \
    --options configs/data/ganfake_easy.yaml configs/data/ganfake_easy_order.yaml configs/model/ganfake_pretrain_dytox.yaml \
    --name dytox_easy_m1500_sumblog0.1 \
    --data-path ./datasets/CDDB/  \
    --output-basedir ./datasets/checkpoints  \
    --memory-size 1500 \
    --binary_loss sum_b_log \
    --binary_weight 0.1
```

### LUCIR:
Using LUCIR, train the model with binary/multi labels, with different sequences:
```
cd LUCIR
python lucir_main.py --name lucir_easy_m1500_sumasig0.1 --checkpoints_dir ./datasets/checkpoints  --dataroot ./datasets/CDDB/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0 --batch_size 32 --num_epochs 40 --binary_loss sum_a_sig --binary_weight 0.1
```

### iCaRL:
Using iCaRL, train the model with binary/multi labels, with different sequences:
```
cd iCaRL
python main_icarl_CNND.py --name icarl_easy_m1500_sumasig0.1 --checkpoints_dir ./datasets/checkpoints --model_weights ./datasets/checkpoints/no_aug/model_epoch_best.pth --dataroot ./datasets/CDDB/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --binary_loss sum_a_sig
```



## Citation

When using the code/figures/data/etc., please cite our work
```
@article{li2022continual,
  title={A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials},
  author={Li, Chuqiao and Huang, Zhiwu and Paudel, Danda Pani and Wang, Yabin and Shahbazi, Mohamad and Hong, Xiaopeng and Van Gool, Luc},
  journal={arXiv preprint arXiv:2205.05467},
  year={2022}
}
```
