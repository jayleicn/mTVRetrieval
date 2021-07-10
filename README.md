multilingual TVRetrieval
=====

mTVR: Multilingual Moment Retrieval in Videos. ACL 2021
[Jie Lei](http://www.cs.unc.edu/~jielei/), 
[Tamara L. Berg](http://tamaraberg.com/), [Mohit Bansal](http://www.cs.unc.edu/~mbansal/)


Convolutional Start-End (ConvSE) detector, making the moment predictions efficient and accurate.
The ConvSe module is inspired by edge detectors in image
processing. It learns to detect start (up) and end (down) edges in the 1D query-clip similarity 
signals with two trainable 1D convolution filters, and is the core of XML's high accuracy 
and efficiency.


## Resources
- Data: [mTVR dataset](./data/)
- CodaLab Submission: [standalone_eval/README.md](standalone_eval/README.md)


## Getting started

### Prerequisites
0. Clone this repository
```
git clone https://github.com/jayleicn/mTVRetrieval.git
cd mTVRetrieval
```

1. Prepare feature files

Download [mtvr_feature.tar.gz](https://drive.google.com/file/d/1I4hK91fe80JpdzkI9CC_COfVzRuE-FIg/view?usp=sharing) (24GB). 
After downloading the feature file, extract it to the project directory:
```
tar -xf path/to/mtvr_features.tar.gz -C .
```
You should be able to see `mtvr_features` under your project root directory. 
It contains video features (ResNet, I3D) and text features (subtitle and query, from RoBERTa). 
Please refer the [TVR repo](https://github.com/jayleicn/TVRetrieval) for details on feature extraction. 

2. Install dependencies.

```
# 1, create conda environment
conda create -n mtvr python=3.7 -y
conda activate mtvr
# 2, install PyTorch 
conda install pytorch torchvision -c pytorch
conda activate mtvr 
pip install easydict tqdm tensorboard
pip install h5py==2.9.0
```

3. Add project root to `PYTHONPATH`
```
source setup.sh
```
Note that you need to do this each time you start a new session.

### Training and Inference

1. mXML training
```
bash baselines/mxml/scripts/train.sh 
```

Training using the above config will stop at around epoch 60, around 1 day with a single 2080Ti GPU.
On val set, for VCMR R@1, IoU=0.7, you should be able to get ~2.4 for Chinese and ~2.9 for English. 
The resulting model and config will be saved at a dir:
`baselines/mxml/results/tvr-video_sub-*`.

2. mXML inference

After training, you can inference using the saved model on val or test_public set:
```
bash baselines/mxml/scripts/inference.sh MODEL_DIR_NAME SPLIT_NAME
```
`MODEL_DIR_NAME` is the name of the dir containing the saved model, 
e.g., `tvr-video_sub-*`. 
`SPLIT_NAME` could be `val` or `test_public`. 
By default, this code evaluates all the 3 tasks (VCMR, SVMR, VR), you can change this behavior 
by appending option, e.g. `--tasks VCMR VR` where only VCMR and VR are evaluated. 
The generated predictions will be saved at the same dir as the model, you can evaluate the predictions 
by following the instructions here [Evaluation and Submission](#Evaluation-and-Submission). 

### Evaluation and Submission

We only release ground-truth for train and val splits, to get results on test-public split, 
please submit your results follow the instructions here:
[standalone_eval/README.md](standalone_eval/README.md)


## Citations
If you find this code useful for your research, please cite our paper:
```
@inproceedings{lei2020tvr,
  title={mTVR: Multilingual Moment Retrieval in Videos},
  author={Lei, Jie and Berg, Tamara L and Bansal, Mohit},
  booktitle={ACL},
  year={2021}
}
```

## Acknowledgement
This research is supported by grants and awards from NSF, DARPA and ARO.
This code is built upon [TVRetrieval](https://github.com/jayleicn/TVRetrieval).

## Contact
jielei [at] cs.unc.edu
