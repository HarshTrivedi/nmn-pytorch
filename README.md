Neural Module Network (NMN) for VQA in Pytorch
===================

**Note: This is NOT an official repository for Neural Module Networks.**

NMN is a network that is assembled dynamically by composing shallow network fragments called modules into a deeper structure. These modules are jointly trained to be freely composable. This is a PyTorch implementation of Neural Module Networks for Visual Question Answering. Most Ideas are directly taken from the following paper:


> **[Neural Module Networks:](http://arxiv.org/abs/1511.02799)** Jacob Andreas, Marcus Rohrbach, Trevor Darrell and Dan Klein. CVPR 2016.

Please cite the above paper in case you use this code in your work. The instructions to reproduce the results can be found below, but first some results demo:

Demo:
-----

More results can be seen with [visualize_model.ipynb]( https://github.com/HarshTrivedi/nmn-pytorch/blob/master/visualize_model.ipynb ).

![Demo1](README-Images/demo1.png?raw=true)  | ![Demo2](README-Images/demo2.png?raw=true)
:------------------------------------------:|:-----------------------------------------:
![Demo3](README-Images/demo3.png?raw=true)  | ![Demo4](README-Images/demo4.png?raw=true)



Download Data:
--------------

You need to download Images, Annotations and Questions from VQA website. And you need to download VGG model file used to preprocess the images. To save you some efforts of making sure downloaded files are appropriate placed in directory structure, I have prepared few ```download.txt```'s'

Run the following command in root directory ``` find . | grep download.txt```. You should be able to see the following directories containing ```download.txt```:

```
./preprocessing/lib/download.txt
./raw_data/Annotations/download.txt
./raw_data/Images/download.txt
./raw_data/Questions/download.txt
```

Each download.txt has specific instruction with ```wget``` command that you need to run in the respective directory. Make sure files are as expected as mentioned in corresponding ```download.txt``` after downloading data.


Proprocessing:
-------------

```preprocessing``` directory contains the scripts required to preprocess the ```raw_data```. This preprocessed data is stored in ``` preprocessed_data```. You need to run the following scripts in order:

```
1. python preprocessing/pick_subset.py [ Optional: If you want to operate on spcific question-type ]
2. python preprocessing/build_answer_vocab.py
3. python preprocessing/build_layouts.py 
4. python preprocessing/build_module_input_vocab.py
5. python preprocessing/extract_image_vgg_features.py
```


Run Experiments:
---------------

You can start training the model with ```python train_cmp_nn_vqa.py```. 
The accuracy/loss logs will be piped to ```logs/cmp_nn_vqa.log```. 
Once training is done, the selected model will be automatically saved at ```saved_models/cmp_nn_vqa.pt```


Visualize Results:
------------------

The results can be visualized by running ```visualize_model.ipynb``` and selecting model name which was saved.








