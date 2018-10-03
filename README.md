## Introduction 
![DQN agent](assets/show_result.gif)
This repository include the code needed to train an agent for [Unity's Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment.

## Installation
Clone the repository and initialize the submodules.
```
git clone https://github.com/O1SA/banana-collector.git
git submodule init 
git submodule update
cd banana-collector 
```
Install the project requirements.
```
pip install -r requirements.txt
```

Install Unity ml-agents.
```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
```

Download the Banana Collectors environment.
```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
unzip Banana_Linux_NoVis.zip
mv Banana_Linux_NoVis/ data/
rm Banana_Linux_NoVis.zip
```

## Training 
To train the model use this [notebook](Train_DQN.ipynb). If the installation is already completed, skip the Installation section and run all the cells of the notebook. 

The hyperparameters and training characteristics are all grouped in one cell. Refer to the page [report](Report.md) for a description of all the variables as well as their baseline values. All variables can easily be modified before starting the training.

## Results 
A [trained model](saved_models/model_dqn.ckpt) with an average score over 100 episodes of 16.93 is included in this repository.

For a more complete description of the results, refer to the [report](Report.md) page. 

To visualise the agent, use this [notebook](Result_DQN.ipynb).