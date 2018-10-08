# Vector Observation Space

The environment state space has 37 dimensions and four discrete actions are available to the agent. A reward of +1 and -1 is provided for collecting respectively a yellow banana and a blue banana.

In order to solve the environment, an agent must get an average score of +13 over 100 consecutive episodes.

## Setup on AWS

Download the Banana Collectors environment.

```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
unzip Banana_Linux_NoVis.zip
mv Banana_Linux_NoVis/ data/
rm Banana_Linux_NoVis.zip
```

## Training 
To train the model use this [notebook](Train_DQN.ipynb). If the installation is already completed, skip the related section and run all the cells of the notebook. 

The hyperparameters and training characteristics are all grouped in one cell. Refer to the page [report](Report.md) for a description of all the variables as well as their baseline values. All variables can easily be modified before starting the training.

## Results 
A [trained model](saved_models/model_dqn.ckpt) with an average score over 100 episodes of 16.93 is included in this repository.

For a more complete description of the results, refer to the [report](Report.md) page. 

To visualise the agent, use this [notebook](Result_DQN.ipynb).