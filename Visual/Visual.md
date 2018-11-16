# Visual Observations

## Environment

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

## Setup on AWS

Create a spot graphic compute instance with community AMI: ami-18642967.

Clone this repository.

```
git clone https://github.com/ostamand/banana-collector.git
git submodule update
git submodule init
```

Activate the pytorch conda environment.

```
source activate pytorch_p36
```

Install ml-agents using the library already included in the AMI.

```
cd ml-agents/python
pip install .
```

Downgrade IPython and install tqm
```
pip install -U ipython==6.5.0
pip install tqdm 
```

Start the X Server and make ubuntu use it for display.

```
sudo /usr/bin/X :0 &
export DISPLAY=:0
```

Download the Unity Visual Banana Linux environment.

```
cd ~/banana-collector/Visual
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip
unzip VisualBanana_Linux.zip
```

## Training 

For a list of argument availabe for the train script use the `--help` argument 

```
python train.py --help
```

For example, to run the training for 1000 episodes, reload the environment each 500 and log every 10, the following command would be used:

```
python train.py --env_file VisualBanana_Linux/Banana.x86_64 --episodes 1000 --log_every 10 --reload_every 500 --save_thresh 5
```

To download files locally after training.

```
scp -i path/to/key user@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:path/to/file .
```

## Results 

A [trained model](saved_models/p_dqn.ckpt) with an average score over 100 episodes of 13.59 is included in this repository.

The environment is solved in 815 episodes. 

For a more complete description of the results, refer to the [report](Report.md) page. 