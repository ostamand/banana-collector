# Unity Banana Collector 
![DQN agent](assets/show_result.gif)
This repository includes the code needed to train agents to solve Udacity navigation project based on [Unity's Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment.

The environment has two disctinct scenes. Refer to each specific page for setup, training and results.

- [Vector Observation Space](Vector.md)
- [Visual Observations](Visual/Visual.md)

## Installation

Clone the repository and initialize the submodules.

```
git clone https://github.com/ostamand/banana-collector.git
cd banana-collector 
git submodule init 
git submodule update
```

Create a virtual environment and activate it.

```
python -m venv banana_0.4.0b
source banana_0.4.0b/bin/activate
```

Install Unity ml-agents.
```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
```

Install the project requirements.

```
pip install -r requirements.txt
```
### Optional 

Install a new kernel.

```
ipython kernel install --user --name=banana_0.4.0b
```

If necessary, to delete the kernel use

```
jupyter kernelspec uninstall banana_0.4.0b
```