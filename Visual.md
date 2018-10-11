# Visual Banana Collector 

# Visual Observations

## Setup on AWS

Create a spot graphic compute instance with community AMI: ami-18642967.

Start the X Server.

```
sudo /usr/bin/X :0 &
```

Make the ubuntu use X Server for display.

```
export DISPLAY=:0
```


Downgrade IPython
```
pip install -U ipython==6.5.0
```