## Usage
```
mkdir {checkpoints, figures}
python train.py
```
## Experience
Do not set batch size too large, such as 1024, which will mislead the training. I finally set the batch size 32 and the model performs well.
