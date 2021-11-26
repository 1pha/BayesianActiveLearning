# Code Usage

## How and What to run
For detailed usage of configuration, please look up [`config.py`](config.py). For some reason, `-h` option does not work now.

### Naive Training
To train naive models, use the command below. Some configuration should be given.
+ `--model_name_or_path` Use one of `bilstm`, `cnnlstm` or `bert` for models.
+ `--load_size` For preset configurations for models, you can choose `base` or `large` for these. `large` option does not optimize properly since the sequence is small and model is comparatively more gigantic that its input.
+ `--batch_size` default is 256. Increase if you have nice looking VRAM hardware.
+ `--fp16` Use mixed precision. Default=True
+ `--init_pct` How much data to use. In 0 < init_pct <= 1 float

```bash
python code/run.py --model_name_or_path=bilstm --load_size=base --do_train=True --do_valid=True
```

### Active Learning Training
You should explcitily decalre few things
+ `--active_learning=True` Without this flag, `run.py` will run with NaiveTrainer
+ `--acquisition` How to calculate uncertainty. You have `random`, `lc`(=least confidence), `margin`(=margin of confidence), `bald`, `batchbald` option.
+ `--acquisition_period` How should 
+ `--approximation` With `bald` and `batchbald` in `acquisition`, you should use `mcdropout` for this. `ensemble` is not implemented yet :).

One example might be as follows.
```bash
python code/run.py --model_name_or_path=bilstm  --active_learning=True --acquisition=lc --load_size=base --do_train=True --do_valid=True
```