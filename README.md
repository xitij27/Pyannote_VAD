# Pyannet VAD

We implement the [pyannet](https://github.com/pyannote/pyannote-audio/tree/develop) to evaluate the performance on [Alitmeeting](https://www.openslr.org/119/) and [Dihard-3](https://catalog.ldc.upenn.edu/LDC2022S12) datasets.

## Alimeeting dataset

### Inference with our trained model

Please open the script `run_infer_ali.sh`, specify your conda environment name and some paths:

- `source`: directory that contains audio files in `.wav` format;
- `rttm_dir`: directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `pretrained`: path of our trained model, which can be found at `./trained_models/best_ckpt_ali`;

Then, run it wirh following command:

```shell
bash run_infer_ali.sh
```

### Train your own model

We also provide a script `run_train_ali.sh` to train pyannet on Alimeeting dataset.
To run it, please first open the configuration file `config/config_ali.yaml` and specify some paths:

- `train_path`: training data directory that contains audio files in `.wav` format;
- `train_rttm_path`: training data directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `eval_path`: test data directory that contains audio files in `.wav` format;
- `eval_rttm_path`: test data directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `noise_path`: directory that contains noise audio files for data augmentation;
- `snr`: a list of signal-to-noise ratios (dB) for random selection;
- `output_directory`: directory that contains experiment log file and saved checkpoints;
- `checkpoint_path`: optional pre-trained model, if not specified the model would be trained from scratch (we leave it blank when training on Alimeeting dataset);
- `max_epoch`: number of training epochs;
- `gpus:`: a list of your GPU ids;

Then, run it with following command:

```shell
bash run_train_ali.sh
```


## Dihard-3 dataset

### Inference with our trained model

Please open the script `run_infer_dh.sh`, specify your conda environment name and some paths:

- `source`: directory that contains audio files in `.flac` format;
- `rttm_dir`: directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `pretrained`: path of our trained model, which can be found at `./trained_models/best_ckpt_dh`;

Then, run it wirh following command:

```shell
bash run_infer_dh.sh
```

### Train your own model

We also provide a script `run_train_dh.sh` to train pyannet on Dihard-3 dataset.
To run it, please first open the configuration file `config/config_dh.yaml` and specify some paths:

- `train_path`: training data directory that contains audio files in `.flac` format;
- `train_rttm_path`: training data directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `eval_path`: test data directory that contains audio files in `.flac` format;
- `eval_rttm_path`: test data directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `output_directory`: directory that contains experiment log file and saved checkpoints;
- `checkpoint_path`: optional pre-trained model, if not specified the model would be trained from scratch (since there is only Dihard-3 dev set available for training, we load the Alimeeting trained model instead of training from scratch);
- `max_epoch`: number of training epochs;
- `gpus:`: a list of your GPU ids;

Then, run it with following command:

```shell
bash run_train_dh.sh
```