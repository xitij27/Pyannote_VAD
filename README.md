# PyanNet VAD

Voice activity detection with PyanNet, the SincNet → BiLSTM → feed-forward architecture from [pyannote.audio](https://github.com/pyannote/pyannote-audio/tree/develop). This repo trains and evaluates it on [AliMeeting](https://www.openslr.org/119/) (far-field) and [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12).

Part of the final year project [Robust Voice Activity Detection using DNN Approaches](https://github.com/xitij27/Robust-Voice-Activity-Detection-using-DNN-Approaches).

## Repository layout

```
Pyannote_VAD/
├── config/
│   ├── config_ali.yaml      # training config for AliMeeting
│   └── config_dh.yaml       # training config for DIHARD III
├── dataloader/
│   └── data_loader.py       # chunked dataset with labels from RTTM files
├── model/
│   ├── PyanNet.py           # the model used by every script
│   └── PyanNet_se.py        # variant with a SpeechBrain MetricGAN+ enhancement front end (not used by any script)
├── trainer/
│   ├── trainer.py           # training step, checkpoint save/load (used by train_*.py)
│   └── trainer_se.py        # same, defaults to PyanNet_se (not used by any script)
├── trained_models/
│   ├── best_ckpt_ali        # trained checkpoint for AliMeeting
│   └── best_ckpt_dh         # trained checkpoint for DIHARD III
├── utils_vad/               # data-preparation helpers, see below
├── train_ali.py
├── train_dh.py
├── inference_ali.py
└── inference_dh.py
```

## Requirements

The repo has no requirements file. The scripts import `torch`, `torchaudio`, `pyannote.audio`, `pyannote.core`, `pyannote.database`, `einops`, `librosa`, `soundfile`, `scipy`, `scikit-learn`, `numpy` and `pyyaml`. No versions are pinned.

`utils_vad/utils.py` uses `np.float`, which was removed in NumPy 1.24, so install `numpy<1.24`.

## Inference with the trained checkpoints

Run from the repository root:

```bash
# AliMeeting: a .wav file, or a folder of .wav files
python inference_ali.py /path/to/wav_dir trained_models/best_ckpt_ali --rttm_dir /path/to/rttm_dir -d cpu

# DIHARD III: a .flac file, or a folder of .flac files
python inference_dh.py /path/to/flac_dir trained_models/best_ckpt_dh --rttm_dir /path/to/rttm_dir -d cpu
```

Arguments:

- `source` (first positional argument): one audio file or a folder. Given a folder, `inference_ali.py` reads `*.wav` and `inference_dh.py` reads `*.flac`.
- `pretrained` (second positional argument): path to the checkpoint.
- `--rttm_dir`: a folder holding one reference `<name>.rttm` per audio file. Both scripts need it. Only `inference_dh.py` marks it as required, but `inference_ali.py` fails without it.
- `-d/--device`: pass `-d cpu`. The default is `cuda` when a GPU is visible, but the audio is never moved to the GPU, so that default fails with a device-mismatch error.
- `-o/--overlap_percent` (default `0`) and `-sr/--sample_rate` (default `16000`) work as named.
- `--data_dir`, `--output_dir`, `--save_wav`, `-m`, `-e`, `--duration_ms`, `--min_vad_ms` and `-v` are parsed but have no effect. The scripts write no masked audio and no RTTM files.

Settings fixed in the code:

|                 | `inference_ali.py`                      | `inference_dh.py`                       |
|-----------------|-----------------------------------------|-----------------------------------------|
| Audio reader    | `torchaudio` (keeps the first channel)  | `soundfile` (mono only; asserts 1-D)    |
| Window length   | 5 s                                     | 1 s                                     |
| Speech threshold| 0.5                                     | 0.5                                     |

Per-file results and the averaged results are logged to the console. File names are taken from the path with `split('/')`, so use `/`-separated paths (Linux or macOS).

## Training

Always pass the config explicitly:

```bash
python train_ali.py -c config/config_ali.yaml
python train_dh.py  -c config/config_dh.yaml
```

`train_ali.py` defaults to `config/config.yaml`, which doesn't exist. `train_dh.py` defaults to `config/config_dh.yaml`.

### What's missing before training can run

1. **Modules from outside this repo.** `dataloader/data_loader.py` imports `KaldiData` and `load_wav` from `utils.kaldi_data`, and `read_rttm` from `utils.feature`. Neither module is in this repo. The loader looks for them in a parent `code_and_model/` folder, found by cutting the current working directory at `code_and_model`.
2. **`dataloader/data_loader_aug.py`.** `train_ali.py` imports this loader, which adds noise using `noise_path` and `snr`. It isn't in the repo. The `data_loader.py` that is here takes neither argument.
3. **Data format.** `train_path` and `eval_path` are passed to `KaldiData`. The loader reads `.wavs` and `.reco2dur` from that object, so these paths must be Kaldi-style data directories with `wav.scp` and `reco2dur` files, not plain folders of audio. `train_rttm_path` and `eval_rttm_path` hold one `<recording>.rttm` per recording.
4. **A CUDA GPU.** The evaluation loop always calls `torch.nn.parallel.data_parallel` with the configured `gpus`. The CPU setting (`gpus: [-1]`) is only handled inside `trainer.step`.

### Config keys

| Key | Used by | Meaning |
|-----|---------|---------|
| `train_path`, `eval_path` | both | Kaldi-style data directories (see above) |
| `train_rttm_path`, `eval_rttm_path` | both | folders of reference RTTMs |
| `noise_path`, `snr` | `train_ali.py` only | passed to the missing noise-augmenting loader |
| `output_directory` | both | where the log file `Stat` and checkpoints go |
| `checkpoint_path` | `train_ali.py` only | resume from this checkpoint if it isn't empty |
| `max_epoch`, `batch_size`, `epochs_per_eval`, `iters_per_log`, `seed` | both | training loop |
| `chunk_size`, `chunk_step`, `rate` | both | training chunk length and stride (seconds), and the sample rate |
| `num_workers` | `train_ali.py` only | `DataLoader` workers (commented out in `train_dh.py`) |
| `gpus` | both | GPU ids |
| `optimizer_param` | both | `Adam` or `RAdam`, learning rate, gradient-norm clip |
| `model_config` | both | keyword arguments for `PyanNet` |
| `use_pre_compute_feature`, `frame_len`, `frame_shift` | neither | present in the configs but unused |

`train_dh.py` ignores `checkpoint_path`. Its checkpoint-loading code is commented out, and so is the key in `config_dh.yaml`, so DIHARD training starts from scratch.

### Differences between the two training scripts

|                          | `train_ali.py`                         | `train_dh.py`                                      |
|--------------------------|----------------------------------------|----------------------------------------------------|
| Dataset class            | `data_loader_aug.Dataset` (missing)    | `data_loader.Dataset`                              |
| Chunk / stride in config | 5 s / 4 s                              | 1 s / 0.4 s                                        |
| GPUs in config           | `[0, 1]`                               | `[0]`                                              |
| Resumes from checkpoint  | yes                                    | no                                                 |
| Checkpoints saved        | every 5 epochs                         | after the final epoch only                         |
| Errors during an epoch   | raised                                 | caught; the rest of that epoch is skipped and the errors are printed at the end |

`data_loader.Dataset` returns `None` for any chunk whose length isn't exactly `chunk_size`. The `train_dh.py` collate function filters those out, and skips a batch if every chunk in it was dropped.

Checkpoints are saved as `<output_directory>/<MM-DD_HH-MM>_<epoch>`. Each is a dict with `model`, `optimizer` and `iteration` keys, the format the inference scripts load.

### Model and loss

Both configs use SincNet with stride 10, a 2-layer bidirectional LSTM with 128 units, two 128-unit linear layers and a single sigmoid output, on 16 kHz mono audio. The loss is binary cross-entropy between the frame outputs and the RTTM speech mask, with the mask downsampled to the frame rate.

## Data-preparation helpers (`utils_vad/`)

None of these are called by the training or inference scripts. Most default to hard-coded cluster paths.

| File | Purpose |
|------|---------|
| `utils.py` | `downsample()`, used by training and inference |
| `prepare_rttm.sh` | AliMeeting: writes `wav.scp`, `utt2spk`, `spk2utt` and `text`, and converts TextGrid annotations to per-recording RTTMs with `make_textgrid_rttm.py`. Set the paths at the top of the script |
| `make_textgrid_rttm.py` | Converts one TextGrid file to RTTM |
| `make_spkdiar.py` | Writes a Kaldi-style data directory (`wav.scp`, `utt2spk`, `spk2utt`, `rttm`, `segments`, `reco2dur`) from a folder of `.wav` files and a combined `rttm` file |
| `prepare_cmu_data.py`, `gen_random_mixture_config.py`, `make_mixture.py`, `AudioTool.py` | Simulate multi-speaker mixtures from the CMU Arctic corpus with room acoustics and added noise. `make_mixture.py` also imports `vad_silero` from a sibling `silero_vad` folder that isn't in this repo |

## How the metrics are computed

- **Prediction:** frame scores above 0.5 count as speech, upsampled to one value per audio sample. With `--overlap_percent > 0`, overlapping windows are OR-ed together.
- **Reference:** one value per sample, from the RTTM segments.
- **Accuracy:** the fraction of samples where prediction and reference agree.
- **False alarm:** samples predicted as speech that the reference marks as non-speech, as a fraction of *all* samples.
- **Missed detection:** samples predicted as non-speech that the reference marks as speech, as a fraction of *all* samples.
- **ROC-AUC:** `sklearn.metrics.roc_auc_score` on the binary decisions, not on the model scores.
- **Averaging:** each file is scored separately and the scores are averaged, so every file counts equally.
- **`RTF`:** printed as *total audio duration ÷ model processing time*, which is how many times faster than real time the model runs. This is the inverse of the usual RTF definition. The value labelled "speech duration" in the log is the total padded audio length, not the amount of speech.

## Attribution

- `model/PyanNet.py` is modified from pyannote.audio, as its header comment says.
- `utils_vad/AudioTool.py`, `prepare_cmu_data.py`, `gen_random_mixture_config.py` and `make_mixture.py` carry author headers from an upstream OneDiarization codebase.
