
import os
import gc
import torch
import logging
from glob import glob
import numpy as np

from pyannote.core import Segment, Timeline, Annotation

import sys
egs_path = os.path.dirname(__file__)
sys.path.append(egs_path)

from utils_vad.utils import downsample
from sklearn.metrics import roc_auc_score
import soundfile as sf
import numpy as np
import time

def load_model(pretrained, device):
    from model.PyanNet import PyanNet

    model = PyanNet(num_channels=1)
    device = torch.device(device)

    try:
        states = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(states['model'])
        logging.info(f'{pretrained} loaded')
        del states
    except:
        raise RuntimeError(f'Unable to load pretrained model: {pretrained}')

    return model.to(device)

def inference_single(source, model, output_dir='', overlap=0, merge_dist_ms=128, expand_ms=128, chunk_ms=1000, min_spec_threshold=3, save_wav=False, min_vad_ms=100, sampling_rate = 16000, device="cpu", threshold=0.5, rttm_dir=''):
    import torchaudio

    chunk_sample = int(sampling_rate * chunk_ms / 1000)
    step_sample = int(sampling_rate * chunk_ms * (1 - overlap) / 1000)
    
    device = torch.device(device)
    
    global chunk_size, stride  # Declare as global variables
    chunk_size = str(chunk_ms/1000)
    stride = str(chunk_ms * (1 - overlap) / 1000)
    ### 1. wav format
    # audio, sr = torchaudio.load(source)

    ### 2. flac format
    audio, sr = sf.read(source)
    audio = torch.from_numpy(audio).float()
    assert len(audio.shape) == 1, audio.shape
    audio = audio.unsqueeze(0)

    len_audio = audio.shape[-1]

    if len_audio % chunk_sample > 0:
        len_pad = chunk_sample - len_audio % chunk_sample
        pad_audio = torch.zeros(audio.shape[0], len_pad)
        audio = torch.cat([audio, pad_audio], dim=-1)

    if sampling_rate != sr:
        logging.warning(f'Resample from {sr} to {sampling_rate}')
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=sampling_rate)

    # convert to single channel
    if len(audio.shape) > 1:
        audio = audio[0, :]
        
    vad_mask = torch.zeros(audio.shape)
    
    small_noise = torch.randn(audio.shape) * 1e-6

    t0 = time.time()

    for st_sample in range(0, len(audio), step_sample):
        audio_chunk = audio[st_sample:st_sample+chunk_sample]
        
        frame_mask = model(audio_chunk[None, None, :])
        # logging.info("chunk {}: {}, {}".format(st_sample, torch.min(frame_mask), torch.max(frame_mask)))
        frame_mask = torch.where(frame_mask > threshold, 1, 0)
        sample_mask = downsample(frame_mask, audio_chunk[None, :]).squeeze().float()
        
        vad_mask[..., st_sample:st_sample+chunk_sample] += sample_mask

    t = time.time() - t0
        
    vad_mask = torch.where(vad_mask > 1, 1, vad_mask)
    
    masked_audio = vad_mask * audio + (1 - vad_mask) * small_noise
    
    # if output_dir == '':
    #     out_name = os.path.join(os.path.dirname(source), os.path.basename(source).replace(".wav", "_vad.wav"))
    # else:
    #     out_name = os.path.join(output_dir, os.path.basename(source))
    #
    # # logging.info(f"{source} after vad: {masked_audio}")
    # torchaudio.save(out_name, masked_audio[None, :], sampling_rate, encoding='PCM_S', bits_per_sample=16)

    ### compute vad accuracy
    gt_mask = torch.zeros_like(vad_mask)
    utt_id = source.strip().split('/')[-1].split('.')[0]
    f_rttm = open(f'{rttm_dir}/{utt_id}.rttm', 'r')
    for line in f_rttm.readlines():
        tokens = line.strip().split()
        start, end = int(float(tokens[3]) * sr), int((float(tokens[3]) + float(tokens[4])) * sr)
        assert end <= gt_mask.shape[0], (end, gt_mask.shape)
        gt_mask[start-1: end] = 1
    acc = (vad_mask == gt_mask).sum() / gt_mask.shape[0]
    fa = ((vad_mask == 1) & (gt_mask == 0)).sum() / gt_mask.shape[0]
    missing = ((vad_mask == 0) & (gt_mask == 1)).sum() / gt_mask.shape[0]
    roc_auc = roc_auc_score(gt_mask.numpy(), vad_mask.numpy())

    logging.info(f'{utt_id}: Acc = {acc:.3f}, False alarm = {fa:.3f}, Missing detection = {missing:.3f}, ROC-AUC score = {roc_auc:.3f}')

    return acc, fa, missing, roc_auc, t, gt_mask.shape[-1]/16000


def main(args):
    verbose = args.verbose
    logging.info(f'source={args.source}, pretrained={args.pretrained}, data_dir={args.data_dir}')

    total_acc, total_fa, total_missing, total_roc_auc, num_samples = 0, 0, 0, 0, 0
    total_t1, total_t2 = 0, 0

    if os.path.isdir(args.source):
        for source in glob(os.path.join(args.source, "*.flac")):
            if not '_masked-' in source:
                model = load_model(args.pretrained, args.device)
                logging.info(f'Processing {source}')
                acc, fa, missing, roc_auc, t1, t2 = inference_single(source, model, args.data_dir,
                                 args.overlap_percent, args.merge_dist_ms, args.expand_ms, save_wav=args.save_wav, min_vad_ms=args.min_vad_ms, sampling_rate=args.sample_rate, rttm_dir=args.rttm_dir)
                logging.info(f'processing time: {t1:.3f}, speech duration: {t2:.3f}')
                total_acc += acc
                total_fa += fa
                total_missing += missing
                total_roc_auc += roc_auc
                num_samples += 1
                total_t1 += t1
                total_t2 += t2
                del model
                gc.collect()
        total_acc /= num_samples
        total_fa /= num_samples
        total_missing /= num_samples
        total_roc_auc /= num_samples
    else:
        model = load_model(args.pretrained, args.device)
        logging.info(f'Processing {args.source}')
        acc, fa, missing, roc_auc, t1, t2 = inference_single(args.source,
                         model, args.data_dir, args.overlap_percent, args.merge_dist_ms, args.expand_ms, save_wav=args.save_wav, min_vad_ms=args.min_vad_ms, sampling_rate=args.sample_rate, rttm_dir=args.rttm_dir)
        logging.info(f'processing time: {t1:.3f}, speech duration: {t2:.3f}')
        total_acc += acc
        total_fa += fa
        total_missing += missing
        total_roc_auc += roc_auc
        total_t1 += t1
        total_t2 += t2
        del model
        gc.collect()

    logging.info('======================== Results ========================')
    logging.info(f'Chunk Size = {chunk_size}s, stride = {stride}s')
    logging.info(f'Acc = {total_acc:.3f}, Missing detection = {total_missing:.3f}, False alarm = {total_fa:.3f}, ROC-AUC score = {total_roc_auc:.3f}')
    logging.info(f'Processing time = {total_t1:.3f}, speech duration = {total_t2:.3f}, RTF = {total_t2 / total_t1:.3f}')
    logging.info('Inference completed!')

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'source',
        type=str,
        help='single audio file or folder containing audio files to infer',
    )
    parser.add_argument(
        'pretrained',
        type=str,
        help='pretrained checkpoint file',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default="",
        help='directory to store results',
        required=False,
    )
    parser.add_argument(
        '--rttm_dir',
        type=str,
        default="",
        help='directory to gt rttm',
        required=True,
    )
    parser.add_argument(
        '-sr', '--sample_rate',
        type=int,
        default=16000,
        help='audio source sampling rate',
        required=False
    )
    parser.add_argument(
        '-o', '--overlap_percent',
        type=float,
        default=0,
        help='Percentage of overlap in chunks',
        required=False
    )
    parser.add_argument(
        '-m', '--merge_dist_ms',
        type=int,
        default=128,
        help='Merge segments with gap within number of samples',
        required=False
    )
    parser.add_argument(
        '-e', '--expand_ms',
        type=int,
        default=128,
        help='Expand segment boundaries',
        required=False
    )
    parser.add_argument(
        '--duration_ms',
        type=int,
        default=25,
        help='Fixed data length for inference',
        required=False
    )
    parser.add_argument(
        '--min_vad_ms',
        type=int,
        default=0,
        help='Fixed data length for inference',
        required=False
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='folder to save output',
        required=False
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for model inference"
    )
    parser.add_argument(
        "--save_wav", 
        type=bool,
        default=True,
        help="Save masked wav file", 
        required=False
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="show debug info", required=False)
    args = parser.parse_args()
    logging.info(f'{args}\n')

    main(args)