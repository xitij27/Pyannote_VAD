#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /OneDiarization/egs/far_feild_vad/utils_vad/prepare_cmu_data.py
Description  : Prepare CMU Arctic corpus for VAD simulation
Author       : cwg cwg@hnu.edu.cn
Version      : 0.0.1
LastEditors  : cwghnu cwg@hnu.edu.cn
LastEditTime : 2023-03-31 15:51:20
Copyright (c) 2023 by cwg, All Rights Reserved. 
'''

import os
from glob import glob
import argparse

egs_path = os.path.dirname(os.path.dirname(__file__))

def prepare_cmu_data(wav_dir):
    scp_dir = os.path.dirname(wav_dir)

    with open(os.path.join(scp_dir, "wav.scp"), 'w') as f_wav, open(os.path.join(scp_dir, "spk2utt"), 'w') as f_spk2utt:
        dict_spk2utt = dict()
        for source in glob(os.path.join(wav_dir, "**/*.wav"), recursive=True):
            wav_full_path = os.path.join(wav_dir, source)
            path_list = wav_full_path.split(os.sep)
            speaker_id, wav_name = path_list[-3], path_list[-1].split(".wav")[0]
            wav_id = speaker_id+"_"+wav_name

            f_wav.write("{} {}\n".format(wav_id, wav_full_path))

            if speaker_id in dict_spk2utt.keys():
                dict_spk2utt[speaker_id].append(wav_id)
            else:
                dict_spk2utt[speaker_id] = [wav_id]

        for key, value in dict_spk2utt.items():
            f_spk2utt.write("{} {}\n".format(key, " ".join(value)))

def prepare_noise_rir_wavscp(noise_dir):

    scp_dir = os.path.dirname(noise_dir)
    with open(os.path.join(scp_dir, "noise.scp"), 'w') as f_noise:
        for source in glob(os.path.join(noise_dir, "**/*.wav"), recursive=True):
            wav_full_path = os.path.join(noise_dir, source)
            wav_id = os.path.basename(wav_full_path).split(".wav")[0]

            f_noise.write("{} {}\n".format(wav_id, wav_full_path))


if __name__ == "__main__":
    
    wav_dir = os.path.join(egs_path, "/home4/huyuchen/raw_data/cmu_arctic")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
        
    noise_dir = os.path.join(egs_path, "/home4/huyuchen/raw_data/RIRS_NOISES/pointsource_noises")
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav_dir',
        type=str,
        default=wav_dir,
        help='wav file directory',
        required=False
    )
    parser.add_argument(
        '--noise_dir',
        type=str,
        default=noise_dir,
        help='noise file directory',
        required=False
    )
    args = parser.parse_args()

    prepare_cmu_data(args.wav_dir)
    
    # prepare noise wavs from MUSAN
    prepare_noise_rir_wavscp(args.noise_dir)