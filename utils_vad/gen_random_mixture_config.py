#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /OneDiarization/egs/far_feild_vad/utils_vad/gen_random_mixture_config.py
Description  : Generate config for converstation talk simulation. Before this, CMU Arctic corpus should be downloaded on /OneDiarization/egs/far_field_vad/datasets/cmu_data.
               The directory should be like this:
               OneDiarization
                ├─ egs
                │  └─ far_field_vad
                │     ├─ datasets
                │     │  ├─ cmu_data
                │     │  │  ├─ cmu_link.txt
                │     │  │  ├─ cmu_us_aew_arctic
                │     │  │  │  ├─ etc
                │     │  │  │  │  └─ txt.done.data
                │     │  │  │  └─ wav
                │     │  │  │     ├─ arctic_a0001.wav
                │     │  │  │     ├─ arctic_a0002.wav
                │     │  │  │     ├─ arctic_a0003.wav
                │     │  │  │     ├─ arctic_a0004.wav
                │     │  │  │     ├─ arctic_a0005.wav
                │     │  │  │     ├─ arctic_a0006.wav
                │     │  │  │     ├─ arctic_a0007.wav
                
Author       : cwg cwg@hnu.edu.cn
Version      : 0.0.1
LastEditors  : cwghnu cwg@hnu.edu.cn
LastEditTime : 2023-03-12 12:49:19
Copyright (c) 2023 by cwg, All Rights Reserved. 
'''

import numpy as np
import os
import itertools
import random
import json
import sys
import pyroomacoustics as pra
import argparse

egs_path = os.path.dirname(os.path.dirname(__file__))

def load_wav_scp(wav_scp_file):
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}

def load_spk2utt(spk2utt_file):
    """ returns dictionary { spkid: list of uttids } """
    if not os.path.exists(spk2utt_file):
        return None
    lines = [line.strip().split() for line in open(spk2utt_file)]
    return {x[0]: x[1:] for x in lines}

def min_distance(spk_pos, spk_pos_list):
    spk_pos = np.array(spk_pos)                 # [3]
    spk_pos_list = np.array(spk_pos_list)       # [num_spks, 3]

    dis_diff = spk_pos[None, :] - spk_pos_list
    min_distance_value = np.min(np.sqrt(np.sum(dis_diff**2, axis=-1)))

    return min_distance_value
    

def gen_mixture_config(
        wav_scp, 
        noise_scp,
        spk2utt, 
        output_scp,
        n_mixtures=10, 
        n_speakers=[2,3,4], 
        min_utts=10, 
        max_utts=20, 
        sil_scales=[2.0, 3.0, 5.0, 10.0],
        noise_snr=-20,
        target_db = [-50, -30],
        rir_time = [0.1, 1.0],
        room_size = [[3.0, 8.0], [3.0, 8.0], [3.0, 4.0]],
        mic_spk_dis = [0.05, 0.1],
        min_dis_spk = 0.5,
        pick_up_ratio = [0.1, 0.4]
    ):
    wavs = load_wav_scp(wav_scp)
    noises = load_wav_scp(noise_scp)

    spk2utt = load_spk2utt(spk2utt)

    all_speakers = list(spk2utt.keys())
    all_noises = list(noises.keys())

    mixtures = []

    with open(output_scp, "w") as f_out:
        for it in range(n_mixtures):
            # recording ids are mix_0000001, mix_0000002, ...
            recid = 'mix_{:07d}'.format(it + 1)
            n_speaker = random.sample(n_speakers, 1)[0]
            sil_scale = random.sample(sil_scales, 1)[0]
            # randomly select speakers, a background noise and a SNR
            speakers = random.sample(all_speakers, n_speaker)
            noise_snr = noise_snr
            noise_id = random.sample(all_noises, 1)[0]
            
            e_absorption = None
            while e_absorption is None:
                utt_rir = random.uniform(rir_time[0], rir_time[1])
                utt_room_size = [random.uniform(room_size[idx][0], room_size[idx][1]) for idx in range(3)]
                try:
                    e_absorption, max_order = pra.inverse_sabine(utt_rir, utt_room_size)
                except Exception as e:
                    pass
                
            
            mixture = {'speakers': []}
            spk_poses = []
            for speaker in speakers:
                # randomly select the number of utterances
                n_utts = np.random.randint(min_utts, max_utts + 1)
                # utts = spk2utt[speaker][:n_utts]
                cycle_utts = itertools.cycle(spk2utt[speaker])
                # random start utterance
                roll = np.random.randint(0, len(spk2utt[speaker]))
                for i in range(roll):
                    next(cycle_utts)
                utts = [next(cycle_utts) for i in range(n_utts)]
                # randomly select wait time before appending utterance
                intervals = np.random.exponential(sil_scale, size=n_utts)

                # generate speakers' position
                spk_mic_distance = random.uniform(0.5, min(utt_room_size[0]//2, utt_room_size[1]//2)-0.5)
                spk_angle = random.uniform(0, 2*np.pi)
                spk_pos = [utt_room_size[0]//2 + np.cos(spk_angle)*spk_mic_distance, utt_room_size[1]//2 + np.sin(spk_angle)*spk_mic_distance, utt_room_size[2]//2+random.uniform(-0.3, 0.3)]
                if len(spk_poses) > 0:
                    while min_distance(spk_pos, spk_poses) < min_dis_spk:
                        spk_mic_distance = random.uniform(0.5, min(utt_room_size[0]//2, utt_room_size[1]//2)-0.5)
                        spk_angle = random.uniform(0, 2*np.pi)
                        spk_pos = [utt_room_size[0]//2 + np.cos(spk_angle)*spk_mic_distance, utt_room_size[1]//2 + np.sin(spk_angle)*spk_mic_distance, utt_room_size[2]//2+random.uniform(-0.3, 0.3)]
                spk_poses.append(spk_pos)

                spk_audio_db = random.uniform(target_db[0], target_db[1])
                mixture['speakers'].append({
                        'spkid': speaker,
                        'utts': [wavs[utt] for utt in utts],
                        'intervals': intervals.tolist(),
                        'audio_db': spk_audio_db
                        })
            
            mic_pose = [utt_room_size[0]//2, utt_room_size[1]//2, utt_room_size[2]//2]
            
            noise_mic_distance = random.uniform(0.5, min(utt_room_size[0]//2, utt_room_size[1]//2)-0.5)
            noise_angle = random.uniform(0, 2*np.pi)
            noise_pos = [utt_room_size[0]//2 + np.cos(noise_angle)*noise_mic_distance, utt_room_size[1]//2 + np.sin(noise_angle)*noise_mic_distance, utt_room_size[2]//2+random.uniform(-0.3, 0.3)]
            while min_distance(noise_pos, spk_poses) < min_dis_spk:
                noise_mic_distance = random.uniform(0.5, min(utt_room_size[0]//2, utt_room_size[1]//2)-0.5)
                noise_angle = random.uniform(0, 2*np.pi)
                noise_pos = [utt_room_size[0]//2 + np.cos(noise_angle)*noise_mic_distance, utt_room_size[1]//2 + np.sin(noise_angle)*noise_mic_distance, utt_room_size[2]//2+random.uniform(-0.3, 0.3)]
            
            mixture['rir_time'] = utt_rir
            mixture['noise'] = noises[noise_id]
            mixture['snr'] = noise_snr
            mixture['recid'] = recid
            mixture['room_size'] = utt_room_size
            mixture['spk_poses'] = spk_poses
            mixture['mic_pose'] = mic_pose
            mixture['noise_pose'] = noise_pos

            # print(recid, json.dumps(mixture))
            f_out.write("{} {}\n".format(recid, json.dumps(mixture)))

if __name__ == "__main__":
    subset="eval"
    wav_scp_path = os.path.join(egs_path, "/home4/huyuchen/raw_data/mixture/wav.scp")
    noise_scp_path = os.path.join(egs_path, "/home4/huyuchen/raw_data/mixture/noise.scp")
    spk2utt_path = os.path.join(egs_path, "/home4/huyuchen/raw_data/mixture/spk2utt")
    mix_scp_path = os.path.join(egs_path, f"/home4/huyuchen/raw_data/mixture/{subset}/mixture.scp")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav_scp',
        type=str,
        default=wav_scp_path,
        help='wav.scp path',
        required=False
    )
    parser.add_argument(
        '--noise_scp',
        type=str,
        default=noise_scp_path,
        help='noise.scp path',
        required=False
    )
    parser.add_argument(
        '--spk2utt',
        type=str,
        default=spk2utt_path,
        help='spk2utt path',
        required=False
    )
    parser.add_argument(
        '--mixture_scp',
        type=str,
        default=mix_scp_path,
        help='output config path',
        required=False
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.mixture_scp)):
        os.makedirs(os.path.dirname(args.mixture_scp))

    gen_mixture_config(args.wav_scp, args.noise_scp, args.spk2utt, args.mixture_scp, n_mixtures=3)