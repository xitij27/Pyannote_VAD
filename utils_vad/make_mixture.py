#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /OneDiarization/egs/far_feild_vad/utils_vad/make_mixture.py
Description  : Generate mixture wav file according to config generated from gen_random_mixture_config.py. Note: this script can only process 16kHz audio.
Author       : cwghnu cwg@hnu.edu.cn
Version      : 0.0.1
LastEditors  : cwghnu cwg@hnu.edu.cn
LastEditTime : 2023-04-02 15:55:00
Copyright (c) 2023 by cwg, All Rights Reserved. 
'''

import argparse
import os
import numpy as np
import math
import soundfile as sf
import json
import soundfile as sf
from scipy import signal
import rir_generator
from AudioTool import AudioTool
# import gpuRIR
import pyroomacoustics as pra
from pyannote.core import Annotation, Segment
from pydub import AudioSegment
import random

import sys
egs_silero_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'silero_vad')
sys.path.append(egs_silero_path)
sys.path.append("/home3/huyuchen/pytorch_workplace/OneDiarization/egs/silero_vad")
from vad_silero import vad_silero

egs_path = os.path.dirname(os.path.dirname(__file__))
egs_path = "/home3/huyuchen/pytorch_workplace/OneDiarization/egs/far_feild_vad"

def make_mixture(mixture_scp_file, out_data_dir, out_wav_dir, rate=16000):
    '''
    @description: 
    @param mixture_scp_file [str]: the full path of mixture.scp
    @param out_data_dir [str]: directory to store segments/utt2spk/wav.scp files
    @param out_wav_dir [str]:  directory to store generated wav files
    @param rate [int]: sampling rate
    @return [None]
    '''

    # open output data files
    segments_f = open(out_data_dir + '/segments', 'w')
    utt2spk_f = open(out_data_dir + '/utt2spk', 'w')
    wav_scp_f = open(out_data_dir + '/wav.scp', 'w')
    spk2utt_f = open(out_data_dir + '/spk2utt', 'w')
    reco2dur_f = open(out_data_dir + '/reco2dur', 'w')

    rttm_dir = os.path.join(out_data_dir, "rttm")
    if not os.path.exists(rttm_dir):
        os.makedirs(rttm_dir)
        
    vad = vad_silero(return_last=False)	

    for line in open(mixture_scp_file):
        recid, jsonstr = line.strip().split(None, 1)
        indata = json.loads(jsonstr)
        
        print("Simulating {}".format(recid))

        rttm_rec_f = open(os.path.join(rttm_dir, recid+".rttm"), 'w')
        rec_annotation = Annotation()

        wavfn = indata['recid']
        # recid now include out_wav_dir
        # recid = os.path.join(out_wav_dir, wavfn).replace('/','_')
        recid = wavfn
        noise_snr = indata['snr']
        utt_rir = indata['rir_time']
        utt_room_size = indata['room_size']
        mixture = []

        try:
            e_absorption, max_order = pra.inverse_sabine(utt_rir, utt_room_size)
        except Exception as e:
            print(e)
            continue
            
        room = pra.ShoeBox(
            utt_room_size,
            fs=rate,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )

        for idx_speaker, speaker in enumerate(indata['speakers']):
            spkid = speaker['spkid']
            spkid = spkid + "_{}".format(idx_speaker)
            utts = speaker['utts']
            intervals = speaker['intervals']
            
            audio_db = speaker['audio_db']

            data = []
            pos = 0
            for interval, utt in zip(intervals, utts):
                # append silence interval data
                silence = np.zeros(int(interval * rate))
                data.append(silence)
                rec = utt
                st = 0
                et = None

                speech, _  = sf.read(rec, start=st, stop=et)
                speech = AudioTool.normalize(speech, target_db=audio_db)

                data.append(speech)
                # calculate start/end position in samples
                startpos = pos + len(silence)
                endpos = startpos + len(speech)

                start_time = startpos / rate
                end_time = endpos / rate

                for _slice in vad.get_voice_frames_offline(rec):
                    start_s = _slice['start'] + start_time
                    end_s = _slice['end'] + start_time
                    rec_annotation[Segment(start_s, end_s)] = spkid
                    
                    # write segments and utt2spk
                    uttid = '{}_{}_{:07d}_{:07d}'.format(
                            spkid, recid, int(start_s * 100),
                            int(end_s * 100))
                    print(uttid, recid,
                        start_s, end_s, file=segments_f)
                    print(uttid, spkid, file=utt2spk_f)
                    print(spkid, uttid, file=spk2utt_f)

                # update position for next utterance
                pos = endpos
            data = np.concatenate(data)

            mixture.append(data)

        # save rttm file
        rec_annotation.write_rttm(rttm_rec_f)
        rttm_rec_f.close()
        
        # fitting to the maximum-length speaker data, then mix all speakers
        maxlen = max(len(x) for x in mixture)
        mixture = [np.pad(x, (0, maxlen - len(x)), 'constant') for x in mixture]
        signal_power_list = [np.mean(data**2) for data in mixture]
        
        # scale noise
        signal_power_average = np.min(signal_power_list)
        noise_data, sr = sf.read(indata['noise'])
        if maxlen > len(noise_data):
            noise_data = np.pad(noise_data, (0, maxlen - len(noise_data)), 'wrap')
        else:
            noise_data = noise_data[:maxlen]
        noise_power = np.mean(noise_data**2)
        scale = math.sqrt(math.pow(10, - noise_snr / 10) * signal_power_average / noise_power)
        noise_data = noise_data * scale
        room.add_source(indata['noise_pose'], signal=noise_data)

        # add spk signal
        for idx_spk in range(len(mixture)):
            spk_noise = np.random.randn(maxlen)
            noise_power = np.mean(spk_noise**2)
            scale = math.sqrt(math.pow(10, - 25 / 10) * signal_power_average / noise_power)
            mixture[idx_spk] += spk_noise * scale
            room.add_source(indata['spk_poses'][idx_spk], signal=mixture[idx_spk])

        speech_rev = []
        room.add_microphone_array(pra.MicrophoneArray(np.array(indata['mic_pose'])[None, :].T, fs=rate))
        
        room.compute_rir()
        
        mixture = np.concatenate([mixture, noise_data[None, :]], axis=0)     # [num_spks + noise, num_samples]
        
        spk_rirs = room.rir # [num_mic, num_spks, samples]
        maxlen = []
        for idx_mic in range(len(spk_rirs)):
            for idx_spk in range(len(spk_rirs[idx_mic])):
                maxlen.append(len(spk_rirs[idx_mic][idx_spk]))
        maxlen = max(maxlen)
        for idx_mic in range(len(spk_rirs)):
            for idx_spk in range(len(spk_rirs[idx_mic])):
                spk_rirs[idx_mic][idx_spk] = np.pad(spk_rirs[idx_mic][idx_spk], (0, maxlen-len(spk_rirs[idx_mic][idx_spk])), 'constant')
                
            spk_rirs[idx_mic] = spk_rirs[idx_mic] / np.max(spk_rirs[idx_mic])
        spk_rirs = np.array(spk_rirs)
        spk_rirs = np.transpose(spk_rirs, axes=(1,2,0))

        speech_rev = []     # [num_spk, num_samples, num_mics]
        for idx_spk in range(len(mixture)):
            spk_reverb = signal.convolve(mixture[idx_spk, :, None], spk_rirs[idx_spk, :, :])   # [num_samples, num_mics]
            speech_rev.append(spk_reverb)
        speech_rev = np.stack(speech_rev)
        
        # [num_samples, num_spk]
        mixture_mc = np.sum(speech_rev, axis=0)
        if np.max(mixture_mc) > 1:
            mixture_mc = mixture_mc / np.max(mixture_mc)

        # # output the wav file and write wav.scp
        outfname = '{}.wav'.format(wavfn)
        outpath = os.path.join(out_wav_dir, outfname)
        sf.write(outpath, mixture_mc, rate)
        # mixture_mc, _ = sf.read(outpath)
        print(recid, os.path.abspath(outpath), file=wav_scp_f)
        print(recid, "{:.6f}".format(len(mixture_mc)/rate), file=reco2dur_f)

    wav_scp_f.close()
    segments_f.close()
    utt2spk_f.close()
    reco2dur_f.close()
    
    del vad

if __name__ == "__main__":
    subset = "eval"
    mix_scp_path = os.path.join(egs_path, f"/home4/huyuchen/raw_data/mixture/{subset}/mixture.scp")
    out_wav_dir = os.path.join(egs_path, f"/home4/huyuchen/raw_data/mixture/{subset}/wav")
    out_data_dir = os.path.join(egs_path, f"/home4/huyuchen/raw_data/mixture/{subset}/data")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mixture_scp',
        type=str,
        default=mix_scp_path,
        help='mixture config path',
        required=False
    )
    parser.add_argument(
        '--out_wav_dir',
        type=str,
        default=out_wav_dir,
        help='mixture wav dir',
        required=False
    )
    parser.add_argument(
        '--out_data_dir',
        type=str,
        default=out_data_dir,
        help='mixture kaldi style config dir',
        required=False
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
        
    if not os.path.exists(args.out_wav_dir):
        os.makedirs(args.out_wav_dir)

    make_mixture(args.mixture_scp, args.out_data_dir, args.out_wav_dir)