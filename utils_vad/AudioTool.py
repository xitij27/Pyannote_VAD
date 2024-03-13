#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /OneDiarization/egs/far_feild_vad/utils_vad/AudioTool.py
Description  :  
Author       : cwg cwg@hnu.edu.cn
Version      : 0.0.1
LastEditors  : cwghnu cwg@hnu.edu.cn
LastEditTime : 2023-03-31 18:03:27
Copyright (c) 2023 by cwg, All Rights Reserved. 
'''

import numpy as np


class AudioTool(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize(audio, target_db=-20, max_gain_db=300.0):
        gain = target_db - AudioTool.rms_db(audio)
        if gain > max_gain_db:
            raise ValueError("target_db can not be greater than max_gain_db (%f dB)" % (target_db, max_gain_db))
        return AudioTool.gain_db(audio, min(max_gain_db, gain))
    
    @staticmethod
    def align_audio2maxdb(audio):
        # audio: [num_samples, num_channels]
        rms_db_list = []
        for idx in range(audio.shape[-1]):
            rms_db_list.append(AudioTool.rms_db(audio[:, idx]))

        max_db = np.max(rms_db_list)
        for idx in range(audio.shape[-1]):
            audio[:, idx] = AudioTool.normalize(audio[:, idx], target_db=max_db)

        return audio
    
    @staticmethod
    def align_audio2maxvalue(audio):
        # audio: [num_samples, num_channels]
        max_value_list = []
        for idx in range(audio.shape[-1]):
            max_value_list.append(np.max(np.abs(audio[:, idx])))

        max_value = np.max(max_value_list)
        for idx in range(audio.shape[-1]):
            audio[:, idx] = audio[:, idx] / np.max(np.abs(audio[:, idx])) * max_value

        return audio

    @staticmethod
    def gain_db(audio, gain):
        audio *= 10. ** (gain / 20.)
        return audio

    @staticmethod
    def rms_db(audio):
        mean_square = np.mean(audio ** 2)
        return 10 * np.log10(mean_square)