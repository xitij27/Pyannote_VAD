import os
import sys
import torch
import random
import numpy as np
import librosa
import torch

proj_dir = os.path.join(os.getcwd().split("code_and_model")[0], "code_and_model")
sys.path.append(proj_dir)

from utils.kaldi_data import KaldiData, load_wav
from utils.feature import read_rttm

from itertools import permutations
from pyannote.core import Segment


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)

def random_dict(dict_in):
    dict_in_keys = list(dict_in.keys())
    random.shuffle(dict_in_keys)
    new_dict = {}
    for key in dict_in_keys:
        new_dict[key] = dict_in.get(key)
    return new_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_path, 
            rttm_path,
            chunk_size=5, 
            chunk_step=4, 
            rate=16000, 
        ):

        self.data_path = data_path
        self.rttm_path = rttm_path
        self.chunk_size = chunk_size
        self.chunk_step = chunk_step
        self.rate = rate

        self.kaldi_obj = KaldiData(self.data_path)
        
        self.total_chunk = 0
        self.chunk_indices = []
        for rec in self.kaldi_obj.wavs:
            num_chunks = int((self.kaldi_obj.reco2dur[rec] - self.chunk_size + self.chunk_step) // self.chunk_step)
            # self.total_chunk += num_chunks
            
            data, _ = load_wav(self.kaldi_obj.wavs[rec], 0, int(self.rate))
            num_channel = data.shape[-1]
            if num_channel > 100:
                num_channel = 1

            for idx_channel in range(num_channel):
                for idx_chunk in range(num_chunks):
                    if idx_chunk * self.chunk_step + self.chunk_size < self.kaldi_obj.reco2dur[rec]:
                        self.chunk_indices.append(
                            (rec, idx_channel, idx_chunk * self.chunk_step,
                             idx_chunk * self.chunk_step + self.chunk_size)
                        )
                        self.total_chunk += 1

        print("[Dataset Msg] total number of chunks: {}".format(self.total_chunk))

    # def __init__(
    #         self, 
    #         data_path, 
    #         rttm_path,
    #         chunk_size=5, 
    #         chunk_step=4, 
    #         rate=16000, 
    #     ):
    #     self.data_path = data_path
    #     self.rttm_path = rttm_path
    #     self.chunk_size = chunk_size
    #     self.chunk_step = chunk_step
    #     self.rate = rate

    #     self.kaldi_obj = KaldiData(self.data_path)
        
    #     self.chunk_indices = []
    #     fixed_length = 16000  # Based on sample rate

    #     for rec in self.kaldi_obj.wavs:
    #         data, _ = load_wav(self.kaldi_obj.wavs[rec], 0, int(self.rate))
    #         num_channel = data.shape[-1]
    #         if num_channel > 100:
    #             num_channel = 1

    #         for idx_channel in range(num_channel):
    #             idx_start = 0
    #             while idx_start + self.chunk_size <= self.kaldi_obj.reco2dur[rec]:
    #                 idx_end = idx_start + self.chunk_size
    #                 if idx_end - idx_start == fixed_length / self.rate:
    #                     idx_start += self.chunk_step
    #                     continue  # Skip this chunk
    #                 self.chunk_indices.append((rec, idx_channel, idx_start, idx_end))
    #                 idx_start += self.chunk_step

    #     self.total_chunk = len(self.chunk_indices)
    #     print("[Dataset Msg] Total number of chunks: {}".format(self.total_chunk))

    def get_mask_from_rttm(self, rec_id, num_sample, time_start, time_end):
        
        rttm_path = os.path.join(self.rttm_path, rec_id + ".rttm")
        
        rttm_content = read_rttm(rttm_path)
        
        mask = np.zeros(int(num_sample), np.int32)
        
        target_segment = Segment(time_start, time_end)
        for line in rttm_content:
            line_content = line.split(" ")
            st, dur, label = float(line_content[3]), float(line_content[4]), line_content[7]
            curr_segment = Segment(st, st+dur)
            if target_segment.intersects(curr_segment):
                overlap_segment = target_segment & curr_segment
                start_sample = int((overlap_segment.start - time_start) * self.rate)
                end_sample = int((overlap_segment.end - time_start) * self.rate)
                mask[max(0, start_sample):end_sample] = 1
            
        return mask

    # Original getitem
    # def __getitem__(self, index):

    #     utt_id, idx_channel, idx_start, idx_end = self.chunk_indices[index]
        
    #     data_signal, sr = load_wav(self.kaldi_obj.wavs[utt_id], int(idx_start*self.rate), int(idx_end*self.rate))
    #     if len(data_signal.shape) == 2:
    #         data_signal = data_signal[:, idx_channel:idx_channel+1].T     # [num_channel, num_samples]
    #     else:
    #         data_signal = data_signal[None, :]
    #     data_signal = torch.from_numpy(data_signal).float()
    #     mask = self.get_mask_from_rttm(utt_id, data_signal.shape[-1], idx_start, idx_end)
    #     mask = torch.from_numpy(mask)
    #     print(f"data_signal {index} shape: {data_signal.shape} | mask shape: {mask.shape}")

    #     return {
    #         "feat": data_signal,     # [num_channel, num_sample]
    #         "label": mask.long(),                  # [num_frame]     time domain mask
    #     }
    
    def __len__(self):
        return self.total_chunk

    # this takes of lot of time to train
    # def __getitem__(self, index):
    #     utt_id, idx_channel, idx_start, idx_end = self.chunk_indices[index]

    #     # try:
    #     data_signal, sr = load_wav(self.kaldi_obj.wavs[utt_id], int(idx_start * self.rate), int(idx_end * self.rate))
    
    #     if len(data_signal.shape) == 2:
    #         data_signal = data_signal[:, idx_channel:idx_channel + 1].T  # [num_channel, num_samples]
    #     else:
    #         data_signal = data_signal[None, :]

    #     # Pad or truncate the audio features to a fixed length
    #     fixed_length = 16000  # Based on sample rate. Need to change accordingly
    #     num_samples = data_signal.shape[-1]
   
    #     # if num_samples < fixed_length:
    #     #     data_signal = np.pad(data_signal, ((0, 0), (0, fixed_length - num_samples)), mode='constant')
    #     # elif num_samples > fixed_length:
    #     #     data_signal = data_signal[:, :fixed_length]

    #     data_signal = torch.from_numpy(data_signal).float()
    #     mask = self.get_mask_from_rttm(utt_id, data_signal.shape[-1], idx_start, idx_end)
    #     mask = torch.from_numpy(mask)

    #     print(f"data_signal {index} shape: {data_signal.shape} | mask shape: {mask.shape}")
        
    #     # except Exception as e:
    #     #     print(f"Error processing file '{utt_id}' at index {index}: Channel {idx_channel}, Start {idx_start}, End {idx_end}")
    #     #     print(f"Error Details: {e}")
    #     return {
    #         "feat": data_signal,  # [num_channel, num_sample]
    #         "label": mask.long(),  # [num_frame]     time domain mask
    #     }

    def __getitem__(self, index):

        utt_id, idx_channel, idx_start, idx_end = self.chunk_indices[index]
        data_signal, sr = load_wav(self.kaldi_obj.wavs[utt_id], int(idx_start * self.rate), int(idx_end * self.rate))

        if len(data_signal.shape) == 2:
            data_signal = data_signal[:, idx_channel:idx_channel + 1].T  # [num_channel, num_samples]
        else:
            data_signal = data_signal[None, :]

        # Check if the length of the audio segment is equal to the desired chunk size
        fixed_length = self.chunk_size * self.rate  # Based on sample rate
        num_samples = data_signal.shape[-1]
        if num_samples != fixed_length:
            return None

        data_signal = torch.from_numpy(data_signal).float()
        mask = self.get_mask_from_rttm(utt_id, data_signal.shape[-1], idx_start, idx_end)
        mask = torch.from_numpy(mask)

        return {
            "feat": data_signal,  # [num_channel, num_sample]
            "label": mask.long(),  # [num_frame]     time domain mask
        }



def test_dataset():

    evalset = Dataset(
        data_path = '/home/users/ntu/kshitij0/scratch/datasets/third_dihard_challenge_eval/data/domains/meeting/flac',
        rttm_path = '/home/users/ntu/kshitij0/scratch/datasets/third_dihard_challenge_eval/data/domains/meeting/rttm'
    )

    for i in range(evalset.total_chunk):
        evalset.__getitem__(i)

if __name__ == "__main__":
    test_dataset()