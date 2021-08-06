import os

from numpy.core.fromnumeric import size
import scipy.io.wavfile as wav
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from glob import glob


class Audio_dataset:
    def __init__(self, audio_names, targets, max_audio_length = 300000 , max_target_length = 300, batch_size = 10) :
        self.audios = audio_names
        self.targets = targets
        self.max_audio_length = max_audio_length
        self.max_target_length =  max_target_length
        self.batch_size = batch_size
        self.batches = len(self.audio_names)//self.batch_size
    def __iter__(self):
        audios = np.empty((self.batch_size, self.max_audio_length), dtype=np.int16)
        lengths = np.empty((self.batch_size,), dtype=np.int32)
        targets = [None] * self.batch_size
        i,cur_size = 0,0 
        for filename in self.audios:
            sample_rate, audio = wav.read(filename)
            assert sample_rate == 16000
            assert audio.dtype == np.int16
            target_phrase = self.targets[i]
            i += 1
            if len(audio) > self.max_audio_length or len(target_phrase) > self.max_target_length:
                continue
            audios[i, :len(audio)] = audio
            lengths[i] = len(audio)
            targets[i] = target_phrase
            cur_size += 1
            if cur_size == self.batch_size:
                cur_size = 0
                yield audios, lengths, targets