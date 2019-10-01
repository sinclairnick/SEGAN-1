import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm

from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('--file_name', type=str, required=True, help='audio file name')
    parser.add_argument('--generator', type=str, required=True, help='generator to test file on')
    
    opt = parser.parse_args()
    FILE_NAME = opt.file_name
    GENERATOR = opt.generator

    generator = Generator()
    generator.load_state_dict(torch.load(GENERATOR, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        generator.cuda()
    generator.eval()
    
    noisy_slices = slice_signal(FILE_NAME, window_size, 1, sample_rate)
    enhanced_speech = []
    for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
        z = nn.init.normal(torch.Tensor(1, 1024, 8))
        noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        if torch.cuda.is_available():
            noisy_slice, z = noisy_slice.cuda(), z.cuda()
        noisy_slice, z = Variable(noisy_slice), Variable(z)
        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

    enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
    file_name = os.path.join(os.path.dirname(FILE_NAME),
                             'enhanced_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, enhanced_speech.T)
