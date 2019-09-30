import argparse
import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils import AudioDataset, emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')
    parser.add_argument('--loss_path', default='../loss.csv', help='save path for loss info file (csv)')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    current_epoch = 0
    discriminator = Discriminator()
    generator = Generator()

    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)  
    
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))

    # load existing models if they exist
    if os.path.exists('epochs'):
        files = os.listdir('epochs')
        if 'generator.tar' in files and 'discriminator.tar' in files:
            g_checkpoint = torch.load('generator.tar')
            generator.load_state_dict(g_checkpoint['model_state_dict'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
            g_loss = g_checkpoint['loss']
            
            d_checkpoint = torch.load('discriminator.tar')
            discriminator.load_state_dict(d_checkpoint['model_state_dict'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
            d_loss = d_checkpoint['loss']
            
            current_epoch = g_checkpoint['epoch']
    # initialize training mode in torch
    generator.train()
    discriminator.train()

    for epoch in range(current_epoch, NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            noisy_loss.backward()

            d_loss = clean_loss + noisy_loss
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()
            train_bar.set_description(
                'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.item(), noisy_loss.item(), g_loss.item(), g_cond_loss.item()))
        if not os.path.exists(opt.loss_path):
            with open(opt.loss_path, 'w') as f:
                f.write('epoch,g_loss_,g_cond_loss,total_g_loss,clean_loss,noisy_loss,total_d_loss\n')
        with open(opt.loss_path, 'a') as file:
            file.write('{},{},{},{},{},{},{}\n'.format(epoch+1, g_loss_, g_cond_loss, g_loss, clean_loss, noisy_loss, d_loss))

        # TEST model
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_noisy in test_bar:
            z = nn.init.normal(torch.Tensor(test_noisy.size(0), 1024, 8))
            if torch.cuda.is_available():
                test_noisy, z = test_noisy.cuda(), z.cuda()
            test_noisy, z = Variable(test_noisy), Variable(z)
            fake_speech = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
            fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

            for idx in range(fake_speech.shape[0]):
                generated_sample = fake_speech[idx]
                file_name = os.path.join('results',
                                         '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
                wavfile.write(file_name, sample_rate, generated_sample.T)

        # save the model parameters for each epoch
        g_path = os.path.join('epochs', 'generator.tar')
        d_path = os.path.join('epochs', 'discriminator.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            'loss': g_loss,
            }, g_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': d_optimizer.state_dict(),
            'loss': d_loss,
            }, d_path)
    # save model after epochs complete
    torch.save(generator.state_dict(), os.path.join('epochs', 'generator-final.pkl'))
    torch.save(discriminator.state_dict(), os.path.join('epochs', 'discriminator-final.pkl'))
