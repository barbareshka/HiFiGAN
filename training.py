import argparse
import glob
import itertools
import json
import os
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from meldataset import MelDataset, MelTunedDataset, mel_spectrogram, get_dataset_filelist
from generator import Generator
from discriminator import MPD, MSD
from losses import f_loss, g_loss, d_loss

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def scan(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def train(a, h):
    # Initialize all structures/files/folders
    steps = 0
    t_files, val_files = get_dataset_filelist(a)
    device = torch.device('cuda:{:d}'.format(0))
    generator = Generator(h).to(device)
    mpd = MPD().to(device)
    msd = MSD().to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)
    if os.path.isdir(a.checkpoint_path):
        check1 = scan(a.checkpoint_path, 'g_')
        check2 = scan(a.checkpoint_path, 'do_')

    if check1 is not None and check2 is not None:
        state_dict_g = torch.load(check1, map_location=device)
        state_dict_do = torch.load(check2, map_location=device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    else:
        state_dict_do = None
        last_epoch = -1

    # Optimizers
    opt_c1 = generator.parameters()
    opt_c2 = itertools.chain(msd.parameters(), mpd.parameters())
    adam_opt_g = torch.optim.AdamW(opt_c1, h['learning_rate'], betas=[h['adam_b1'], h['adam_b2']])
    adam_opt_d = torch.optim.AdamW(opt_c2, h['learning_rate'], betas=[h['adam_b1'], h['adam_b2']])

    if state_dict_do is not None:
        adam_opt_d.load_state_dict(state_dict_do['optim_d'])
        adam_opt_g.load_state_dict(state_dict_do['optim_g'])

    # Datasets
    if a.fine_tuning:
        trainset = MelTunedDataset(t_files, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], n_cache_reuse=0,
                          shuffle=True, fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    else: 
        trainset = MelDataset(t_files, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], n_cache_reuse=0,
                          shuffle=True, fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)

    if a.fine_tuning:
        validset = MelTunedDataset(val_files, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], False, False, n_cache_reuse=0,
                          fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    else:
        validset = MelDataset(val_files, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], False, False, n_cache_reuse=0,
                          fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    # Loaders
    train_sampler = None

    train_loader = DataLoader(trainset, num_workers=h['num_workers'], shuffle=False,
                              sampler=train_sampler,
                              batch_size=h['batch_size'],
                              pin_memory=True,
                              drop_last=True)

    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=1,
                              pin_memory=True,
                              drop_last=True)

    # Schedulers
    gen_sched = torch.optim.lr_scheduler.ExponentialLR(adam_opt_g, gamma=h['lr_decay'], last_epoch=last_epoch)
    disc_sched = torch.optim.lr_scheduler.ExponentialLR(adam_opt_d, gamma=h['lr_decay'], last_epoch=last_epoch)

    # Training
    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            x, y, x_mel, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True)).unsqueeze(1)
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))

            y_gen = generator(x)
            mel_conf = {'n_fft': h['n_fft'], 'num_mels' : h['num_mels'], 'sampling_rate' : h['sampling_rate'], 
                        'hop_size': h['hop_size'], 'win_size' : h['win_size'], 'fmin' : h['fmin'],
                        'fmax' : h['fmax_for_loss'] }
            y_gen_mel = mel_spectrogram(y_gen.squeeze(1), mel_conf)

            adam_opt_d.zero_grad()

            # MPD & MSD
            y_disc_r, y_disc_g, _, _ = mpd(y, y_gen.detach())
            loss_disc_f, _, _ = d_loss(y_disc_r, y_disc_g)
            y_disc_r, y_disc_g, _, _ = msd(y, y_gen.detach())
            loss_disc_s, _, _ = d_loss(y_disc_r, y_disc_g)

            total_disc_loss = loss_disc_s + loss_disc_f

            total_disc_loss.backward()
            adam_opt_d.step()

            # Generator
            adam_opt_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_gen_mel) * 45

            y_disc_r, y_disc_g, vecs_r, vecs_g = mpd(y, y_gen)
            loss_gen_f, _ = g_loss(y_disc_g)
            loss_fm_f = f_loss(vecs_r, vecs_g)

            y_disc_r, y_disc_g, vecs_r, vecs_g = msd(y, y_gen)
            loss_fm_s = f_loss(vecs_r, vecs_g)
            loss_gen_s, _ = g_loss(y_disc_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            adam_opt_g.step()
            
            if steps % 5 == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_gen_mel).item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}'.
                      format(steps, loss_gen_all, mel_error))

            # checkpointing
            if steps % 3000 == 0 and steps != 0:
                path1 = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                path2 = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                torch.save({'generator': (generator.module if h['num_gpus'] > 1 else generator).state_dict()}, path1)
                torch.save({'mpd': (mpd.module if h['num_gpus'] > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h['num_gpus'] > 1
                                                         else msd).state_dict(),
                                     'optim_g': adam_opt_g.state_dict(), 'optim_d': adam_opt_d.state_dict(), 'steps': steps,
                                     'epoch': epoch}, path2)

            # Validation
            if steps % 3000 == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, x_mel, y_mel = batch
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                        y_gen = generator(x.to(device))
                        mel_conf = {'n_fft': h['n_fft'], 'num_mels' : h['num_mels'], 'sampling_rate' : h['sampling_rate'], 
                        'hop_size': h['hop_size'], 'win_size' : h['win_size'], 'fmin' : h['fmin'],
                        'fmax' : h['fmax_for_loss']}
                        y_gen_mel = mel_spectrogram(y_gen.squeeze(1), mel_conf)
                        val_err_tot += F.l1_loss(y_mel, y_gen_mel).item()
                    val_err = val_err_tot / (j + 1)
                generator.train()

            steps += 1
        gen_sched.step()
        disc_sched.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='wavs')
    parser.add_argument('--input_mels_dir', default='mels')
    parser.add_argument('--input_training_file', default='data/training.txt')
    parser.add_argument('--input_validation_file', default='data/validation.txt')
    parser.add_argument('--checkpoint_path', default='logits')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--training_epochs', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    params = json.loads(data)
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    train(a, params)


if __name__ == '__main__':
    main()
