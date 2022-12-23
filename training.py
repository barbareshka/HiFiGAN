import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from meldataset import MelDataset, MelTunedDataset, mel_spectrogram, get_dataset_filelist
from hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from tools import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, AttrDict

torch.backends.cudnn.benchmark = True


def train(a, h):
    if h['num_gpus'] > 1:
        init_process_group(backend=h['dist_config']['dist_backend'], init_method=h['dist_config']['dist_url'],
                           world_size=h['dist_config']['world_size'] * h['num_gpus'], rank=0)

    torch.cuda.manual_seed(h['seed'])
    device = torch.device('cuda:{:d}'.format(0))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    #print(generator)
    os.makedirs(a.checkpoint_path, exist_ok=True)
    #print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h['num_gpus'] > 1:
        generator = DistributedDataParallel(generator, device_ids=[0]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[0]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[0]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h['learning_rate'], betas=[h['adam_b1'], h['adam_b2']])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h['learning_rate'], betas=[h['adam_b1'], h['adam_b2']])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h['lr_decay'], last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h['lr_decay'], last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)
    if a.fine_tuning:
        trainset = MelTunedDataset(training_filelist, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], n_cache_reuse=0,
                          shuffle=True, fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    else: 
        trainset = MelDataset(training_filelist, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], n_cache_reuse=0,
                          shuffle=True, fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)

    train_sampler = None

    train_loader = DataLoader(trainset, num_workers=h['num_workers'], shuffle=False,
                              sampler=train_sampler,
                              batch_size=h['batch_size'],
                              pin_memory=True,
                              drop_last=True)

    if a.fine_tuning:
        validset = MelTunedDataset(validation_filelist, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], False, False, n_cache_reuse=0,
                          fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    else:
        validset = MelDataset(validation_filelist, h['segment_size'], h['n_fft'], h['num_mels'],
                          h['hop_size'], h['win_size'], h['sampling_rate'], h['fmin'], h['fmax'], False, False, n_cache_reuse=0,
                          fmax_loss=h['fmax_for_loss'], device=device, base_mels_path=a.input_mels_dir)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        print("Epoch: {}".format(epoch+1))

        if h['num_gpus'] > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h['n_fft'], h['num_mels'], h['sampling_rate'], h['hop_size'], h['win_size'],
                                          h['fmin'], h['fmax_for_loss'])

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()
            
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}'.
                      format(steps, loss_gen_all, mel_error))

                # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h['num_gpus'] > 1 else generator).state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h['num_gpus'] > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h['num_gpus'] > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})


                # Validation
            if steps % a.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h['n_fft'], h['num_mels'], h['sampling_rate'],
                                                      h['hop_size'], h['win_size'],
                                                      h['fmin'], h['fmax_for_loss'])
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                    val_err = val_err_tot / (j+1)

                generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='logits')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    params = json.loads(data)

    torch.manual_seed(params['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params['seed'])
        params['num_gpus'] = torch.cuda.device_count()
        params['batch_size'] = int(params['batch_size'] / params['num_gpus'])
        print('Batch size per GPU :', params['batch_size'])
    else:
        pass
    
    train(a, params)


if __name__ == '__main__':
    main()
