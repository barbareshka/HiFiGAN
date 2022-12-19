# HiFiGAN
(Based on the article [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646))

Information from the artical:
> HiFi-GAN consists of one generator and two discriminators: multi-scale and multi-period discriminators. The generator and discriminators are trained adversarially, > along with two additional losses for
> improving training stability and model performance.

> The generator is a fully convolutional neural network. It uses a mel-spectrogram as input and
> upsamples it through transposed convolutions until the length of the output sequence matches the
> temporal resolution of raw waveforms. Every transposed convolution is followed by a multi-receptive
> field fusion (MRF) module, which we describe in the next paragraph. Figure 1 shows the architecture
> of the generator

> To this end, we propose the multi-period discriminator (MPD) consisting of several sub-discriminators
> each handling a portion of periodic signals of input audio. Additionally, to capture consecutive patterns
> and long-term dependencies, we use the multi-scale discriminator (MSD) proposed in MelGAN, which consecutively evaluates audio samples at different levels. 
