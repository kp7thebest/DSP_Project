# Three-Stem Audio Separation: The Cocktail Fork Problem

Course project for Digital Signal Processing at BITS Pilani (Sep–Nov 2025). We studied
*The Cocktail Fork Problem* (Petermann et al., ICASSP 2022), which splits a soundtrack into
**speech**, **music**, and **sound effects**. We then ran the pretrained Multi-Resolution
CrossNet (MRX) model on a test mix we built ourselves using the Divide and Remaster (DnR) recipe.

**Authors:** Ishan Dhingra, Karthik Periasamy

_Work in progress._

## Background

### The problem

A soundtrack mixture is modelled as

```
y(t) = x_speech(t) + x_music(t) + x_sfx(t)
```

The goal is to recover each of the three stems from `y`. Mask-based models do this in the
STFT domain: they predict a real-valued mask `M_j` for each source and reconstruct
`x_j = iSTFT(M_j ⊙ Y)`.

### Divide and Remaster (DnR) dataset

- **Speech:** LibriSpeech, rebuilt at 44.1 kHz from the original LibriVox recordings
- **Music:** Free Music Archive (FMA)
- **SFX:** FSD50K, split into foreground and background sounds
- 60 s mixtures: no overlap within a class, natural overlap across classes
- Loudness set in LUFS following film/broadcast practice: speech loudest, then foreground SFX,
  music, and background SFX
- 3,406 train / 487 validation / 973 test mixtures

## MRX: Multi-Resolution CrossNet

MRX extends CrossNet-Open-Unmix (X-UMX) with several STFT resolutions (32, 64 and 256 ms
windows that share one hop size):

1. Compute the STFT of the mixture at each resolution and project each magnitude spectrogram to 512 dims.
2. Average across resolutions, run three BLSTM stacks, and average their outputs.
3. Decode one mask per source per resolution (3 sources × 3 resolutions = 9 masks).
4. Apply each mask to its own spectrogram, take the iSTFT, and sum the results across resolutions.

Short windows capture transients like consonants and impacts. Long windows resolve harmonics
in music and ambience.

## Running the pretrained model

We used MERL's official implementation,
[merlresearch/cocktail-fork-separation](https://github.com/merlresearch/cocktail-fork-separation),
which ships pretrained MRX checkpoints through git LFS:

```bash
git lfs install
git clone https://github.com/merlresearch/cocktail-fork-separation.git
cd cocktail-fork-separation
pip install -r requirements.txt
python separate.py --audio-path ../DSP_Project/audio/custom_mix/mixture.wav --out-dir out/custom_mix
```

This writes `speech.wav`, `music.wav` and `sfx.wav` to the output directory. The input
should be sampled at 44.1 kHz.
