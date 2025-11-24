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

## Reported results

SI-SDR (dB) at 44.1 kHz, from Table 2 of the paper:

| Model         | Music   | Speech   | SFX     |
|---------------|---------|----------|---------|
| No processing | −6.8    | 1.0      | −5.0    |
| Conv-TasNet   | 0.3     | 8.5      | 2.0     |
| UMX (64 ms)   | 3.1     | 11.7     | 4.4     |
| XUMX (64 ms)  | 3.5     | 11.7     | 5.1     |
| **MRX**       | **4.2** | **12.3** | **5.7** |

## Custom test mixture

`audio/custom_mix/mixture.wav` is a ~56 s soundtrack (44.1 kHz, stereo) that we built in
Ableton Live following the DnR mixing rules:

- clean, isolated speech, music and SFX sources
- loudness-matched, with speech sitting highest in the mix
- staggered clips with partial overlap across classes and none within a class

## Reference

D. Petermann, G. Wichern, Z.-Q. Wang, and J. Le Roux, "The Cocktail Fork Problem: Three-Stem
Audio Separation for Real-World Soundtracks," *Proc. ICASSP*, 2022.
[doi:10.1109/ICASSP43922.2022.9746005](https://doi.org/10.1109/ICASSP43922.2022.9746005) ·
[project page](https://cocktail-fork.github.io)
