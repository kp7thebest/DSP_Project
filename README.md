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
