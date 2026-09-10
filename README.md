# AWR-Net

**Decoupling Anatomy and Appearance for 3D Fetal Brain Ultrasound Synthesis**

AWR-Net synthesizes 3D fetal brain ultrasound (US) volumes from anatomical masks in two
stages, trained on different data and in different domains:

Stage 1 is frozen while Stage 2 trains, and Stage 2 may only apply a bounded residual, so
appearance adaptation can change local intensity and texture but cannot redraw the anatomy
the mask specifies. Inference needs only a mask, so masks derived from fetal MRI —
normal or abnormal — can condition US synthesis without paired MRI–US data.

![Example A](assets/ExampleA_axial_sweep.gif)
![Example B](assets/ExampleB_axial_sweep.gif)
![Example C](assets/ExampleC_axial_sweep.gif)

Axial sweeps through three cases, each conditioned only on an MRI-derived mask: the source
MR volume, the anatomical mask that drives synthesis, the Stage-1 coarse output, and the
Stage-2 refinement. Example A is a normal brain; B and C carry ventriculomegaly of
increasing degree. The ventricular anatomy in the mask is preserved through both stages,
while Stage 2 adds the speckle, boundary contrast and depth-dependent attenuation that
make the volume read as ultrasound. No real ultrasound of these subjects exists — the mask
is the only input.

> **Status.** Code will be released upon publication.

| Label | Structure | | Label | Structure |
|---:|---|---|---:|---|
| 1 / 2 | cerebral hemisphere (CER) | | 8 / 9 | choroid plexus (CP) |
| 3 / 4 | cerebellar hemisphere (CBH) | | 10 | midbrain (MB) |
| 5 | vermis (VER) | | 11 | cavum septi pellucidi (CSP) |
| 6 / 7 | lateral ventricle (LV) | | 12 / 13 | deep central region (DCR) |

`DCR` merges the basal ganglia, thalamic region and adjacent deep white matter, which are
hard to separate reliably in 3D US. Map MRI protocols into this space first: cortical and
white-matter tissues into left/right `CER`, deep gray nuclei and thalamic regions into
`DCR`, and refine the brainstem label down to `MB`.

## Usage

```bash
bash bash/run_awrnet.sh                               # the whole pipeline
EXP_NAME=my_run bash bash/run_awrnet.sh               # name the run
STAGES="coarse_test refine" bash bash/run_awrnet.sh   # inference only
```

## Acknowledgements

Builds on [guided-diffusion](https://github.com/openai/guided-diffusion) (MIT, Stage-1
U-Net and diffusion), [WaveCNet](https://github.com/LiQiufu/WaveCNet) (CC BY-NC-SA 4.0,
3D Haar DWT/IDWT) and [pix2pix](https://github.com/phillipi/pix2pix) (Stage-2 PatchGAN).
