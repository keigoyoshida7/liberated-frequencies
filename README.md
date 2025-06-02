<p align="center">
<img src="1.png"/>
</p>

# liberated-frequencies

<h4>
    <a href="https://keigoyoshida.jp/room20.html">Homepage</a> |
    <a href="https://www.instagram.com/p/DGqFPAOO3oo/">Archives</a> |
    <a href="https://flyingtokyo.com/open-call/">This project Support by METI and Rhizomatiks as "Flying Tokyo 2024"</a> |

</h4>

liberated frequencies, explores unprecedented soundscapes that defy our traditional auditory pleasures by "liberating" AI from the limitations of human-defined ‘pleasing'.<br>
The AI continuously learns in real-time from the highest-rated sounds. Utilizing this sound data, the AI predicts and generates the subsequent auditory experiences, creating an evolving and immersive soundscape.<br>
The subject in the soundscape wears EEG sensors that measure real-time theta waves (4-8 Hz) of her brain activity. <br>According to Sammler et al. (2007), increased activity in this frequency band is typically associated with intensified auditory pleasure.<br>
However, in response to this heightened brain-based pleasure, the AI—continuously learning from the real-time EEG data—intentionally disrupts the experience.<br>
It transforms the generated sounds, subtly altering pitches, waveforms, tempos and syncopations, gradually diverging from the original sound patterns the subject found pleasurable.<br>
This deliberate shift invites the viewer to explore the boundaries of discomfort, challenging the conventional auditory aesthetics inherently favored by human perception.<br>
Do these deliberately 'liberated' sounds merely traumatize the human senses, or do they open a gateway to new auditory expressions and possibilities?<br>

**Core features of liberated frequencies include:**

-   Deviation from Wavenet (https://arxiv.org/abs/1609.03499) based sound outouts.
-   Brainwave lawdata receive via OSC from [[Open BCI](https://github.com/OpenBCI/OpenBCI_GUI)]
-   Sound Input and Output with tarageting channels.

For more like concepts, please visit the [liberated frequencies documentation](https://keigoyoshida.jp/room20.html).

## Python quick start

Pre-built pip packages support Ubuntu 20.04+, macOS 10.15+ and Windows 10+
(64-bit) with Python 3.8-3.11.

```bash
# Install
pip install open3d       # or
pip install open3d-cpu   # Smaller CPU only wheel on x86_64 Linux (v0.17+)

# Verify installation
python -c "import open3d as o3d; print(o3d.__version__)"

# Python API
python -c "import open3d as o3d; \
           mesh = o3d.geometry.TriangleMesh.create_sphere(); \
           mesh.compute_vertex_normals(); \
           o3d.visualization.draw(mesh, raw_mode=True)"

# Open3D CLI
open3d example visualization/draw
```

## Communication channels

-   [Instagram DM](https://www.instagram.com/keigoyoshida_/): bug reports,
    feature requests,discussions etc.

## Citation

Please cite our work if you use liberated frequencies.

```bib
    author    = {Keigo Yoshida and Rinko Oka and Ryuji Kurokawa (Arsaffix)},
    title     = {liberated frequencies},
    journal   = {none},
    year      = {2024},
```
