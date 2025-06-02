<p align="center">
<img src="https://raw.githubusercontent.com/isl-org/Open3D/main/docs/_static/open3d_logo_horizontal.png" width="320" />
</p>

# liberated-frequencies

<h4>
    <a href="https://www.open3d.org">Homepage</a> |
    <a href="https://www.open3d.org/docs">Archives</a> |
    <a href="https://www.open3d.org/docs/release/getting_started.html">This project Support by METI and Rhizomatiks as "Flying Tokyo 2024"</a> |

</h4>

Our Audio Visual Performance, liberated frequencies, explores unprecedented soundscapes that defy our traditional auditory pleasures
by "liberating" AI from the limitations of human-defined ‘pleasing'.
Before the production, our team gathered glitch, experimental, voice and noise sounds, which a subject later rated based on the pleasure they evoked.
During performance, the AI continuously learns in real-time from the highest-rated sounds. Utilizing this sound data, the AI predicts and generates the subsequent auditory experiences, creating an evolving and immersive soundscape.
The subject in the soundscape wears EEG sensors that measure real-time theta waves (4-8 Hz) of her brain activity. According to Sammler et al. (2007), increased activity in this frequency band is typically associated with intensified auditory pleasure.
However, in response to this heightened brain-based pleasure, the AI—continuously learning from the real-time EEG data—intentionally disrupts the experience.
It transforms the generated sounds, subtly altering pitches, waveforms, tempos and syncopations, gradually diverging from the original sound patterns the subject found pleasurable.
This deliberate shift invites the viewer to explore the boundaries of discomfort, challenging the conventional auditory aesthetics inherently favored by human perception.
Do these deliberately 'liberated' sounds merely traumatize the human senses,
or do they open a gateway to new auditory expressions and possibilities?

[![Ubuntu CI](https://github.com/isl-org/Open3D/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Ubuntu+CI%22)
[![macOS CI](https://github.com/isl-org/Open3D/actions/workflows/macos.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22macOS+CI%22)
[![Windows CI](https://github.com/isl-org/Open3D/actions/workflows/windows.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Windows+CI%22)

**Core features of Open3D include:**

-   Deviation from Wavenet (https://arxiv.org/abs/1609.03499) based sound outouts.
-   Brainwave lawdata receive via OSC.
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

To get the latest features in Open3D, install the
[development pip package](https://www.open3d.org/docs/latest/getting_started.html#development-version-pip).
To compile Open3D from source, refer to
[compiling from source](https://www.open3d.org/docs/release/compilation.html).


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
