<p align="center">
<img src="https://raw.githubusercontent.com/isl-org/Open3D/main/docs/_static/open3d_logo_horizontal.png" width="320" />
</p>

# liberated-frequencies

<h4>
    <a href="https://www.open3d.org">Homepage</a> |
    <a href="https://www.open3d.org/docs">Docs</a> |
    <a href="https://www.open3d.org/docs/release/getting_started.html">Quick Start</a> |
    <a href="https://www.open3d.org/docs/release/compilation.html">Compile</a> |
    <a href="https://www.open3d.org/docs/release/index.html#python-api-index">Python</a> |
    <a href="https://www.open3d.org/docs/release/cpp_api.html">C++</a> |
    <a href="https://github.com/isl-org/Open3D-ML">Open3D-ML</a> |
    <a href="https://github.com/isl-org/Open3D/releases">Viewer</a> |
    <a href="https://www.open3d.org/docs/release/contribute/contribute.html">Contribute</a> |
    <a href="https://www.youtube.com/channel/UCRJBlASPfPBtPXJSPffJV-w">Demo</a> |
    <a href="https://github.com/isl-org/Open3D/discussions">Forum</a>
</h4>

Open3D is an open-source library that supports rapid development of software
that deals with 3D data. The Open3D frontend exposes a set of carefully selected
data structures and algorithms in both C++ and Python. The backend is highly
optimized and is set up for parallelization. We welcome contributions from
the open-source community.

[![Ubuntu CI](https://github.com/isl-org/Open3D/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Ubuntu+CI%22)
[![macOS CI](https://github.com/isl-org/Open3D/actions/workflows/macos.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22macOS+CI%22)
[![Windows CI](https://github.com/isl-org/Open3D/actions/workflows/windows.yml/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Windows+CI%22)

**Core features of Open3D include:**

-   Deviation from Wavenet (https://github.com/isl-org/Open3D/assets/41028320/e9b8645a-a823-4d78-8310-e85207bbc3e4) based sound outouts.
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
