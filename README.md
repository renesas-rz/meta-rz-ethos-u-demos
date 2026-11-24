# meta-rz-ethos-u-demos

OpenEmbedded/Yocto layer providing Ethos-U demo applications for the
Renesas RZ/G3E SMARC Evaluation platform.

The layer is built on top of the `meta-rz-ethos-u` layer from the
[rz-ethos-u](https://github.com/renesas-rz/rz-ethos-u) repository, which
also provides the kas configuration used to build it.

## Included demos

| Recipe | Description |
| --- | --- |
| `image-classification` | MobileNet V2 image classification using TensorFlow Lite with the Ethos-U delegate. Single picture, slideshow and live USB camera modes, with the result shown on the console or on a Wayland display. |

## Dependencies
This layer depends on `meta-rz-ethos-u` from the
[rz-ethos-u](https://github.com/renesas-rz/rz-ethos-u) repository, and on
everything that layer requires (meta-renesas, poky, meta-arm and
meta-openembedded, including `meta-multimedia` for GStreamer/OpenCV). The
image must include the Renesas graphics (Wayland/Weston) support, e.g.
`core-image-weston`.

Note: `meta-rz-ethos-u-examples` is not a dependency for
`meta-rz-ethos-u-demos`.

## Build Instructions
The kas configuration `kas/features/ethos-u-demos.yml` lives in the
[rz-ethos-u](https://github.com/renesas-rz/rz-ethos-u) repository and fetches
this layer from GitHub:

```bash
git clone https://github.com/renesas-rz/rz-ethos-u
cd rz-ethos-u
./kas-container build kas/machines/smarc-rzg3e.yml:\
kas/images/core-image-weston.yml:\
kas/features/graphics.yml:\
kas/features/codecs.yml:\
kas/features/ethos-u-demos.yml
```

To build with a local copy of this layer instead (e.g. for development),
change the `meta-rz-ethos-u-demos` entry in the kas fragment to use a
local `path:` instead of `url:`/`branch:`.

The feature can also be enabled from `./kas-container menu` under
`Features`->`Demo Ethos-U Applications`.

To use the layer in an existing bitbake setup instead, add it and install
the application:

```bash
bitbake-layers add-layer meta-rz-ethos-u-demos
echo 'IMAGE_INSTALL:append = " image-classification"' >> conf/local.conf
```

## Running the demos
On the target run `image-classification`; `image-classification -h` lists
the available options. The full documentation (`README_USB_CAMERA.md` and the
`image-classification(1)` manual page) is in the `image-classification-doc`
package.

## License
See [COPYING.MIT](COPYING.MIT).
