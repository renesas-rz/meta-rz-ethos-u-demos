# Enable OpenCL (used by image-classification -g) when mali-library
# provides it, i.e. "opencl" is in both DISTRO and MACHINE_FEATURES.
RZ_OPENCL = "${@bb.utils.contains('COMBINED_FEATURES', 'opencl', '1', '', d)}"

PACKAGECONFIG:append = "${@' opencl' if d.getVar('RZ_OPENCL') else ''}"

# opencl-headers clashes with the headers installed by mali-library, so
# depend on mali-library's virtual/libopencl instead.
DEPENDS:remove = "${@'opencl-headers virtual/opencl-icd' if d.getVar('RZ_OPENCL') else ''}"
DEPENDS:append = "${@' virtual/libopencl' if d.getVar('RZ_OPENCL') else ''}"
