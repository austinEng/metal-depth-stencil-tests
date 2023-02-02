# Metal Depth Stencil Tests
Various Metal drivers have issues with depth stencil textures, particularly when they have a non-zero number of mip levels and/or array layers.

Some results from local testing are summarized in this [spreadsheet](https://docs.google.com/spreadsheets/d/1G2LuDNlZU1cAuxafvAYHf_HSlQHBYWQXeeBMqfDkvm0/edit?usp=sharing&resourcekey=0-NNvOW6xRqGKayqroZynYdA).

The tests write and read back from a depth stencil texture using various methods. The tests also have a parameterization that performs an intermediate texture-to-texture copy before the readback.

## How to run
Build with `./build.sh`. Then run `./bin/depth_stencil_tests`. Pass `--help` to see options.

# Issues
Below is my attempt to parse out what some of the problems are. It is incomplete as there are a plethora of failures.

## Failures on 2021 15-Inch Macbook Pro, MacOS 13.1 (M1 Max)
Tests which read back data using the depth test for Depth16Unorm fail. Interestingly, they pass when the tested mip level is non-zero. They only fail when level=0.

## Validation Layers Inconsistency
MacOS 10.13 throws errors if you try to attach just a depth or stencil attachment when the format is a combined depth-stencil format:
```
[MTLRenderPipelineDescriptorInternal validateWithDevice:]:2408: failed `depthAttachmentPixelFormat (MTLPixelFormatInvalid) and (MTLPixelFormatDepth32Float_Stencil8) must match.'
```
Newer OSes don't require this, but additional tests will fail if you don't specify both attachments.
All the tests I ran set both attachments, for consistency.

## Reading / Writing Using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)

Fails on various platforms:
 - MacOS 13.1, 12.1 AMD, both Stencil8 and Depth32FloatStencil8:
   - write: StencilLoadOpStoreOpOffsetWithView
   - write: StencilOpStoreOpOffsetWithView
 - MacOS 13.1, 12.5, 12.1, 11.4, 10.13 Intel, all depth formats:
   - write: DepthLoadOpStoreOpOffsetWithView
 - MacOS 13.1, 12.5 with stencil8
   - read: ShaderReadStencilOffsetWithView
   - read: CopyT2T-ShaderReadStencilOffsetWithView

Visual inspection shows that oftene, the 0th subresource is always selected.

## All tests fail with Stencil8 on Mac AMD 12.1 (1002:6821)
This format seems to be entirely broken on this platform.

## Many operations only work with the 0th mip level and 0th array layer
These are indicated by yellow boxes in the [spreadsheet](https://docs.google.com/spreadsheets/d/1G2LuDNlZU1cAuxafvAYHf_HSlQHBYWQXeeBMqfDkvm0/edit?usp=sharing&resourcekey=0-NNvOW6xRqGKayqroZynYdA).
 - Reading data with the depth test on Mac AMD.
 - Sampling stencil from Depth32FloatStencil8 on Mac AMD (Stencil8 works).
 - Sampling stencil on Mac Intel
 - Nearly all opertions with stencil on Intel with MacOS 10.13.

## Texture-to-texture copies of depth sometimes work, sometimes do not
This occurs on Mac Intel.
 - In MacOS 13.1, writing depth with DepthLoadOpStoreOp and reading with the DepthTest works.
   Performing and intermediate texture-to-texture copy makes it work only with the 0th mip level and array layer.
 - Conversely in MacOS 12.1, writing depth with DepthLoadOpStoreOp and reading it in the shader does not work for any subresources.
   Performing an intermediate texture-to-texture copy makes it work for all subresources.
