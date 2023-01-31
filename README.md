# Metal Depth Stencil Tests
Various Metal drivers have issues with depth stencil textures, particularly when they have a non-zero number of mip levels and/or array layers.

Some results from local testing are summarized in this [spreadsheet](https://docs.google.com/spreadsheets/d/1G2LuDNlZU1cAuxafvAYHf_HSlQHBYWQXeeBMqfDkvm0/edit).

Notably, all the tests pass on Apple M1 Max, or when they are run using only 1 mip level and 1 array layer (args: `--levels 1 --layers 1`).

The tests write and read back from a depth stencil texture using various methods.

Write methods:
 - CopyFromBufferToStencil
 - StencilLoadOpStoreOp
 - StencilLoadOpStoreOpOffsetWithView
 - CopyFromBufferToStencilThenStencilLoadOpStoreOp
 - StencilLoadOpStoreOpThenCopyFromBufferToStencil
 - StencilOpStoreOp
 - StencilOpStoreOpOffsetWithView

Read methods:
 - CopyFromStencilToBuffer
 - StencilTest
 - StencilTestOffsetWithView
 - ShaderRead
 - ShaderReadOffsetWithView
 - CopyT2T-* (same as all the read methods, with an intermediate texture copy)

## How to run
Build with `./build.sh`. Then run `./bin/stencil_tests`. Pass `--help` to see options.

## Failures on 2017 15-Inch Macbook Pro, MacOS 13.1

### Reading a X32_Stencil8 non-zero subresource texture view of Depth32FloatStencil8 on AMD in the shader
 - Texture is created as TextureType2DArray with MTLPixelFormatDepth32FloatStencil8
 - It is viewed as Texture2D MTLPixelFormatX32_Stencil8 by selecting a single layer and mip level using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - The results are wrong. Visual inspection shows the read is only reading data from the 0th array layer.
 - Works for Stencil8
 - Works of the mip is selecting using the `lod` arg in the shader.
 - Broken both either if the view is at a non-zero mip offset or non-zero slice offset.

### Reading a Stencil8 texture view of Stencil8 on Intel at a subresource offset
 - Texture is created as TextureType2DArray with MTLPixelFormatStencil8
 - It is viewed as Texture2D Stencil8 by selecting a single mip and layer using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - Works if the layer and level are selected using the `array_index` and `lod`.
 - The results are wrong. Visual inspection shows the read is always reading back zeros.

### Reading a multi-subresource Depth32FloatStencil8/Stencil8 texture on Intel using the stencil test
 - Texture is created as TextureType2DArray with either MTLPixelFormatDepth32FloatStencil8 or MTLPixelFormatStencil8
 - It is attached as a stencil attachment
  - either using `.level` and `.slice` of [`MTLRenderPassAttachmentDescriptor`](https://developer.apple.com/documentation/metal/mtlrenderpassattachmentdescriptor?language=objc)
  - or creating a Texture2D view by selecting a single mip and layer using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - The results are wrong. Visual inspection shows the read is only reading data from the 0th layer and level.

### Reading a multi-mip Depth32FloatStencil8/Stencil8 texture view on AMD using the stencil test
 - Texture is created as TextureType2DArray with either MTLPixelFormatDepth32FloatStencil8 or MTLPixelFormatStencil8
 - It is attached as a stencil attachment as a Texture2D view by selecting a single mip using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - The results are wrong. Visual inspection shows the read is only reading data from the 0th mip level.

### Writing a multi-mip Depth32FloatStencil8/Stencil8 texture view attachment on AMD
 - Texture is created as TextureType2DArray with either MTLPixelFormatDepth32FloatStencil8 or MTLPixelFormatStencil8
 - It is attached as a stencil attachment as a Texture2D view by selecting a single mip using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - Write to the texture using the StoreOp.
 - The results are wrong. Visual inspection shows the data is a repeating pattern:
      ```
      00 00 00 00 00 00 00 00
      ff 3b ff 3b ff 3b ff 3b
      00 00 00 00 00 00 00 00
      ff 3b ff 3b ff 3b ff 3b
      00 00 00 00 00 00 00 00
      ff 3b ff 3b ff 3b ff 3b
      00 00 00 00 00 00 00 00
      ff 3b ff 3b ff 3b ff 3b
      ```

### Reading a X32_Stencil8 non-zero subresource of multi-mip Depth32FloatStencil8 on Intel in the shader
 - Texture is created as TextureType2DArray with MTLPixelFormatDepth32FloatStencil8
 - A single mip is selected:
  - either passing the `lod` argument explicily in the shader
  - or creating a Texture2D MTLPixelFormatX32_Stencil8 view and selecting a single mip using [newTextureViewWithPixelFormat:textureType:levels:slices:](https://developer.apple.com/documentation/metal/mtltexture/1515409-newtextureviewwithpixelformat)
 - The results are wrong. Sometimes it's wrong values, sometimes it's always 0.
 - Works for Stencil8.
 - With 1 array layer, works at mips 0, 1, 2. Fails at 3+.
 - With 3 array layers and 4 mips, it fails except for level=0, layer=0.
