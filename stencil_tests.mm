#include <array>
#include <tuple>
#include <vector>
#include <cstdint>

#include <Metal/Metal.h>

constexpr unsigned int width = 16;
constexpr unsigned int height = 16;
unsigned int arrayLayers = 3;
unsigned int mipmapLevels = 4;
bool verbose = false;

using TextureData = std::vector<std::vector<std::vector<uint8_t>>>;

#define WRITE_METHODS(X, ...)                                     \
  X(CopyFromBufferToStencil, __VA_ARGS__)                         \
  X(StencilLoadOpStoreOp, __VA_ARGS__)                            \
  X(StencilLoadOpStoreOpTextureView, __VA_ARGS__)                 \
  X(CopyFromBufferToStencilThenStencilLoadOpStoreOp, __VA_ARGS__) \
  X(StencilLoadOpStoreOpThenCopyFromBufferToStencil, __VA_ARGS__) \
  X(StencilOpStoreOp, __VA_ARGS__)                                \
  X(StencilOpStoreOpTextureView, __VA_ARGS__)

#define READ_METHODS(X, ...)              \
  X(CopyFromStencilToBuffer, __VA_ARGS__) \
  X(StencilTest, __VA_ARGS__)             \
  X(StencilTestTextureView, __VA_ARGS__)  \
  X(ShaderRead, __VA_ARGS__)              \
  X(ShaderReadTextureView, __VA_ARGS__)

#define DECL_ENUM(X, ...) X,

enum class WriteMethod {
  WRITE_METHODS(DECL_ENUM)
};

enum class ReadMethod {
  READ_METHODS(DECL_ENUM)
};

#define DECL_ENUM_LIST(X, Cls) Cls::X,

WriteMethod writeMethods[] = { WRITE_METHODS(DECL_ENUM_LIST, WriteMethod) };
ReadMethod readMethods[] = { READ_METHODS(DECL_ENUM_LIST, ReadMethod) };

#define TO_STRING_CASE(X, Cls) case Cls::X: return #X;

const char* ToString(WriteMethod wm) {
  switch (wm) {
    WRITE_METHODS(TO_STRING_CASE, WriteMethod)
  }
}

const char* ToString(ReadMethod rm) {
  switch (rm) {
    READ_METHODS(TO_STRING_CASE, ReadMethod)
  }
}

const char* ToString(MTLPixelFormat pixelFormat) {
  switch (pixelFormat) {
    case MTLPixelFormatStencil8:
      return "Stencil8";
    case MTLPixelFormatDepth32Float_Stencil8:
      return "Depth32FloatStencil8";
    default:
      __builtin_unreachable();
  }
}

#undef TO_STRING_CASE
#undef DECL_ENUM_LIST
#undef DECL_ENUM
#undef READ_METHODS
#undef WRITE_METHODS

void WriteStencilWithCopy(
  id<MTLDevice> device,
  id<MTLCommandQueue> commandQueue,
  id<MTLTexture> dsTex,
  TextureData* data
) {
  uint8_t stencilValue = 0;

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      id<MTLBuffer> buffer = [device newBufferWithLength:(width >> level) * (height >> level)
                                                 options:MTLResourceStorageModeManaged];
      auto* contents = static_cast<uint8_t*>([buffer contents]);
      for (size_t i = 0; i < (*data)[level][layer].size(); ++i) {
        stencilValue += 1;

        (*data)[level][layer][i] = stencilValue;
        contents[i] = stencilValue;

        stencilValue = stencilValue % 255;
      }
      [buffer didModifyRange:NSMakeRange(0, [buffer length])];
      [blitEncoder
             copyFromBuffer:buffer
               sourceOffset:0
          sourceBytesPerRow:(width >> level)
        sourceBytesPerImage:(width >> level) * (height >> level)
                 sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                  toTexture:dsTex
           destinationSlice:layer
           destinationLevel:level
          destinationOrigin:MTLOriginMake(0, 0, 0)
                    options:MTLBlitOptionStencilFromDepthStencil];
    }
  }
  [blitEncoder endEncoding];
  [commandBuffer commit];
}

void WriteContentsWithLoadOp(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLTexture> dsTex,
    bool offsetWithTextureView,
    TextureData* data) {
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      MTLRenderPassDescriptor* rpDesc =  [MTLRenderPassDescriptor renderPassDescriptor];
      rpDesc.stencilAttachment.clearStencil = 1 + ((level * arrayLayers + layer) % 255);
      rpDesc.stencilAttachment.loadAction = MTLLoadActionClear;
      rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
      if (offsetWithTextureView) {
        rpDesc.stencilAttachment.texture =
          [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                   textureType:MTLTextureType2D
                                        levels:NSMakeRange(level, 1)
                                        slices:NSMakeRange(layer, 1)];
      } else {
        rpDesc.stencilAttachment.texture = dsTex;
        rpDesc.stencilAttachment.level = level;
        rpDesc.stencilAttachment.slice = layer;
      }

      id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
      [renderEncoder endEncoding];

      std::fill((*data)[level][layer].begin(),
                (*data)[level][layer].end(),
                rpDesc.stencilAttachment.clearStencil);
    }
  }
  [commandBuffer commit];
}

void WriteContentsWithStencilOp(
  id<MTLDevice> device,
  id<MTLCommandQueue> commandQueue,
  id<MTLTexture> dsTex,
  bool offsetWithTextureView,
  TextureData* data
) {
  auto* shader = @R"(
    struct RasterizerData {
      float4 position [[position]];
    };

    vertex RasterizerData vert_main(uint vertexID [[vertex_id]]) {
      float2 pos[6] = {
        float2(-1.0f, -1.0f),
        float2(0.0f, -1.0f),
        float2(-1.0f, 0.0f),
        float2(-1.0f, 0.0f),
        float2(0.0f, -1.0f),
        float2(0.0f, 0.0f)
      };
      RasterizerData out;
      out.position = float4(pos[vertexID], 0.0f, 1.0f);
      return out;
    }

    fragment float4 frag_main(RasterizerData in [[stage_in]]) {
      return float4();
    }
  )";

  NSError* error = nullptr;
  id<MTLLibrary> lib = [device newLibraryWithSource:shader
                                            options:nil
                                              error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }
  MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
  pipelineStateDesc.stencilAttachmentPixelFormat = [dsTex pixelFormat];
  pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
  pipelineStateDesc.fragmentFunction= [lib newFunctionWithName:@"frag_main"];
  id<MTLRenderPipelineState> pipelineState =
    [device newRenderPipelineStateWithDescriptor:pipelineStateDesc
                                           error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }

  MTLDepthStencilDescriptor* dsDesc = [MTLDepthStencilDescriptor new];
  dsDesc.frontFaceStencil = [MTLStencilDescriptor new];
  dsDesc.frontFaceStencil.stencilCompareFunction = MTLCompareFunctionAlways;
  dsDesc.frontFaceStencil.stencilFailureOperation = MTLStencilOperationKeep;
  dsDesc.frontFaceStencil.depthFailureOperation = MTLStencilOperationKeep;
  dsDesc.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationInvert;
  dsDesc.frontFaceStencil.readMask = 0xFFFFFFFF;
  dsDesc.frontFaceStencil.writeMask = 0xFFFFFFFF;

  id<MTLDepthStencilState> dsState = [device newDepthStencilStateWithDescriptor:dsDesc];

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      MTLRenderPassDescriptor* rpDesc =  [MTLRenderPassDescriptor renderPassDescriptor];
      rpDesc.stencilAttachment.clearStencil = 1 + ((level * arrayLayers + layer) % 255);
      rpDesc.stencilAttachment.loadAction = MTLLoadActionClear;
      rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
      if (offsetWithTextureView) {
        rpDesc.stencilAttachment.texture =
          [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                   textureType:MTLTextureType2D
                                        levels:NSMakeRange(level, 1)
                                        slices:NSMakeRange(layer, 1)];
      } else {
        rpDesc.stencilAttachment.texture = dsTex;
        rpDesc.stencilAttachment.level = level;
        rpDesc.stencilAttachment.slice = layer;
      }

      std::fill((*data)[level][layer].begin(),
                (*data)[level][layer].end(),
                rpDesc.stencilAttachment.clearStencil);

      id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
      [renderEncoder setRenderPipelineState:pipelineState];
      [renderEncoder setDepthStencilState:dsState];
      [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
      [renderEncoder setCullMode:MTLCullModeNone];
      [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6];
      [renderEncoder endEncoding];

      // Emulate the stencil invert operation to update the
      // expected texture data.
      for (int y = (height >> level) / 2; y < (height >> level); ++y) {
        for (unsigned x = 0; x < (width >> level) / 2; ++x) {
          unsigned i = y * (width >> level) + x;
          (*data)[level][layer][i] =
            ~(*data)[level][layer][i];
        }
      }
    }
  }
  [commandBuffer commit];
}

void EncodeCheckR8DataWithCopy(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer, id<MTLTexture> dsTex, const TextureData& data) {
  id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      id<MTLBuffer> buffer = [device newBufferWithLength:(width >> level) * (height >> level)
                                                 options:MTLResourceStorageModeShared];
      [blitEncoder
                  copyFromTexture:dsTex
                      sourceSlice:layer
                      sourceLevel:level
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                        toBuffer:buffer
                destinationOffset:0
          destinationBytesPerRow:(width >> level)
        destinationBytesPerImage:(width >> level) * (height >> level)
                          options:(
                            [dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                              ? MTLBlitOptionStencilFromDepthStencil
                              : MTLBlitOptionNone
                          )];

      [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        auto* contents = static_cast<uint8_t*>([buffer contents]);
        if (memcmp(contents, data[level][layer].data(), data[level][layer].size()) != 0) {
          NSLog(@"\tlevel %d, layer %d - FAILED!", level, layer);
          if (verbose) {
            printf("\n\tExpected:\n");
            for (unsigned y = 0; y < (height >> level); ++y) {
              printf("\t");
              for (unsigned x = 0; x < (width >> level); ++x) {
                unsigned i = y * (width >> level) + x;
                printf("%02x ", data[level][layer][i]);
              }
              printf("\n");
            }
            printf("\n\tGot:\n");
            for (unsigned y = 0; y < (height >> level); ++y) {
              printf("\t");
              for (unsigned x = 0; x < (width >> level); ++x) {
                unsigned i = y * (width >> level) + x;
                printf("%02x ", contents[i]);
              }
              printf("\n");
            }

            printf("\n");
          }
        } else if (verbose) {
          NSLog(@"\tlevel %d, layer %d - OK", level, layer);
        }
      }];
    }
  }
  [blitEncoder endEncoding];
}

void CheckStencilWithCopy(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLTexture> dsTex, const TextureData& data) {
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  EncodeCheckR8DataWithCopy(device, commandBuffer, dsTex, data);
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

void CheckStencilWithShader(
  id<MTLDevice> device,
  id<MTLCommandQueue> commandQueue,
  id<MTLTexture> dsTex,
  bool offsetWithTextureView,
  const TextureData& data) {

  id<MTLTexture> stencilView;
  MTLPixelFormat viewFormat;
  if ([dsTex pixelFormat] == MTLPixelFormatStencil8) {
    viewFormat = MTLPixelFormatStencil8;
    stencilView = dsTex;
  } else {
    viewFormat = MTLPixelFormatX32_Stencil8;
    stencilView = [dsTex newTextureViewWithPixelFormat:viewFormat];
  }

  auto* shader_with_texture_array = @R"(
    struct RasterizerData {
      float4 position [[position]];
    };

    // Draw a fullscreen quad
    vertex RasterizerData vert_main(uint vertexID [[vertex_id]]) {
      float2 pos[6] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(-1.0f, 1.0f),
        float2(-1.0f, 1.0f),
        float2(1.0f, -1.0f),
        float2(1.0f, 1.0f)
      };
      RasterizerData out;
      out.position = float4(pos[vertexID], 0.0f, 1.0f);
      return out;
    }

    // Sample stencil data and write it to the attachment.
    fragment uint frag_main(
        RasterizerData in [[stage_in]],
        constant uint& lod [[buffer(0)]],
        constant uint& array_index [[buffer(1)]],
        metal::texture2d_array<uint, metal::access::sample> stencil_texture [[texture(0)]]
    ) {
      return stencil_texture.read(uint2(in.position.xy), lod, array_index).x;
    }
  )";

  auto* shader_no_texture_array = @R"(
    struct RasterizerData {
      float4 position [[position]];
    };

    // Draw a fullscreen quad
    vertex RasterizerData vert_main(uint vertexID [[vertex_id]]) {
      float2 pos[6] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(-1.0f, 1.0f),
        float2(-1.0f, 1.0f),
        float2(1.0f, -1.0f),
        float2(1.0f, 1.0f)
      };
      RasterizerData out;
      out.position = float4(pos[vertexID], 0.0f, 1.0f);
      return out;
    }

    // Sample stencil data and write it to the attachment.
    fragment uint frag_main(
        RasterizerData in [[stage_in]],
        metal::texture2d<uint, metal::access::sample> stencil_texture [[texture(0)]]
    ) {
      return stencil_texture.read(uint2(in.position.xy)).x;
    }
  )";

  NSError* error = nullptr;
  id<MTLLibrary> lib = [
    device newLibraryWithSource:(offsetWithTextureView ? shader_no_texture_array : shader_with_texture_array)
                        options:nil
                          error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }
  MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
  pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
  pipelineStateDesc.fragmentFunction= [lib newFunctionWithName:@"frag_main"];
  pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
  id<MTLRenderPipelineState> pipelineState =
    [device newRenderPipelineStateWithDescriptor:pipelineStateDesc
                                           error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

  MTLTextureDescriptor* colorTargetDesc = [MTLTextureDescriptor new];
  colorTargetDesc.textureType = MTLTextureType2DArray;
  colorTargetDesc.pixelFormat = MTLPixelFormatR8Uint;
  colorTargetDesc.width = width;
  colorTargetDesc.height = height;
  colorTargetDesc.depth = 1;
  colorTargetDesc.arrayLength = arrayLayers;
  colorTargetDesc.mipmapLevelCount = mipmapLevels;
  colorTargetDesc.storageMode = MTLStorageModePrivate;
  colorTargetDesc.usage = MTLTextureUsageRenderTarget;

  id<MTLTexture> colorTarget = [device newTextureWithDescriptor:colorTargetDesc];

  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
      rpDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      rpDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpDesc.colorAttachments[0].texture = colorTarget;
      rpDesc.colorAttachments[0].level = level;
      rpDesc.colorAttachments[0].slice = layer;

      id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
      [renderEncoder setRenderPipelineState:pipelineState];
      [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
      [renderEncoder setCullMode:MTLCullModeNone];
      if (offsetWithTextureView) {
        stencilView = [dsTex newTextureViewWithPixelFormat:viewFormat
                                               textureType:MTLTextureType2D
                                                    levels:NSMakeRange(level, 1)
                                                    slices:NSMakeRange(layer, 1)];
        [renderEncoder setFragmentTexture:stencilView atIndex:0];
      } else {
        [renderEncoder setFragmentBytes:&level length:sizeof(level) atIndex:0];
        [renderEncoder setFragmentBytes:&layer length:sizeof(layer) atIndex:1];
        [renderEncoder setFragmentTexture:stencilView atIndex:0];
      }
      [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6];
      [renderEncoder endEncoding];
    }
  }
  EncodeCheckR8DataWithCopy(device, commandBuffer, colorTarget, data);
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

// Read stencil via the stencil test bit by bit into 8 color textures (one for each bit).
// Sample the 8 textures and render to one texture, effectively recomposing the stencil data.
// Check that the recomposed data is correct.
void CheckStencilWithStencilTest(
  id<MTLDevice> device,
  id<MTLCommandQueue> commandQueue,
  id<MTLTexture> dsTex,
  bool offsetWithTextureView,
  const TextureData& data
) {
  auto* read_stencil_bit_shader = @R"(
    struct RasterizerData {
      float4 position [[position]];
    };

    // Draw a fullscreen quad
    vertex RasterizerData vert_main(uint vertexID [[vertex_id]]) {
      float2 pos[6] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(-1.0f, 1.0f),
        float2(-1.0f, 1.0f),
        float2(1.0f, -1.0f),
        float2(1.0f, 1.0f)
      };
      RasterizerData out;
      out.position = float4(pos[vertexID], 0.0f, 1.0f);
      return out;
    }

    fragment uint frag_main(RasterizerData in [[stage_in]]) {
      return 1;
    }
  )";

  NSError* error = nullptr;
  id<MTLLibrary> lib = [
    device newLibraryWithSource:read_stencil_bit_shader
                        options:nil
                          error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }
  MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
  pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
  pipelineStateDesc.fragmentFunction= [lib newFunctionWithName:@"frag_main"];
  pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
  pipelineStateDesc.stencilAttachmentPixelFormat = [dsTex pixelFormat];
  id<MTLRenderPipelineState> readbackPipelineState =
    [device newRenderPipelineStateWithDescriptor:pipelineStateDesc
                                           error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }

  auto* recompose_shader = @R"(
    struct RasterizerData {
      float4 position [[position]];
    };

    // Draw a fullscreen quad
    vertex RasterizerData vert_main(uint vertexID [[vertex_id]]) {
      float2 pos[6] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(-1.0f, 1.0f),
        float2(-1.0f, 1.0f),
        float2(1.0f, -1.0f),
        float2(1.0f, 1.0f)
      };
      RasterizerData out;
      out.position = float4(pos[vertexID], 0.0f, 1.0f);
      return out;
    }

    fragment uint frag_main(
      RasterizerData in [[stage_in]],
      metal::texture2d<uint, metal::access::sample> b0 [[texture(0)]],
      metal::texture2d<uint, metal::access::sample> b1 [[texture(1)]],
      metal::texture2d<uint, metal::access::sample> b2 [[texture(2)]],
      metal::texture2d<uint, metal::access::sample> b3 [[texture(3)]],
      metal::texture2d<uint, metal::access::sample> b4 [[texture(4)]],
      metal::texture2d<uint, metal::access::sample> b5 [[texture(5)]],
      metal::texture2d<uint, metal::access::sample> b6 [[texture(6)]],
      metal::texture2d<uint, metal::access::sample> b7 [[texture(7)]]) {

      return (
        (b0.read(uint2(in.position.xy)).x << 0) +
        (b1.read(uint2(in.position.xy)).x << 1) +
        (b2.read(uint2(in.position.xy)).x << 2) +
        (b3.read(uint2(in.position.xy)).x << 3) +
        (b4.read(uint2(in.position.xy)).x << 4) +
        (b5.read(uint2(in.position.xy)).x << 5) +
        (b6.read(uint2(in.position.xy)).x << 6) +
        (b7.read(uint2(in.position.xy)).x << 7)
      );
    }
  )";
  lib = [
    device newLibraryWithSource:recompose_shader
                        options:nil
                          error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }
  pipelineStateDesc = [MTLRenderPipelineDescriptor new];
  pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
  pipelineStateDesc.fragmentFunction= [lib newFunctionWithName:@"frag_main"];
  pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
  id<MTLRenderPipelineState> recomposePipelineState =
    [device newRenderPipelineStateWithDescriptor:pipelineStateDesc
                                           error:&error];
  if (error != nullptr) {
    NSLog(@"%@", error);
    return;
  }

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

  MTLTextureDescriptor* readbackTexDesc = [MTLTextureDescriptor new];
  readbackTexDesc.textureType = MTLTextureType2DArray;
  readbackTexDesc.pixelFormat = MTLPixelFormatR8Uint;
  readbackTexDesc.width = width;
  readbackTexDesc.height = height;
  readbackTexDesc.depth = 1;
  readbackTexDesc.arrayLength = arrayLayers;
  readbackTexDesc.mipmapLevelCount = mipmapLevels;
  readbackTexDesc.storageMode = MTLStorageModePrivate;
  readbackTexDesc.usage = MTLTextureUsageRenderTarget;

  id<MTLTexture> readbackTex = [device newTextureWithDescriptor:readbackTexDesc];

  for (uint32_t level = 0; level < mipmapLevels; ++level) {
    MTLTextureDescriptor* colorTargetDesc = [MTLTextureDescriptor new];
    colorTargetDesc.textureType = MTLTextureType2D;
    colorTargetDesc.pixelFormat = MTLPixelFormatR8Uint;
    colorTargetDesc.width = width >> level;
    colorTargetDesc.height = height >> level;
    colorTargetDesc.depth = 1;
    colorTargetDesc.arrayLength = 1;
    colorTargetDesc.mipmapLevelCount = 1;
    colorTargetDesc.storageMode = MTLStorageModePrivate;
    colorTargetDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
      std::array<id<MTLTexture>, 8> colorTargets;
      for (uint32_t i = 0; i < colorTargets.size(); ++i) {
        colorTargets[i] = [device newTextureWithDescriptor:colorTargetDesc];
      }
      MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
      rpDesc.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0);
      rpDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

      rpDesc.stencilAttachment.loadAction = MTLLoadActionLoad;
      rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
      if (offsetWithTextureView) {
        rpDesc.stencilAttachment.texture =
          [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                  textureType:MTLTextureType2D
                                        levels:NSMakeRange(level, 1)
                                        slices:NSMakeRange(layer, 1)];
      } else {
        rpDesc.stencilAttachment.texture = dsTex;
        rpDesc.stencilAttachment.level = level;
        rpDesc.stencilAttachment.slice = layer;
      }

      MTLDepthStencilDescriptor* dsDesc = [MTLDepthStencilDescriptor new];
      dsDesc.frontFaceStencil = [MTLStencilDescriptor new];

      // Read back the stencil bit-by-bit, writing the value to a color attachment.
      for (uint32_t b = 0; b < 8; ++b) {
        rpDesc.colorAttachments[0].texture = colorTargets[b];

        id<MTLRenderCommandEncoder> renderEncoder =
          [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];

        dsDesc.frontFaceStencil.stencilCompareFunction = MTLCompareFunctionEqual;
        dsDesc.frontFaceStencil.stencilFailureOperation = MTLStencilOperationKeep;
        dsDesc.frontFaceStencil.depthFailureOperation = MTLStencilOperationKeep;
        dsDesc.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationKeep;
        dsDesc.frontFaceStencil.readMask = 1 << b;
        dsDesc.frontFaceStencil.writeMask = 0;

        id<MTLDepthStencilState> dsState =
          [device newDepthStencilStateWithDescriptor:dsDesc];
        [renderEncoder setDepthStencilState:dsState];

        [renderEncoder setRenderPipelineState:readbackPipelineState];
        [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [renderEncoder setCullMode:MTLCullModeNone];
        [renderEncoder setStencilReferenceValue:(1 << b)];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                          vertexStart:0
                          vertexCount:6];
        [renderEncoder endEncoding];
      }

      // Now, merge the 8 textures into readbackTex.
      rpDesc.colorAttachments[0].texture = readbackTex;
      rpDesc.colorAttachments[0].level = level;
      rpDesc.colorAttachments[0].slice = layer;
      rpDesc.stencilAttachment = nil;

      id<MTLRenderCommandEncoder> renderEncoder =
          [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
      [renderEncoder setRenderPipelineState:recomposePipelineState];
      for (uint32_t b = 0; b < 8; ++b) {
        [renderEncoder setFragmentTexture:colorTargets[b] atIndex:b];
      }
      [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
      [renderEncoder setCullMode:MTLCullModeNone];
      [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                       vertexStart:0
                       vertexCount:6];
      [renderEncoder endEncoding];
    }
  }
  EncodeCheckR8DataWithCopy(device, commandBuffer, readbackTex, data);
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

MTLTextureUsage GetRequiredUsage(WriteMethod wm, ReadMethod rm) {
  MTLTextureUsage usage = 0;
  switch (wm) {
    case WriteMethod::CopyFromBufferToStencil:
      break;
    case WriteMethod::StencilLoadOpStoreOp:
    case WriteMethod::StencilLoadOpStoreOpTextureView:
    case WriteMethod::CopyFromBufferToStencilThenStencilLoadOpStoreOp:
    case WriteMethod::StencilLoadOpStoreOpThenCopyFromBufferToStencil:
    case WriteMethod::StencilOpStoreOp:
    case WriteMethod::StencilOpStoreOpTextureView:
      usage |= MTLTextureUsageRenderTarget;
      break;
  }

  switch (rm) {
    case ReadMethod::CopyFromStencilToBuffer:
      break;
    case ReadMethod::ShaderRead:
    case ReadMethod::ShaderReadTextureView:
      usage |= MTLTextureUsagePixelFormatView | MTLTextureUsageShaderRead;
      break;
    case ReadMethod::StencilTestTextureView:
    case ReadMethod::StencilTest:
      usage |= MTLTextureUsageRenderTarget;
      break;
  }
  return usage;
}

int main(int argc, char* argv[]) {
#define HAS_SWITCH(flag) strncmp(argv[i], #flag, sizeof(#flag) - 1) == 0
  const char* writeMethod = nullptr;
  const char* readMethod = nullptr;
  for (int i = 0; i < argc; ++i) {
    if (HAS_SWITCH(--verbose)) {
      verbose = true;
    }
    if (HAS_SWITCH(--levels)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "`levels` switch should be followed by number. Example: --levels 4\n");
        return 1;
      }
      mipmapLevels = atoi(argv[i + 1]);
    }
    if (HAS_SWITCH(--layers)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "`layers` switch should be followed by number. Example: --layers 3\n");
        return 1;
      }
      arrayLayers = atoi(argv[i + 1]);
    }
    if (HAS_SWITCH(--write)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "`write` switch should be followed by write method. Example: --write CopyFromBufferToStencil\n");
        return 1;
      }
      writeMethod = argv[i + 1];
    }
    if (HAS_SWITCH(--read)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "`read` switch should be followed by read method. Example: --read CheckStencilWithStencilTest\n");
        return 1;
      }
      readMethod = argv[i + 1];
    }
  }
#undef HAS_SWITCH

  for (id<MTLDevice> device : MTLCopyAllDevices()) {
    if (![[device name] containsString:@"AMD"]) {
      continue;
    }
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSLog(@"==================================================================\n\n");
    NSLog(@"Testing with %@\n\n", [device name]);

    for (WriteMethod wm : writeMethods) {
      if (writeMethod != nullptr && strcmp(writeMethod, ToString(wm)) != 0) {
        continue;
      }
      for (ReadMethod rm : readMethods) {
        if (readMethod != nullptr && strcmp(readMethod, ToString(rm)) != 0) {
          continue;
        }
        for (bool copyT2T : {false, true}) {
          for (MTLPixelFormat pixelFormat : {MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatStencil8}) {
            MTLTextureDescriptor* dsTexDesc = [MTLTextureDescriptor new];
            dsTexDesc.textureType = MTLTextureType2DArray;
            dsTexDesc.pixelFormat = pixelFormat;
            dsTexDesc.width = width;
            dsTexDesc.height = height;
            dsTexDesc.depth = 1;
            dsTexDesc.arrayLength = arrayLayers;
            dsTexDesc.mipmapLevelCount = mipmapLevels;
            dsTexDesc.storageMode = MTLStorageModePrivate;
            dsTexDesc.usage = GetRequiredUsage(wm, rm);

            id<MTLTexture> dsTex = [device newTextureWithDescriptor:dsTexDesc];

            if (copyT2T) {
              NSLog(@"Testing %s %s then CopyT2T then %s on %@", ToString(pixelFormat), ToString(wm), ToString(rm), [device name]);
            } else {
              NSLog(@"Testing %s %s then %s on %@", ToString(pixelFormat), ToString(wm), ToString(rm), [device name]);
            }
            NSLog(@"-----------------------------------------------------------------");

            TextureData data;
            data.resize(mipmapLevels);
            for (uint32_t level = 0; level < mipmapLevels; ++level) {
              data[level].resize(arrayLayers);
              for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
                data[level][layer].resize((width >> level) * (height >> level));
              }
            }
            switch (wm) {
              case WriteMethod::CopyFromBufferToStencil:
                WriteStencilWithCopy(device, commandQueue, dsTex, &data);
                break;
              case WriteMethod::StencilLoadOpStoreOp:
                WriteContentsWithLoadOp(device, commandQueue, dsTex, /* offsetWithTextureView */ false, &data);
                break;
              case WriteMethod::StencilLoadOpStoreOpTextureView:
                WriteContentsWithLoadOp(device, commandQueue, dsTex, /* offsetWithTextureView */ true, &data);
                break;
              case WriteMethod::CopyFromBufferToStencilThenStencilLoadOpStoreOp:
                WriteStencilWithCopy(device, commandQueue, dsTex, &data);
                WriteContentsWithLoadOp(device, commandQueue, dsTex, /* offsetWithTextureView */ false, &data);
                break;
              case WriteMethod::StencilLoadOpStoreOpThenCopyFromBufferToStencil:
                WriteContentsWithLoadOp(device, commandQueue, dsTex, /* offsetWithTextureView */ false, &data);
                WriteStencilWithCopy(device, commandQueue, dsTex, &data);
                break;
              case WriteMethod::StencilOpStoreOp:
                WriteContentsWithStencilOp(device, commandQueue, dsTex, /* offsetWithTextureView */ false, &data);
                break;
              case WriteMethod::StencilOpStoreOpTextureView:
                WriteContentsWithStencilOp(device, commandQueue, dsTex, /* offsetWithTextureView */ true, &data);
                break;
            }

            if (copyT2T) {
              id<MTLTexture> intermediateTex = [device newTextureWithDescriptor:dsTexDesc];

              id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
              id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
              for (uint32_t level = 0; level < mipmapLevels; ++level) {
                for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
                  [blitEncoder
                    copyFromTexture:dsTex
                        sourceSlice:layer
                        sourceLevel:level
                      sourceOrigin:MTLOriginMake(0,0,0)
                        sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                          toTexture:intermediateTex
                  destinationSlice:layer
                    destinationLevel:level
                  destinationOrigin:MTLOriginMake(0,0,0)];
                }
              }
              [blitEncoder endEncoding];
              [commandBuffer commit];

              dsTex = intermediateTex;
            }

            switch (rm) {
              case ReadMethod::CopyFromStencilToBuffer:
                CheckStencilWithCopy(device, commandQueue, dsTex, data);
                break;
              case ReadMethod::ShaderRead:
                CheckStencilWithShader(device, commandQueue, dsTex, /* offsetWithTextureView */ false, data);
                break;
              case ReadMethod::ShaderReadTextureView:
                CheckStencilWithShader(device, commandQueue, dsTex, /* offsetWithTextureView */ true, data);
                break;
              case ReadMethod::StencilTest:
                CheckStencilWithStencilTest(device, commandQueue, dsTex, /* offsetWithTextureView */ false, data);
                break;
              case ReadMethod::StencilTestTextureView:
                CheckStencilWithStencilTest(device, commandQueue, dsTex, /* offsetWithTextureView */ true, data);
                break;
            }
          }
        }
      }
    }
  }

  return 0;
}