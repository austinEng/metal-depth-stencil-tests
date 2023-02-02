#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <Metal/Metal.h>

#define LOG(FORMAT, ...) \
    printf("%s\n", [[NSString stringWithFormat:FORMAT, ##__VA_ARGS__] UTF8String]);

constexpr unsigned int width = 16;
constexpr unsigned int height = 16;
unsigned int arrayLayers = 3;
unsigned int mipmapLevels = 4;
bool verbose = false;
constexpr float depthTolerance = 0.005;
bool deterministic = false;

int getInitialDataValue() {
    if (deterministic) {
        return 0;
    }
    return rand();
}

// Workaround required on MacOS 10.13.
bool UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats = true;

// reset between iterations.
bool failed = false;
bool failedFor0thSubresource = false;

enum DataType {
    F32 = 0,
    U16 = 1,
    U8 = 2,
};

struct SubresourceData {
    std::tuple<std::vector<float>, std::vector<uint16_t>, std::vector<uint8_t>> data;

    void resize(size_t size) {
        std::get<F32>(data).resize(size);
        std::get<U16>(data).resize(size);
        std::get<U8>(data).resize(size);
    }

    template <DataType D>
    const auto& Get() const {
        return std::get<D>(data);
    }

    template <DataType D>
    auto& Get() {
        return std::get<D>(data);
    }
};

// Indexed by level, then layer, then (y * width + x).
using TextureData = std::vector<std::vector<SubresourceData>>;

#define WRITE_METHODS(X, ...)                                 \
    X(CopyFromBufferToDepth, __VA_ARGS__)                     \
    X(CopyFromSingleSubresourceTextureToDepth, __VA_ARGS__)   \
    X(DepthLoadOpStoreOp, __VA_ARGS__)                        \
    X(DepthLoadOpStoreOpOffsetWithView, __VA_ARGS__)          \
    X(CopyFromBufferToStencil, __VA_ARGS__)                   \
    X(CopyFromSingleSubresourceTextureToStencil, __VA_ARGS__) \
    X(StencilLoadOpStoreOp, __VA_ARGS__)                      \
    X(StencilLoadOpStoreOpOffsetWithView, __VA_ARGS__)        \
    X(StencilOpStoreOp, __VA_ARGS__)                          \
    X(StencilOpStoreOpOffsetWithView, __VA_ARGS__)

#define READ_METHODS(X, ...)                      \
    X(CopyFromDepthToBuffer, __VA_ARGS__)         \
    X(CopyFromStencilToBuffer, __VA_ARGS__)       \
    X(DepthTest, __VA_ARGS__)                     \
    X(DepthTestOffsetWithView, __VA_ARGS__)       \
    X(ShaderReadDepth, __VA_ARGS__)               \
    X(ShaderReadDepthOffsetWithView, __VA_ARGS__) \
    X(StencilTest, __VA_ARGS__)                   \
    X(StencilTestOffsetWithView, __VA_ARGS__)     \
    X(ShaderReadStencil, __VA_ARGS__)             \
    X(ShaderReadStencilOffsetWithView, __VA_ARGS__)

#define DECL_ENUM(X, ...) X,

enum class WriteMethod { WRITE_METHODS(DECL_ENUM) };

enum class ReadMethod { READ_METHODS(DECL_ENUM) };

#define DECL_ENUM_LIST(X, Cls) Cls::X,

constexpr WriteMethod writeMethods[] = {WRITE_METHODS(DECL_ENUM_LIST, WriteMethod)};
constexpr ReadMethod readMethods[] = {READ_METHODS(DECL_ENUM_LIST, ReadMethod)};

#define TO_STRING_CASE(X, Cls) \
    case Cls::X:               \
        return #X;

const char* ToString(WriteMethod wm) {
    switch (wm) { WRITE_METHODS(TO_STRING_CASE, WriteMethod) }
}

const char* ToString(ReadMethod rm) {
    switch (rm) { READ_METHODS(TO_STRING_CASE, ReadMethod) }
}

const char* ToString(MTLPixelFormat pixelFormat) {
    switch (pixelFormat) {
        case MTLPixelFormatDepth16Unorm:
            return "Depth16Unorm";
        case MTLPixelFormatDepth32Float:
            return "Depth32Float";
        case MTLPixelFormatDepth32Float_Stencil8:
            return "Depth32FloatStencil8";
        case MTLPixelFormatStencil8:
            return "Stencil8";
        default:
            __builtin_unreachable();
    }
}

uint32_t DepthByteSize(MTLPixelFormat pixelFormat) {
    switch (pixelFormat) {
        case MTLPixelFormatDepth32Float_Stencil8:
        case MTLPixelFormatDepth32Float:
            return 4;
        case MTLPixelFormatDepth16Unorm:
            return 2;
        default:
            __builtin_unreachable();
    }
}

#undef TO_STRING_CASE
#undef DECL_ENUM_LIST
#undef DECL_ENUM
#undef READ_METHODS
#undef WRITE_METHODS

void WriteDepthWithCopy(id<MTLDevice> device,
                        id<MTLCommandBuffer> commandBuffer,
                        id<MTLTexture> dsTex,
                        TextureData* data) {
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            uint32_t bytesPerRow = DepthByteSize([dsTex pixelFormat]) * (width >> level);
            id<MTLBuffer> buffer = [device newBufferWithLength:bytesPerRow * (height >> level)
                                                       options:MTLResourceStorageModeShared];
            if ([dsTex pixelFormat] == MTLPixelFormatDepth16Unorm) {
                auto* contents = static_cast<uint16_t*>([buffer contents]);
                uint16_t depthValue = getInitialDataValue() % 0xFFFF;
                for (size_t i = 0; i < (*data)[level][layer].Get<U16>().size(); ++i) {
                    depthValue += 1;
                    (*data)[level][layer].Get<U16>()[i] = depthValue;
                    (*data)[level][layer].Get<F32>()[i] = float(depthValue) / float(0xFFFF);
                    contents[i] = depthValue;
                    depthValue = depthValue % 0xFFFF;
                }
            } else {
                auto* contents = static_cast<float*>([buffer contents]);
                uint16_t depthValue = getInitialDataValue() % 0xFFFF;
                for (size_t i = 0; i < (*data)[level][layer].Get<F32>().size(); ++i) {
                    depthValue += 1;
                    (*data)[level][layer].Get<U16>()[i] = depthValue;
                    (*data)[level][layer].Get<F32>()[i] = float(depthValue) / float(0xFFFF);
                    contents[i] = float(depthValue) / float(0xFFFF);
                    depthValue = depthValue % 0xFFFF;
                }
            }
            [blitEncoder copyFromBuffer:buffer
                           sourceOffset:0
                      sourceBytesPerRow:bytesPerRow
                    sourceBytesPerImage:bytesPerRow * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:dsTex
                       destinationSlice:layer
                       destinationLevel:level
                      destinationOrigin:MTLOriginMake(0, 0, 0)
                                options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                             ? MTLBlitOptionDepthFromDepthStencil
                                             : MTLBlitOptionNone)];
        }
    }
    [blitEncoder endEncoding];
}

void WriteDepthWithTextureCopy(id<MTLDevice> device,
                               id<MTLCommandBuffer> commandBuffer,
                               id<MTLTexture> dsTex,
                               TextureData* data) {
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            uint32_t bytesPerRow = DepthByteSize([dsTex pixelFormat]) * (width >> level);
            id<MTLBuffer> buffer = [device newBufferWithLength:bytesPerRow * (height >> level)
                                                       options:MTLResourceStorageModeManaged];
            if ([dsTex pixelFormat] == MTLPixelFormatDepth16Unorm) {
                auto* contents = static_cast<uint16_t*>([buffer contents]);
                uint16_t depthValue = getInitialDataValue() % 0xFFFF;
                for (size_t i = 0; i < (*data)[level][layer].Get<U16>().size(); ++i) {
                    depthValue += 1;
                    (*data)[level][layer].Get<U16>()[i] = depthValue;
                    (*data)[level][layer].Get<F32>()[i] = float(depthValue) / float(0xFFFF);
                    contents[i] = depthValue;
                    depthValue = depthValue % 0xFFFF;
                }
            } else {
                auto* contents = static_cast<float*>([buffer contents]);
                uint16_t depthValue = getInitialDataValue() % 0xFFFF;
                for (size_t i = 0; i < (*data)[level][layer].Get<F32>().size(); ++i) {
                    depthValue += 1;
                    (*data)[level][layer].Get<U16>()[i] = depthValue;
                    (*data)[level][layer].Get<F32>()[i] = float(depthValue) / float(0xFFFF);
                    contents[i] = float(depthValue) / float(0xFFFF);
                    depthValue = depthValue % 0xFFFF;
                }
            }

            [buffer didModifyRange:NSMakeRange(0, [buffer length])];

            MTLTextureDescriptor* intermediateTexDesc = [MTLTextureDescriptor new];
            intermediateTexDesc.textureType = MTLTextureType2D;
            intermediateTexDesc.pixelFormat = [dsTex pixelFormat];
            intermediateTexDesc.width = (width >> level);
            intermediateTexDesc.height = (height >> level);
            intermediateTexDesc.depth = 1;
            intermediateTexDesc.arrayLength = 1;
            intermediateTexDesc.mipmapLevelCount = 1;
            intermediateTexDesc.storageMode = MTLStorageModePrivate;
            id<MTLTexture> intermediateTex = [device newTextureWithDescriptor:intermediateTexDesc];

            [blitEncoder copyFromBuffer:buffer
                           sourceOffset:0
                      sourceBytesPerRow:bytesPerRow
                    sourceBytesPerImage:bytesPerRow * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:intermediateTex
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)
                                options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                             ? MTLBlitOptionDepthFromDepthStencil
                                             : MTLBlitOptionNone)];
            [blitEncoder copyFromTexture:intermediateTex
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                               toTexture:dsTex
                        destinationSlice:layer
                        destinationLevel:level
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
        }
    }
    [blitEncoder endEncoding];
}

void WriteStencilWithCopy(id<MTLDevice> device,
                          id<MTLCommandBuffer> commandBuffer,
                          id<MTLTexture> dsTex,
                          TextureData* data) {
    // Randomize initial value to mitigate reading results from a previous test iteration.
    uint8_t stencilValue = getInitialDataValue() % 255;

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            id<MTLBuffer> buffer = [device newBufferWithLength:(width >> level) * (height >> level)
                                                       options:MTLResourceStorageModeManaged];
            auto* contents = static_cast<uint8_t*>([buffer contents]);
            for (size_t i = 0; i < (*data)[level][layer].Get<U8>().size(); ++i) {
                stencilValue += 1;

                (*data)[level][layer].Get<U8>()[i] = stencilValue;
                contents[i] = stencilValue;

                stencilValue = stencilValue % 255;
            }
            [buffer didModifyRange:NSMakeRange(0, [buffer length])];
            [blitEncoder copyFromBuffer:buffer
                           sourceOffset:0
                      sourceBytesPerRow:(width >> level)
                    sourceBytesPerImage:(width >> level) * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:dsTex
                       destinationSlice:layer
                       destinationLevel:level
                      destinationOrigin:MTLOriginMake(0, 0, 0)
                                options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                             ? MTLBlitOptionStencilFromDepthStencil
                                             : MTLBlitOptionNone)];
        }
    }
    [blitEncoder endEncoding];
}

void WriteStencilWithTextureCopy(id<MTLDevice> device,
                                 id<MTLCommandBuffer> commandBuffer,
                                 id<MTLTexture> dsTex,
                                 TextureData* data) {
    // Randomize initial value to mitigate reading results from a previous test iteration.
    uint8_t stencilValue = getInitialDataValue() % 255;

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            id<MTLBuffer> buffer = [device newBufferWithLength:(width >> level) * (height >> level)
                                                       options:MTLResourceStorageModeManaged];
            auto* contents = static_cast<uint8_t*>([buffer contents]);
            for (size_t i = 0; i < (*data)[level][layer].Get<U8>().size(); ++i) {
                stencilValue += 1;

                (*data)[level][layer].Get<U8>()[i] = stencilValue;
                contents[i] = stencilValue;

                stencilValue = stencilValue % 255;
            }
            [buffer didModifyRange:NSMakeRange(0, [buffer length])];

            MTLTextureDescriptor* intermediateTexDesc = [MTLTextureDescriptor new];
            intermediateTexDesc.textureType = MTLTextureType2D;
            intermediateTexDesc.pixelFormat = [dsTex pixelFormat];
            intermediateTexDesc.width = (width >> level);
            intermediateTexDesc.height = (height >> level);
            intermediateTexDesc.depth = 1;
            intermediateTexDesc.arrayLength = 1;
            intermediateTexDesc.mipmapLevelCount = 1;
            intermediateTexDesc.storageMode = MTLStorageModePrivate;
            id<MTLTexture> intermediateTex = [device newTextureWithDescriptor:intermediateTexDesc];

            [blitEncoder copyFromBuffer:buffer
                           sourceOffset:0
                      sourceBytesPerRow:(width >> level)
                    sourceBytesPerImage:(width >> level) * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:intermediateTex
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)
                                options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                             ? MTLBlitOptionStencilFromDepthStencil
                                             : MTLBlitOptionNone)];
            [blitEncoder copyFromTexture:intermediateTex
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                               toTexture:dsTex
                        destinationSlice:layer
                        destinationLevel:level
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
        }
    }
    [blitEncoder endEncoding];
}

void WriteDepthWithLoadOp(id<MTLDevice> device,
                          id<MTLCommandBuffer> commandBuffer,
                          id<MTLTexture> dsTex,
                          bool offsetWithView,
                          TextureData* data) {
    // Randomize initial value to mitigate reading results from a previous test iteration.
    uint16_t depthStartValue = getInitialDataValue() % 0xFFFF;

    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            uint16_t depthClearValue =
                1 + ((level * arrayLayers + layer + depthStartValue) % 0xFFFF);
            ;

            MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.depthAttachment.clearDepth = float(depthClearValue) / float(0xFFFF);
            rpDesc.depthAttachment.loadAction = MTLLoadActionClear;
            rpDesc.depthAttachment.storeAction = MTLStoreActionStore;
            if (offsetWithView) {
                rpDesc.depthAttachment.texture =
                    [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                             textureType:MTLTextureType2D
                                                  levels:NSMakeRange(level, 1)
                                                  slices:NSMakeRange(layer, 1)];
            } else {
                rpDesc.depthAttachment.texture = dsTex;
                rpDesc.depthAttachment.level = level;
                rpDesc.depthAttachment.slice = layer;
            }
            if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
                UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
                rpDesc.stencilAttachment.texture = rpDesc.depthAttachment.texture;
                rpDesc.stencilAttachment.level = rpDesc.depthAttachment.level;
                rpDesc.stencilAttachment.slice = rpDesc.depthAttachment.slice;
                rpDesc.stencilAttachment.loadAction = MTLLoadActionLoad;
                rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
            }

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder endEncoding];

            std::fill((*data)[level][layer].Get<U16>().begin(),
                      (*data)[level][layer].Get<U16>().end(), depthClearValue);
            std::fill((*data)[level][layer].Get<F32>().begin(),
                      (*data)[level][layer].Get<F32>().end(), rpDesc.depthAttachment.clearDepth);
        }
    }
}

void WriteStencilWithLoadOp(id<MTLDevice> device,
                            id<MTLCommandBuffer> commandBuffer,
                            id<MTLTexture> dsTex,
                            bool offsetWithView,
                            TextureData* data) {
    // Randomize initial value to mitigate reading results from a previous test iteration.
    uint8_t stencilStartValue = getInitialDataValue() % 255;

    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.stencilAttachment.clearStencil =
                1 + ((level * arrayLayers + layer + stencilStartValue) % 255);
            rpDesc.stencilAttachment.loadAction = MTLLoadActionClear;
            rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
            if (offsetWithView) {
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
            if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
                UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
                rpDesc.depthAttachment.texture = rpDesc.stencilAttachment.texture;
                rpDesc.depthAttachment.level = rpDesc.stencilAttachment.level;
                rpDesc.depthAttachment.slice = rpDesc.stencilAttachment.slice;
                rpDesc.depthAttachment.loadAction = MTLLoadActionLoad;
                rpDesc.depthAttachment.storeAction = MTLStoreActionStore;
            }

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder endEncoding];

            std::fill((*data)[level][layer].Get<U8>().begin(),
                      (*data)[level][layer].Get<U8>().end(), rpDesc.stencilAttachment.clearStencil);
        }
    }
}

void WriteContentsWithStencilOp(id<MTLDevice> device,
                                id<MTLCommandBuffer> commandBuffer,
                                id<MTLTexture> dsTex,
                                bool offsetWithView,
                                TextureData* data) {
    auto* shader = @R"(
#include <metal_stdlib>
using namespace metal;

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
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.stencilAttachmentPixelFormat = [dsTex pixelFormat];
    if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
        UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
        pipelineStateDesc.depthAttachmentPixelFormat = [dsTex pixelFormat];
    }
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    id<MTLRenderPipelineState> pipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
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

    // Randomize initial value to mitigate reading results from a previous test iteration.
    uint8_t stencilStartValue = getInitialDataValue() % 255;

    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.stencilAttachment.clearStencil =
                1 + ((level * arrayLayers + layer + stencilStartValue) % 255);
            rpDesc.stencilAttachment.loadAction = MTLLoadActionClear;
            rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
            if (offsetWithView) {
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

            std::fill((*data)[level][layer].Get<U8>().begin(),
                      (*data)[level][layer].Get<U8>().end(), rpDesc.stencilAttachment.clearStencil);

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:pipelineState];
            [renderEncoder setDepthStencilState:dsState];
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];

            // Emulate the stencil invert operation to update the
            // expected texture data.
            for (unsigned y = (height >> level) / 2; y < (height >> level); ++y) {
                for (unsigned x = 0; x < (width >> level) / 2; ++x) {
                    unsigned i = y * (width >> level) + x;
                    (*data)[level][layer].Get<U8>()[i] = ~(*data)[level][layer].Get<U8>()[i];
                }
            }
        }
    }
}

void EncodeCheckR8DataWithCopy(id<MTLDevice> device,
                               id<MTLCommandBuffer> commandBuffer,
                               id<MTLTexture> dsTex,
                               const TextureData& data) {
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            id<MTLBuffer> buffer = [device newBufferWithLength:(width >> level) * (height >> level)
                                                       options:MTLResourceStorageModeShared];
            [blitEncoder copyFromTexture:dsTex
                             sourceSlice:layer
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                                toBuffer:buffer
                       destinationOffset:0
                  destinationBytesPerRow:(width >> level)
                destinationBytesPerImage:(width >> level) * (height >> level)
                                 options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                              ? MTLBlitOptionStencilFromDepthStencil
                                              : MTLBlitOptionNone)];

            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
                auto* contents = static_cast<uint8_t*>([buffer contents]);
                if (memcmp(contents, data[level][layer].Get<U8>().data(),
                           data[level][layer].Get<U8>().size()) != 0) {
                    failed = true;
                    if (level == 0 && layer == 0) {
                        failedFor0thSubresource = true;
                    }
                    if (verbose) {
                        LOG(@"\tlevel %d, layer %d - FAILED!", level, layer);
                        printf("\n\tExpected:\n");
                        for (unsigned y = 0; y < (height >> level); ++y) {
                            printf("\t");
                            for (unsigned x = 0; x < (width >> level); ++x) {
                                unsigned i = y * (width >> level) + x;
                                printf("%02x ", data[level][layer].Get<U8>()[i]);
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
                    LOG(@"\tlevel %d, layer %d - OK", level, layer);
                }
            }];
        }
    }
    [blitEncoder endEncoding];
}

void EncodeCheckDepthWithCopy(id<MTLDevice> device,
                              id<MTLCommandBuffer> commandBuffer,
                              id<MTLTexture> dsTex,
                              const TextureData& data) {
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            uint32_t bytesPerRow = DepthByteSize([dsTex pixelFormat]) * (width >> level);
            id<MTLBuffer> buffer = [device newBufferWithLength:bytesPerRow * (height >> level)
                                                       options:MTLResourceStorageModeShared];
            [blitEncoder copyFromTexture:dsTex
                             sourceSlice:layer
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                                toBuffer:buffer
                       destinationOffset:0
                  destinationBytesPerRow:bytesPerRow
                destinationBytesPerImage:bytesPerRow * (height >> level)
                                 options:([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8
                                              ? MTLBlitOptionDepthFromDepthStencil
                                              : MTLBlitOptionNone)];

            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
                if ([dsTex pixelFormat] == MTLPixelFormatDepth16Unorm) {
                    auto* contents = static_cast<uint16_t*>([buffer contents]);
                    bool checkFailed = false;
                    for (size_t i = 0; i < data[level][layer].Get<U16>().size(); ++i) {
                        int diff = int(contents[i]) - int(data[level][layer].Get<U16>()[i]);
                        if (diff < -1 || diff > 1) {
                            checkFailed = true;
                            break;
                        }
                    }
                    if (checkFailed) {
                        failed = true;
                        if (level == 0 && layer == 0) {
                            failedFor0thSubresource = true;
                        }
                        if (verbose) {
                            LOG(@"\tlevel %d, layer %d - FAILED!", level, layer);
                            printf("\n\tExpected close to:\n");
                            for (unsigned y = 0; y < (height >> level); ++y) {
                                printf("\t");
                                for (unsigned x = 0; x < (width >> level); ++x) {
                                    unsigned i = y * (width >> level) + x;
                                    printf("%04x ", data[level][layer].Get<U16>()[i]);
                                }
                                printf("\n");
                            }
                            printf("\n\tGot:\n");
                            for (unsigned y = 0; y < (height >> level); ++y) {
                                printf("\t");
                                for (unsigned x = 0; x < (width >> level); ++x) {
                                    unsigned i = y * (width >> level) + x;
                                    printf("%04x ", contents[i]);
                                }
                                printf("\n");
                            }

                            printf("\n");
                        }
                    } else if (verbose) {
                        LOG(@"\tlevel %d, layer %d - OK", level, layer);
                    }
                } else {
                    auto* contents = static_cast<float*>([buffer contents]);
                    // if (memcmp(contents, data[level][layer].Get<F32>().data(),
                    //            data[level][layer].Get<F32>().size()) != 0) {
                    bool checkFailed = false;
                    for (size_t i = 0; i < data[level][layer].Get<F32>().size(); ++i) {
                        float diff = contents[i] - data[level][layer].Get<F32>()[i];
                        if (diff > depthTolerance / 2. || diff < -depthTolerance / 2.) {
                            checkFailed = true;
                            break;
                        }
                    }
                    if (checkFailed) {
                        failed = true;
                        if (level == 0 && layer == 0) {
                            failedFor0thSubresource = true;
                        }
                        if (verbose) {
                            LOG(@"\tlevel %d, layer %d - FAILED!", level, layer);
                            printf("\n\tExpected close to:\n");
                            for (unsigned y = 0; y < (height >> level); ++y) {
                                printf("\t");
                                for (unsigned x = 0; x < (width >> level); ++x) {
                                    unsigned i = y * (width >> level) + x;
                                    printf("%08x ", *reinterpret_cast<const uint32_t*>(
                                                        &data[level][layer].Get<F32>()[i]));
                                }
                                printf("\n");
                            }
                            printf("\n\tGot:\n");
                            for (unsigned y = 0; y < (height >> level); ++y) {
                                printf("\t");
                                for (unsigned x = 0; x < (width >> level); ++x) {
                                    unsigned i = y * (width >> level) + x;
                                    printf("%08x ",
                                           *reinterpret_cast<const uint32_t*>(&contents[i]));
                                }
                                printf("\n");
                            }

                            printf("\n");
                        }
                    } else if (verbose) {
                        LOG(@"\tlevel %d, layer %d - OK", level, layer);
                    }
                }
            }];
        }
    }
    [blitEncoder endEncoding];
}

void CheckDepthWithCopy(id<MTLDevice> device,
                        id<MTLCommandBuffer> commandBuffer,
                        id<MTLTexture> dsTex,
                        const TextureData& data) {
    EncodeCheckDepthWithCopy(device, commandBuffer, dsTex, data);
}

void CheckStencilWithCopy(id<MTLDevice> device,
                          id<MTLCommandBuffer> commandBuffer,
                          id<MTLTexture> dsTex,
                          const TextureData& data) {
    EncodeCheckR8DataWithCopy(device, commandBuffer, dsTex, data);
}

void CheckDepthWithShader(id<MTLDevice> device,
                          id<MTLCommandBuffer> commandBuffer,
                          id<MTLTexture> dsTex,
                          bool offsetWithView,
                          const TextureData& data) {
    id<MTLTexture> depthView = dsTex;

    auto* shader_with_texture_array = @R"(
#include <metal_stdlib>
using namespace metal;

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

    // Sample depth data and write it to the attachment.
    fragment float frag_main(
        RasterizerData in [[stage_in]],
        constant uint& lod [[buffer(0)]],
        constant uint& array_index [[buffer(1)]],
        metal::texture2d_array<float, metal::access::sample> depth_texture [[texture(0)]]
    ) {
      return depth_texture.read(uint2(in.position.xy), array_index, lod).x;
    }
  )";

    auto* shader_no_texture_array = @R"(
#include <metal_stdlib>
using namespace metal;

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

    // Sample depth data and write it to the attachment.
    fragment float frag_main(
        RasterizerData in [[stage_in]],
        metal::texture2d<float, metal::access::sample> depth_texture [[texture(0)]]
    ) {
      return depth_texture.read(uint2(in.position.xy)).x;
    }
  )";

    NSError* error = nullptr;
    id<MTLLibrary> lib = [device
        newLibraryWithSource:(offsetWithView ? shader_no_texture_array : shader_with_texture_array)
                     options:nil
                       error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    id<MTLRenderPipelineState> pipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }

    MTLTextureDescriptor* colorTargetDesc = [MTLTextureDescriptor new];
    colorTargetDesc.textureType = MTLTextureType2DArray;
    colorTargetDesc.pixelFormat = MTLPixelFormatR32Float;
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

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:pipelineState];
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            if (offsetWithView) {
                depthView = [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                                     textureType:MTLTextureType2D
                                                          levels:NSMakeRange(level, 1)
                                                          slices:NSMakeRange(layer, 1)];
                [renderEncoder setFragmentTexture:depthView atIndex:0];
            } else {
                [renderEncoder setFragmentBytes:&level length:sizeof(level) atIndex:0];
                [renderEncoder setFragmentBytes:&layer length:sizeof(layer) atIndex:1];
                [renderEncoder setFragmentTexture:depthView atIndex:0];
            }
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];
        }
    }
    EncodeCheckDepthWithCopy(device, commandBuffer, colorTarget, data);
}

void CheckStencilWithShader(id<MTLDevice> device,
                            id<MTLCommandBuffer> commandBuffer,
                            id<MTLTexture> dsTex,
                            bool offsetWithView,
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
#include <metal_stdlib>
using namespace metal;

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
      return stencil_texture.read(uint2(in.position.xy), array_index, lod).x;
    }
  )";

    auto* shader_no_texture_array = @R"(
#include <metal_stdlib>
using namespace metal;

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
    id<MTLLibrary> lib = [device
        newLibraryWithSource:(offsetWithView ? shader_no_texture_array : shader_with_texture_array)
                     options:nil
                       error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
    id<MTLRenderPipelineState> pipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }

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

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:pipelineState];
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            if (offsetWithView) {
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
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];
        }
    }
    EncodeCheckR8DataWithCopy(device, commandBuffer, colorTarget, data);
}

// Upload the expected data into two float32 textures, one with slightly
// greater values and one with slightly smaller values. Perform two depth
// tests to check that the depth data is between the two.
void CheckDepthWithDepthTest(id<MTLDevice> device,
                             id<MTLCommandBuffer> commandBuffer,
                             id<MTLTexture> dsTex,
                             bool offsetWithView,
                             const TextureData& data) {
    auto* shader = @R"(
#include <metal_stdlib>
using namespace metal;

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

    struct FragmentOut {
        uint color [[color(0)]];
        float depth [[depth(any)]];
    };
    fragment FragmentOut frag_main(
        RasterizerData in [[stage_in]],
        metal::texture2d<float, metal::access::sample> expected_depth [[texture(0)]]
    ) {
      FragmentOut out;
      out.color = 1;
      out.depth = expected_depth.read(uint2(in.position.xy)).x;
      return out;
    }
  )";

    NSError* error = nullptr;
    id<MTLLibrary> lib = [device newLibraryWithSource:shader options:nil error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
    pipelineStateDesc.depthAttachmentPixelFormat = [dsTex pixelFormat];
    if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
        UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
        pipelineStateDesc.stencilAttachmentPixelFormat = [dsTex pixelFormat];
    }
    id<MTLRenderPipelineState> pipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }

    MTLDepthStencilDescriptor* dsDesc = [MTLDepthStencilDescriptor new];
    dsDesc.depthWriteEnabled = false;

    dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    id<MTLDepthStencilState> dsState0 = [device newDepthStencilStateWithDescriptor:dsDesc];

    dsDesc.depthCompareFunction = MTLCompareFunctionGreaterEqual;
    id<MTLDepthStencilState> dsState1 = [device newDepthStencilStateWithDescriptor:dsDesc];

    for (uint32_t level = 0; level < mipmapLevels; ++level) {
        for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
            MTLTextureDescriptor* expectedDataTexDesc = [MTLTextureDescriptor new];
            expectedDataTexDesc.textureType = MTLTextureType2D;
            expectedDataTexDesc.pixelFormat = MTLPixelFormatR32Float;
            expectedDataTexDesc.width = (width >> level);
            expectedDataTexDesc.height = (height >> level);
            expectedDataTexDesc.depth = 1;
            expectedDataTexDesc.arrayLength = 1;
            expectedDataTexDesc.mipmapLevelCount = 1;
            expectedDataTexDesc.storageMode = MTLStorageModePrivate;
            expectedDataTexDesc.usage = MTLTextureUsageShaderRead;

            id<MTLTexture> expectedDataTex0 = [device newTextureWithDescriptor:expectedDataTexDesc];
            id<MTLTexture> expectedDataTex1 = [device newTextureWithDescriptor:expectedDataTexDesc];

            // Upload the expected depth data to a float32 texture.
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            uint32_t bytesPerRow = DepthByteSize([dsTex pixelFormat]) * (width >> level);
            id<MTLBuffer> buffer0 = [device newBufferWithLength:bytesPerRow * (height >> level)
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> buffer1 = [device newBufferWithLength:bytesPerRow * (height >> level)
                                                        options:MTLResourceStorageModeShared];
            {
                auto* contents0 = static_cast<float*>([buffer0 contents]);
                auto* contents1 = static_cast<float*>([buffer1 contents]);
                for (size_t i = 0; i < data[level][layer].Get<F32>().size(); ++i) {
                    // Some tolerance since we may be comparing float32 with depth16unorm.
                    contents0[i] = data[level][layer].Get<F32>()[i] - depthTolerance / 2.0;
                    contents1[i] = data[level][layer].Get<F32>()[i] + depthTolerance / 2.0;
                }
            }

            [blitEncoder copyFromBuffer:buffer0
                           sourceOffset:0
                      sourceBytesPerRow:bytesPerRow
                    sourceBytesPerImage:bytesPerRow * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:expectedDataTex0
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blitEncoder copyFromBuffer:buffer1
                           sourceOffset:0
                      sourceBytesPerRow:bytesPerRow
                    sourceBytesPerImage:bytesPerRow * (height >> level)
                             sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                              toTexture:expectedDataTex1
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blitEncoder endEncoding];

            MTLTextureDescriptor* colorTargetDesc = [MTLTextureDescriptor new];
            colorTargetDesc.textureType = MTLTextureType2DArray;
            colorTargetDesc.pixelFormat = MTLPixelFormatR8Uint;
            colorTargetDesc.width = (width >> level);
            colorTargetDesc.height = (height >> level);
            colorTargetDesc.depth = 1;
            colorTargetDesc.arrayLength = 1;
            colorTargetDesc.mipmapLevelCount = 1;
            colorTargetDesc.storageMode = MTLStorageModePrivate;
            colorTargetDesc.usage = MTLTextureUsageRenderTarget;

            id<MTLTexture> colorTarget0 = [device newTextureWithDescriptor:colorTargetDesc];
            id<MTLTexture> colorTarget1 = [device newTextureWithDescriptor:colorTargetDesc];

            MTLRenderPassDescriptor* rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rpDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            rpDesc.depthAttachment.loadAction = MTLLoadActionLoad;
            rpDesc.depthAttachment.storeAction = MTLStoreActionStore;
            if (offsetWithView) {
                rpDesc.depthAttachment.texture =
                    [dsTex newTextureViewWithPixelFormat:[dsTex pixelFormat]
                                             textureType:MTLTextureType2D
                                                  levels:NSMakeRange(level, 1)
                                                  slices:NSMakeRange(layer, 1)];
            } else {
                rpDesc.depthAttachment.texture = dsTex;
                rpDesc.depthAttachment.level = level;
                rpDesc.depthAttachment.slice = layer;
            }
            if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
                UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
                rpDesc.stencilAttachment.texture = rpDesc.stencilAttachment.texture;
                rpDesc.stencilAttachment.level = rpDesc.stencilAttachment.level;
                rpDesc.stencilAttachment.slice = rpDesc.stencilAttachment.slice;
                rpDesc.stencilAttachment.loadAction = MTLLoadActionLoad;
                rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
            }

            rpDesc.colorAttachments[0].texture = colorTarget0;
            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:pipelineState];
            [renderEncoder setDepthStencilState:dsState0];
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            [renderEncoder setFragmentTexture:expectedDataTex0 atIndex:0];
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];

            rpDesc.colorAttachments[0].texture = colorTarget1;
            renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:pipelineState];
            [renderEncoder setDepthStencilState:dsState1];
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            [renderEncoder setFragmentTexture:expectedDataTex1 atIndex:0];
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];

            blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromTexture:colorTarget0
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                                toBuffer:buffer0
                       destinationOffset:0
                  destinationBytesPerRow:(width >> level)
                destinationBytesPerImage:(width >> level) * (height >> level)];
            [blitEncoder copyFromTexture:colorTarget1
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width >> level, height >> level, 1)
                                toBuffer:buffer1
                       destinationOffset:0
                  destinationBytesPerRow:(width >> level)
                destinationBytesPerImage:(width >> level) * (height >> level)];
            [blitEncoder endEncoding];

            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
                auto* contents0 = static_cast<uint8_t*>([buffer0 contents]);
                auto* contents1 = static_cast<uint8_t*>([buffer1 contents]);

                bool foundError = false;
                for (unsigned y = 0; y < (height >> level) && !foundError; ++y) {
                    for (unsigned x = 0; x < (width >> level) && !foundError; ++x) {
                        unsigned i = y * (width >> level) + x;
                        if (!contents0[i] || !contents1[i]) {
                            foundError = true;
                        }
                    }
                }

                if (foundError) {
                    failed = true;
                    if (level == 0 && layer == 0) {
                        failedFor0thSubresource = true;
                    }
                    if (verbose) {
                        LOG(@"\tlevel %d, layer %d - FAILED!", level, layer);
                        printf("\n\tIncorrect at the following locations:\n");
                        for (unsigned y = 0; y < (height >> level); ++y) {
                            printf("\t");
                            for (unsigned x = 0; x < (width >> level); ++x) {
                                unsigned i = y * (width >> level) + x;
                                if (!contents0[i] || !contents1[i]) {
                                    printf("x ");
                                } else {
                                    printf(". ");
                                }
                            }
                            printf("\n");
                        }
                        printf("\n");
                    }
                } else if (verbose) {
                    LOG(@"\tlevel %d, layer %d - OK", level, layer);
                }
            }];
        }
    }
}

// Read stencil via the stencil test bit by bit into 8 color textures (one for each bit).
// Sample the 8 textures and render to one texture, effectively recomposing the stencil data.
// Check that the recomposed data is correct.
void CheckStencilWithStencilTest(id<MTLDevice> device,
                                 id<MTLCommandBuffer> commandBuffer,
                                 id<MTLTexture> dsTex,
                                 bool offsetWithView,
                                 const TextureData& data) {
    auto* read_stencil_bit_shader = @R"(
#include <metal_stdlib>
using namespace metal;

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
    id<MTLLibrary> lib = [device newLibraryWithSource:read_stencil_bit_shader
                                              options:nil
                                                error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    MTLRenderPipelineDescriptor* pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
    pipelineStateDesc.stencilAttachmentPixelFormat = [dsTex pixelFormat];
    if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
        UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
        pipelineStateDesc.depthAttachmentPixelFormat = [dsTex pixelFormat];
    }
    id<MTLRenderPipelineState> readbackPipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }

    auto* recompose_shader = @R"(
#include <metal_stdlib>
using namespace metal;

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
    lib = [device newLibraryWithSource:recompose_shader options:nil error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }
    pipelineStateDesc = [MTLRenderPipelineDescriptor new];
    pipelineStateDesc.vertexFunction = [lib newFunctionWithName:@"vert_main"];
    pipelineStateDesc.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    pipelineStateDesc.colorAttachments[0].pixelFormat = MTLPixelFormatR8Uint;
    id<MTLRenderPipelineState> recomposePipelineState =
        [device newRenderPipelineStateWithDescriptor:pipelineStateDesc error:&error];
    if (error != nullptr) {
        LOG(@"%@", error);
        return;
    }

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
            rpDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rpDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            rpDesc.stencilAttachment.loadAction = MTLLoadActionLoad;
            rpDesc.stencilAttachment.storeAction = MTLStoreActionStore;
            if (offsetWithView) {
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
            if ([dsTex pixelFormat] == MTLPixelFormatDepth32Float_Stencil8 &&
                UseBothDepthAndStencilAttachmentsForCombinedDepthStencilFormats) {
                rpDesc.depthAttachment.texture = rpDesc.stencilAttachment.texture;
                rpDesc.depthAttachment.level = rpDesc.stencilAttachment.level;
                rpDesc.depthAttachment.slice = rpDesc.stencilAttachment.slice;
                rpDesc.depthAttachment.loadAction = MTLLoadActionLoad;
                rpDesc.depthAttachment.storeAction = MTLStoreActionStore;
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
                [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                [renderEncoder endEncoding];
            }

            // Now, merge the 8 textures into readbackTex.
            rpDesc.colorAttachments[0].texture = readbackTex;
            rpDesc.colorAttachments[0].level = level;
            rpDesc.colorAttachments[0].slice = layer;
            rpDesc.stencilAttachment = nil;
            rpDesc.depthAttachment = nil;

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [renderEncoder setRenderPipelineState:recomposePipelineState];
            for (uint32_t b = 0; b < 8; ++b) {
                [renderEncoder setFragmentTexture:colorTargets[b] atIndex:b];
            }
            [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [renderEncoder setCullMode:MTLCullModeNone];
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [renderEncoder endEncoding];
        }
    }
    EncodeCheckR8DataWithCopy(device, commandBuffer, readbackTex, data);
}

MTLTextureUsage GetRequiredUsage(WriteMethod wm, ReadMethod rm) {
    MTLTextureUsage usage = 0;
    switch (wm) {
        case WriteMethod::CopyFromBufferToDepth:
        case WriteMethod::CopyFromSingleSubresourceTextureToDepth:
        case WriteMethod::CopyFromBufferToStencil:
        case WriteMethod::CopyFromSingleSubresourceTextureToStencil:
            break;
        case WriteMethod::DepthLoadOpStoreOp:
        case WriteMethod::DepthLoadOpStoreOpOffsetWithView:
        case WriteMethod::StencilLoadOpStoreOp:
        case WriteMethod::StencilLoadOpStoreOpOffsetWithView:
        case WriteMethod::StencilOpStoreOp:
        case WriteMethod::StencilOpStoreOpOffsetWithView:
            usage |= MTLTextureUsageRenderTarget;
            break;
    }

    switch (rm) {
        case ReadMethod::CopyFromDepthToBuffer:
        case ReadMethod::CopyFromStencilToBuffer:
            break;
        case ReadMethod::ShaderReadDepth:
        case ReadMethod::ShaderReadDepthOffsetWithView:
            usage |= MTLTextureUsageShaderRead;
            break;
        case ReadMethod::ShaderReadStencil:
        case ReadMethod::ShaderReadStencilOffsetWithView:
            usage |= MTLTextureUsagePixelFormatView | MTLTextureUsageShaderRead;
            break;
        case ReadMethod::DepthTest:
        case ReadMethod::DepthTestOffsetWithView:
        case ReadMethod::StencilTestOffsetWithView:
        case ReadMethod::StencilTest:
            usage |= MTLTextureUsageRenderTarget;
            break;
    }
    return usage;
}

// Extracts an integer property from a registry entry.
uint32_t GetEntryProperty(io_registry_entry_t entry, CFStringRef name) {
    uint32_t value = 0;

    // Recursively search registry entry and its parents for property name
    // The data should release with CFRelease
    auto data = static_cast<CFDataRef>(
        IORegistryEntrySearchCFProperty(entry, kIOServicePlane, name, kCFAllocatorDefault,
                                        kIORegistryIterateRecursively | kIORegistryIterateParents));

    if (data == nullptr) {
        return value;
    }

    // CFDataGetBytePtr() is guaranteed to return a read-only pointer
    value = *reinterpret_cast<const uint32_t*>(CFDataGetBytePtr(data));
    return value;
}

void printUsage() {
    printf(R"(USAGE: depth_stencil_tests [options]

OPTIONS:
  --verbose		Print information about failed subresources and contents
  --levels <value>	Override the number of mip levels. (default: 4)
  --layers <value>	Override the number of array levels. (default: 3)
  --write <name>	Filter to only test one write method
  --read <name>		Filter to only test one read method
  --format <name>	Filter to only test one pixel format
  --gpu <name>		Filter to only test one GPU. Matches if the provided name is contained in the MTLDevice name.
  --deterministic	Fill textures with deterministic data. Makes results reproducable, but may cause unexpected
			passes if out-of-bounds reads happen to read the "correct" data.
)");
}

int main(int argc, char* argv[]) {
#define HAS_SWITCH(flag) strncmp(argv[i], #flag, sizeof(#flag) - 1) == 0
    const char* writeMethod = nullptr;
    const char* readMethod = nullptr;
    const char* gpuName = nullptr;
    const char* formatName = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (HAS_SWITCH(--help)) {
            printUsage();
            return 1;
        }
        if (HAS_SWITCH(--verbose)) {
            verbose = true;
        }
        if (HAS_SWITCH(--levels)) {
            if (i + 1 >= argc) {
                fprintf(stderr,
                        "`levels` switch should be followed by number. Example: --levels 4\n");
                return 1;
            }
            mipmapLevels = atoi(argv[i + 1]);
        }
        if (HAS_SWITCH(--layers)) {
            if (i + 1 >= argc) {
                fprintf(stderr,
                        "`layers` switch should be followed by number. Example: --layers 3\n");
                return 1;
            }
            arrayLayers = atoi(argv[i + 1]);
        }
        if (HAS_SWITCH(--write)) {
            if (i + 1 >= argc) {
                fprintf(stderr, "`write` switch should be followed by write method. Example: "
                                "--write CopyFromBufferToStencil\n");
                return 1;
            }
            writeMethod = argv[i + 1];
        }
        if (HAS_SWITCH(--read)) {
            if (i + 1 >= argc) {
                fprintf(stderr, "`read` switch should be followed by read method. Example: --read "
                                "CheckStencilWithStencilTest\n");
                return 1;
            }
            readMethod = argv[i + 1];
        }
        if (HAS_SWITCH(--format)) {
            if (i + 1 >= argc) {
                fprintf(stderr, "`format` switch should be followed by format name. Example: "
                                "--format Depth32FloatStencil8\n");
                return 1;
            }
            formatName = argv[i + 1];
        }
        if (HAS_SWITCH(--gpu)) {
            if (i + 1 >= argc) {
                fprintf(stderr,
                        "`gpu` switch should be followed by gpu name. Example: --gpu AMD\n");
                return 1;
            }
            gpuName = argv[i + 1];
        }
        if (HAS_SWITCH(--deterministic)) {
            deterministic = true;
        }
    }
#undef HAS_SWITCH

    auto osVersion = [[NSProcessInfo processInfo] operatingSystemVersionString];
    for (id<MTLDevice> device : MTLCopyAllDevices()) {
        if (gpuName != nullptr &&
            ![[device name] containsString:[NSString stringWithUTF8String:gpuName]]) {
            continue;
        }

        auto matchingDict = IORegistryEntryIDMatching([device registryID]);
        if (matchingDict == nullptr) {
            return 1;
        }

        // IOServiceGetMatchingService will consume the reference on the matching dictionary,
        // so we don't need to release the dictionary.
        auto acceleratorEntry = IOServiceGetMatchingService(kIOMasterPortDefault, matchingDict);
        if (acceleratorEntry == IO_OBJECT_NULL) {
            return 1;
        }

        // Get the parent entry that will be the IOPCIDevice
        io_registry_entry_t deviceEntry;
        if (IORegistryEntryGetParentEntry(acceleratorEntry, kIOServicePlane, &deviceEntry) !=
            kIOReturnSuccess) {
            return 1;
        }

        uint32_t vendorId = GetEntryProperty(deviceEntry, CFSTR("vendor-id"));
        uint32_t deviceId = GetEntryProperty(deviceEntry, CFSTR("device-id"));

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        if (verbose) {
            LOG(@"==================================================================\n\n");
            LOG(@"Testing with %@ %04x:%04x\n\n", [device name], vendorId, deviceId);
        }

        for (WriteMethod wm : writeMethods) {
            if (writeMethod != nullptr && strcmp(writeMethod, ToString(wm)) != 0) {
                continue;
            }
            for (ReadMethod rm : readMethods) {
                if (readMethod != nullptr && strcmp(readMethod, ToString(rm)) != 0) {
                    continue;
                }
                for (bool copyT2T : {false, true}) {
                    for (MTLPixelFormat pixelFormat : {
                             MTLPixelFormatDepth16Unorm,
                             MTLPixelFormatDepth32Float,
                             MTLPixelFormatDepth32Float_Stencil8,
                             MTLPixelFormatStencil8,
                         }) {
                        if (formatName != nullptr &&
                            strcmp(formatName, ToString(pixelFormat)) != 0) {
                            continue;
                        }

                        const bool isStencil =
                            std::string(ToString(pixelFormat)).find("Stencil") != std::string::npos;
                        const bool isDepth =
                            std::string(ToString(pixelFormat)).find("Depth") != std::string::npos;

                        const bool writeDepth =
                            std::string(ToString(wm)).find("Depth") != std::string::npos;
                        const bool writeStencil =
                            std::string(ToString(wm)).find("Stencil") != std::string::npos;
                        const bool readDepth =
                            std::string(ToString(rm)).find("Depth") != std::string::npos;
                        const bool readStencil =
                            std::string(ToString(rm)).find("Stencil") != std::string::npos;

                        // Skip configurations that don't make sense.
                        if (!isDepth && (writeDepth || readDepth)) {
                            continue;
                        }
                        if (!isStencil && (writeStencil || readStencil)) {
                            continue;
                        }
                        if (writeDepth && !readDepth) {
                            continue;
                        }
                        if (writeStencil && !readStencil) {
                            continue;
                        }

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
                            LOG(@"%@, %@ (%04x:%04x), %s, %s, CopyT2T-%s", osVersion, [device name],
                                vendorId, deviceId, ToString(pixelFormat), ToString(wm),
                                ToString(rm));
                        } else {
                            LOG(@"%@, %@ (%04x:%04x), %s, %s, %s", osVersion, [device name],
                                vendorId, deviceId, ToString(pixelFormat), ToString(wm),
                                ToString(rm));
                        }

                        TextureData data;
                        data.resize(mipmapLevels);
                        for (uint32_t level = 0; level < mipmapLevels; ++level) {
                            data[level].resize(arrayLayers);
                            for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
                                data[level][layer].resize((width >> level) * (height >> level));
                            }
                        }

                        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

                        failed = false;
                        failedFor0thSubresource = false;
                        switch (wm) {
                            case WriteMethod::CopyFromBufferToDepth:
                                WriteDepthWithCopy(device, commandBuffer, dsTex, &data);
                                break;
                            case WriteMethod::CopyFromSingleSubresourceTextureToDepth:
                                WriteDepthWithTextureCopy(device, commandBuffer, dsTex, &data);
                                break;
                            case WriteMethod::CopyFromBufferToStencil:
                                WriteStencilWithCopy(device, commandBuffer, dsTex, &data);
                                break;
                            case WriteMethod::CopyFromSingleSubresourceTextureToStencil:
                                WriteStencilWithTextureCopy(device, commandBuffer, dsTex, &data);
                                break;
                            case WriteMethod::DepthLoadOpStoreOp:
                                WriteDepthWithLoadOp(device, commandBuffer, dsTex,
                                                     /* offsetWithView */ false, &data);
                                break;
                            case WriteMethod::DepthLoadOpStoreOpOffsetWithView:
                                WriteDepthWithLoadOp(device, commandBuffer, dsTex,
                                                     /* offsetWithView */ true, &data);
                                break;
                            case WriteMethod::StencilLoadOpStoreOp:
                                WriteStencilWithLoadOp(device, commandBuffer, dsTex,
                                                       /* offsetWithView */ false, &data);
                                break;
                            case WriteMethod::StencilLoadOpStoreOpOffsetWithView:
                                WriteStencilWithLoadOp(device, commandBuffer, dsTex,
                                                       /* offsetWithView */ true, &data);
                                break;
                            case WriteMethod::StencilOpStoreOp:
                                WriteContentsWithStencilOp(device, commandBuffer, dsTex,
                                                           /* offsetWithView */ false, &data);
                                break;
                            case WriteMethod::StencilOpStoreOpOffsetWithView:
                                WriteContentsWithStencilOp(device, commandBuffer, dsTex,
                                                           /* offsetWithView */ true, &data);
                                break;
                        }

                        if (copyT2T) {
                            id<MTLTexture> intermediateTex =
                                [device newTextureWithDescriptor:dsTexDesc];

                            id<MTLBlitCommandEncoder> blitEncoder =
                                [commandBuffer blitCommandEncoder];
                            for (uint32_t level = 0; level < mipmapLevels; ++level) {
                                for (uint32_t layer = 0; layer < arrayLayers; ++layer) {
                                    [blitEncoder copyFromTexture:dsTex
                                                     sourceSlice:layer
                                                     sourceLevel:level
                                                    sourceOrigin:MTLOriginMake(0, 0, 0)
                                                      sourceSize:MTLSizeMake(width >> level,
                                                                             height >> level, 1)
                                                       toTexture:intermediateTex
                                                destinationSlice:layer
                                                destinationLevel:level
                                               destinationOrigin:MTLOriginMake(0, 0, 0)];
                                }
                            }
                            [blitEncoder endEncoding];

                            dsTex = intermediateTex;
                        }

                        switch (rm) {
                            case ReadMethod::CopyFromDepthToBuffer:
                                CheckDepthWithCopy(device, commandBuffer, dsTex, data);
                                break;
                            case ReadMethod::CopyFromStencilToBuffer:
                                CheckStencilWithCopy(device, commandBuffer, dsTex, data);
                                break;
                            case ReadMethod::ShaderReadDepth:
                                CheckDepthWithShader(device, commandBuffer, dsTex,
                                                     /* offsetWithView */ false, data);
                                break;
                            case ReadMethod::ShaderReadDepthOffsetWithView:
                                CheckDepthWithShader(device, commandBuffer, dsTex,
                                                     /* offsetWithView */ true, data);
                                break;
                            case ReadMethod::ShaderReadStencil:
                                CheckStencilWithShader(device, commandBuffer, dsTex,
                                                       /* offsetWithView */ false, data);
                                break;
                            case ReadMethod::ShaderReadStencilOffsetWithView:
                                CheckStencilWithShader(device, commandBuffer, dsTex,
                                                       /* offsetWithView */ true, data);
                                break;
                            case ReadMethod::DepthTest:
                                CheckDepthWithDepthTest(device, commandBuffer, dsTex,
                                                        /* offsetWithView */ false, data);
                                break;
                            case ReadMethod::DepthTestOffsetWithView:
                                CheckDepthWithDepthTest(device, commandBuffer, dsTex,
                                                        /* offsetWithView */ true, data);
                                break;
                            case ReadMethod::StencilTest:
                                CheckStencilWithStencilTest(device, commandBuffer, dsTex,
                                                            /* offsetWithView */ false, data);
                                break;
                            case ReadMethod::StencilTestOffsetWithView:
                                CheckStencilWithStencilTest(device, commandBuffer, dsTex,
                                                            /* offsetWithView */ true, data);
                                break;
                        }

                        [commandBuffer commit];
                        [commandBuffer waitUntilCompleted];

                        if (failed && !failedFor0thSubresource) {
                            LOG(@"FAILED (non-zero subresource)");
                        } else if (failed) {
                            LOG(@"FAILED");
                        } else {
                            LOG(@"OK");
                        }
                    }
                }
            }
        }
    }

    return 0;
}
