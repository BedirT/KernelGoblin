#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel_goblin/trellis2/uv_raster.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifndef KG_METALLIB_PATH
#error "KG_METALLIB_PATH must point to the compiled Metal library"
#endif

namespace kernel_goblin::trellis2 {
namespace {

std::runtime_error metal_error(const char* message, NSError* error = nil) {
  if (error == nil) return std::runtime_error(message);
  return std::runtime_error(std::string(message) + ": " +
                            error.localizedDescription.UTF8String);
}

NSUInteger align256(NSUInteger value) { return (value + 255) & ~NSUInteger(255); }

template <typename T>
id<MTLBuffer> make_buffer(id<MTLDevice> device, const std::vector<T>& values) {
  return [device newBufferWithBytes:values.data()
                             length:values.size() * sizeof(T)
                            options:MTLResourceStorageModeShared];
}

}  // namespace

struct MetalUVRaster::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLRenderPipelineState> pipeline;

  Impl() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw metal_error("no Metal device is available");
      queue = [device newCommandQueue];
      if (queue == nil) throw metal_error("could not create Metal command queue");

      NSError* error = nil;
      NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:KG_METALLIB_PATH]];
      id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
      if (library == nil) throw metal_error("could not load UV raster metallib", error);
      id<MTLFunction> vertex = [library newFunctionWithName:@"uv_raster_vertex"];
      id<MTLFunction> fragment = [library newFunctionWithName:@"uv_raster_fragment"];
      if (vertex == nil || fragment == nil) throw metal_error("UV raster functions are missing");

      MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
      descriptor.vertexFunction = vertex;
      descriptor.fragmentFunction = fragment;
      descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
      descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;
      pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
      if (pipeline == nil) throw metal_error("could not create UV raster pipeline", error);
    }
  }
};

MetalUVRaster::MetalUVRaster() : impl_(std::make_unique<Impl>()) {}
MetalUVRaster::~MetalUVRaster() = default;
MetalUVRaster::MetalUVRaster(MetalUVRaster&&) noexcept = default;
MetalUVRaster& MetalUVRaster::operator=(MetalUVRaster&&) noexcept = default;

std::string MetalUVRaster::device_name() const {
  @autoreleasepool { return impl_->device.name.UTF8String; }
}

UVRasterResult MetalUVRaster::rasterize(
    const std::vector<float>& positions,
    const std::vector<float>& uvs,
    const std::vector<std::uint32_t>& faces,
    std::uint32_t width,
    std::uint32_t height) const {
  // Reuse the reference validator before allocating GPU resources.
  (void)uv_raster_cpu(positions, uvs, faces, width, height);
  UVRasterResult result;
  result.width = width;
  result.height = height;
  const auto pixel_count = static_cast<std::size_t>(width) * height;
  result.positions.assign(pixel_count * 4, 0.0f);
  result.face_ids.assign(pixel_count, 0);
  if (faces.empty()) return result;

  @autoreleasepool {
    id<MTLBuffer> position_buffer = make_buffer(impl_->device, positions);
    id<MTLBuffer> uv_buffer = make_buffer(impl_->device, uvs);
    id<MTLBuffer> face_buffer = make_buffer(impl_->device, faces);
    if (position_buffer == nil || uv_buffer == nil || face_buffer == nil) {
      throw metal_error("could not allocate UV raster input buffers");
    }

    MTLTextureDescriptor* position_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:width height:height mipmapped:NO];
    position_desc.usage = MTLTextureUsageRenderTarget;
    position_desc.storageMode = MTLStorageModePrivate;
    MTLTextureDescriptor* id_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
        width:width height:height mipmapped:NO];
    id_desc.usage = MTLTextureUsageRenderTarget;
    id_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> position_texture = [impl_->device newTextureWithDescriptor:position_desc];
    id<MTLTexture> id_texture = [impl_->device newTextureWithDescriptor:id_desc];
    if (position_texture == nil || id_texture == nil) {
      throw metal_error("could not allocate UV raster render targets");
    }

    const NSUInteger position_row = align256(width * 4 * sizeof(float));
    const NSUInteger id_row = align256(width * sizeof(std::uint32_t));
    id<MTLBuffer> position_readback = [impl_->device
        newBufferWithLength:position_row * height options:MTLResourceStorageModeShared];
    id<MTLBuffer> id_readback = [impl_->device
        newBufferWithLength:id_row * height options:MTLResourceStorageModeShared];
    if (position_readback == nil || id_readback == nil) {
      throw metal_error("could not allocate UV raster readback buffers");
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = position_texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.colorAttachments[1].texture = id_texture;
    pass.colorAttachments[1].loadAction = MTLLoadActionClear;
    pass.colorAttachments[1].storeAction = MTLStoreActionStore;
    pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);

    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    id<MTLRenderCommandEncoder> render = [command renderCommandEncoderWithDescriptor:pass];
    if (command == nil || render == nil) throw metal_error("could not create render encoder");
    [render setRenderPipelineState:impl_->pipeline];
    [render setCullMode:MTLCullModeNone];
    [render setVertexBuffer:position_buffer offset:0 atIndex:0];
    [render setVertexBuffer:uv_buffer offset:0 atIndex:1];
    [render setVertexBuffer:face_buffer offset:0 atIndex:2];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:faces.size()];
    [render endEncoding];

    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) throw metal_error("could not create UV raster blit encoder");
    const MTLSize size = MTLSizeMake(width, height, 1);
    [blit copyFromTexture:position_texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:size
                 toBuffer:position_readback destinationOffset:0
        destinationBytesPerRow:position_row destinationBytesPerImage:position_row * height];
    [blit copyFromTexture:id_texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:size
                 toBuffer:id_readback destinationOffset:0
        destinationBytesPerRow:id_row destinationBytesPerImage:id_row * height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw metal_error("Metal UV raster command failed", command.error);
    }

    for (std::uint32_t y = 0; y < height; ++y) {
      std::memcpy(result.positions.data() + static_cast<std::size_t>(y) * width * 4,
                  static_cast<const std::byte*>(position_readback.contents) + y * position_row,
                  width * 4 * sizeof(float));
      std::memcpy(result.face_ids.data() + static_cast<std::size_t>(y) * width,
                  static_cast<const std::byte*>(id_readback.contents) + y * id_row,
                  width * sizeof(std::uint32_t));
    }
  }
  return result;
}

}  // namespace kernel_goblin::trellis2
