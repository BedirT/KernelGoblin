#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel_goblin/trellis2/z_order.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

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

void validate_count(std::size_t count) {
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("Metal Z-order dispatch supports at most UINT32_MAX elements");
  }
}

id<MTLBuffer> input_buffer(id<MTLDevice> device,
                           const std::vector<std::uint32_t>& values) {
  return [device newBufferWithBytes:values.data()
                             length:values.size() * sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared];
}

id<MTLBuffer> output_buffer(id<MTLDevice> device, std::size_t count) {
  return [device newBufferWithLength:count * sizeof(std::uint32_t)
                             options:MTLResourceStorageModeShared];
}

void dispatch(id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> pipeline,
              NSArray<id<MTLBuffer>>* buffers,
              std::uint32_t count) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil) throw metal_error("could not create Metal command encoder");

  [encoder setComputePipelineState:pipeline];
  for (NSUInteger index = 0; index < buffers.count; ++index) {
    [encoder setBuffer:buffers[index] offset:0 atIndex:index];
  }
  [encoder setBytes:&count length:sizeof(count) atIndex:4];

  const NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw metal_error("Metal Z-order dispatch failed", command.error);
  }
}

std::vector<std::uint32_t> copy_buffer(id<MTLBuffer> buffer, std::size_t count) {
  std::vector<std::uint32_t> result(count);
  std::memcpy(result.data(), buffer.contents, count * sizeof(std::uint32_t));
  return result;
}

}  // namespace

struct MetalZOrder::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> encode_pipeline;
  id<MTLComputePipelineState> decode_pipeline;

  Impl() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw metal_error("no Metal device is available");
      queue = [device newCommandQueue];
      if (queue == nil) throw metal_error("could not create Metal command queue");

      NSError* error = nil;
      NSString* path = [NSString stringWithUTF8String:KG_METALLIB_PATH];
      NSURL* url = [NSURL fileURLWithPath:path];
      id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
      if (library == nil) throw metal_error("could not load Z-order metallib", error);

      id<MTLFunction> encode_function = [library newFunctionWithName:@"z_order_encode"];
      id<MTLFunction> decode_function = [library newFunctionWithName:@"z_order_decode"];
      if (encode_function == nil || decode_function == nil) {
        throw metal_error("Z-order functions are missing from metallib");
      }
      encode_pipeline = [device newComputePipelineStateWithFunction:encode_function error:&error];
      if (encode_pipeline == nil) throw metal_error("could not create encode pipeline", error);
      error = nil;
      decode_pipeline = [device newComputePipelineStateWithFunction:decode_function error:&error];
      if (decode_pipeline == nil) throw metal_error("could not create decode pipeline", error);
    }
  }
};

MetalZOrder::MetalZOrder() : impl_(std::make_unique<Impl>()) {}
MetalZOrder::~MetalZOrder() = default;
MetalZOrder::MetalZOrder(MetalZOrder&&) noexcept = default;
MetalZOrder& MetalZOrder::operator=(MetalZOrder&&) noexcept = default;

std::string MetalZOrder::device_name() const {
  @autoreleasepool {
    return impl_->device.name.UTF8String;
  }
}

std::vector<std::uint32_t> MetalZOrder::encode(
    const std::vector<std::uint32_t>& x,
    const std::vector<std::uint32_t>& y,
    const std::vector<std::uint32_t>& z) const {
  if (x.size() != y.size() || x.size() != z.size()) {
    throw std::invalid_argument("x, y, and z must have identical lengths");
  }
  validate_count(x.size());
  if (x.empty()) return {};

  @autoreleasepool {
    id<MTLBuffer> x_buffer = input_buffer(impl_->device, x);
    id<MTLBuffer> y_buffer = input_buffer(impl_->device, y);
    id<MTLBuffer> z_buffer = input_buffer(impl_->device, z);
    id<MTLBuffer> codes_buffer = output_buffer(impl_->device, x.size());
    if (x_buffer == nil || y_buffer == nil || z_buffer == nil || codes_buffer == nil) {
      throw metal_error("could not allocate Metal encode buffers");
    }
    dispatch(impl_->queue, impl_->encode_pipeline,
             @[ x_buffer, y_buffer, z_buffer, codes_buffer ],
             static_cast<std::uint32_t>(x.size()));
    return copy_buffer(codes_buffer, x.size());
  }
}

Coordinates MetalZOrder::decode(const std::vector<std::uint32_t>& codes) const {
  validate_count(codes.size());
  if (codes.empty()) return {};

  @autoreleasepool {
    id<MTLBuffer> codes_buffer = input_buffer(impl_->device, codes);
    id<MTLBuffer> x_buffer = output_buffer(impl_->device, codes.size());
    id<MTLBuffer> y_buffer = output_buffer(impl_->device, codes.size());
    id<MTLBuffer> z_buffer = output_buffer(impl_->device, codes.size());
    if (codes_buffer == nil || x_buffer == nil || y_buffer == nil || z_buffer == nil) {
      throw metal_error("could not allocate Metal decode buffers");
    }
    dispatch(impl_->queue, impl_->decode_pipeline,
             @[ codes_buffer, x_buffer, y_buffer, z_buffer ],
             static_cast<std::uint32_t>(codes.size()));
    return {
        copy_buffer(x_buffer, codes.size()),
        copy_buffer(y_buffer, codes.size()),
        copy_buffer(z_buffer, codes.size()),
    };
  }
}

}  // namespace kernel_goblin::trellis2
