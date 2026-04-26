#pragma once
// lunara/runtime/executor.h
// Loads compiled Triton/PTX kernels and provides a synchronous execution interface over CUDA streams.

#include "llvm/Support/Error.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lunara {
namespace runtime {

/// Opaque handle to a loaded GPU kernel.
struct KernelHandle {
  void *module_handle = nullptr;  ///< CUmodule
  void *func_handle   = nullptr;  ///< CUfunction
  std::string name;
};

/// Grid/block launch config.
struct LaunchConfig {
  unsigned grid_x  = 1, grid_y  = 1, grid_z  = 1;
  unsigned block_x = 128, block_y = 1, block_z = 1;
  unsigned shared_mem_bytes = 0;
  void *stream = nullptr;  ///< CUstream; nullptr = default stream
};

/// Simple executor that manages a pool of loaded kernels.
class KernelExecutor {
public:
  KernelExecutor();
  ~KernelExecutor();

  /// Load a cubin file and register all kernels in it.
  llvm::Error loadCUBIN(const std::string &path);

  /// Load a single PTX source string.
  llvm::Error loadPTX(const std::string &ptx, const std::string &kernel_name);

  /// Retrieve a previously loaded kernel by name.
  llvm::Expected<KernelHandle*> getKernel(const std::string &name);

  /// Launch a kernel with the given config and argument buffer.
  llvm::Error launch(KernelHandle &kernel,
                     const LaunchConfig &config,
                     const std::vector<void*> &args);

  /// Synchronise the default stream.
  llvm::Error synchronize();

  /// Number of loaded kernels.
  size_t numKernels() const { return kernels_.size(); }

private:
  std::unordered_map<std::string, std::unique_ptr<KernelHandle>> kernels_;
  std::vector<void*> modules_;  ///< CUmodule handles to free on destruction
};

} // namespace runtime
} // namespace lunara
