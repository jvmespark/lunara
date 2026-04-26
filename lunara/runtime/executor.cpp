// lunara/runtime/executor.cpp
#include "lunara/runtime/executor.h"
#include "lunara/utils/logging.h"

#include "llvm/Support/Error.h"
#include <fstream>

#ifdef LUNARA_ENABLE_CUDA
#include <cuda.h>
#define CU_CHECK(expr)                                                  \
  do {                                                                  \
    CUresult _rc = (expr);                                              \
    if (_rc != CUDA_SUCCESS) {                                          \
      const char *_s;                                                   \
      cuGetErrorString(_rc, &_s);                                       \
      return llvm::make_error<llvm::StringError>(                       \
          std::string(#expr) + " failed: " + _s,                       \
          llvm::inconvertibleErrorCode());                              \
    }                                                                   \
  } while (0)
#endif

namespace lunara {
namespace runtime {

KernelExecutor::KernelExecutor() {
#ifdef LUNARA_ENABLE_CUDA
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuCtxCreate(&ctx, 0, dev);
  LUNARA_LOG(info) << "CUDA context initialised";
#else
  LUNARA_LOG(warn) << "KernelExecutor: CUDA not compiled in — stubs only";
#endif
}

KernelExecutor::~KernelExecutor() {
#ifdef LUNARA_ENABLE_CUDA
  for (auto *m : modules_)
    cuModuleUnload(static_cast<CUmodule>(m));
#endif
}

llvm::Error KernelExecutor::loadCUBIN(const std::string &path) {
#ifdef LUNARA_ENABLE_CUDA
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    return llvm::make_error<llvm::StringError>(
        "Cannot open cubin: " + path, llvm::inconvertibleErrorCode());
  std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

  CUmodule mod;
  CUresult rc = cuModuleLoadData(&mod, buf.data());
  if (rc != CUDA_SUCCESS) {
    const char *s;
    cuGetErrorString(rc, &s);
    return llvm::make_error<llvm::StringError>(
        std::string("cuModuleLoadData: ") + s,
        llvm::inconvertibleErrorCode());
  }
  modules_.push_back(static_cast<void*>(mod));
  LUNARA_LOG(info) << "Loaded cubin: " << path;

  // Enumerate kernels (relies on the caller registering by name for now)
  return llvm::Error::success();
#else
  (void)path;
  return llvm::make_error<llvm::StringError>(
      "CUDA not compiled in", llvm::inconvertibleErrorCode());
#endif
}

llvm::Error KernelExecutor::loadPTX(const std::string &ptx,
                                    const std::string &kernel_name) {
#ifdef LUNARA_ENABLE_CUDA
  CUmodule mod;
  CUresult rc = cuModuleLoadData(&mod, ptx.c_str());
  if (rc != CUDA_SUCCESS) {
    const char *s;
    cuGetErrorString(rc, &s);
    return llvm::make_error<llvm::StringError>(
        std::string("cuModuleLoadData(PTX): ") + s,
        llvm::inconvertibleErrorCode());
  }
  modules_.push_back(static_cast<void*>(mod));

  auto kh = std::make_unique<KernelHandle>();
  kh->name = kernel_name;
  kh->module_handle = static_cast<void*>(mod);

  CUfunction fn;
  rc = cuModuleGetFunction(&fn, mod, kernel_name.c_str());
  if (rc != CUDA_SUCCESS) {
    const char *s; cuGetErrorString(rc, &s);
    return llvm::make_error<llvm::StringError>(
        "cuModuleGetFunction '" + kernel_name + "': " + s,
        llvm::inconvertibleErrorCode());
  }
  kh->func_handle = static_cast<void*>(fn);
  kernels_[kernel_name] = std::move(kh);
  LUNARA_LOG(info) << "Loaded PTX kernel: " << kernel_name;
  return llvm::Error::success();
#else
  (void)ptx; (void)kernel_name;
  return llvm::make_error<llvm::StringError>(
      "CUDA not compiled in", llvm::inconvertibleErrorCode());
#endif
}

llvm::Expected<KernelHandle*>
KernelExecutor::getKernel(const std::string &name) {
  auto it = kernels_.find(name);
  if (it == kernels_.end())
    return llvm::make_error<llvm::StringError>(
        "Kernel not found: " + name, llvm::inconvertibleErrorCode());
  return it->second.get();
}

llvm::Error KernelExecutor::launch(KernelHandle &kernel,
                                   const LaunchConfig &config,
                                   const std::vector<void*> &args) {
#ifdef LUNARA_ENABLE_CUDA
  CUfunction fn = static_cast<CUfunction>(kernel.func_handle);
  CUstream   stream = static_cast<CUstream>(config.stream);

  CUresult rc = cuLaunchKernel(
      fn,
      config.grid_x, config.grid_y, config.grid_z,
      config.block_x, config.block_y, config.block_z,
      config.shared_mem_bytes,
      stream,
      const_cast<void**>(args.data()),
      nullptr);
  if (rc != CUDA_SUCCESS) {
    const char *s; cuGetErrorString(rc, &s);
    return llvm::make_error<llvm::StringError>(
        "cuLaunchKernel '" + kernel.name + "': " + s,
        llvm::inconvertibleErrorCode());
  }
  LUNARA_LOG(debug) << "Launched " << kernel.name
                    << " grid=(" << config.grid_x << ","
                    << config.grid_y << "," << config.grid_z << ")";
  return llvm::Error::success();
#else
  (void)kernel; (void)config; (void)args;
  return llvm::make_error<llvm::StringError>(
      "CUDA not compiled in", llvm::inconvertibleErrorCode());
#endif
}

llvm::Error KernelExecutor::synchronize() {
#ifdef LUNARA_ENABLE_CUDA
  CUresult rc = cuStreamSynchronize(nullptr);
  if (rc != CUDA_SUCCESS) {
    const char *s; cuGetErrorString(rc, &s);
    return llvm::make_error<llvm::StringError>(
        std::string("cuStreamSynchronize: ") + s,
        llvm::inconvertibleErrorCode());
  }
#endif
  return llvm::Error::success();
}

} // namespace runtime
} // namespace lunara
