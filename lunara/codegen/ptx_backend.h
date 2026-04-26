#pragma once
// lunara/codegen/ptx_backend.h
// Drives LLVM's NVPTX backend to compile MLIR GPU dialect → PTX → CUBIN.
// This is used when the Triton path is disabled or for low-level ops.

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include <string>

namespace lunara {
namespace codegen {

struct PTXOptions {
  std::string gpu_arch       = "sm_80";   ///< e.g. sm_80, sm_90
  int         opt_level      = 3;
  bool        fast_math      = true;
  bool        use_fp16_math  = true;
  bool        emit_line_info = false;
  /// If non-empty, also compile PTX → cubin using ptxas.
  std::string ptxas_path;
};

/// Lower MLIR GPU dialect ops → PTX text.
llvm::Expected<std::string>
compileToPTX(mlir::ModuleOp mod, const PTXOptions &opts = {});

/// Lower MLIR GPU dialect ops → cubin binary (requires ptxas).
llvm::Expected<std::vector<char>>
compileToCUBIN(mlir::ModuleOp mod, const PTXOptions &opts = {});

/// Write PTX string to a file.
llvm::Error writePTX(const std::string &ptx, const std::string &path);

/// Load a cubin and return a handle (wraps cuModuleLoad).
/// Returns opaque void* handle; caller must call unloadCUBIN.
llvm::Expected<void*> loadCUBIN(const std::vector<char> &cubin);
void unloadCUBIN(void *handle);

} // namespace codegen
} // namespace lunara
