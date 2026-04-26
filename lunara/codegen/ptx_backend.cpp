// lunara/codegen/ptx_backend.cpp
#include "lunara/codegen/ptx_backend.h"
#include "lunara/utils/logging.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <fstream>

using namespace mlir;
using namespace llvm;

namespace lunara {
namespace codegen {

// ── LLVM NVPTX initialisation (call once) ─────────────────────────────────
static void initNVPTX() {
  static bool done = false;
  if (done) return;
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  done = true;
}

// ── Build NVPTX TargetMachine ──────────────────────────────────────────────
static Expected<std::unique_ptr<TargetMachine>>
makeTargetMachine(const PTXOptions &opts) {
  initNVPTX();

  std::string triple = "nvptx64-nvidia-cuda";
  std::string err;
  const Target *target = TargetRegistry::lookupTarget(triple, err);
  if (!target)
    return make_error<StringError>(
        "NVPTX target not found: " + err, inconvertibleErrorCode());

  TargetOptions topts;
  if (opts.fast_math) {
    topts.AllowFPOpFusion    = FPOpFusion::Fast;
    topts.UnsafeFPMath       = true;
    topts.NoNaNsFPMath       = true;
  }

  CodeGenOptLevel cg_opt;
  switch (opts.opt_level) {
  case 0: cg_opt = CodeGenOptLevel::None;       break;
  case 1: cg_opt = CodeGenOptLevel::Less;       break;
  case 2: cg_opt = CodeGenOptLevel::Default;    break;
  default:cg_opt = CodeGenOptLevel::Aggressive; break;
  }

  auto tm = std::unique_ptr<TargetMachine>(
      target->createTargetMachine(
          triple,
          opts.gpu_arch,   // CPU = sm_XX
          "+ptx75",        // features
          topts,
          Reloc::PIC_,
          std::nullopt,
          cg_opt));
  if (!tm)
    return make_error<StringError>(
        "Failed to create NVPTX TargetMachine", inconvertibleErrorCode());
  return tm;
}

// ── MLIR module → LLVM IR module ──────────────────────────────────────────
static Expected<std::unique_ptr<llvm::Module>>
toLLVM(ModuleOp mlirMod) {
  MLIRContext *ctx = mlirMod.getContext();
  registerGPUDialectTranslation(*ctx);
  registerNVVMDialectTranslation(*ctx);

  llvm::LLVMContext llvmCtx;
  auto llvmMod = translateModuleToLLVMIR(mlirMod, llvmCtx);
  if (!llvmMod)
    return make_error<StringError>(
        "MLIR→LLVM IR translation failed", inconvertibleErrorCode());
  return llvmMod;
}

// ── Compile LLVM module → PTX string ──────────────────────────────────────
static Expected<std::string>
llvmToPTX(llvm::Module &llvmMod, const PTXOptions &opts) {
  auto tmOrErr = makeTargetMachine(opts);
  if (!tmOrErr) return tmOrErr.takeError();
  auto &tm = *tmOrErr;

  llvmMod.setDataLayout(tm->createDataLayout());

  std::string ptx;
  raw_string_ostream os(ptx);
  legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, os, nullptr,
                               CodeGenFileType::AssemblyFile))
    return make_error<StringError>(
        "Cannot emit PTX assembly", inconvertibleErrorCode());
  pm.run(llvmMod);
  os.flush();
  return ptx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

Expected<std::string>
compileToPTX(ModuleOp mod, const PTXOptions &opts) {
  LUNARA_LOG(info) << "Compiling to PTX (arch=" << opts.gpu_arch << ")";

  auto llvmModOrErr = toLLVM(mod);
  if (!llvmModOrErr) return llvmModOrErr.takeError();

  return llvmToPTX(**llvmModOrErr, opts);
}

Expected<std::vector<char>>
compileToCUBIN(ModuleOp mod, const PTXOptions &opts) {
  auto ptxOrErr = compileToPTX(mod, opts);
  if (!ptxOrErr) return ptxOrErr.takeError();

  if (opts.ptxas_path.empty())
    return make_error<StringError>(
        "ptxas_path must be set to compile to cubin",
        inconvertibleErrorCode());

  // Write PTX to tmp file
  SmallString<128> ptx_path, cubin_path;
  sys::fs::createTemporaryFile("lunara_ptx", ".ptx", ptx_path);
  sys::fs::createTemporaryFile("lunara_cubin", ".cubin", cubin_path);

  {
    std::ofstream f(ptx_path.str().str());
    f << *ptxOrErr;
  }

  // Run ptxas
  std::string ptxas = opts.ptxas_path;
  std::vector<StringRef> args{ptxas,
      "-arch", opts.gpu_arch,
      "-o", cubin_path,
      ptx_path};
  std::string errMsg;
  int rc = sys::ExecuteAndWait(ptxas, args,
                                std::nullopt, {}, 0, 0, &errMsg);
  sys::fs::remove(ptx_path);
  if (rc != 0) {
    sys::fs::remove(cubin_path);
    return make_error<StringError>(
        "ptxas failed: " + errMsg, inconvertibleErrorCode());
  }

  // Read cubin
  std::ifstream f(cubin_path.str().str(), std::ios::binary);
  std::vector<char> cubin((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
  sys::fs::remove(cubin_path);
  LUNARA_LOG(info) << "cubin size: " << cubin.size() << " bytes";
  return cubin;
}

Error writePTX(const std::string &ptx, const std::string &path) {
  std::ofstream f(path);
  if (!f) return make_error<StringError>(
      "Cannot write PTX to " + path, inconvertibleErrorCode());
  f << ptx;
  return Error::success();
}

Expected<void*> loadCUBIN(const std::vector<char> &cubin) {
#ifdef LUNARA_ENABLE_CUDA
  CUmodule mod;
  CUresult rc = cuModuleLoadData(&mod, cubin.data());
  if (rc != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorString(rc, &str);
    return make_error<StringError>(
        std::string("cuModuleLoadData failed: ") + str,
        inconvertibleErrorCode());
  }
  return static_cast<void*>(mod);
#else
  (void)cubin;
  return make_error<StringError>(
      "CUDA support not compiled in", inconvertibleErrorCode());
#endif
}

void unloadCUBIN(void *handle) {
#ifdef LUNARA_ENABLE_CUDA
  if (handle) cuModuleUnload(static_cast<CUmodule>(handle));
#else
  (void)handle;
#endif
}

} // namespace codegen
} // namespace lunara
