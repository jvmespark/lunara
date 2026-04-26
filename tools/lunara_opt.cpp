// tools/lunara-opt/lunara_opt.cpp
// Main compiler driver:  ONNX/MLIR  →  optimised MLIR  →  Triton Python  →  PTX
//
// Usage examples:
//   lunara-opt model.onnx --emit=triton --out=kernels/
//   lunara-opt model.mlir --emit=ptx --arch=sm_80 --out=model.ptx
//   lunara-opt model.onnx --print-pipeline --opt=2

#include "lunara/frontend/onnx_importer.h"
#include "lunara/passes/pipeline.h"
#include "lunara/codegen/ptx_backend.h"
#include "lunara/utils/logging.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

// ── CLI options ───────────────────────────────────────────────────────────────
static cl::opt<std::string> InputFile(
    cl::Positional, cl::desc("<input .onnx or .mlir>"), cl::Required);

static cl::opt<std::string> OutputDir(
    "out", cl::desc("Output directory / file"), cl::init("lunara_out"));

static cl::opt<std::string> EmitFormat(
    "emit",
    cl::desc("Output format: mlir | triton | ptx | cubin"),
    cl::init("triton"));

static cl::opt<std::string> GpuArch(
    "arch", cl::desc("GPU architecture (e.g. sm_80)"), cl::init("sm_80"));

static cl::opt<int> OptLevel(
    "opt", cl::desc("Optimisation level 0-3"), cl::init(2));

static cl::opt<bool> PrintPipeline(
    "print-pipeline", cl::desc("Print the pass pipeline and exit"),
    cl::init(false));

static cl::opt<bool> PrintIRAfterEach(
    "print-ir-after-each",
    cl::desc("Print MLIR IR after each pass"),
    cl::init(false));

static cl::opt<bool> Verbose(
    "v", cl::desc("Verbose logging"), cl::init(false));

// ── Helpers ───────────────────────────────────────────────────────────────────

static OwningOpRef<ModuleOp>
parseInputFile(MLIRContext &ctx, const std::string &path) {
  // ONNX path
  if (path.size() > 5 && path.substr(path.size() - 5) == ".onnx") {
    LUNARA_LOG(info) << "Importing ONNX model: " << path;
    auto modOrErr = lunara::frontend::importONNXModel(ctx, path);
    if (!modOrErr) {
      llvm::errs() << "ONNX import failed: "
                   << toString(modOrErr.takeError()) << "\n";
      return {};
    }
    return std::move(*modOrErr);
  }

  // MLIR path
  LUNARA_LOG(info) << "Parsing MLIR file: " << path;
  std::string errMsg;
  auto srcMgr = std::make_shared<SourceMgr>();
  auto fileOrErr = openInputFile(path, &errMsg);
  if (!fileOrErr) {
    llvm::errs() << "Cannot open file: " << errMsg << "\n";
    return {};
  }
  srcMgr->AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  return parseSourceFile<ModuleOp>(srcMgr, &ctx);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  registerAllDialects(getGlobalDialectRegistry());
  registerAllPasses();
  cl::ParseCommandLineOptions(argc, argv, "Lunara ML Compiler\n");

  // ── Build pipeline options ─────────────────────────────────────────────────
  lunara::passes::PipelineOptions opts;
  opts.opt_level           = OptLevel;
  opts.print_ir_after_each = PrintIRAfterEach;
  opts.enable_triton       = (EmitFormat != "ptx");

  if (PrintPipeline) {
    MLIRContext ctx;
    PassManager pm(&ctx);
    lunara::passes::buildLunaraPipeline(pm, opts);
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  // ── Setup context ──────────────────────────────────────────────────────────
  DialectRegistry registry;
  registerAllDialects(registry);
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // ── Parse input ───────────────────────────────────────────────────────────
  auto mod = parseInputFile(ctx, InputFile);
  if (!mod) return 1;

  // ── Run pipeline ──────────────────────────────────────────────────────────
  if (!lunara::passes::runPipeline(*mod, opts)) {
    llvm::errs() << "Compilation pipeline failed.\n";
    return 1;
  }

  // ── Emit output ───────────────────────────────────────────────────────────
  if (EmitFormat == "mlir") {
    std::string outPath = OutputDir + "/output.mlir";
    std::error_code ec;
    raw_fd_ostream out(outPath, ec);
    if (ec) { llvm::errs() << "Cannot write " << outPath << "\n"; return 1; }
    mod->print(out);
    llvm::outs() << "Written: " << outPath << "\n";

  } else if (EmitFormat == "ptx") {
    lunara::codegen::PTXOptions ptx_opts;
    ptx_opts.gpu_arch = GpuArch;
    ptx_opts.opt_level = OptLevel;
    auto ptxOrErr = lunara::codegen::compileToPTX(*mod, ptx_opts);
    if (!ptxOrErr) {
      llvm::errs() << "PTX codegen failed: "
                   << toString(ptxOrErr.takeError()) << "\n";
      return 1;
    }
    std::string outPath = OutputDir + "/output.ptx";
    if (auto err = lunara::codegen::writePTX(*ptxOrErr, outPath)) {
      llvm::errs() << toString(std::move(err)) << "\n";
      return 1;
    }
    llvm::outs() << "Written: " << outPath << "\n";

  } else if (EmitFormat == "triton") {
    // The Triton emitter is Python-side; the pipeline has annotated
    // the IR with lunara.kernel_kind attributes for the Python driver.
    llvm::outs() << "Run:  python -m lunara.driver --ir " 
                 << OutputDir << "/output.mlir\n";
    // Also dump the optimised MLIR for the Python driver.
    std::error_code ec;
    std::string outPath = OutputDir + "/output.mlir";
    raw_fd_ostream out(outPath, ec);
    mod->print(out);
    llvm::outs() << "Intermediate MLIR written: " << outPath << "\n";
  } else {
    llvm::errs() << "Unknown emit format: " << EmitFormat << "\n";
    return 1;
  }

  return 0;
}
