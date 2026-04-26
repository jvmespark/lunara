#pragma once
// lunara/frontend/onnx_importer.h
// Ingests an ONNX model and produces an MLIR module in the StableHLO dialect.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include <string>

namespace lunara {
namespace frontend {

/// Options that control ONNX import behaviour.
struct ImportOptions {
  bool infer_shapes      = true;   ///< Run ONNX shape inference before import
  bool fold_constants    = true;   ///< Constant-fold during import
  bool verbose           = false;
  int  opset_version     = 17;
};

/// Import an ONNX protobuf model file and lower it to StableHLO inside
/// \p context.  Returns a OwningOpRef<ModuleOp> on success.
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
importONNXModel(mlir::MLIRContext &context,
                const std::string &onnx_path,
                const ImportOptions &opts = {});

/// Import from an in-memory ONNX serialised bytes.
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
importONNXBytes(mlir::MLIRContext &context,
                llvm::ArrayRef<char> bytes,
                const ImportOptions &opts = {});

} // namespace frontend
} // namespace lunara
