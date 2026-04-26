#pragma once
// lunara/dialects/stablehlo_to_linalg.h
// Converts StableHLO dialect ops into MLIR linalg-on-tensors representation.
// This is the lowering step that exposes loop structure needed by
// the tiling and fusion passes.

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir { class RewritePatternSet; class TypeConverter; }

namespace lunara {
namespace dialects {

/// Populate pattern set with StableHLO -> linalg conversion patterns.
/// Covers: dot_general, convolution, reduce, element-wise ops, reshape,
///         transpose, broadcast, compare, select, scatter, gather.
void populateStableHLOToLinalgPatterns(
    mlir::RewritePatternSet &patterns,
    mlir::TypeConverter     &typeConverter);

/// Pass: lower an entire module from StableHLO to linalg + arith + tensor.
std::unique_ptr<mlir::Pass> createConvertStableHLOToLinalgPass();

/// Pass: canonicalise and DCE after lowering.
std::unique_ptr<mlir::Pass> createLinalgCleanupPass();

} // namespace dialects
} // namespace lunara
