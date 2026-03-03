//======- GemminiToLLVMIRTranslation.cpp - Translate Gemmini to LLVM IR--====//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the Gemmini dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace buddy;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the Gemmini dialect to LLVM IR.
class GemminiDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName.starts_with("gemmini.intr.")) {
      llvm::Module *module = moduleTranslation.getLLVMModule();
      std::string intrinsicName = "llvm.riscv.";
      llvm::StringRef mnemonic = opName.drop_front(13);
      for (char c : mnemonic) {
        if (c == '_')
          intrinsicName += '.';
        else
          intrinsicName += c;
      }

      llvm::SmallVector<llvm::Type *> argTypes;
      llvm::SmallVector<llvm::Value *> args;
      for (Value operand : op->getOperands()) {
        llvm::Value *value = moduleTranslation.lookupValue(operand);
        args.push_back(value);
        argTypes.push_back(value->getType());
      }

      llvm::FunctionType *ftype =
          llvm::FunctionType::get(builder.getVoidTy(), argTypes, false);
      llvm::FunctionCallee callee =
          module->getOrInsertFunction(intrinsicName, ftype);
      builder.CreateCall(callee, args);
      return success();
    }

    return failure();
  }
};
} // end namespace

void buddy::registerGemminiDialectTranslation(DialectRegistry &registry) {
  registry.insert<gemmini::GemminiDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, gemmini::GemminiDialect *dialect) {
        dialect->addInterfaces<GemminiDialectLLVMIRTranslationInterface>();
      });
}

void buddy::registerGemminiDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGemminiDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
