//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation ---===//
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value"))
      .getInt();
}

acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}

// Pack spad address: rows << (addrLen+16) | cols << addrLen | spAddr
Value packSpadAddr(Value rows, Value cols, Value spAddr, int64_t addrLen,
                   Location loc, ConversionPatternRewriter &rewriter) {
  Value addrLenPlus16 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(addrLen + 16));
  Value addrLenConst = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(addrLen));
  Value rowsShifted = rewriter.create<arith::ShLIOp>(loc, rows, addrLenPlus16);
  Value colsShifted = rewriter.create<arith::ShLIOp>(loc, cols, addrLenConst);
  Value or1 = rewriter.create<arith::OrIOp>(loc, rowsShifted, colsShifted);
  // Mask spAddr to 32 bits to prevent sign-extension issues on RV64.
  // Constants like dSpAddrStart (0x80000000) have bit 31 set, which causes
  // the RV64 lui instruction to sign-extend to 0xFFFFFFFF80000000,
  // corrupting the rows/cols fields packed in the upper 32 bits.
  Value mask = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(0xFFFFFFFF));
  Value maskedAddr = rewriter.create<arith::AndIOp>(loc, spAddr, mask);
  return rewriter.create<arith::OrIOp>(loc, or1, maskedAddr);
}

template <typename IntrOp = Mvin_IntrOp>
void gemminiMvinOffset(const Value &mem, Value offset, Value spAddr, Value cols,
                       Value rows, int64_t addrLen,
                       ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offset);
  Value spad = packSpadAddr(rows, cols, spAddr, addrLen, loc, rewriter);
  rewriter.create<IntrOp>(loc, configPtr, spad);
}

void gemminiMvoutOffset(const Value &mem, Value offset, Value spAddr,
                        Value cols, Value rows, int64_t addrLen,
                        ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offset);
  Value spad = packSpadAddr(rows, cols, spAddr, addrLen, loc, rewriter);
  rewriter.create<Mvout_IntrOp>(loc, configPtr, spad);
}

} // namespace

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct GemminiFlushLowering : public ConvertOpToLLVMPattern<FlushOp> {
  using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rs2Attr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(flushOp, skip, rs2);
    return success();
  }
};

struct GemminiConfigStLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
  using ConvertOpToLLVMPattern<ConfigStOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    float scale = configStOp.getScale().convertToFloat();
    Location loc = configStOp.getLoc();
    uint64_t rs1 = ((uint64_t)configStOp.getActivation() << 2) | CONFIG_ST;
    Value value1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    // rs2 = acc_scale_bits << 32 | (uint32_t)stride
    uint64_t scaleBits =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale);
    Value scaleBitsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(scaleBits));
    Value shift32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(32));
    Value scaleShifted =
        rewriter.create<arith::ShLIOp>(loc, scaleBitsVal, shift32);
    Value mask32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(0xFFFFFFFF));
    Value strideMasked =
        rewriter.create<arith::AndIOp>(loc, strideValue, mask32);
    Value value2 =
        rewriter.create<arith::OrIOp>(loc, scaleShifted, strideMasked);
    rewriter.replaceOpWithNewOp<ConfigSt_IntrOp>(configStOp, value1, value2);
    return success();
  }
};

struct GemminiConfigLdLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
  using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
  explicit GemminiConfigLdLowering(LLVMTypeConverter &typeConverter,
                                   int64_t dim)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim) {}
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t blockMvinStride = configLdOp.getBlockMvinStride();
    if (blockMvinStride == (uint64_t)-1)
      blockMvinStride = dim;
    uint64_t pixelRepeats = configLdOp.getPixelRepeats();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   (blockMvinStride << 16) | pixelRepeats << 8 |
                   configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 |
                   CONFIG_LD;
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    rewriter.replaceOpWithNewOp<ConifgLd_IntrOp>(configLdOp, rs1value,
                                                 rs2Value);
    return success();
  }

private:
  int64_t dim;
};

struct GemminiConfigExLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
  using ConvertOpToLLVMPattern<ConfigExOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSetOnlyStrides() << 7 |
        configExOp.getSysAct() << 3 | configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.replaceOpWithNewOp<ConfigEX_IntrOp>(configExOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiConfigNormLowering : public ConvertOpToLLVMPattern<ConfigNormOp> {
  using ConvertOpToLLVMPattern<ConfigNormOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigNormOp configNormOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = configNormOp.getLoc();
    uint64_t rs1 = (((uint64_t)((uint32_t)configNormOp.getQConst())) << 32) |
                   (configNormOp.getQConstType() & 1) << 18 |
                   (configNormOp.getSetStatsIdOnly() & 1) << 17 |
                   (configNormOp.getActMsb() & 1) << 16 |
                   configNormOp.getStatsId() << 8 | CONFIG_BERT;
    uint64_t rs2 = (((uint64_t)((uint32_t)configNormOp.getIgeluQc())) << 32) |
                   ((uint64_t)((uint32_t)configNormOp.getIgeluQb()));
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ConfigNorm_IntrOp>(configNormOp, rs1Value,
                                                   rs2Value);
    return success();
  }
};

struct GemminiMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  explicit GemminiMvinLowering(LLVMTypeConverter &typeConverter,
                               int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        dyn_cast<MemRefType>(mvinOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    Value rows = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[0]));
    Value cols = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[1]));
    Value spad = packSpadAddr(rows, cols, spadAddrValue, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, indexCastOp, spad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvin2Lowering : public ConvertOpToLLVMPattern<Mvin2Op> {
  using ConvertOpToLLVMPattern<Mvin2Op>::ConvertOpToLLVMPattern;
  explicit GemminiMvin2Lowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin2Op mvin2Op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvin2Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        dyn_cast<MemRefType>(mvin2Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvin2Op.getAddr();
    Value rows = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[0]));
    Value cols = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[1]));
    Value spad = packSpadAddr(rows, cols, spadAddrValue, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Mvin2_IntrOp>(mvin2Op, indexCastOp, spad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvin3Lowering : public ConvertOpToLLVMPattern<Mvin3Op> {
  using ConvertOpToLLVMPattern<Mvin3Op>::ConvertOpToLLVMPattern;
  explicit GemminiMvin3Lowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(Mvin3Op mvin3Op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvin3Op.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        dyn_cast<MemRefType>(mvin3Op.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvin3Op.getAddr();
    Value rows = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[0]));
    Value cols = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[1]));
    Value spad = packSpadAddr(rows, cols, spadAddrValue, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Mvin3_IntrOp>(mvin3Op, indexCastOp, spad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  explicit GemminiMvoutLowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = mvoutOp.getLoc();
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddr = mvoutOp.getAddr();
    MemRefType memRefType =
        dyn_cast<MemRefType>(mvoutOp.getOperandTypes().front());
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    Value rows = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[0]));
    Value cols = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(memRefShape[1]));
    Value newSpad = packSpadAddr(rows, cols, spadAddr, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, indexCastOp, newSpad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiPreloadZerosLowering
    : public ConvertOpToLLVMPattern<PreloadZerosOp> {
  using ConvertOpToLLVMPattern<PreloadZerosOp>::ConvertOpToLLVMPattern;
  explicit GemminiPreloadZerosLowering(LLVMTypeConverter &typeConverter,
                                       int64_t dim, int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadZerosOp preloadZerosOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value addr = preloadZerosOp.getAddr();
    Value cRows = preloadZerosOp.getCRows();
    Value cCols = preloadZerosOp.getCCols();
    Location loc = preloadZerosOp.getLoc();
    // rs1 is all static: dim rows, dim cols, garbage addr (-1)
    uint64_t rs1 = (uint64_t)dim << (addrLen + 16) | (uint64_t)dim << addrLen |
                   (uint64_t)-1;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    // rs2 uses dynamic values
    Value rs2Value = packSpadAddr(cRows, cCols, addr, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadZerosOp, rs1Value,
                                                rs2Value);
    return success();
  }

private:
  int64_t dim;
  int64_t addrLen;
};

struct GemminiPreloadLowering : public ConvertOpToLLVMPattern<PreloadOp> {
  using ConvertOpToLLVMPattern<PreloadOp>::ConvertOpToLLVMPattern;
  explicit GemminiPreloadLowering(LLVMTypeConverter &typeConverter,
                                  int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadOp preloadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value bdAddr = preloadOp.getBdAddr();
    Value cAddr = preloadOp.getCAddr();
    Value bdCols = preloadOp.getBdCols();
    Value bdRows = preloadOp.getBdRows();
    Value cCols = preloadOp.getCCols();
    Value cRows = preloadOp.getCRows();
    Location loc = preloadOp.getLoc();
    Value rs1 = packSpadAddr(bdRows, bdCols, bdAddr, addrLen, loc, rewriter);
    Value rs2 = packSpadAddr(cRows, cCols, cAddr, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadOp, rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputePreloadedLowering
    : public ConvertOpToLLVMPattern<ComputePreloadedOp> {
  using ConvertOpToLLVMPattern<ComputePreloadedOp>::ConvertOpToLLVMPattern;
  explicit GemminiComputePreloadedLowering(LLVMTypeConverter &typeConverter,
                                           int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputePreloadedOp computePreloadedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computePreloadedOp.getAAddr();
    Value bdAddr = computePreloadedOp.getBdAddr();
    Value aRows = computePreloadedOp.getARows();
    Value aCols = computePreloadedOp.getACols();
    Value bdRows = computePreloadedOp.getBdRows();
    Value bdCols = computePreloadedOp.getBdCols();
    Location loc = computePreloadedOp.getLoc();
    Value rs1 = packSpadAddr(aRows, aCols, aAddr, addrLen, loc, rewriter);
    Value rs2 = packSpadAddr(bdRows, bdCols, bdAddr, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<ComputePreloaded_IntrOp>(computePreloadedOp,
                                                         rs1, rs2);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputeAccumulatedLowering
    : public ConvertOpToLLVMPattern<ComputeAccumulatedOp> {
  using ConvertOpToLLVMPattern<ComputeAccumulatedOp>::ConvertOpToLLVMPattern;
  explicit GemminiComputeAccumulatedLowering(LLVMTypeConverter &typeConverter,
                                             int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputeAccumulatedOp computeAccumulatedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computeAccumulatedOp.getAAddr();
    Value bdAddr = computeAccumulatedOp.getBdAddr();
    Value aRows = computeAccumulatedOp.getARows();
    Value aCols = computeAccumulatedOp.getACols();
    Value bdRows = computeAccumulatedOp.getBdRows();
    Value bdCols = computeAccumulatedOp.getBdCols();
    Location loc = computeAccumulatedOp.getLoc();
    Value rs1 = packSpadAddr(aRows, aCols, aAddr, addrLen, loc, rewriter);
    Value rs2 = packSpadAddr(bdRows, bdCols, bdAddr, addrLen, loc, rewriter);
    rewriter.replaceOpWithNewOp<ComputeAccumulated_IntrOp>(
        computeAccumulatedOp, rs1, rs2);

    return success();
  }

private:
  int64_t addrLen;
};

class GemminiTileMatMulLowering : public ConvertOpToLLVMPattern<TileMatMulOp> {
  // i, j, k, padI, padJ, padK are i64 Values (dynamic).
  void gemminiLoopWs(Value i, Value j, Value k, Value padI, Value padJ,
                     Value padK, Value &a, Value &b, Value &d, Value &c,
                     size_t aRowStride, size_t bRowStride, size_t dRowStride,
                     size_t cRowStride, bool aTranspose, bool bTranspose,
                     bool fullC, bool lowD, bool exAccumulate, int act,
                     TileMatMulOp &tileMatMulOp,
                     ConversionPatternRewriter &rewriter) const {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = a.getLoc();
    auto ci = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI64IntegerAttr(v));
    };

    // loopWsConfigBounds instruction.
    // rs1 = padK << 32 | padJ << 16 | padI
    Value rs1Value = rewriter.create<arith::OrIOp>(
        loc,
        rewriter.create<arith::OrIOp>(
            loc, rewriter.create<arith::ShLIOp>(loc, padK, ci(32)),
            rewriter.create<arith::ShLIOp>(loc, padJ, ci(16))),
        padI);
    // rs2 = k << 32 | j << 16 | i
    Value rs2Value = rewriter.create<arith::OrIOp>(
        loc,
        rewriter.create<arith::OrIOp>(
            loc, rewriter.create<arith::ShLIOp>(loc, k, ci(32)),
            rewriter.create<arith::ShLIOp>(loc, j, ci(16))),
        i);
    rewriter.create<LoopWsConfigBounds_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigAddrsAB instruction.
    rewriter.create<LoopWsConfigAddrsAB_IntrOp>(loc, a, b);
    // loopWsConfigAddrsDC instruction
    rewriter.create<LoopWsConfigAddrsDC_IntrOp>(loc, d, c);
    // loopWsConfigStridesAB instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(aRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(bRowStride));
    rewriter.create<LoopWsConfigStridesAB_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigStrideDC instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(dRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(cRowStride));
    rewriter.create<LoopWsConfigStridesDC_IntrOp>(loc, rs1Value, rs2Value);
    uint64_t rs1Static =
        (uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
    uint64_t rs2Static = bTranspose << 1 | aTranspose;
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs1Static));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs2Static));
    rewriter.create<LoopWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  // i, j, k, padI, padJ, padK are i64 Values (dynamic).
  void spTiledMatmulWs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, Value i, Value j, Value k,
                       Value padI, Value padJ, Value padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       ConversionPatternRewriter &rewriter) const {

    gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, d, c, strideA, strideB,
                  repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose,
                  fullC, lowD, !noBias, act, tileMatMulOp, rewriter);
  }

  // Helper: create a 2-level nested scf.for. Returns {outerIV, innerIV,
  // outerLoop}. Leaves insertion point inside the innermost loop body.
  struct NestedLoopInfo {
    Value outerIV, innerIV;
    scf::ForOp outerLoop;
  };
  NestedLoopInfo
  createNestedForLoops(Location loc, int64_t ubI, int64_t stepI, int64_t ubJ,
                       int64_t stepJ,
                       ConversionPatternRewriter &rewriter) const {
    auto idx = [&](int64_t v) {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, idx(0), idx(ubI), idx(stepI));
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, idx(0), idx(ubJ), idx(stepJ));
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    return {outerLoop.getInductionVar(), innerLoop.getInductionVar(),
            outerLoop};
  }

  // Value overload: bounds are already index-typed Values.
  NestedLoopInfo
  createNestedForLoops(Location loc, Value ubI, Value stepI, Value ubJ,
                       Value stepJ,
                       ConversionPatternRewriter &rewriter) const {
    auto idx = [&](int64_t v) {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto outerLoop =
        rewriter.create<scf::ForOp>(loc, idx(0), ubI, stepI);
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop =
        rewriter.create<scf::ForOp>(loc, idx(0), ubJ, stepJ);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    return {outerLoop.getInductionVar(), innerLoop.getInductionVar(),
            outerLoop};
  }

  // Tiling functions
  // i, j, k, padI, padJ, padK are i64 Values (dynamic tile dimensions from
  // the outer scf.for loops).
  void spTiledMatmulOs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, Value i, Value j, Value k,
                       Value padI, Value padJ, Value padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       ConversionPatternRewriter &rewriter) const {
    const int64_t aSpAddrStart = 0;
    const int64_t dSpAddrStart = 1 << (addrLen - 1);
    const int64_t cSpAddrStart =
        (3 << (addrLen - 2)) | (fullC << (addrLen - 3));

    const int64_t maxBlockLen = MAX_BYTES / (dim * 1);
    const int64_t maxBlockLenAcc = MAX_BYTES / (dim * 4);

    Location loc = a.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    bool dAddrNull = llvm::dyn_cast<arith::ConstantOp>(d.getDefiningOp()) &&
                     getNumberFromValue(d) == 0;
    bool cAddrNull = llvm::dyn_cast<arith::ConstantOp>(c.getDefiningOp()) &&
                     getNumberFromValue(c) == 0;

    // Helper to create i64 constant Values
    auto ci = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI64IntegerAttr(v));
    };
    // Helper to cast index to i64
    auto toI64 = [&](Value idx) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, i64Type, idx);
    };
    // Helper to cast i64 to index
    auto toIndex = [&](Value v) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                 v);
    };
    auto idxConst = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };

    // bSpAddrStart = BANK_NUM * bankRows - k * j * dim (dynamic)
    Value bSpAddrStartVal = rewriter.create<arith::SubIOp>(
        loc, ci(BANK_NUM * bankRows),
        rewriter.create<arith::MulIOp>(
            loc, rewriter.create<arith::MulIOp>(loc, k, j), ci(dim)));

    // aBlocks = min(k, maxBlockLen)
    Value aBlocksVal = rewriter.create<arith::SelectOp>(
        loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, k,
                                       ci(maxBlockLen)),
        k, ci(maxBlockLen));
    // bBlocks = min(j, maxBlockLen)
    Value bBlocksVal = rewriter.create<arith::SelectOp>(
        loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, j,
                                       ci(maxBlockLen)),
        j, ci(maxBlockLen));
    // dBlocks = min(j, maxBlockLenAcc)
    Value dBlocksVal = rewriter.create<arith::SelectOp>(
        loc,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, j,
                                       ci(maxBlockLenAcc)),
        j, ci(maxBlockLenAcc));

    // i-1, j-1, k-1 (used in multiple places)
    Value iMinus1 = rewriter.create<arith::SubIOp>(loc, i, ci(1));
    Value jMinus1 = rewriter.create<arith::SubIOp>(loc, j, ci(1));
    Value kMinus1 = rewriter.create<arith::SubIOp>(loc, k, ci(1));

    // Move-in D
    if (!dAddrNull && !noBias) {
      const size_t dStride = repeatingBias ? 0 : strideD * sizeOfAccT;
      rewriter.create<ConfigLdOp>(loc, ci(dStride),
                                  llvm::APFloat((float)dScaleFactor), false, 0);

      auto info = createNestedForLoops(loc, toIndex(i), idxConst(1),
                                       toIndex(j), toIndex(dBlocksVal),
                                       rewriter);
      Value i0 = toI64(info.outerIV);
      Value j0 = toI64(info.innerIV);

      // biasRow = repeatingBias ? 0 : i0
      Value biasRow = repeatingBias ? ci(0) : i0;
      // offset = (biasRow * strideD + j0) * dim * sizeOfAccT
      Value offset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<arith::MulIOp>(loc, biasRow, ci(strideD)),
              j0),
          ci(dim * sizeOfAccT));
      // dSpAddrAcc = dSpAddrStart + (i0 * j + j0) * dim
      Value dSpAddrAcc = rewriter.create<arith::AddIOp>(
          loc, ci(dSpAddrStart),
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::AddIOp>(
                  loc, rewriter.create<arith::MulIOp>(loc, i0, j), j0),
              ci(dim)));
      // blocks = min(dBlocks, j - j0)
      Value jMinusJ0 = rewriter.create<arith::SubIOp>(loc, j, j0);
      Value blocks = rewriter.create<arith::SelectOp>(
          loc,
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                         jMinusJ0, dBlocksVal),
          jMinusJ0, dBlocksVal);
      // cols = blocks * dim - (j0 + blocks >= j ? padJ : 0)
      Value j0PlusBlocks = rewriter.create<arith::AddIOp>(loc, j0, blocks);
      Value atEnd = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, j0PlusBlocks, j);
      Value padJOrZero =
          rewriter.create<arith::SelectOp>(loc, atEnd, padJ, ci(0));
      Value cols = rewriter.create<arith::SubIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, blocks, ci(dim)),
          padJOrZero);
      // rows = dim - (i0 == i-1 ? padI : 0)
      Value isLastI = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, i0, iMinus1);
      Value rows = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastI, padI, ci(0)));

      gemminiMvinOffset(d, offset, dSpAddrAcc, cols, rows, addrLen, rewriter);
      rewriter.setInsertionPointAfter(info.outerLoop);
    }

    // Move-in B
    rewriter.create<ConfigLdOp>(loc, ci(strideB),
                                llvm::APFloat((float)bScaleFactor), false, 0);
    {
      auto info = createNestedForLoops(loc, toIndex(j), toIndex(bBlocksVal),
                                       toIndex(k), idxConst(1), rewriter);
      Value j0 = toI64(info.outerIV);
      Value k0 = toI64(info.innerIV);

      // offset = (k0 * strideB + j0) * dim * sizeOfElemT
      Value offset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<arith::MulIOp>(loc, k0, ci(strideB)), j0),
          ci(dim * sizeOfElemT));
      // bSpAddr = bSpAddrStart + (k0 * j + j0) * dim
      Value bSpAddr = rewriter.create<arith::AddIOp>(
          loc, bSpAddrStartVal,
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::AddIOp>(
                  loc, rewriter.create<arith::MulIOp>(loc, k0, j), j0),
              ci(dim)));
      // blocks = min(bBlocks, j - j0)
      Value jMinusJ0 = rewriter.create<arith::SubIOp>(loc, j, j0);
      Value blocks = rewriter.create<arith::SelectOp>(
          loc,
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                         jMinusJ0, bBlocksVal),
          jMinusJ0, bBlocksVal);
      // cols = blocks * dim - (j0 + blocks >= j ? padJ : 0)
      Value j0PlusBlocks = rewriter.create<arith::AddIOp>(loc, j0, blocks);
      Value atEnd = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, j0PlusBlocks, j);
      Value padJOrZero =
          rewriter.create<arith::SelectOp>(loc, atEnd, padJ, ci(0));
      Value cols = rewriter.create<arith::SubIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, blocks, ci(dim)),
          padJOrZero);
      // rows = dim - (k0 == k-1 ? padK : 0)
      Value isLastK = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, k0, kMinus1);
      Value rows = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastK, padK, ci(0)));

      gemminiMvinOffset(b, offset, bSpAddr, cols, rows, addrLen, rewriter);
      rewriter.setInsertionPointAfter(info.outerLoop);
    }

    // Move-in A
    rewriter.create<ConfigLdOp>(loc, ci(strideA),
                                llvm::APFloat((float)aScaleFactor), false, 0);
    {
      auto info = createNestedForLoops(loc, toIndex(i), idxConst(1),
                                       toIndex(k), toIndex(aBlocksVal),
                                       rewriter);
      Value i0 = toI64(info.outerIV);
      Value k0 = toI64(info.innerIV);

      // offset = (i0 * strideA + k0) * dim * sizeOfElemT
      Value offset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<arith::MulIOp>(loc, i0, ci(strideA)), k0),
          ci(dim * sizeOfElemT));
      // aSpAddr = aSpAddrStart + (i0 * k + k0) * dim
      Value aSpAddr = rewriter.create<arith::AddIOp>(
          loc, ci(aSpAddrStart),
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::AddIOp>(
                  loc, rewriter.create<arith::MulIOp>(loc, i0, k), k0),
              ci(dim)));
      // blocks = min(aBlocks, k - k0)
      Value kMinusK0 = rewriter.create<arith::SubIOp>(loc, k, k0);
      Value blocks = rewriter.create<arith::SelectOp>(
          loc,
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                         kMinusK0, aBlocksVal),
          kMinusK0, aBlocksVal);
      // cols = blocks * dim - (k0 + blocks >= k ? padK : 0)
      Value k0PlusBlocks = rewriter.create<arith::AddIOp>(loc, k0, blocks);
      Value atEnd = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, k0PlusBlocks, k);
      Value padKOrZero =
          rewriter.create<arith::SelectOp>(loc, atEnd, padK, ci(0));
      Value cols = rewriter.create<arith::SubIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, blocks, ci(dim)),
          padKOrZero);
      // rows = dim - (i0 == i-1 ? padI : 0)
      Value isLastI = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, i0, iMinus1);
      Value rows = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastI, padI, ci(0)));

      gemminiMvinOffset(a, offset, aSpAddr, cols, rows, addrLen, rewriter);
      rewriter.setInsertionPointAfter(info.outerLoop);
    }

    // Compute: outer i0 × j0 loops, with k0 peeled for first iteration
    {
      auto outerI = rewriter.create<scf::ForOp>(loc, idxConst(0), toIndex(i),
                                                 idxConst(1));
      rewriter.setInsertionPointToStart(outerI.getBody());
      auto outerJ = rewriter.create<scf::ForOp>(loc, idxConst(0), toIndex(j),
                                                 idxConst(1));
      rewriter.setInsertionPointToStart(outerJ.getBody());

      Value i0 = toI64(outerI.getInductionVar());
      Value j0 = toI64(outerJ.getInductionVar());

      // Common subexpressions for this (i0, j0) pair
      Value cSpAddr = rewriter.create<arith::AddIOp>(
          loc, ci(cSpAddrStart),
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::AddIOp>(
                  loc, rewriter.create<arith::MulIOp>(loc, i0, j), j0),
              ci(dim)));

      // Dimension computations that depend on i0, j0 (not k0)
      Value isLastI = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, i0, iMinus1);
      Value isLastJ = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, j0, jMinus1);
      Value aRows = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastI, padI, ci(0)));
      Value bCols = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastJ, padJ, ci(0)));
      Value cCols = bCols;
      Value cRows = aRows;

      // noBiasNewMatrix mask for outSpAddr
      Value noBiasNewMatrixMask = ci(~((int64_t)1 << (addrLen - 2)));
      bool applyNoBiasMask = noBias && !dAddrNull;

      // Helper to emit preload+compute intrinsics for a given k0 value
      auto emitPreloadCompute = [&](Value k0, bool isFirstK) {
        Value aSpAddr = rewriter.create<arith::AddIOp>(
            loc, ci(aSpAddrStart),
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::AddIOp>(
                    loc, rewriter.create<arith::MulIOp>(loc, i0, k), k0),
                ci(dim)));
        Value bSpAddr = rewriter.create<arith::AddIOp>(
            loc, bSpAddrStartVal,
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::AddIOp>(
                    loc, rewriter.create<arith::MulIOp>(loc, k0, j), j0),
                ci(dim)));

        // outSpAddr = (k0 == k-1) ? cSpAddr : GARBAGE_ADDR
        Value isLastK = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, k0, kMinus1);
        Value outSpAddr = rewriter.create<arith::SelectOp>(
            loc, isLastK, cSpAddr, ci(GARBAGE_ADDR));

        // noBiasNewMatrix: clear bit (addrLen-2)
        if (applyNoBiasMask) {
          Value masked = rewriter.create<arith::AndIOp>(loc, outSpAddr,
                                                        noBiasNewMatrixMask);
          outSpAddr =
              rewriter.create<arith::SelectOp>(loc, isLastK, masked, outSpAddr);
        }

        // aCols, bRows depend on k0
        Value aCols = rewriter.create<arith::SubIOp>(
            loc, ci(dim),
            rewriter.create<arith::SelectOp>(loc, isLastK, padK, ci(0)));
        Value bRows = aCols; // same formula

        Value garbageAddr = ci(GARBAGE_ADDR);
        Value dimOp = ci(dim);

        // Emit dialect-level ops (lowering patterns handle bit-packing)
        rewriter.create<PreloadOp>(loc, garbageAddr, outSpAddr, dimOp, dimOp,
                                   cCols, cRows);
        if (isFirstK) {
          rewriter.create<ComputePreloadedOp>(loc, aSpAddr, bSpAddr, aRows,
                                              aCols, bRows, bCols);
        } else {
          rewriter.create<ComputeAccumulatedOp>(loc, aSpAddr, bSpAddr, aRows,
                                                aCols, bRows, bCols);
        }
      };

      // Peel k0 = 0
      emitPreloadCompute(ci(0), /*isFirstK=*/true);

      // k0 = 1..k-1 (scf.for naturally executes 0 iterations when k==1)
      auto kLoop = rewriter.create<scf::ForOp>(loc, idxConst(1), toIndex(k),
                                                idxConst(1));
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value k0 = toI64(kLoop.getInductionVar());
      emitPreloadCompute(k0, /*isFirstK=*/false);
      rewriter.setInsertionPointAfter(kLoop);

      rewriter.setInsertionPointAfter(outerI);
    }

    // Move-out C
    if (!cAddrNull) {
      const size_t sizeof_C = fullC ? sizeOfAccT : sizeOfElemT;

      auto info = createNestedForLoops(loc, toIndex(i), idxConst(1),
                                       toIndex(j), idxConst(1), rewriter);
      Value i0 = toI64(info.outerIV);
      Value j0 = toI64(info.innerIV);

      // offset = (i0 * strideC + j0) * dim * sizeof_C
      Value offset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<arith::MulIOp>(loc, i0, ci(strideC)), j0),
          ci(dim * sizeof_C));
      // cSpAddr = cSpAddrStart + (i0 * j + j0) * dim
      Value cSpAddr = rewriter.create<arith::AddIOp>(
          loc, ci(cSpAddrStart),
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::AddIOp>(
                  loc, rewriter.create<arith::MulIOp>(loc, i0, j), j0),
              ci(dim)));
      // cCols = dim - (j0 == j-1 ? padJ : 0)
      Value isLastJ = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, j0, jMinus1);
      Value cCols = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastJ, padJ, ci(0)));
      // cRows = dim - (i0 == i-1 ? padI : 0)
      // NOTE: original code had bug: compared i0 == j-1, preserving that
      Value isLastIForRows = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, i0, jMinus1);
      Value cRows = rewriter.create<arith::SubIOp>(
          loc, ci(dim),
          rewriter.create<arith::SelectOp>(loc, isLastIForRows, padI, ci(0)));

      gemminiMvoutOffset(c, offset, cSpAddr, cCols, cRows, addrLen, rewriter);
      rewriter.setInsertionPointAfter(info.outerLoop);
    }
  }

  void tiledMatmulOuter(
      size_t dimI, size_t dimJ, size_t dimK, Value &A, Value &B, Value &D,
      Value &C, size_t strideA, size_t strideB, size_t strideD, size_t strideC,
      scale_t aScaleFactor, scale_t bScaleFactor, scale_acc_t dScaleFactor,
      size_t tileI, size_t tileJ, size_t tileK, int act, acc_scale_t scale,
      acc_scale_t bertScale, bool repeatingBias, bool aTranspose,
      bool bTranspose, bool fullC, bool lowD, uint8_t weightA, int dataflow,
      TileMatMulOp &tileMatMulOp, ConversionPatternRewriter &rewriter) const {
    const size_t dimIPadded = (dimI / dim + (dimI % dim != 0)) * dim;
    const size_t dimJPadded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    const size_t dimKPadded = (dimK / dim + (dimK % dim != 0)) * dim;
    const size_t I0 =
        dimIPadded / (tileI * dim) + (dimIPadded % (tileI * dim) != 0);
    const size_t J0 =
        dimJPadded / (tileJ * dim) + (dimJPadded % (tileJ * dim) != 0);
    const size_t K0 =
        dimKPadded / (tileK * dim) + (dimKPadded % (tileK * dim) != 0);
    const size_t lastI =
        dimIPadded % (tileI * dim) == 0 ? tileI : (dimIPadded / dim) % tileI;
    const size_t lastJ =
        dimJPadded % (tileJ * dim) == 0 ? tileJ : (dimJPadded / dim) % tileJ;
    const size_t lastK =
        dimKPadded % (tileK * dim) == 0 ? tileK : (dimKPadded / dim) % tileK;
    const size_t paddingI = dimIPadded - dimI;
    const size_t paddingJ = dimJPadded - dimJ;
    const size_t paddingK = dimKPadded - dimK;
    const bool noBias = false;
    const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
    const size_t sizeofC = fullC ? sizeOfAccT : sizeOfElemT;
    Location loc = tileMatMulOp.getLoc();
    llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
    rewriter.create<ConfigExOp>(loc, /*dataflow = */ dataflow,
                                /*sysAct = */ act & 3,
                                /* sysShift = */ 0, accScaleIdentity);
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(bScaleFactor),
                                false, 1);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideD * sizeofD));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)dScaleFactor), lowD, 2);

    /*
      Add config norm op
    */
    if (act == IGELU) {
      const float sqrt_2 = 1.41421356237;
      const float S = bertScale;
      const float S_erf = (-0.2888 * ((S * S) / 2));

      const uint32_t qb = -1.769 / (S / sqrt_2);
      const uint32_t qc = 1.0 / S_erf;
      rewriter.create<ConfigNormOp>(loc, 0, 0, 0, 0, 0, qb, qc);
    }

    if (act == SOFTMAX) {
      const float a = 0.3585;
      const float b = 1.353;
      const float c = 0.344;

      const uint32_t qln2 = (int)(0.693147 / bertScale);
      const uint32_t qln2_inv = 65536 / qln2;
      const uint32_t qb = b / bertScale;
      const uint32_t qc = c / (a * bertScale * bertScale);
      rewriter.create<ConfigNormOp>(loc, qln2, 0, 0, 1, 0, qb, qc);
      rewriter.create<ConfigNormOp>(loc, qln2_inv, 1, 0, 1, 0, qb, qc);
    }

    // Helpers for building arith ops inside the outer tiling loops
    IntegerType i64Type = rewriter.getI64Type();
    auto ci = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc,
                                                rewriter.getI64IntegerAttr(v));
    };
    auto toI64 = [&](Value idx) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, i64Type, idx);
    };
    auto idxConst = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };

    // Outer tiling loops: i0 in [0, I0), j0 in [0, J0), k0 in [0, K0)
    auto i0Loop =
        rewriter.create<scf::ForOp>(loc, idxConst(0), idxConst(I0), idxConst(1));
    rewriter.setInsertionPointToStart(i0Loop.getBody());

    auto j0Loop =
        rewriter.create<scf::ForOp>(loc, idxConst(0), idxConst(J0), idxConst(1));
    rewriter.setInsertionPointToStart(j0Loop.getBody());

    auto k0Loop =
        rewriter.create<scf::ForOp>(loc, idxConst(0), idxConst(K0), idxConst(1));
    rewriter.setInsertionPointToStart(k0Loop.getBody());

    Value i0v = toI64(i0Loop.getInductionVar());
    Value j0v = toI64(j0Loop.getInductionVar());
    Value k0v = toI64(k0Loop.getInductionVar());

    // pre = (k0 == 0) ? D + biasOffset : 0
    {
      Value isFirstK = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, k0v, ci(0));
      // biasRow = repeatingBias ? 0 : i0 * tileI * dim
      Value biasRow =
          repeatingBias
              ? ci(0)
              : rewriter.create<arith::MulIOp>(loc, i0v, ci(tileI * dim));
      // offset = (biasRow * strideD + j0 * tileJ * dim) * sizeofD
      Value preOffset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<arith::MulIOp>(loc, biasRow, ci(strideD)),
              rewriter.create<arith::MulIOp>(loc, j0v, ci(tileJ * dim))),
          ci(sizeofD));
      Value preAddr =
          rewriter.create<arith::AddIOp>(loc, i64Type, D, preOffset);
      Value pre =
          rewriter.create<arith::SelectOp>(loc, isFirstK, ci(0), preAddr);

      // out = (k0 == K0-1) ? C + outOffset : 0
      Value isLastK0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, k0v, ci(K0 - 1));
      // offset = (i0 * tileI * dim * strideC + j0 * tileJ * dim) * sizeofC
      Value outOffset = rewriter.create<arith::MulIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc,
              rewriter.create<arith::MulIOp>(loc, i0v, ci(tileI * dim * strideC)),
              rewriter.create<arith::MulIOp>(loc, j0v, ci(tileJ * dim))),
          ci(sizeofC));
      Value outAddr =
          rewriter.create<arith::AddIOp>(loc, i64Type, C, outOffset);
      Value out =
          rewriter.create<arith::SelectOp>(loc, isLastK0, outAddr, ci(0));

      // i = (i0 < I0-1) ? tileI : lastI  (and similarly for j, k)
      Value isNotLastI0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, i0v, ci(I0 - 1));
      Value isNotLastJ0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, j0v, ci(J0 - 1));
      Value isNotLastK0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, k0v, ci(K0 - 1));
      Value iVal = rewriter.create<arith::SelectOp>(loc, isNotLastI0,
                                                     ci(tileI), ci(lastI));
      Value jVal = rewriter.create<arith::SelectOp>(loc, isNotLastJ0,
                                                     ci(tileJ), ci(lastJ));
      Value kVal = rewriter.create<arith::SelectOp>(loc, isNotLastK0,
                                                     ci(tileK), ci(lastK));

      // padI = (i0 == I0-1) ? paddingI : 0  (and similarly for j, k)
      Value isLastI0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, i0v, ci(I0 - 1));
      Value isLastJ0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, j0v, ci(J0 - 1));
      Value padIVal = rewriter.create<arith::SelectOp>(loc, isLastI0,
                                                        ci(paddingI), ci(0));
      Value padJVal = rewriter.create<arith::SelectOp>(loc, isLastJ0,
                                                        ci(paddingJ), ci(0));
      Value padKVal = rewriter.create<arith::SelectOp>(loc, isLastK0,
                                                        ci(paddingK), ci(0));

      // Compute pointer `a`
      Value aOffset;
      if (aTranspose) {
        // offset = (k0 * tileK * dim * strideA + i0 * tileI * dim) * sizeOfElemT
        aOffset = rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, k0v, ci(tileK * dim * strideA)),
                rewriter.create<arith::MulIOp>(loc, i0v, ci(tileI * dim))),
            ci(sizeOfElemT));
      } else {
        // offset = (i0 * tileI * dim * strideA + k0 * tileK * dim) * sizeOfElemT
        aOffset = rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, i0v, ci(tileI * dim * strideA)),
                rewriter.create<arith::MulIOp>(loc, k0v, ci(tileK * dim))),
            ci(sizeOfElemT));
      }
      Value aPtr = rewriter.create<arith::AddIOp>(loc, i64Type, A, aOffset);

      // Compute pointer `b`
      Value bOffset;
      if (bTranspose) {
        // offset = (j0 * tileJ * dim * strideB + k0 * tileK * dim) * sizeOfElemT
        bOffset = rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, j0v, ci(tileJ * dim * strideB)),
                rewriter.create<arith::MulIOp>(loc, k0v, ci(tileK * dim))),
            ci(sizeOfElemT));
      } else {
        // offset = (k0 * tileK * dim * strideB + j0 * tileJ * dim) * sizeOfElemT
        bOffset = rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::AddIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, k0v, ci(tileK * dim * strideB)),
                rewriter.create<arith::MulIOp>(loc, j0v, ci(tileJ * dim))),
            ci(sizeOfElemT));
      }
      Value bPtr = rewriter.create<arith::AddIOp>(loc, i64Type, B, bOffset);

      if (dataflow == OUTPUT_STATIONARY) {
        spTiledMatmulOs(aPtr, bPtr, pre, out, aScaleFactor, bScaleFactor,
                        dScaleFactor, iVal, jVal, kVal, padIVal, padJVal,
                        padKVal, strideA, strideB, strideD, strideC,
                        aTranspose, bTranspose, fullC, lowD, noBias,
                        repeatingBias, act, tileMatMulOp, rewriter);
      } else { // WS
        spTiledMatmulWs(aPtr, bPtr, pre, out, aScaleFactor, bScaleFactor,
                        dScaleFactor, iVal, jVal, kVal, padIVal, padJVal,
                        padKVal, strideA, strideB, strideD, strideC,
                        aTranspose, bTranspose, fullC, lowD, noBias,
                        repeatingBias, act, tileMatMulOp, rewriter);
      }
    }
    rewriter.setInsertionPointAfter(i0Loop);
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileMatMulOp, flushValue,
                                              flushValue);
    return;
  }

  size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
    return (I * K + K * J) * dim;
  }

  size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
    return (I * J) * dim;
  }

public:
  using ConvertOpToLLVMPattern<TileMatMulOp>::ConvertOpToLLVMPattern;
  explicit GemminiTileMatMulLowering(LLVMTypeConverter &typeConverter,
                                     int64_t dim, int64_t addrLen,
                                     int64_t accRows, int64_t bankRows,
                                     size_t sizeOfElemT, size_t sizeOfAccT)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen),
        accRows(accRows), bankRows(bankRows), sizeOfElemT(sizeOfElemT),
        sizeOfAccT(sizeOfAccT) {}
  LogicalResult
  matchAndRewrite(TileMatMulOp tileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    size_t dbPartitionRows = ((BANK_NUM * bankRows / 2) / 2);
    size_t dbMatsInPartition = (dbPartitionRows / dim);
    size_t dbMatsInAcc((accRows / 2) / dim);
    size_t dbMaxTileIJ((size_t)sqrt(dbMatsInAcc));
    size_t dbMaxTileK(dbMatsInPartition / dbMaxTileIJ);

    Value aArray = tileMatMulOp.getAArray();
    Value bArray = tileMatMulOp.getBArray();
    Value cArray = tileMatMulOp.getCArray();
    Value dArray = tileMatMulOp.getDArray();
    MemRefType aArrayType = dyn_cast<MemRefType>(aArray.getType());
    MemRefType bArrayType = dyn_cast<MemRefType>(bArray.getType());
    MemRefType cArrayType = dyn_cast<MemRefType>(cArray.getType());
    MemRefType dArrayType = dyn_cast<MemRefType>(dArray.getType());
    StridedLayoutAttr aArrayLayout =
        dyn_cast<StridedLayoutAttr>(aArrayType.getLayout());
    StridedLayoutAttr bArrayLayout =
        dyn_cast<StridedLayoutAttr>(bArrayType.getLayout());
    StridedLayoutAttr cArrayLayout =
        dyn_cast<StridedLayoutAttr>(cArrayType.getLayout());
    SmallVector<Type> resultType = {rewriter.getIndexType()};
    TypeRange typeRange(resultType);
    Location loc = tileMatMulOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value aArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                aArray);
    if (aArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, aArrayLayout.getOffset() * sizeOfElemT);
      aArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, aArrayExtractOp, offset);
    }
    Value aArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
    Value bArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                bArray);
    if (bArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, bArrayLayout.getOffset() * sizeOfElemT);
      bArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, bArrayExtractOp, offset);
    }
    Value bArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
    Value cArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                cArray);
    if (cArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, cArrayLayout.getOffset() * sizeOfElemT);
      cArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, cArrayExtractOp, offset);
    }
    Value cArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
    Value dArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                dArray);
    Value dArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);
    llvm::ArrayRef<int64_t> aArrayShape = aArrayType.getShape();
    llvm::ArrayRef<int64_t> bArrayShape = bArrayType.getShape();
    llvm::ArrayRef<int64_t> cArrayShape = cArrayType.getShape();
    llvm::ArrayRef<int64_t> dArrayShape = dArrayType.getShape();
    size_t dimI = aArrayShape[0];
    size_t dimK = aArrayShape[1];
    size_t dimJ = bArrayShape[1];
    size_t strideA = aArrayShape[1];
    size_t strideB = bArrayShape[1];
    size_t strideC = cArrayShape[1];
    size_t strideD = dArrayShape[1];
    scale_t aScaleFactor = tileMatMulOp.getAScaleFactor().convertToFloat();
    scale_t bScaleFactor = tileMatMulOp.getBScaleFactor().convertToFloat();
    scale_acc_t dScaleFactor = tileMatMulOp.getDScaleFactor().convertToFloat();
    int act = tileMatMulOp.getAct();
    acc_scale_t scale = tileMatMulOp.getAccScale().convertToFloat();
    acc_scale_t bertScale = tileMatMulOp.getBertScale().convertToFloat();
    bool repeatingBias = tileMatMulOp.getRepeatingBias();
    bool aTranspose = tileMatMulOp.getATranspose();
    bool bTranspose = tileMatMulOp.getBTranspose();
    bool fullC = tileMatMulOp.getFullC();
    bool lowD = tileMatMulOp.getLowD();
    uint8_t weightA = tileMatMulOp.getWeightA();
    size_t dimIPaded = (dimI / dim + (dimI % dim != 0)) * dim;
    size_t dimJPaded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    size_t dimKPaded = (dimK / dim + (dimK % dim != 0)) * dim;
    size_t maxSpadRows = BANK_NUM * bankRows / 2;
    size_t maxAccRows = accRows / 2;
    size_t tileI, tileJ, tileK;
    if (act == LAYERNORM || act == SOFTMAX) {
      tileI = 1;
      tileJ = dimJPaded / dim;
      tileK = 1;
    } else {
      tileI = dimIPaded / dim < dbMaxTileIJ ? dimIPaded / dim : dbMaxTileIJ;
      tileJ = dimJPaded / dim < dbMaxTileIJ ? dimJPaded / dim : dbMaxTileIJ;
      tileK = dimKPaded / dim < dbMaxTileK ? dimKPaded / dim : dbMaxTileK;
    }
    while (true) {
      bool increased = false;

      if (tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * dim <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * dim <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows &&
          (tileK + 1) * dim <= dimKPaded) {
        tileK++;
        increased = true;
      }
      if (!increased)
        break;
    }
    int dataflow = tileMatMulOp.getDataflow();

    tiledMatmulOuter(dimI, dimJ, dimK, aArrayindexCastOp, bArrayindexCastOp,
                     dArrayindexCastOp, cArrayindexCastOp, strideA, strideB,
                     strideD, strideC, aScaleFactor, bScaleFactor, dScaleFactor,
                     tileI, tileJ, tileK, act, scale, bertScale, repeatingBias,
                     aTranspose, bTranspose, fullC, lowD, weightA, dataflow,
                     tileMatMulOp, rewriter);
    return success();
  };

private:
  int64_t dim;
  int64_t addrLen;
  int64_t accRows;
  int64_t bankRows;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
};

class GemminiTileConvLowering : public ConvertOpToLLVMPattern<TileConvOp> {

  void gemminiLoopConvWs(
      int batchSize, int inDim, int inChannels, int outChannels, int outDim,
      int poolOutDim, int stride, int padding, int kernelDim,
      int kernelDilation, int poolSize, int poolStride, int poolPadding,
      int batches, int porows, int pocols, int pochs, int krows, int kcols,
      int kchs, int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
      int pupad, int pdpad, int orows, int ocols, Value &weights, Value &output,
      Value &bias, Value &input, bool noBias, bool noPool, bool downsample,
      bool writ180, bool inputDilated, int act, bool transOutput1203,
      bool transWeight1203, bool transWeight0132, bool transInput3120,
      int maxPixelsPerRow, bool dw, TileConvOp &tileConvOp,
      ConversionPatternRewriter &rewriter) const {
    Location loc = tileConvOp.getLoc();
    // loopConvWsConfig1
    uint64_t rs1 = (uint64_t)outChannels << 48 | (uint64_t)inChannels << 32 |
                   (uint64_t)inDim << 16 | (uint64_t)batchSize;
    uint64_t rs2 = (uint64_t)padding << 48 | (uint64_t)stride << 32 |
                   (uint64_t)poolOutDim << 16 | (uint64_t)outDim;
    TypedAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    TypedAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig1_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig2
    rs1 = (uint64_t)kernelDim << 48 | (uint64_t)poolSize << 32 |
          (uint64_t)poolStride << 16 | (uint64_t)poolPadding;
    rs2 = (uint64_t)batches << 48 | (uint64_t)porows << 32 |
          (uint64_t)pocols << 16 | (uint64_t)pochs;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig2_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig3
    rs1 = (uint64_t)krows << 48 | (uint64_t)kcols << 32 | (uint64_t)kchs << 16 |
          (uint64_t)lpad;
    rs2 = (uint64_t)rpad << 48 | (uint64_t)upad << 32 | (uint64_t)dpad << 16 |
          (uint64_t)plpad;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig3_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig4
    rs1 = (uint64_t)orows << 48 | (uint64_t)prpad << 32 |
          (uint64_t)pupad << 16 | (uint64_t)pdpad;
    rs2 = (uint64_t)kernelDilation << 16 | (uint64_t)ocols;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig4_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsconfig5
    rewriter.create<LoopConvWsConfig5_IntrOp>(loc, weights, output);
    // loopConvWsconfig6
    rewriter.create<LoopConvWsConfig6_IntrOp>(loc, bias, input);
    // loopConvWs
    rs1 = (uint64_t)maxPixelsPerRow << 8 | dw << 6 | transInput3120 << 5 |
          transWeight0132 << 4 | transWeight1203 << 3 | transOutput1203 << 2 |
          writ180 << 1 | noBias;
    rs2 = act << 3 | inputDilated << 2 | downsample << 1 | noPool;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void spTiledConv(int batchSize, int inRowDim, int inColDim, int inChannels,
                   int outChannels, int outRowDim, int outColDim,
                   int poolOutRowDim, int poolOutColDim, int stride,
                   int padding, int kernelDim, int kernelDilation, int inStride,
                   int weightStride, int outStride, int poolSize,
                   int poolStride, int poolPadding, int batches, int porows,
                   int pocols, int pochs, int krows, int kcols, int kchs,
                   int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
                   int pupad, int pdpad, Value &input, Value &weights,
                   Value &output, Value &bias, int act, acc_scale_t scale,
                   bool wrot180, bool transOutput1203, bool transInput3120,
                   bool transWeight1203, bool transWeight0132, bool noBias,
                   bool noPool, bool downsample, bool inputDilated, bool dw,
                   TileConvOp &tileConvOp,
                   ConversionPatternRewriter &rewriter) const {

    Location loc = tileConvOp.getLoc();
    if (dw) {
      kchs = 1;
      pochs = 1;
    }

    const int orows = porows * poolStride + poolSize - 1 - pupad - pdpad;
    const int ocols = pocols * poolStride + poolSize - 1 - plpad - prpad;
    const int ochs = pochs;

    // Calculate image dimensions
    // Note: "irows" and "icols" includes padding
    const int dilatedKrows = krows + (kernelDilation - 1) * (krows - 1);
    const int dilatedKcols = kcols + (kernelDilation - 1) * (kcols - 1);
    int irows = orows * stride + dilatedKrows - 1;
    int icols = ocols * stride + dilatedKcols - 1;
    int irowsUnpadded = irows - upad - dpad;
    int icolsUnpadded = icols - lpad - rpad;

    const int ichs = kchs;

#define UNDILATED(x) ((inputDilated) ? (((x) + 1) / 2) : (x))

    if (inputDilated) {
      irowsUnpadded = (irowsUnpadded + 1) / 2;
      icolsUnpadded = (icolsUnpadded + 1) / 2;

      irows = irowsUnpadded + UNDILATED(upad) + UNDILATED(dpad);
      icols = icolsUnpadded + UNDILATED(lpad) + UNDILATED(rpad);
    }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
    const bool transposed =
        transOutput1203 || transInput3120 || transWeight1203 || transWeight0132;
    int maxPixelsPerRow = transposed || wrot180 || downsample || inputDilated ||
                                  kernelDilation > 1 || ichs > dim
                              ? 1
                              : dim / ichs;
    if (maxPixelsPerRow > kcols)
      maxPixelsPerRow = kcols;
#else
    const int maxPixelsPerRow = 1;
#endif
    // Calculate spad address offsets
    const int outChannelsPerBank = ochs / dim + (ochs % dim != 0);
    const int inChannelsPerBank = kchs / dim + (kchs % dim != 0);
    const int bRows = transWeight0132
                          ? inChannelsPerBank * kcols * krows * ochs
                          : outChannelsPerBank * kcols * krows * kchs;

    static uint32_t dSpAddrRow = 0;
    static uint32_t cSpAddrRow = 0;

    const uint32_t aSpAddrStart = 0;
    const uint32_t bSpAddrStart = BANK_NUM * bankRows - bRows;
    const uint32_t dSpAddrStart = (1 << (addrLen - 1)) + dSpAddrRow;
    const uint32_t cSpAddrStart = (3 << (addrLen - 2)) + cSpAddrRow;

    if (bias != 0) {
      dSpAddrRow = (dSpAddrRow + accRows / 2) % accRows;
    }

    if (output != 0) {
      cSpAddrRow = (cSpAddrRow + accRows / 2) % accRows;
    }
    if (inRowDim == inColDim && outRowDim == outColDim &&
        poolOutRowDim == poolOutColDim) {
      gemminiLoopConvWs(
          batchSize, inRowDim, inChannels, outChannels, outRowDim,
          poolOutRowDim, stride, padding, kernelDim, kernelDilation, poolSize,
          poolStride, poolPadding, batches, porows, pocols, pochs, krows, kcols,
          kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows,
          ocols, weights, output, bias, input, noBias, noPool, downsample,
          wrot180, inputDilated, act, transOutput1203, transWeight1203,
          transWeight0132, transInput3120, maxPixelsPerRow, dw, tileConvOp,
          rewriter);
      return;
    }
    if (!noPool) {
      llvm::outs() << "Pooling with rectangular convolutions is currently not "
                      "supported.\n";
      return;
    }
    // Only rectangular convolutions will use the following C code
    // mvin bias
    const size_t maxBlockLen = MAX_BYTES / (dim * 1);
    const size_t maxBlockLenAcc = MAX_BYTES / (dim * 4);
    if (bias != NULL) {
      // TODO we probably don't need quite this many nested loops for this part
      const int maxOchsPerMvin =
          ochs < (int)(maxBlockLenAcc * dim) ? ochs : maxBlockLenAcc * dim;
      Value zeroValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0));
      rewriter.create<ConfigLdOp>(loc, zeroValue,
                                  llvm::APFloat((float)MVIN_SCALE_IDENTITY),
                                  false, 2, batches * orows * ocols);
      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += dim) {
            const int I = ocols - ocol > dim ? dim : ocols - ocol;
            for (int och = 0; och < ochs; och += maxOchsPerMvin) {
              const int J =
                  ochs - och > maxOchsPerMvin ? maxOchsPerMvin : ochs - och;
              const uint32_t dSpAddr = dSpAddrStart +
                                       (och / dim) * batches * orows * ocols +
                                       b * orows * ocols + orow * ocols + ocol;
              auto cci = [&](int64_t v) -> Value {
                return rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getI64IntegerAttr(v));
              };
              if (noBias) {
                gemminiMvinOffset<Mvin3_IntrOp>(zeroValue, cci(0 * sizeOfAccT),
                                                cci(dSpAddr), cci(J), cci(I),
                                                addrLen, rewriter);
              } else {
                gemminiMvinOffset<Mvin3_IntrOp>(bias, cci(och * sizeOfAccT),
                                                cci(dSpAddr), cci(J), cci(I),
                                                addrLen, rewriter);
              }
            }
          }
    }
    // mvin input
    if (input != NULL) {
      int maxChsPerMvin =
          ichs < (int)(maxBlockLen * dim) ? ichs : maxBlockLen * dim;
      if (transInput3120) {
        maxChsPerMvin =
            batches < (int)(maxBlockLen * dim) ? batches : maxBlockLen * dim;
      }
      const int dramStride =
          transInput3120 ? batchSize * sizeOfElemT : inChannels * sizeOfElemT;
      const int spadStride =
          transInput3120
              ? ichs * (irows >> downsample) * (icols >> downsample)
              : batches * (irows >> downsample) * (icols >> downsample);
      Value strideValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(dramStride << downsample));
      rewriter.create<ConfigLdOp>(loc, strideValue,
                                  llvm::APFloat((float)MVIN_SCALE_IDENTITY),
                                  false, 0, spadStride, maxPixelsPerRow);
      const int b_it = transInput3120 ? maxChsPerMvin : 1;
      const int ich_it = transInput3120 ? 1 : maxChsPerMvin;
      for (int b = 0; b < batches; b += b_it)
        for (int irow = -UNDILATED(upad);
             irow < irowsUnpadded + UNDILATED(dpad); irow += 1 + downsample) {
          const int irowPadded = irow + UNDILATED(upad);
          for (int icol = -UNDILATED(lpad);
               icol < icolsUnpadded + UNDILATED(rpad);) {
            // TODO There might be some unnecessary mvins here at the edge of
            // the image
            int I = icolsUnpadded - icol > (dim << downsample)
                        ? (dim << downsample)
                        : icolsUnpadded - icol;
            if (icol < 0) {
              I = -icol > dim ? dim : -icol;
            } else if (icol >= icolsUnpadded) {
              I = icolsUnpadded + UNDILATED(rpad) - icol > dim
                      ? dim
                      : icolsUnpadded + UNDILATED(rpad) - icol;
            }
            const int icolPadded = icol + UNDILATED(lpad);
            for (int ich = 0; ich < ichs; ich += ich_it) {
              int K = ichs - ich > maxChsPerMvin ? maxChsPerMvin : ichs - ich;
              if (transInput3120) {
                K = batches - b > maxChsPerMvin ? maxChsPerMvin : batches - b;
              }
#define DS(x) ((x) >> (downsample))
              uint32_t aSpAddr = aSpAddrStart +
                                 (ich / dim) * batches * DS(irows) * DS(icols) +
                                 b * DS(irows) * DS(icols) +
                                 DS(irowPadded) * DS(icols) + DS(icolPadded);
              if (transInput3120) {
                aSpAddr = aSpAddrStart +
                          (b / dim) * ichs * DS(irows) * DS(icols) +
                          ich * DS(irows) * DS(icols) +
                          DS(irowPadded) * DS(icols) + DS(icolPadded);
              }
              const bool is_zeros = irow < 0 || irow >= irowsUnpadded ||
                                    icol < 0 || icol >= icolsUnpadded;
              size_t offset =
                  (b * inRowDim * inColDim + irow * inColDim + icol) *
                      inStride +
                  ich;
              Value memAddr = input;
              if (is_zeros) {
                memAddr = rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getI64IntegerAttr(0));
                offset = 0;
              } else if (transInput3120) {
                offset = (ich * inRowDim * inColDim + irow * inColDim + icol) *
                             batchSize +
                         b;
              }
              {
                auto cci = [&](int64_t v) -> Value {
                  return rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI64IntegerAttr(v));
                };
                gemminiMvinOffset(memAddr, cci(offset * sizeOfElemT),
                                  cci(aSpAddr), cci(K), cci(I >> downsample),
                                  addrLen, rewriter);
              }
            }
            icol += I;
          }
        }
    }
    // mvin weights
    if (weights != NULL) {
      int max_chs_per_mvin =
          ochs < (int)(maxBlockLen * dim) ? ochs : maxBlockLen * dim;
      if (transWeight0132) {
        max_chs_per_mvin =
            kchs < (int)(maxBlockLen * dim) ? kchs : maxBlockLen * dim;
      }
      size_t dramStride = weightStride * sizeOfElemT;
      if (dw) {
        dramStride = sizeOfElemT;
      } else if (transWeight1203) {
        dramStride = kernelDim * kernelDim * outChannels * sizeOfElemT;
      } else if (transWeight0132) {
        dramStride = inChannels * sizeOfElemT;
      }
      const size_t spadBlockStride =
          transWeight0132 ? krows * kcols * ochs : krows * kcols * kchs;
      Value dramStrideValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(dramStride));
      rewriter.create<ConfigLdOp>(loc, dramStrideValue,
                                  llvm::APFloat((float)MVIN_SCALE_IDENTITY),
                                  false, 1, spadBlockStride);

      const size_t och_it = transWeight0132 ? dim : max_chs_per_mvin;
      const size_t kch_it = transWeight0132 ? max_chs_per_mvin : dim;
      for (int och = 0; och < ochs; och += och_it) {
        for (int krow = 0; krow < krows; krow++)
          for (int kcol = 0; kcol < kcols; kcol++)
            for (int kch = 0; kch < kchs; kch += kch_it) {
              int K = kchs - kch > dim ? dim : kchs - kch;
              int J =
                  ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
              if (transWeight0132) {
                K = ochs - och > dim ? dim : ochs - och;
                J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin
                                                  : kchs - kch;
              }
              uint32_t bSpAddr = bSpAddrStart +
                                 (och / dim) * krows * kcols * kchs +
                                 krow * kcols * kchs + kcol * kchs + kch;
              if (transWeight0132) {
                bSpAddr = bSpAddrStart + (kch / dim) * krows * kcols * ochs +
                          krow * kcols * ochs + kcol * ochs + och;
              }
              size_t offset =
                  (krow * kernelDim * inChannels + kcol * inChannels + kch) *
                      weightStride +
                  och;
              if (dw) {
                offset = krow * kernelDim + kcol;
              } else if (transWeight1203) {
                offset =
                    (kch * kernelDim * kernelDim + krow * kernelDim + kcol) *
                        outChannels +
                    och;
              } else if (transWeight0132) {
                offset = (krow * kernelDim * outChannels + kcol * outChannels +
                          och) *
                             inChannels +
                         kch;
              }
              {
                auto cci = [&](int64_t v) -> Value {
                  return rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI64IntegerAttr(v));
                };
                gemminiMvinOffset<Mvin2_IntrOp>(
                    weights, cci(offset * sizeOfElemT), cci(bSpAddr), cci(J),
                    cci(K), addrLen, rewriter);
              }
            }
      }
    }
    // Compute
    {
      const int b_it = transInput3120 ? dim : 1;
      const int ocol_it = transInput3120 ? 1 : (dim << inputDilated);
      if (transInput3120) {
        rewriter.create<ConfigExOp>(loc, /*dataflow = */ OUTPUT_STATIONARY,
                                    /*act = */ 0, /*shift = */ 0,
                                    /*scale = */ llvm::APFloat((float)0),
                                    /*cStride = */ orows * ocols,
                                    /*aStride = */ irows * icols,
                                    /*aTranspose = */ 0, /*bTranspose*/ 0,
                                    /*setOnlyStrides = */ true);
      }
      for (int och = 0; och < ochs; och += dim) {
        for (int krow = 0; krow < krows; krow++) {
          for (int kcol = 0; kcol < kcols; kcol += maxPixelsPerRow) {
            for (int kch = 0; kch < kchs; kch += dim) {
              bool newWeights = true;
              for (int b = 0; b < batches; b += b_it) {
                for (int orow = 0; orow < orows; orow++) {
                  // Skip some kernel rows due to input-dilation
                  if (inputDilated &&
                      ((krow * kernelDilation + orow * stride - upad) % 2 !=
                       0)) {
                    continue;
                  }
                  for (int ocol = 0; ocol < ocols;) {
                    // Skip some cols dimensions due to input-dilation
                    if (inputDilated &&
                        ((kcol + ocol * stride - lpad) % 2 != 0)) {
                      ocol++;
                      continue;
                    }
                    int irow = orow * stride + krow * kernelDilation;
                    int icol = ocol * stride + kcol * kernelDilation;
                    if (inputDilated) {
                      irow = (irow + 1) / 2;
                      icol = (icol + 1) / 2;
                    }
                    const int pixels = kcols - kcol > maxPixelsPerRow
                                           ? maxPixelsPerRow
                                           : kcols - kcol;
                    const uint32_t cSpAddr =
                        cSpAddrStart + (och / dim) * batches * orows * ocols +
                        b * orows * ocols + orow * ocols + ocol;
                    // Over here, construct a new matrix
                    //
                    // Let us assume that we only ever operate on
                    // one pixel in one row.
                    // Thus, krows == kcols == 1
                    //
                    // Then, for every set of I, J, and K values
                    //     - I = ocols
                    //     - J = ochs
                    //     - K = kchs
                    int I = UNDILATED(ocols - ocol > (dim << inputDilated)
                                          ? (dim << inputDilated)
                                          : ocols - ocol);
                    const int J = ochs - och > dim ? dim : ochs - och;
                    const int K =
                        pixels * (kchs - kch > dim ? dim : kchs - kch);
                    if (transInput3120) {
                      I = batches - b > dim ? dim : batches - b;
                    }
                    uint32_t aSpAddr =
                        aSpAddrStart +
                        (kch / dim) * batches * DS(irows) * DS(icols) +
                        b * DS(irows) * DS(icols) + DS(irow) * DS(icols) +
                        DS(icol);
                    if (transInput3120) {
                      aSpAddr = aSpAddrStart +
                                (b / dim) * kchs * DS(irows) * DS(icols) +
                                kch * DS(irows) * DS(icols) +
                                DS(irow) * DS(icols) + DS(icol);
                    }
                    const int krow_ = wrot180 ? krows - krow - 1 : krow;
                    const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;
                    uint32_t bSpAddr =
                        bSpAddrStart + (och / dim) * krows * kcols * kchs +
                        krow_ * kcols * kchs + kcol_ * kchs + kch;
                    if (transWeight0132) {
                      bSpAddr = bSpAddrStart +
                                (kch / dim) * krows * kcols * ochs +
                                krow_ * kcols * ochs + kcol_ * ochs + och;
                    }
                    const uint32_t perSpAddr =
                        newWeights ? bSpAddr : GARBAGE_ADDR;

                    Value garbageAddrOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(GARBAGE_ADDR));
                    Value iOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(I));
                    Value jOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(J));
                    Value kOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(K));
                    Value perSpAddrOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(perSpAddr));
                    Value aSpAddrOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(aSpAddr));
                    Value cSpAddrOp = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(cSpAddr));

                    rewriter.create<PreloadOp>(loc, perSpAddrOp, cSpAddrOp, kOp,
                                               jOp, iOp, jOp);
                    if (newWeights) {
                      rewriter.create<ComputePreloadedOp>(
                          loc, aSpAddrOp, garbageAddrOp, iOp, kOp, iOp, jOp);
                    } else {
                      rewriter.create<ComputeAccumulatedOp>(
                          loc, aSpAddrOp, garbageAddrOp, iOp, kOp, iOp, jOp);
                    }
                    ocol += ocol_it;
                    newWeights = false;
                  }
                }
              }
            }
          }
        }
      }
    }
#undef DS
#undef UNDILATED
    // mvout output
    if (output != NULL) {
      if (noPool) {
        for (int b = 0; b < batches; b++)
          for (int orow = 0; orow < orows; orow++)
            for (int ocol = 0; ocol < ocols; ocol += dim) {
              const int I = ocols - ocol > dim ? dim : ocols - ocol;
              for (int och = 0; och < ochs; och += dim) {
                const int J = ochs - och > dim ? dim : ochs - och;
                const uint32_t cSpAddr =
                    cSpAddrStart + (och / dim) * batches * orows * ocols +
                    b * orows * ocols + orow * ocols + ocol;
                size_t outOffset =
                    (b * outRowDim * outColDim + orow * outColDim + ocol) *
                        outStride +
                    och;
                if (transOutput1203) {
                  outOffset =
                      (orow * outColDim * batchSize + ocol * batchSize + b) *
                          outChannels +
                      och;
                }
                {
                  auto cci = [&](int64_t v) -> Value {
                    return rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(v));
                  };
                  gemminiMvoutOffset(output, cci(outOffset * sizeOfElemT),
                                     cci(cSpAddr), cci(J), cci(I), addrLen,
                                     rewriter);
                }
              }
            }
      } else {
        printf("Pooling with rectangular convolutions is currently not "
               "supported.\n");
        exit(1);
      }
    }
  }

  void tiledConv(int batchSize, int inRowDim, int inColDim, int inChannels,
                 int outChannels, int outRowDim, int outColDim, int stride,
                 int inputDilation, int kernelDilation, int padding,
                 int kernelDim, int inStride, int weightStride, int outStride,
                 bool wrot180, bool transOutput1203, bool transInput3120,
                 bool transWeight1203, bool transWeight0132, int batches,
                 int porows, int pocols, int pochs, int krows, int kcols,
                 int kchs, const Value &input, const Value &weights,
                 const Value &bias, Value &output, int act, acc_scale_t scale,
                 int poolSize, int poolStride, int poolPadding,
                 TileConvOp &tileConvOp,
                 ConversionPatternRewriter &rewriter) const {
    bool noBias = false;
    bool noPool = poolStride == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const bool downsample = stride == 2 && kernelDim == 1 &&
                            inRowDim % 2 == 0 && inColDim % 2 == 0 &&
                            padding == 0 && noPool && inputDilation == 1 &&
                            !transInput3120;
    const int inputDilated = inputDilation == 2;
    int64_t stDramStride = transOutput1203
                               ? batchSize * outChannels * sizeOfElemT
                               : outChannels * sizeOfElemT;
    Location loc = tileConvOp.getLoc();
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(stDramStride));
    rewriter.create<ConfigStOp>(loc, strideValue, act, llvm::APFloat(scale));
    rewriter.create<ConfigExOp>(
        loc, /*dataflow = */ WEIGHT_STATIONARY, /*act = */ 0, /*shift = */ 0,
        /*scale = */ llvm::APFloat((float)0), /*cStride = */ inputDilation,
        /*aStride = */ stride >> downsample,
        /*aTranspose = */ transInput3120, /*bTranspose*/ transWeight0132,
        /*setOnlyStrides = */ false);
    const int poolOutRowDim =
        (outRowDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const int poolOutColDim =
        (outColDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const int dilatedInRowDim = inRowDim + (inputDilation - 1) * (inRowDim - 1);
    const int dilatedInColDim = inColDim + (inputDilation - 1) * (inColDim - 1);

    int porowEnd = poolOutRowDim;

    for (int b = 0; b < batchSize; b += batches) {
      for (int porow = 0; porow < porowEnd; porow += porows) {
        const int orow = porow * poolStride - poolPadding;
        for (int pocol = 0; pocol < poolOutColDim; pocol += pocols) {
          const int ocol = pocol * poolStride - poolPadding;
          for (int poch = 0; poch < outChannels; poch += pochs) {
            for (int krow = 0; krow < kernelDim; krow += krows) {
              const int orow_floored = orow < 0 ? 0 : orow;

              int irow =
                  orow_floored * stride + krow * kernelDilation - padding;
              for (int kcol = 0; kcol < kernelDim; kcol += kcols) {
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                int icol =
                    ocol_floored * stride + kcol * kernelDilation - padding;

                for (int kch = 0; kch < inChannels; kch += kchs) {
                  TypedAttr offsetAttr = rewriter.getI64IntegerAttr(
                      ((b * poolOutRowDim * poolOutColDim +
                        porow * poolOutColDim + pocol) *
                           outChannels +
                       poch) *
                      sizeOfElemT);
                  Value offsetValue =
                      rewriter.create<arith::ConstantOp>(loc, offsetAttr);
                  Value out = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), output,
                      offsetValue);
                  if (transOutput1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((porow * poolOutColDim * batchSize +
                          pocol * batchSize + b) *
                             outChannels +
                         poch) *
                        sizeOfElemT);
                    offsetValue =
                        rewriter.create<arith::ConstantOp>(loc, offsetAttr);
                    out = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                         rewriter.getI64Type(),
                                                         output, offsetValue);
                  }

                  if (krow + krows < kernelDim || kcol + kcols < kernelDim ||
                      kch + kchs < inChannels) {
                    out = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), rewriter.getI64IntegerAttr(0));
                  }
                  Value pochValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),
                      rewriter.getI64IntegerAttr(poch * sizeOfAccT));
                  Value bias_ = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), bias,
                      pochValue);
                  if (krow > 0 || kcol > 0 || kch > 0) {
                    bias_ = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), rewriter.getI64IntegerAttr(0));
                  }

                  const int batches_ =
                      batchSize - b > batches ? batches : batchSize - b;
                  const int porows_ = poolOutRowDim - porow > porows
                                          ? porows
                                          : poolOutRowDim - porow;
                  const int pocols_ = poolOutColDim - pocol > pocols
                                          ? pocols
                                          : poolOutColDim - pocol;
                  const int pochs_ =
                      outChannels - poch > pochs ? pochs : outChannels - poch;
                  const int krows_ =
                      kernelDim - krow > krows ? krows : kernelDim - krow;
                  const int kcols_ =
                      kernelDim - kcol > kcols ? kcols : kernelDim - kcol;
                  const int kchs_ =
                      inChannels - kch > kchs ? kchs : inChannels - kch;

                  const int ocols_ = pocols_ * poolStride + poolSize - 1;
                  const int orows_ = porows_ * poolStride + poolSize - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad =
                      ocol + ocols_ > outColDim ? ocol + ocols_ - outColDim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad =
                      orow + orows_ > outRowDim ? orow + orows_ - outRowDim : 0;

                  const int dilatedKrows_ =
                      krows_ + (kernelDilation - 1) * (krows_ - 1);
                  const int dilatedKcols_ =
                      kcols_ + (kernelDilation - 1) * (kcols_ - 1);

                  const int icols_ =
                      (ocols_ - plpad - prpad) * stride + dilatedKcols_ - 1;
                  const int irows_ =
                      (orows_ - pupad - pdpad) * stride + dilatedKrows_ - 1;

                  int lpad = icol < 0 ? -icol : 0;
                  int rpad = icol + icols_ > dilatedInColDim
                                 ? icol + icols_ - dilatedInColDim
                                 : 0;
                  int upad = irow < 0 ? -irow : 0;
                  int dpad = irow + irows_ > dilatedInRowDim
                                 ? irow + irows_ - dilatedInRowDim
                                 : 0;

                  if (inputDilated) {
                    lpad += lpad == 0 && icol % 2 != 0;
                    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                    upad += upad == 0 && irow % 2 != 0;
                    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                  }

                  int krow_ = krow;
                  int kcol_ = kcol;
                  if (wrot180) {
                    krow_ = kernelDim - krow - krows_;
                    kcol_ = kernelDim - kcol - kcols_;
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((krow_ * kernelDim * inChannels + kcol_ * inChannels +
                        kch) *
                           outChannels +
                       poch) *
                      sizeOfElemT);
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), offsetAttr);
                  Value weightsSlice = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                      offsetValue);
                  if (transWeight1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * kernelDim * kernelDim + krow_ * kernelDim +
                          kcol_) *
                             outChannels +
                         poch) *
                        sizeOfElemT);
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  } else if (transWeight0132) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((krow_ * kernelDim * outChannels +
                          kcol_ * outChannels + poch) *
                             inChannels +
                         kch) *
                        sizeOfElemT);
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((b * inRowDim * inColDim +
                        ((irow + upad) >> inputDilated) * inColDim +
                        ((icol + lpad) >> inputDilated)) *
                           inChannels +
                       kch) *
                      sizeOfElemT);
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), offsetAttr);
                  Value in = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), input,
                      offsetValue);
                  if (transInput3120) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * inRowDim * inColDim +
                          ((irow + upad) >> inputDilated) * inColDim +
                          ((icol + lpad) >> inputDilated)) *
                             batchSize +
                         b) *
                        sizeOfElemT);
                    in = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                        rewriter.getI64Type(),
                                                        input, offsetValue);
                  }

                  spTiledConv(
                      batchSize, inRowDim, inColDim, inChannels, outChannels,
                      outRowDim, outColDim, poolOutRowDim, poolOutColDim,
                      stride, padding, kernelDim, kernelDilation, inStride,
                      weightStride, outStride, poolSize, poolStride,
                      poolPadding, batches_, porows_, pocols_, pochs_, krows_,
                      kcols_, kchs_, lpad, rpad, upad, dpad, plpad, prpad,
                      pupad, pdpad, in, weightsSlice, out, bias_, act, scale,
                      wrot180, transOutput1203, transInput3120, transWeight1203,
                      transWeight0132, noBias, noPool, downsample, inputDilated,
                      false, tileConvOp, rewriter);
                }
              }
            }
          }
        }
      }
    }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileConvOp, flushValue,
                                              flushValue);
  }

  int tiledConvTotalSpadRows(bool acc, int stride, int inputDilation,
                             int kernelDilation, bool downsample,
                             bool transWeight0132, bool transInput3120,
                             int batches, int porows, int pocols, int ochs,
                             int krows, int kcols, int kchs, int poolSize,
                             int poolStride) const {

    const int orows = porows * poolStride + poolSize - 1;
    const int ocols = pocols * poolStride + poolSize - 1;

    const int krowsDilated = krows + (kernelDilation - 1) * (krows - 1);
    const int kcolsDilated = kcols + (kernelDilation - 1) * (kcols - 1);

    int irows = orows * stride + krowsDilated - 1;
    int icols = ocols * stride + kcolsDilated - 1;
    const int ichs = kchs;

    irows = irows / inputDilation + (irows % inputDilation != 0);
    icols = icols / inputDilation + (icols % inputDilation != 0);

    const int inChannelsPerBank = ichs / dim + (ichs % dim != 0);
    const int outChannelsPerBank = ochs / dim + (ochs % dim != 0);
    const int batchesPerBank = batches / dim + (batches % dim != 0);

    const int aRows = transInput3120
                          ? (batchesPerBank * ichs * (irows >> downsample) *
                             (icols >> downsample))
                          : (inChannelsPerBank * batches *
                             (irows >> downsample) * (icols >> downsample));

    const int bRows = transWeight0132
                          ? inChannelsPerBank * kcols * krows * ochs
                          : outChannelsPerBank * kcols * krows * kchs;

    const int cRows = outChannelsPerBank * batches * orows * ocols;

    return acc ? cRows : aRows + bRows;
  }

public:
  using ConvertOpToLLVMPattern<TileConvOp>::ConvertOpToLLVMPattern;
  explicit GemminiTileConvLowering(LLVMTypeConverter &typeConverter,
                                   int64_t dim, int64_t addrLen,
                                   int64_t accRows, int64_t bankRows,
                                   size_t sizeOfElemT, size_t sizeOfAccT)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen),
        accRows(accRows), bankRows(bankRows), sizeOfElemT(sizeOfElemT),
        sizeOfAccT(sizeOfAccT) {}
  LogicalResult
  matchAndRewrite(TileConvOp tileConvOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = tileConvOp.getInput();
    Value output = tileConvOp.getOutput();
    Value weights = tileConvOp.getWeights();
    Value bias = tileConvOp.getBias();
    MemRefType inputType = dyn_cast<MemRefType>(input.getType());
    MemRefType biasType = dyn_cast<MemRefType>(bias.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> biasShape = biasType.getShape();

    Value outRowDimValue = tileConvOp.getOutRowDim();
    int outRowDim = getNumberFromValue(outRowDimValue);
    Value outColDimValue = tileConvOp.getOutColDim();
    int outColDim = getNumberFromValue(outColDimValue);
    Value kernelDimValue = tileConvOp.getKernelDim();
    int kernelDim = getNumberFromValue(kernelDimValue);
    int batchSize = inputShape[0];
    int inRowDim = inputShape[1];
    int inColDim = inputShape[2];
    int inChannels = inputShape[3];
    int outChannels = biasShape[0];
    int stride = tileConvOp.getStride();
    int inputDilation = tileConvOp.getInputDilation();
    int kernelDilation = tileConvOp.getKernelDilation();
    int padding = tileConvOp.getPadding();
    int act = tileConvOp.getAct();
    float scale = tileConvOp.getScale().convertToFloat();
    int poolSize = tileConvOp.getPoolSize();
    int poolStride = tileConvOp.getPoolStride();
    int poolPadding = tileConvOp.getPoolPadding();
    bool wrot180 = tileConvOp.getWrot180();
    bool transOutput1203 = tileConvOp.getTransOutput1203();
    bool transInput3120 = tileConvOp.getTransInput3120();
    bool transWeight1203 = tileConvOp.getTransWeight1203();
    bool transWeight0132 = tileConvOp.getTransWeight0132();
    Location loc = tileConvOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value inputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
    Value inputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, inputExtractOp);
    Value outputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
    Value outputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, outputExtractOp);
    Value biasExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, bias);
    Value biasIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, biasExtractOp);
    Value weightsExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, weights);
    Value weightsIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, weightsExtractOp);
    const bool noPool = poolSize == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const int poolOutRowDim =
        (outRowDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const int poolOutColDim =
        (outColDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const bool downsample = stride == 2 && kernelDim == 1 && padding == 0 &&
                            noPool && inRowDim % 2 == 0 && inColDim % 2 == 0;
    int args[] = {batchSize, poolOutRowDim, poolOutColDim, outChannels,
                  kernelDim, kernelDim,     inChannels};
    const int maxArgs[] = {batchSize, poolOutRowDim, poolOutColDim, outChannels,
                           kernelDim, kernelDim,     inChannels};
    const int orowsIdx = 1;
    const int ocolsIdx = 2;
    const int outChannelsIdx = 3;
    const int inChannelsIdx = 6;
    const int maxSpadRows = (BANK_NUM * bankRows / 2);
    const int maxAccRows = (accRows / 2);
    int spadRows = tiledConvTotalSpadRows(
        false, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    int accRows = tiledConvTotalSpadRows(
        true, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    while (spadRows > maxSpadRows || accRows > maxAccRows) {
      int maxVal = -1;
      int maxIdx = -1;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        if (!(i == ocolsIdx && args[i] <= dim && args[orowsIdx] > 1) &&
            args[i] > maxVal) {
          maxVal = args[i];
          maxIdx = i;
        }
      }

      if (maxIdx == outChannelsIdx || maxIdx == inChannelsIdx) {
        if (args[maxIdx] % dim != 0) {
          args[maxIdx] = (args[maxIdx] / dim) * dim;
        } else {
          args[maxIdx] -= dim;
        }
        args[maxIdx] = args[maxIdx] == 0 ? 1 : args[maxIdx];
      } else {
        args[maxIdx]--;
      }
      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
    }
    bool notIncreased = false;
    while (!notIncreased) {
      notIncreased = true;

      int argsCandidate[] = {args[0], args[1], args[2], args[3],
                             args[4], args[5], args[6]};
      argsCandidate[ocolsIdx]++;

      if (argsCandidate[ocolsIdx] > maxArgs[ocolsIdx])
        continue;

      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);

      if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
        args[ocolsIdx] = argsCandidate[ocolsIdx];
        notIncreased = false;
      }
    }

    bool nothingIncreased = false;
    while (!nothingIncreased) {
      nothingIncreased = true;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        int argsCandidate[] = {args[0], args[1], args[2], args[3],
                               args[4], args[5], args[6]};
        argsCandidate[i]++;

        if (argsCandidate[i] > maxArgs[i])
          continue;
        spadRows = tiledConvTotalSpadRows(
            false, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);
        accRows = tiledConvTotalSpadRows(
            true, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);

        if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
          args[i] = argsCandidate[i];
          nothingIncreased = false;
        }
      }
    }
    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    const int inStride = inChannels;
    const int outStride = outChannels;
    const int weightStride = outChannels;
    tiledConv(batchSize, inRowDim, inColDim, inChannels, outChannels, outRowDim,
              outColDim, stride, inputDilation, kernelDilation, padding,
              kernelDim, inStride, weightStride, outStride, wrot180,
              transOutput1203, transInput3120, transWeight1203, transWeight0132,
              batches, orows, ocols, ochs, krows, kcols, kchs, inputIndexCastOp,
              weightsIndexCastOp, biasIndexCastOp, outputIndexCastOp, act,
              scale, poolSize, noPool ? 0 : poolStride, poolPadding, tileConvOp,
              rewriter);
    return success();
  }

private:
  int64_t dim;
  int64_t addrLen;
  int64_t accRows;
  int64_t bankRows;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiFlushLowering>(converter);
  patterns.add<GemminiConfigStLowering>(converter);
  patterns.add<GemminiConfigLdLowering>(converter, dim);
  patterns.add<GemminiMvinLowering>(converter, addrLen);
  patterns.add<GemminiMvin2Lowering>(converter, addrLen);
  patterns.add<GemminiMvin3Lowering>(converter, addrLen);
  patterns.add<GemminiMvoutLowering>(converter, addrLen);
  patterns.add<GemminiConfigExLowering>(converter);
  patterns.add<GemminiConfigNormLowering>(converter);
  patterns.add<GemminiPreloadZerosLowering>(converter, dim, addrLen);
  patterns.add<GemminiPreloadLowering>(converter, addrLen);
  patterns.add<GemminiComputePreloadedLowering>(converter, addrLen);
  patterns.add<GemminiComputeAccumulatedLowering>(converter, addrLen);
  patterns.add<GemminiTileMatMulLowering>(converter, dim, addrLen, accRows,
                                          bankRows, sizeOfElemT, sizeOfAccT);
  patterns.add<GemminiTileConvLowering>(converter, dim, addrLen, accRows,
                                        bankRows, sizeOfElemT, sizeOfAccT);
}

void mlir::configureGemminiLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<
      Flush_IntrOp, ConfigSt_IntrOp, ConifgLd_IntrOp, ConfigEX_IntrOp,
      Mvin_IntrOp, Mvin2_IntrOp, Mvin3_IntrOp, Mvout_IntrOp, Preload_IntrOp,
      ComputePreloaded_IntrOp, ComputeAccumulated_IntrOp,
      LoopWsConfigBounds_IntrOp, LoopWsConfigAddrsAB_IntrOp,
      LoopWsConfigAddrsDC_IntrOp, LoopWsConfigStridesAB_IntrOp,
      LoopWsConfigStridesDC_IntrOp, LoopWs_IntrOp, LoopConvWsConfig1_IntrOp,
      LoopConvWsConfig2_IntrOp, LoopConvWsConfig3_IntrOp,
      LoopConvWsConfig4_IntrOp, LoopConvWsConfig5_IntrOp,
      LoopConvWsConfig6_IntrOp, LoopConvWs_IntrOp, ConfigNorm_IntrOp>();
  target.addIllegalOp<FlushOp, ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp,
                      Mvin2Op, Mvin3Op, MvoutOp, PrintOp, PreloadZerosOp,
                      PreloadOp, ComputePreloadedOp, ComputeAccumulatedOp,
                      TileMatMulOp, TileConvOp, ConfigNormOp>();
  // SCF ops are created by TileMatMul/TileConv/Print lowerings and must be
  // marked legal so that applyPartialConversion accepts them.  A separate
  // -convert-scf-to-cf pass lowers them to control-flow afterwards.
  target.addLegalOp<scf::ForOp, scf::YieldOp>();
}
