// RUN: buddy-opt %s \
// RUN:     --lower-gemmini="tile_i=1 tile_j=1 tile_k=1" | \
// RUN: FileCheck %s

// Test that custom tile sizes (1,1,1) produce scf.for tiling loops
// with OS dataflow intrinsics on a 32x32 matmul (2x2 tiles of dim=16).

func.func @main() -> i8 {
  %i0 = arith.constant 0 : i8
  %i1I8 = arith.constant 1 : i8
  %i2I32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %aArray = memref.alloc() {alignment = 16} : memref<32x32xi8>
  %bArray = memref.alloc() {alignment = 16} : memref<32x32xi8>
  %cArray = memref.alloc() {alignment = 16} : memref<32x32xi8>
  %dArray = memref.alloc() {alignment = 64} : memref<32x32xi32>
  %dim = memref.dim %aArray, %c0 : memref<32x32xi8>
  scf.for %i = %c0 to %dim step %c1 {
    scf.for %j = %c0 to %dim step %c1 {
      memref.store %i1I8, %aArray[%i, %j] : memref<32x32xi8>
      memref.store %i1I8, %bArray[%i, %j] : memref<32x32xi8>
      memref.store %i2I32, %dArray[%i, %j] : memref<32x32xi32>
    }
  }
  // CHECK: scf.for
  // CHECK: "gemmini.intr.preload"
  // CHECK: "gemmini.intr.compute_preloaded"
  // CHECK-NOT: "gemmini.intr.loop_ws"
  gemmini.tile_matmul %aArray %bArray %cArray %dArray {dataflow = 0} : memref<32x32xi8> memref<32x32xi8> memref<32x32xi8> memref<32x32xi32>
  return %i0 : i8
}
