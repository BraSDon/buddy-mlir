set(LLVM_LINK_LLVM_DYLIB OFF)

set(LLVM_LINK_COMPONENTS
  BuddyRISCVCodeGen
  BuddyRISCVAsmParser
  BuddyRISCVDesc
  BuddyRISCVInfo

  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  BuddyAnalysis
  BuddyAsmParser
  BuddyAsmPrinter
  BuddyCodeGen
  BuddyCore
  BuddyIRReader
  MC
  BuddyMIRParser
  BuddyRemarks
  BuddyScalarOpts
  BuddySelectionDAG
  Support
  BuddyTarget
  BuddyIPO
  BuddyVectorize
  TargetParser
  BuddyTransformUtils
  Passes
  GlobalISel
)

add_llvm_tool(buddy-llc
  ${LLVM_MAIN_SRC_DIR}/tools/llc/llc.cpp
  ${LLVM_MAIN_SRC_DIR}/tools/llc/NewPMDriver.cpp

  DEPENDS
  buddy_intrinsics_gen
  SUPPORT_PLUGINS
  )

target_link_libraries(buddy-llc
  PRIVATE
  LLVMBuddyRISCVCodeGen
  LLVMBuddyRISCVAsmParser
  LLVMBuddyRISCVDesc
  LLVMBuddyRISCVInfo
  LLVMBuddyAnalysis
  LLVMBuddyAsmParser
  LLVMBuddyAsmPrinter
  LLVMBuddyCodeGen
  LLVMBuddyCore
  LLVMBuddyIRReader
  LLVMBuddyMIRParser
  LLVMBuddyRemarks
  LLVMBuddyScalarOpts
  LLVMBuddySelectionDAG
  LLVMBuddyTarget
  LLVMBuddyIPO
  LLVMBuddyVectorize
  LLVMBuddyTransformUtils
)

export_executable_symbols_for_plugins(buddy-llc)
