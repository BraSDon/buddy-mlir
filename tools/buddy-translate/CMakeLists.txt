set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(buddy-translate
  buddy-translate.cpp
  )
llvm_update_compile_flags(buddy-translate)
target_link_libraries(buddy-translate
  PRIVATE
  BuddyToLLVMIRTranslationRegistration
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_link_libraries(buddy-translate)
