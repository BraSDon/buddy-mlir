include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/lib")
add_subdirectory(lib)
set (LLVM_LINK_COMPONENTS 
support
) 
add_llvm_tool(buddy-frontendgen  
frontendgen.cpp
)
target_link_libraries(buddy-frontendgen PRIVATE LLVMfrontendgenlib)
