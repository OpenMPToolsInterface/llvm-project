set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Core
  Object
  Support
  TextAPI
  )

add_llvm_tool(llvm-libtool-darwin
  llvm-libtool-darwin.cpp
)

if(LLVM_INSTALL_CCTOOLS_SYMLINKS)
  add_llvm_tool_symlink(libtool llvm-libtool-darwin)
endif()
