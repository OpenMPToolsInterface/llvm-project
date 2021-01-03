/*
 * Debug.cpp -- OMP debug
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Debug.h"

std::ostream &GdbColor::operator<<(std::ostream &os, GdbColor::Code code) {
  return os << "\033[" << static_cast<int>(code) << "m";
}
