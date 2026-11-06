#pragma once

#include <Python.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// FRAGMENT, not TORCH_LIBRARY: only one definition-counting registration per
// namespace is allowed, and MetaX's mcoplib._C already claims `_C` with a
// plain TORCH_LIBRARY — the two aborted each other at import.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY_FRAGMENT(NAME, MODULE)
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
