// Adapted from MooreThreads/torchada, commit f0d469ff42f7e0bdd89cb0f7ecbbaf06bf56e1b1:
// src/torchada/_graph_rotation_src/graph_exec_aux.cpp
// The upstream MIT notice below applies to this executable-only helper.
// MIT License
//
// Copyright (c) 2025 Moore Threads Technology Co., Ltd ("Moore Threads"). All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <c10/util/Exception.h>
#include <torch_musa/csrc/aten/musa/MUSAGraph.h>
#include <musa_runtime_api.h>

namespace {
using Graph = at::musa::MUSAGraph;

// Protected access is legal here. These are pointers to BASE-class members,
// applied to the real registered MUSAGraph object: no offsets or downcast.
struct Members : Graph {
  static musaGraphExec_t Graph::* exec() { return &Members::graph_exec_; }
  static musaGraph_t Graph::* graph() { return &Members::graph_; }
  static bool Graph::* has_exec() { return &Members::has_graph_exec_; }
};

musaGraphExec_t& executable(Graph& graph) { return graph.*Members::exec(); }
bool& executable_flag(Graph& graph) { return graph.*Members::has_exec(); }

bool has_exec(Graph& graph) {
  const bool live = executable(graph) != nullptr;
  TORCH_CHECK(live == executable_flag(graph),
              "MUSA graph executable pointer/flag mismatch");
  return live;
}

// Caller serializes operations and synchronizes this graph's device first.
// Keep template, pool, capture stream/device and generator states untouched.
void free_exec(Graph& graph) {
  if (!has_exec(graph)) return;
  const auto error = musaGraphExecDestroy(executable(graph));
  TORCH_CHECK(error == musaSuccess, "musaGraphExecDestroy failed: ",
              static_cast<int>(error));
  executable(graph) = nullptr;
  executable_flag(graph) = false;
}

void inst_exec(Graph& graph) {
  if (has_exec(graph)) return;
  const auto captured = graph.*Members::graph();
  TORCH_CHECK(captured != nullptr, "MUSA graph has no retained capture template");
  // An unsuccessful instantiate must not publish a pointer or live flag.
  musaGraphExec_t created = nullptr;
  const auto error = musaGraphInstantiate(&created, captured, nullptr, nullptr, 0);
  TORCH_CHECK(error == musaSuccess, "musaGraphInstantiate failed: ",
              static_cast<int>(error));
  TORCH_CHECK(created != nullptr, "musaGraphInstantiate returned a null executable");
  executable(graph) = created;
  executable_flag(graph) = true;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  // torch_musa already registers the base Graph type through pybind11.
  module.def("free_exec", &free_exec);
  module.def("inst_exec", &inst_exec);
  module.def("has_exec", &has_exec);
}
