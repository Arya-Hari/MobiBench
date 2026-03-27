/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h> // For std::pair, std::string, etc.
 
 #include "executorch/examples/models/llama/runner/runner.h"
 
 namespace py = pybind11;
 
 // This macro creates a Python module.
 // The first argument is the name of the module (my_llama_runner), which must
 // match the name we used in our Python script.
 // The second argument is a variable name (m) for the module object.
 PYBIND11_MODULE(my_llama_runner, m) {
     m.doc() = "Pybind11 bindings for the ExecuTorch Llama runner";
 
     // 1. Define the InferenceMetrics class for Python
     // This makes our C++ struct available as a Python class.
     py::class_<example::InferenceMetrics>(m, "InferenceMetrics")
         .def(py::init<>()) // Expose the default constructor
         // Expose each field of the struct as a read/write property in Python.
         // This allows access like: metrics.wall_time_s
         .def_readwrite("load_time_ms", &example::InferenceMetrics::load_time_ms)
         .def_readwrite("prefill_latency_ms", &example::InferenceMetrics::prefill_latency_ms)
         .def_readwrite("decode_latency_ms", &example::InferenceMetrics::decode_latency_ms)
         .def_readwrite("wall_time_s", &example::InferenceMetrics::wall_time_s)
         .def_readwrite("prefill_tps", &example::InferenceMetrics::prefill_tps)
         .def_readwrite("decode_tps", &example::InferenceMetrics::decode_tps)
         .def_readwrite("avg_cpu_percent", &example::InferenceMetrics::avg_cpu_percent)
         .def_readwrite("peak_memory_mb", &example::InferenceMetrics::peak_memory_mb)
         .def_readwrite("raw_logs", &example::InferenceMetrics::raw_logs);
 
     // 2. Expose the run_inference_with_metrics function to Python
     // The first argument is the Python name of the function.
     // The second argument is a pointer to the C++ function.
     // The third argument is a docstring for the function.
     m.def(
         "run_inference_with_metrics",
         &example::run_inference_with_metrics,
         "Runs Llama inference and returns the output string and a metrics object.");
 }