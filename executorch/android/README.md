# executorch - Custom Llama Evaluation Mod

This is a modified version of the main `executorch` repository.

The primary goal of these changes is to enhance the Llama model example (`examples/models/llama`) with custom evaluation logic, statistics tracking, and new bindings. The modifications appear to be based on or for integration with the MobiBench project.

---

## Summary of Changes

Here is a list of the new and modified files that implement this functionality:

### Modified Files

- `examples/models/llama/eval_llama.py`: Updated the Python evaluation script.  
- `examples/models/llama/main.cpp`: Modified the main C++ entry point for the Llama runner, likely to incorporate new stats or CSV logging.  
- `extension/llm/runner/stats.h`: Added or changed structures for tracking performance statistics.  
- `extension/llm/runner/text_llm_runner.cpp`: Implemented changes for the text runner, likely related to stats.  
- `extension/llm/runner/text_llm_runner.h`: Updated the text runner's header file.  

### Added Files

- `examples/models/llama/csv.h`: A new header file, likely to add functionality for logging evaluation results to a `.csv` file.  
- `examples/models/llama/main.py`: A new Python main script, possibly for a different execution flow or to use new bindings.  
- `examples/models/llama/runner/bindings.cpp`: New C++ bindings, likely to expose functionality to the new `main.py` script.  

---

## Build

### Step 1: Setup
> ⚠️ **Double-check your Python environment:** Make sure to run  
> `conda activate <VENV>` before executing any bash or Python commands.

1. Follow the [ExecuTorch setup tutorial](https://pytorch.org/executorch/main/getting-started-setup) and run:
   ```bash
   ./install_executorch.sh
   ```
2. Install additional dependencies:
   ```bash
   examples/models/llama/install_requirements.sh
   ```

---

### Step 2: Prepare Model

Follow procedure as in https://github.com/pytorch/executorch/tree/main/examples/models/llama

---

## 🧩 Android Build Instructions

To build the **Llama evaluation runner for Android (arm64-v8a)**, follow these steps:

### Step 1: Build ExecuTorch for Android

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -Bcmake-out-android .
```

Then build and install:
```bash
cmake --build cmake-out-android -j16 --target install --config Release
```

---

### Step 2: Build Llama Example for Android

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama
```

Then build:
```bash
cmake --build cmake-out-android/examples/models/llama -j16 --config Release
```

---

### Step 3: Deploy to Android Device

> ⚠️ **Important:** Replace `<model.pte>` and `<tokenizer.model>` with the actual paths to your files.

1. **Create a directory on your phone:**
   ```bash
   adb shell mkdir -p /data/local/tmp/llama
   ```

2. **Push the model file:**
   ```bash
   adb push <model.pte> /data/local/tmp/llama/
   ```

3. **Push the tokenizer file:**
   ```bash
   adb push <tokenizer.model> /data/local/tmp/llama/
   ```

4. **Push the compiled Llama runner binary:**
   ```bash
   adb push cmake-out-android/examples/models/llama/llama_main /data/local/tmp/llama/
   ```

---

## 🧠 Running Dataset Evaluation on Android

After pushing all required files to `/data/local/tmp/llama`, you can evaluate different dataset types using the following commands.

> ⚠️ Ensure the dataset CSVs are also pushed to `/data/local/tmp/llama` before running these commands.

### 1. Summarization
```bash
adb shell "cd /data/local/tmp/llama && ./dataset_testing \
  --model_path ./model.pte \
  --tokenizer_path ./tokenizer.model \
  --dataset_type summarization \
  --csv_path ./summarization_dataset.csv \
  --output_json ./summarization_results.jsonl"
```

### 2. Scientific MCQ (science_qa)
```bash
adb shell "cd /data/local/tmp/llama && ./dataset_testing \
  --model_path ./model.pte \
  --tokenizer_path ./tokenizer.model \
  --dataset_type scientific_mcq \
  --csv_path ./science_qa_dataset.csv \
  --output_json ./science_qa_results.jsonl"
```

### 3. MMLU MCQ
```bash
adb shell "cd /data/local/tmp/llama && ./dataset_testing \
  --model_path ./model.pte \
  --tokenizer_path ./tokenizer.model \
  --dataset_type mmlu_mcq \
  --csv_path ./mmlu_dataset.csv \
  --output_json ./mmlu_results.jsonl"
```

### 4. Context QA
```bash
adb shell "cd /data/local/tmp/llama && ./dataset_testing \
  --model_path ./model.pte \
  --tokenizer_path ./tokenizer.model \
  --dataset_type context_qa \
  --csv_path ./context_qa_dataset.csv \
  --output_json ./context_qa_results.jsonl"
```

---

✅ **You’re now ready to build, deploy, and evaluate Llama on Android devices** using ExecuTorch’s custom evaluation setup.
