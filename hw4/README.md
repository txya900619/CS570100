# Model Compression Homework

## 大綱 (Outline)
We are going to cover ...
01. 作業介紹 (Homework Introduction)
02. 提供的 code (Provided Code)
03. 繳交內容 (Submission Content)
04. 配分標準 (Grading Criteria)

## 1. 作業介紹 (Homework Introduction)

### DEEP COMPRESSION
*   **架構 (Framework):**
    1.  Prune
    2.  Retrain
    3.  Quantize
    4.  Retrain (Note: 這次作業不需要做 quantize retrain - For this assignment, quantize retrain is NOT required for the quantization step)
    5.  Huffman coding
*   **Reference Paper:** [https://arxiv.org/pdf/1510.00149.pdf](https://arxiv.org/pdf/1510.00149.pdf)

### Python library you need
*   **pytorch** (v2.7.0 indicated on slide)
    *   深度學習套件 (Deep learning framework)
    *   記得載 GPU 版 (Remember to install the GPU version)
*   **torchvision** (v0.22.0 indicated on slide)
    *   下載常用測試 data 套件 (Package for common datasets and image transformations)
    *   本次資料使用 cifar10 (CIFAR10 will be used for this assignment)
    *   也可用來下載mnist、cifar100 (Can also be used to download MNIST, CIFAR100)
*   **scikit-learn (sklearn)** (v1.6.1 indicated on slide)
    *   機器學習套件 (Machine learning library)
    *   本次要使用 cluster 的演算法: Kmeans (KMeans clustering algorithm will be used)
    *   (在 quantize 那一步 - In the quantization step)

## 2. 提供的 code (Provided Code)

The provided code includes implementations for:
*   模型 (Model)
*   Prune 部分 (Pruning part)
*   Quantize 部分 (Quantization part)
*   Huffman 部分 (Huffman coding part)

### Assignment File Structure (Initial)
The main scripts and the `net` directory are structured as follows:
```
assignment/
├── net/
│   ├── huffmancoding.py
│   ├── models.py
│   ├── prune.py
│   └── quantization.py
├── huffman_encode.py  // Main script for Huffman encoding/decoding logic (calls functions in net/huffmancoding.py)
├── pruning.py         // Main script for pruning and retraining
├── util.py
└── weight_share.py    // Main script for quantization (calls functions in net/quantization.py)
```

### 模型 (Model)
*   **AlexNet**
*   Located in: `assignment/net/models.py`
*   The `AlexNet` class inherits from `PruningModule`.
    *   Layers: `conv1` to `conv5`, `fc1` to `fc3`.
    *   Includes standard `__init__` and `forward` methods.

### Prune (`assignment/net/prune.py`)
*   This file contains the pruning logic.
*   `prune.py`:
    *   只提供 prune by std 的 fully connected 部分 (壓縮率不可預期) - Currently only provides `prune_by_std` for fully connected layers (unpredictable compression rate).
    *   可選擇實作 prune by percentile 與否 (壓縮率可預期) - Optionally implement `prune_by_percentile` (predictable compression rate).
*   **TODOs in `prune.py`:**
    *   **`prune_by_std(self, s=0.25)`:**
        *   `# TODO: Only fully connected layers were considered, but convolution layers also needed`
    *   **`_prune(self, module, threshold)`:**
        1.  `Use "module.weight.data" to get the weights of a certain layer of the model`
        2.  `Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged`
        3.  `Save the results of the step 2 back to "module.weight.data"`
    *   **`prune_by_percentile(self, q=DEFAULT_PRUNE_RATE)`:**
        *   `# TODO`
        *   `For each layer of weights W (including fc and conv layers) in the model, obtain the (100 – q)th percentile of absolute W as the threshold, and then set the absolute weights less than threshold to 0, and the rest remain unchanged.`
        *   `# Calculate percentile value`
        *   `# Prune the weights and mask`

### Pruning (`assignment/pruning.py`)
*   This script handles: Initial train + prune + prune retrain.
*   **TODO in `train(epochs)` function (within `pruning.py`):**
    *   Inside the loop `for name, p in model.named_parameters():`
        *   `# TODO: zero-out all the gradients corresponding to the pruned weights`
*   **Run:** `python pruning.py`
*   **Output:** `model_after_retraining.ptmodel` (in `saves/` directory, though slide 22 doesn't show path)

### Quantize (`assignment/net/quantization.py`)
*   This file contains the quantization logic (weight sharing).
*   `quantization.py`:
    *   只提供 fully connected 部分 - Currently only provides implementation for fully connected layers.
*   **TODO in `apply_weight_sharing(model, bits=5)` function (for convolution layers `elif len(shape) == 4`):**
    *   `Suppose the weights of a certain convolution layer are called "W"`
    1.  `Get the unpruned (non-zero) weights, "non-zero-W", from "W"`
    2.  `Use KMeans algorithm to cluster "non-zero-W" to (2 ** bits) categories`
    3.  `For weights belonging to a certain category, replace their weights with the centroid value of that category`
    4.  `Save the replaced weights in "module.weight.data", and need to make sure their indices are consistent with the original`
    *   `Finally, the weights of a certain convolution layer will only be composed of (2 ** bits) float numbers and zero`
*   **Run script:** `python weight_share.py` (This script calls `apply_weight_sharing`)
*   **Output:** `saves/model_after_weight_sharing.ptmodel`

### Huffman Coding (`assignment/net/huffmancoding.py`)
*   This file contains Huffman encoding/decoding logic for model parameters.
*   `huffmancoding.py`:
    *   分別有 encode、decode 部分 - Contains both encode and decode parts.
    *   且都只提供 fully connected 部分 - Both currently only support fully connected layers.
*   **TODO in `huffman_encode_conv(param, name, directory)`:**
    *   `You can refer to the code of the function "huffman_encode_fc" below, but note that "csr_matrix" can only be used on 2-dimensional data`
    *   **HINT:**
        *   `Suppose the shape of the weights of a certain convolution layer is (Kn, Ch, W, H)`
        1.  `Call function "csr_matrix" for all (Kn * Ch) two-dimensional matrices (W, H), and get "data", "indices", and "indptr" of all (Kn * Ch) csr_matrix.`
        2.  `Concatenate these 4 parts of all (Kn * Ch) csr_matrices individually into 4 one-dimensional lists, so there will be 4 lists.`
        3.  `Do huffman coding on these 4 lists individually.`
    *   The existing lines for dumping `conv` weights need to be modified.
*   **TODO in `huffman_decode_conv(param, name, directory)`:**
    *   `Decode according to the code of "conv" section you write in the function "huffman_encode_model" (referring to huffman_encode_conv) above, and refer to encode and decode code of "fc"`
    *   The existing lines for loading `conv` weights need to be modified.
*   **Run script:** `python huffman_encode.py` (This script calls functions in `net/huffmancoding.py`)
*   **Observe results:**
    *   壓縮率 (Compression rate) - Layer-wise and total.
    *   Accuracy - After decoding.
    *   Example output (slide 33): `total | 81378600 original bytes | 485484 compressed bytes | 167.62x improvement | 0.60% percent` and `Test set: Average loss: 2.6773, Accuracy: 5642/10000 (56.42%)`

## 3. 繳交內容 (Submission Content)

### TODO Summary
Complete the `# TODO` sections in the following Python files:
*   `assignment/net/prune.py`
*   `assignment/pruning.py`
*   `assignment/net/quantization.py`
*   `assignment/net/huffmancoding.py`

請根據 TODO 的提示內容完成程式 (Please complete the program according to the hints in the TODO sections).

### 繳交規格 (Submission Format)

1.  **Initial file structure:** As described in section 2.
2.  **After running all scripts, your `assignment` folder might look like this:**
    ```
    assignment/
    ├── data/                  (Downloaded by torchvision)
    ├── encodings/             (Generated by huffman_encode.py/huffmancoding.py)
    ├── net/                   (Your modified .py files)
    │   ├── huffmancoding.py
    │   ├── models.py
    │   ├── prune.py
    │   └── quantization.py
    ├── saves/                 (Generated models)
    │   ├── initial_model.ptmodel
    │   ├── model_after_retraining.ptmodel
    │   └── model_after_weight_sharing.ptmodel
    ├── huffman_encode.py
    ├── log.txt                (Generated by util.log)
    ├── pruning.py
    ├── util.py
    └── weight_share.py
    ```
3.  **Files to submit:**
    *   Compress the **`assignment/net/` directory** (containing your modified `huffmancoding.py`, `models.py`, `prune.py`, `quantization.py`) into a `.rar` file.
        *   Name: `學號.rar` (e.g., `108062345.rar`)
    *   Your final compressed model from the `saves/` directory.
        *   File: `model_after_weight_sharing.ptmodel`
        *   Rename to: `學號.ptmodel` (e.g., `108062345.ptmodel`)
    *   **最終繳交這兩項 (Finally, submit these two items):**
        1.  `學號.rar`
        2.  `學號.ptmodel`

## 4. 配分標準 (Grading Criteria)

*   **(70%) Prune model:** 完成 `prune.py`、`pruning.py` 中 #TODO (Complete #TODOs in `prune.py` and `pruning.py`).
*   **(10%) Quantize model:** 完成 `quantization.py` 中 #TODO (Complete #TODOs in `quantization.py`).
*   **(10%) Huffman coding:** 完成 `huffmancoding.py` 中 #TODO (Complete #TODOs in `huffmancoding.py`).
*   **(10%) 壓縮率排名 (Compression Rate Ranking):**
    *   Condition: `accuracy > 58%`.
    *   最終模型 compression rate 越高越好! (The higher the final model compression rate, the better!)
    *   排名前1/3得10分 (Top 1/3 get 10 points)
    *   中1/3得5分 (Middle 1/3 get 5 points)
    *   後1/3得0分 (Bottom 1/3 get 0 points)
    *   會用同學繳回的模型計算,排名改完會公布於 eeclass (Ranking will be calculated based on the models submitted by students and will be announced on eeclass).