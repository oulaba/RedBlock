# RedBlock

## 📌 Overview

This repository provides the official implementation of our paper *"RedBlock: Robust and Efficient Multi-bit Watermarking for LLM-Generated Text via Dual-Scale Redundancy"*.  

## ⚙️ Installation

### 1. Environment Setup

Python **3.9+** is recommended. 
Install depandencies using

```python
pip install -r requirements.txt
```

## Demo

This section introduces an example usage of our watermark scheme. To run on `llama-2-7b`, please login with you hugggingface token that has been granted access. If using 4 colorlist:
- For RedBlock-rh:
  ```
  bash sample-entrypoint.sh
  ```
- For RedBlock-bh, first we need to generate frequency mapping:
  ```
  cd balance_hash
  python3 helper.py -p 4 --path ../token_freq_llama.pkl --save './map_freq_8b.pkl'
  ```
then for quick run,try running
  ```
  bash sample-entrypoint.sh
  ```





