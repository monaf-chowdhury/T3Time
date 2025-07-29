<div align="center">
  <!-- <h1><b> TimeCMA </b></h1> -->
  <!-- <h2><b> Time-CMA </b></h2> -->
  <h2><b> (AAAI'26) (In Submission) T3Time: Tri-Modal Time Series Forecasting via Adaptive Multi-Head Alignment and Residual Fusion </b></h2>
</div>

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2406.01638-B31B1B.svg)](https://arxiv.org/abs/2406.01638)

> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:

<!-- ```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
``` -->

## Abstract
Multivariate time series forecasting (MTSF) seeks to model temporal dynamics among variables to predict future trends. Transformer-based models and large language models (LLMs) have shown promise due to their ability to capture long-range dependencies and patterns. However, current methods often rely on rigid inductive biases, ignore inter variable interactions, or apply static fusion strategies that limit adaptability across forecast horizons. These limitations create bottlenecks in capturing nuanced, horizon-specific relationships in time-series data. To solve this problem, we propose T3Time, a novel trimodal framework consisting of time, spectral, and prompt branches, where the dedicated frequency encoding branch captures the periodic structures along with a gating mechanism that learns prioritization between temporal and spectral features based on the prediction horizon. We also proposed a mechanism which adaptively aggregates multiple cross-modal alignment heads by dynamically weighting the importance of each head based on the features. Extensive experiments on benchmark datasets demonstrate that our model consistently outperforms state-of-the-art baselines, achieving an average reduction of 3.37% in MSE and 2.08% in MAE. Furthermore, it shows strong generalization in few-shot learning settings: with 5% training data, we see a reduction in MSE and MAE by 4.13% and 1.91%, respectively; and with 10% data, by 3.70% and 1.98% on average. Code is available at: https://github.com/monaf-chowdhury/T3Time

<p align="center">
  <img src="attachments/methodology.png" width="100%">
</p>

## Dependencies

* Python 3.11
* PyTorch 2.1.2
* cuda 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_windows.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and [TFB](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view).

## Usages
* ### Last token embedding storage

```bash
bash Store_{data_name}.sh
```

* ### Train and inference
   
```bash
bash {data_name}.sh
```

## Contact Us
For inquiries or further assistance, contact us at [monafabdul15@gmail.com](mailto:monafabdul15@gmail.com) or open an issue on this repository.
