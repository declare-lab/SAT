# SAT: Improving Semi-Supervised Text Classification with Simple Instance-Adaptive Self-Training
This repository contains the official implementation code of the EMNLP 2022 Findings short paper [SAT: Improving Semi-Supervised Text Classification with Simple Instance-Adaptive Self-Training](https://arxiv.org/pdf/2210.12653v1.pdf).


## Usage

1. Set up the environment
```
conda create -n sat python==3.7.5
conda activate sat
cd SAT/
pip3 install -r requirements.txt
```

2. Training
```
cd src/
bash run.sh
```
The parameters.txt shows a list of hyper-parameters.

## Citation
Please cite our paper if you find our work useful for your research:
```bibtex
@inproceedings{chen-etal-2022-sat,
    title = "{SAT}: Improving Semi-Supervised Text Classification with Simple Instance-Adaptive Self-Training",
    author = "Chen, Hui  and  Han, Wei  and  Poria, Soujanya",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.456",
    pages = "6141--6146",
}

```

## Contact
Should you have any questions, feel free to contact [chchenhui1996@gmail.com](chchenhui1996@gmail.com).
