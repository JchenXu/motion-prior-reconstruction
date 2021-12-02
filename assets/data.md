## Data Generation

### Download the pre-processed data
We provide our used pre-processed data from [here](https://drive.google.com/drive/folders/1qvnyoHyG6TYXYrxT4rvpXPdX49Lj-sx8?usp=sharing). Besides, we use the *mpii3d_train_scale12_occ_db.pt* and *3dpw_train_occ_db.pt* provided by [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE), and you can download them [here](https://drive.google.com/drive/folders/1ZP6iNkYvhWKuYAvqnwVYRBoc1xhIneSP)

### Human3.6M
In this project, we also use human3.6m for training and evaluation. Please download the human3.6m and run the script as follows to generate data file:
```bash
python lib/data_utils/h36m_utils.py --dir <your_h36m_path>
```
Note that, we use occluder to generate the *h36m_p1_train_50fps_occ_db.pt* and check [here](https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/asset/data.md) for how to use the occluder. 

### InstarVariety
For InstarVariety, we use the data processed by [VIBE](https://github.com/mkocabas/VIBE/)

### Other Dataset
If you want to add other dataset, we provide the script in *lib/data_utils/* and you can follow the instruction in VIBE and TCMR to generate your data file.
