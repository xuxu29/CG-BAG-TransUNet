# CG-BAG-TransUNet
### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

[Update 2026/02] The official ViT weights appear to have expired. 
You can still download a copy from the [project folder](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) (same to BTCV preprocessed data). After extraction, find the file at:
`../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`

### 2. Prepare data (All data are available!)

All data are available so no need to send emails for data. Please use the [BTCV preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC data](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link).

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. 

```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16  --batch_size 24 --max_epochs 100  --use_bamfe 1 --use_att_gate 1 --att_gate_depth 2 --use_dppf 
```

- Run the test script on synapse dataset. 

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16  --use_bamfe 1 --use_att_gate 1 --att_gate_depth 2 --use_dppf --model_path
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations


```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
