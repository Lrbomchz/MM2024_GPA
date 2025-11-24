## Global Patch-wise Attention is Masterful Facilitator for Masked Image Modeling

This repository includes the supporting code for:

Xi, G., Tian, Y., Yang, M., Zhang, L., Que, X., & Wang, W. (2024, October). Global Patch-wise Attention is Masterful Facilitator for Masked Image Modeling. ACM MM 2024.

## Requirements

python==3.8

torch==1.13.0

torchvision

timm==0.3.2



### Pre-train

We pretrained our models on **2×3090 GPUs**，the **GPA as a teacher** runs 5 days for 200 epochs, while the **GPA as both teacher and feature** runs 8 days for 200 epochs. : (



pretraining GPA to apply GPA as a teacher:

```shell
OMP_NUM_THREADS=1  python  -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py --accum_iter 16 --batch_size 128 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 200 --warmup_epochs 20 --blr 1.5e-4 --weight_decay 0.05 --data_path /path/to/ImageNet --output_dir /path/to/output_dir/ --alpha0 0.0 --alphaT 0.5 --teacher_path /path/to/vit_teacher
```



pretraining GPA to apply GPA as both teacher and feature:

```shell
OMP_NUM_THREADS=1  python  -m torch.distributed.launch --nproc_per_node=2 main_pretrain_learn_att.py --accum_iter 16 --batch_size 128 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 200 --warmup_epochs 20 --blr 1.5e-4 --weight_decay 0.05 --data_path /path/to/ImageNet --output_dir /path/to/output_dir/ --alpha0 0.0 --alphaT 0.5 --teacher_path /path/to/vit_teacher
```



### Fine-tune

```shell
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py --accum_iter 4 --batch_size 128 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.8 --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --dist_eval --data_path /path/to/ImageNet --output_dir /path/to/output_dir/ --finetune /path/to/pretrained_maae_checkpoint
```

