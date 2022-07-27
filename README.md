# SAR Image Noise Despeckling 
### since   2022. 07. 04 
### company GDL, HNU 

# Noise2Self: Blind Denoising by Self-Supervision

This repo demonstrates a framework for blind denoising high-dimensional measurements,
as described in the [paper](https://arxiv.org/abs/1901.11365). It can be used to calibrate 
classical image denoisers and train deep neural nets; 
the same principle works on matrices of single-cell gene expression.


### Traditional Supervised Learning

```
for i, batch in enumerate(data_loader):
    noisy_images, clean_images = batch
    output = model(noisy_images)
    loss = loss_function(output, clean_images)
```

### Self-Supervised Learning

```
from mask import Masker
masker = Masker()
for i, batch in enumerate(data_loader):
    noisy_images, _ = batch
    input, mask = masker.mask(noisy_images, i)
    output = model(input)
    loss = loss_function(output*mask, noisy_images*mask)
```

Dependencies are in the `environment.yml` file.

The remaining notebooks generate figures from the [paper](https://arxiv.org/abs/1901.11365).


### Train Usage( work_train.py )
```
python work_train.py 
    --img-path ./sar_data 
    --img-file-name 200px-Hunmin_jeong-eum.png 
    --epoch 100 
    --lr 0.1
```