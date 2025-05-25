From Pixels to Proof: Detecting and Explaining AI-Generated Artifacts
======================================================================
Task 1:Submission to InterIIT Tech Meet 2024 at IIT Bombay.

Project Overview:
-----------------
Real vs Fake Detection using TinyCLIP + RINE architecture
The models are optimized for 32x32 image inputs to enable efficient deployment on edge devices.

Key Highlights:
---------------
- Combines Computer Vision and NLP
- Uses adversarially trained TinyCLIP for classification
- Supports low-resolution images (32x32) to reduce computational cost
- Dataset includes MSCOCO, Synthbuster, ProGAN, StyleGAN, DALL-E, and others

Installation:
-------------
1. Clone the repo
'''
git clone https://github.com/dev484p/AdversarialDeepFake_Detection.git
'''

3. Download saved Checkpoints from [drive](https://drive.google.com/file/d/1t7qTzaoj5_Bq-pZ6fNTeZeO9E1PGx1oB/view?usp=sharing) and paste it in ckpt/hyp/wkcn directory.
   

Model Architecture:
-------------------
- TinyCLIP + RINE
- Uses Transformer encoder block outputs
- Binary Cross Entropy + Supervised Contrastive Loss for detection
- Dual Clip Architecture With Progressive Adversarial Training (Cosine similarity + classification loss)

Results (Detection Accuracy):
-----------------------------
| Generator         | Accuracy (%) | Average Precision (%) |
|------------------|--------------|------------------------|
| GLIDE            | 78.9         | 90.6                   |
| DALL-E 2         | 62.8         | 70.2                   |
| StyleGAN         | 55.3         | 63.0                   |
| Firefly          | 54.4         | 59.0                   |
| Mean (All)       | 53.5         | 55.1                   |

