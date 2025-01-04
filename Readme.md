
This repository contains the Pytorch implementation of

A Dynamic Multi-Modal Deep Reinforcement Learning Framework for 3D Bin Packing Problem

Anhao Zhao, Caixia Rong, Tianrui Li,Liangcai Lin

The 3D bin packing problem, a notorious NPhard combinatorial optimization challenge with wide-ranging practical implications, focuses on optimizing spatial allocation within a bin through the arrangement of boxes. Previous studies have shown promise in employing Neural Combinatorial Optimization to tackle such intractable problems. However, due to the inherent diversity in observations and the sparse rewards, current learning-based methods for addressing the 3D bin packing problem have yielded less-than-ideal outcomes. In response to this shortfall, we propose a novel Dynamic Multi-modal deep Reinforcement Learning framework tailored specifically for the 3D Bin Packing Problem, coined as DMRLBPP. This framework stands apart from existing learning-based bin-packing approaches in two pivotal respects. Firstly, in order to capture the range of observations, we introduce an innovative dynamic multi-modal encoder, comprising dynamic boxes state and gradient height map sub-encoders, effectively modeling multi-modal information. Secondly, to mitigate the challenge of sparse rewards, we put forth a novel reward function, offering an efficient and adaptable approach for addressing the 3D bin packing problem. Extensive experimental results validate the superiority of our method over all baseline approaches, across instances of varying scales.

On the basis of previous work, we further add the following contents to the model:
1. Compatible with a non-fixed number of box_num
2. Supports multiple container packaging scenarios
3. Introduce large model container selection

This model is for task3 ablation study. Here, we use random container selection and max_size container selection.
Please participate in the paper for details

## Usage

### Preparation

1. Install conda
2. Run `conda env create -f environment.yml` (please change name and prefix according to your preference)

### Train

1. Modify the config file in `config.py` as you need.
2. Run `python main.py`. (for task3 main, switch to main branch)
