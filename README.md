# Self-supervised Multimodal Versatile Networks

The original github repository is: https://github.com/deepmind/deepmind-research/tree/master/mmv

# My contribution:

I have modified the model to just find the embedding at each layer. Download the mmv_tsm_resnet_x2.pkl from the original repository.

# Execution Step

Execute the following command:

```
python find_features.py --checkpoint_path <path to the downloaded .pkl file> --dataset_folder <path where all the videos are present> --output_folder <path to store the outputs of all the videos>
```