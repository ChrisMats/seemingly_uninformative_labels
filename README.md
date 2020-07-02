# Adding seemingly uninformative labels helps in low data regimes
<p align="center">
  <img width="60%" src="generalization_gap.png">
</p>

## Enviroment setup
To install the enviroment we use run:
```conda env create -f environment.yml```

## Usage:
* Training: ```python ./segmentation.py --json_path params.json```
* Testing (using json file): ```python ./segmentation.py --json_path params.json --test```
* Testing (using saved checkpoint): ```python ./segmentation.py --checkpoint CheckpointName --test```
* Fine tune the learning rate: ```python ./segmentation.py --json_path params.json --lr_finder```

## Configuration (json file)

* dataset_params
  * data_location: Location that the datasets are located
  * dataset_location: Location of the dataset inside the data_location (Cityscapes, VOC, CSAWS)
  * crop_size: Patch size for the CSAW-S dataset
  * is_binary: If True the masks are converted to binary masks (main target - background)
  * use_full_training_set: If true all the training examples are used during training
  * how_many_samples: Number of examples to include for training (if use_full_training_set is False)
  * subset_n: Which subset of the full training set to use (if use_full_training_set is False)
  * main_target: The main target (cancer for CSAW-S, person for Cityscapes and Pascal VOC)
  * annotator_id: The annotator's ID for the test set on CSAW-S (int values 1-3)
  * is_coarse: If True, it uses the coarse complementary labels on Cityscapes
  * bootstrap_images: If False, it orders the cities on Cityscapes for the training set
  * n_complementary_labels: Number of complementary labels to include (accepts int or "all")
  * leave_one_out: Which complementary label to exclude for the leave-one-out experiments (if apply is true)
  * test_on_gold_standard: If True, it evaluates on the golden standard (works only with CSAW-S)
  * download_data: Download data for Pascal VOC
  * train_transforms: Defines the augmentations for the training set
  * val_transforms: Defines the augmentations for the validation set
  * test_transforms: Defines the augmentations for the test set
* dataloader_params: Defines the dataloader parameters (batch size etc)
* model_params
  * backbone_type: type of the backbone model (using resnets)
  * segmentation_type: deeplabv3 or FCN
  * pretrained: If True, it uses ImageNet pretrained weights
  * freeze_backbone: If True, it freezes the backbone network
  * goup_norm
    * replace_with_goup_norm: If True, it replaces BatchNorm with GroupNorm
    * num_groups: Number of groups for the GroupNorm
    * keep_stats: If true, it initializes with pretrained statistics on ImageNet
* training_params: Define learning rate, weight decay, learning rate schedule etc.
  * keep val_every and save_every to 1
  * log_every: Number of iterations that validates and saves the model
  * lr_scheduler (MultiStepLR, ReduceLROnPlateau or None)
* system_params: Defines if GPUs are used, which GPUs etc.
* log_params: Project and run name for the logger (we are using [Weights & Biases](https://www.wandb.com/))
* lr_finder: Define the learning rate parameters
  * grid_search_params
    * min_pow, min_pow: The min and max power of 10 for the search
    * resolution: How many different learning rates to try
    * n_epochs: maximum epochs of the training session
    * random_lr: If True, it uses random learning rates withing the accepted range
    * report_intermediate_steps: If True, it logs if validates throughout the training sessions
