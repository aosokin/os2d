from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

cfg = CN()

# GPU or CPU
cfg.is_cuda = True
# fix random seed to this value
cfg.random_seed = 42

cfg.model = CN()
# Backbone network to work with: ResNet50 | ResNet101
cfg.model.backbone_arch = "ResNet50"
# Share weights between feature extractors for class and input images
cfg.model.merge_branch_parameters = True
# Use the TransformNet for the inverse transform - it is slower, but compatible with the models from the weakalign paper
cfg.model.use_inverse_geom_model = True
# Use the TransformNet for the simplified affine transform - only translation and scaling defined by 4 parameters
cfg.model.use_simplified_affine_model = False
# Resize class images with preserving the aspect ratio such that the product of dimentions equals class_image_size squared
cfg.model.class_image_size = 240
# Use Group Normalization instead of BatchNorm
cfg.model.use_group_norm = False
# normalization
cfg.model.normalization_mean = [0.485, 0.456, 0.406]
cfg.model.normalization_std = [0.229, 0.224, 0.225]

cfg.init = CN()
# Path to the file from where to initialize the model (used for both training and evaluation), default - init from scratch
cfg.init.model = ""
# Path to the file from where to initialize the transformation network, overwrites the weights defined in init.model, default - follow init.model
cfg.init.transform = ""

# Training settings
cfg.train = CN()
# train or skip training and just do evaluation
cfg.train.do_training = True
# Batch size
cfg.train.batch_size = 4
# Maximum number of class images in a training batch (to control memory consumption)
cfg.train.class_batch_size = 15

# Name of the training dataset
cfg.train.dataset_name = "grozi-train"
# Scale of the training dataset.
# The longer image side used to sample train patches at training.
# This parameter should be set according to the expected object sizes in the dataset.
cfg.train.dataset_scale = 1280.0
# Cache dataset images in memory vs reading on the fly. Trade-off RAM vs dataloading speed
cfg.train.cache_images = True

cfg.train.objective = CN()
# Objective for the recognition head
# Options:
# "RLL"
# "HingeEmbeddingPosMargin"
# "HingeEmbeddingLinearPosMargin"
# "HingeEmbeddingLinear"
# "HingeEmbedding"
# "BCE"
cfg.train.objective.class_objective = "RLL"
# Margin for the negatives (if the loss needs it); scores normalized to [-1, 1]
cfg.train.objective.neg_margin = 0.5
# Margin for the positives (if the loss needs it); scores normalized to [-1, 1]
cfg.train.objective.pos_margin = 0.6
# Weight factor in front of the localization objective (Smooth-L1)"
cfg.train.objective.loc_weight = 0.2
# An anchor is positive if its IoU with any GT is >= this threshold
cfg.train.objective.positive_iou_threshold = 0.5
# An anchor is negative if its IoU with all GT is < this threshold
cfg.train.objective.negative_iou_threshold = 0.1
# Neg to pos ratio when hard negative mining is done inside the batch
cfg.train.objective.neg_to_pos_ratio = 3
# Weight factor in front of the begative component of the class objective
cfg.train.objective.class_neg_weight = 1.0
# Only for the RLL objective: ratio between the hardest and easiest negatives, this parameter is used instead of the temperature under the exponent
cfg.train.objective.rll_neg_weight_ratio = 0.001
# Flag saying whether to remap class targets after computing the network outputs
cfg.train.objective.remap_classification_targets = True
# Positive and negative IoU thresholds used for remapping
cfg.train.objective.remap_classification_targets_iou_pos = 0.8
cfg.train.objective.remap_classification_targets_iou_neg = 0.4

# Choose which parts of the model to train
cfg.train.model = CN()
# Train feature extractor or keep it frozen
cfg.train.model.train_features = True
# Flag to set BatchNorm in the feature extractor to the eval mode at training
cfg.train.model.freeze_bn = True
# Flag to set BatchNorm in the transformation model extractor to the eval mode at training
cfg.train.model.freeze_bn_transform = True
# Flag to freeze all transformation parameters at the training stage
cfg.train.model.freeze_transform = False
# Freeze this number of blocks of the feature extractor at training
cfg.train.model.num_frozen_extractor_blocks = 0
# Train the transformation model only on positives or on both positives and negatives
cfg.train.model.train_transform_on_negs = False

# data augmentation
cfg.train.augment = CN()
# Size of the cropped image patches used for training
cfg.train.augment.train_patch_width = 600
cfg.train.augment.train_patch_height = 600
# Ammount of scale jitter of the cropped patches, 1.0 - no jitter, the smaller - the more jitter
cfg.train.augment.scale_jitter = 0.7
# Ammount of aspect ratio jitter of the cropped patches, 1.0 - no jitter, the smaller - the more jitter
cfg.train.augment.jitter_aspect_ratio = 0.9
# Use batch flip data augmentation
cfg.train.augment.random_flip_batches = False
# Use color distortion data augmentation
cfg.train.augment.random_color_distortion = False
# Do random crops for class images: small scale and aspect ratio augmentation
cfg.train.augment.random_crop_class_images = False
# When random sampling patches for training this percentage of a positive box should be covered
cfg.train.augment.min_box_coverage = 0.7
# Mine extra class images as all the non-difficult positive examples from the training set
cfg.train.augment.mine_extra_class_images = False

# hard example mining
cfg.train.mining = CN()
# do it or not
cfg.train.mining.do_mining = False
# Remine hard negative classes after this number of steps
cfg.train.mining.mine_hard_patches_iter = 5000
# Save this many hard negative and hard positive patches per image
cfg.train.mining.num_hard_patches_per_image = 10
# Number of random scales to do patch mining, 0 - no randomness, use the ones of evaluation
cfg.train.mining.num_random_pyramid_scales = 2
# Number of random negative labels to test for hard negative mining, -1 - use all
cfg.train.mining.num_random_negative_classes = 200
# NMS threshold when mining hard patches - not to have identical patches
cfg.train.mining.nms_iou_threshold_in_mining = 0.5

# optimization
cfg.train.optim = CN()
# Learning rate
cfg.train.optim.lr = 1e-4
# Number of training steps
cfg.train.optim.max_iter = 200000
# Optimizer
cfg.train.optim.optim_method = "sgd"
cfg.train.optim.weight_decay = 1e-4
cfg.train.optim.sgd_momentum = 0.9
# Clip grad norm at this value
cfg.train.optim.max_grad_norm = 1e+2
# Anneal the learning rate with torch.optim.lr_scheduler.ReduceLROnPlateau strategy
cfg.train.optim.anneal_lr = CN()
# do it or not
# type of annealing: "None", "MultiStepLR", "ReduceLROnPlateau"
cfg.train.optim.anneal_lr.type = "none"
# For MultiStepLR:
cfg.train.optim.anneal_lr.milestones = []
cfg.train.optim.anneal_lr.gamma = 0.1
# For ReduceOnPlateau:
# Monitor this quantity when deciding to anneal learning rate
cfg.train.optim.anneal_lr.quantity_to_monitor = "mAP@0.50_grozi-val-new-cl"
#"min" | "max" depending on whether the quantity of interest is supposed to go up or down
cfg.train.optim.anneal_lr.quantity_mode = "max"
# Quantity improvement has to be at least this much to be significant (this threshold is relative to the characteritic value)
cfg.train.optim.anneal_lr.quantity_epsilon = 1e-2
# Multiply learning rate by this factor when decreasing
cfg.train.optim.anneal_lr.reduce_factor = 0.5
# The minimal value of learning rate
cfg.train.optim.anneal_lr.min_value = 1e-5
# Wait for this number of steps before annealing the learning rate hoping that the quantity will go the right way
cfg.train.optim.anneal_lr.patience = 1000
# Wait for this number of steps before annealing the learning rate initially (e.g. for warm starting)
cfg.train.optim.anneal_lr.initial_patience = 0
# Number of calls to wait before resuming normal operation after lr has been reduced.
cfg.train.optim.anneal_lr.cooldown = 10000
# When deciding to reduce LR use sliding window averages of this width
cfg.train.optim.anneal_lr.quantity_smoothness = 2000
# When reducing LR we can load the best model seen so far, or continue with the current one
cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr = True

# Evaluation parameters
cfg.eval = CN()
# Evaluate the current model after this number of steps
cfg.eval.iter = 5000
# Which dataset to use for evaluation. Can provide multiple datasets.
# The number of values in eval.dataset_names should equal the number of values in eval.dataset_scales or 1
cfg.eval.dataset_names = ["grozi-val-new-cl", "grozi-val-old-cl"]
# Scales of the evaluation datasets. 
# The number of values in eval.dataset_scales should equal the number of values in eval.dataset_names or 1
# The longer image side used to build image pyramids at evaluation.
# This parameter should be set according to the expected object sizes in the dataset. 
cfg.eval.dataset_scales = [1280]
# Cache dataset images in memory vs reading on the fly. Trade-off RAM vs dataloading speed
cfg.eval.cache_images = False
# Do multiscale evaluation - multiply the dataset scale by each of those
cfg.eval.scales_of_image_pyramid = [0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6]
# The number of images for intermediate evaluations on the training set
cfg.eval.train_subset_for_eval_size = 0
# Threshold on IoU to eliminate duplicates with NMS
cfg.eval.nms_iou_threshold = 0.3
# Threshold on scores to filter out low-scoring detections before NMS, default - use all detections
cfg.eval.nms_score_threshold = float("-inf")
# Do NMS on all classes jointly (default - separately for each class)
cfg.eval.nms_across_classes = False
# Threshold on IoU to be used in the mAP evaluation
cfg.eval.mAP_iou_thresholds = [0.5]
# Batch size for the evaluation stage
cfg.eval.batch_size = 1
# At test time wee sometimes need to to geometric augmentations of class images
# This improves matching to objects of different orientations
# Options: "", "rotation90" - 4 rotations, "horflip" - 2 flips
cfg.eval.class_image_augmentation = ""

# Logging parameters
cfg.output = CN()
# Where to store results and models (default: do not save)
cfg.output.path = ""
# Save screen output to log.txt or not
cfg.output.save_log_to_file = False
# Print to the screen after this number of steps
cfg.output.print_iter = 1
# Save the current model after this number of steps
cfg.output.save_iter = 50000
# checkpoint the best model
cfg.output.best_model = CN()
cfg.output.best_model.do_get_best_model = False
cfg.output.best_model.dataset = "" # if empty - use the first validation dataset
cfg.output.best_model.metric = "mAP@0.50"
cfg.output.best_model.mode = "max" # "max" or "min"

# visualizations
cfg.visualization = CN()
# visualizations at eval
cfg.visualization.eval = CN()

cfg.visualization.eval.show_gt_boxes = False

cfg.visualization.eval.show_detections = False
# Threshold on the scores for visualizations, default - show a fixed number of highest scoring detections
cfg.visualization.eval.max_detections = 10
cfg.visualization.eval.score_threshold = float("-inf")

cfg.visualization.eval.show_class_heatmaps = False
# pairs of images and labels to show in this vizualization
cfg.visualization.eval.images_for_heatmaps = []
cfg.visualization.eval.labels_for_heatmaps = []

# path for saving detection results, defult - no saving
cfg.visualization.eval.path_to_save_detections = ""

# visualizations at train
cfg.visualization.train = CN()
cfg.visualization.train.show_gt_boxes_dataloader = False

cfg.visualization.train.show_detections = False
# Threshold on the scores for visualizations, default - show a fixed number of highest scoring detections
cfg.visualization.train.max_detections = 5
cfg.visualization.train.score_threshold = float("-inf")

cfg.visualization.train.show_target_remapping = False

# visualizations when mining patches
cfg.visualization.mining = CN()
cfg.visualization.mining.show_gt_boxes = False

cfg.visualization.mining.show_class_heatmaps = False
# pairs of images and labels to show in this vizualization
cfg.visualization.mining.images_for_heatmaps = []
cfg.visualization.mining.labels_for_heatmaps = []

cfg.visualization.mining.show_mined_patches = False
cfg.visualization.mining.max_detections = 10
cfg.visualization.mining.score_threshold = float("-inf")
