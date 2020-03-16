import logging
import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data.transforms import transforms as maskrcnn_transforms

from os2d.data.dataset import build_dataset_by_name


def build_maskrcnn_model(config_path, weight_path):
    logger = logging.getLogger("detector-retrieval.build_retrievalnet")
    logger.info("Building the maskrcnn-benchmark model...")
    # get the config file
    cfg.merge_from_file(config_path)

    # check the number of classes
    if cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES == 0:
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2

    assert cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES == 2, "We need a one-class detector, but have {0} classes".format(cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

    # build the model
    model = build_detection_model(cfg)

    # load the weights
    logger.info("Loading weights from {}".format(weight_path))
    loaded = torch.load(weight_path, map_location=torch.device("cpu"))
    if "model" not in loaded:
        loaded = dict(model=loaded)
    load_state_dict(model, loaded.pop("model"))

    return model, cfg


def run_maskrcnn_on_images(model, cfg, batch_images):
    # apply maskrcnn normalization
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = maskrcnn_transforms.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255)
    device = batch_images.device
    batch_images = [normalize_transform(img.cpu(), None).to(device=device) for img in batch_images]

    # convert images to maskrcnn format
    batch_images = to_image_list(batch_images, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)

    # model to GPU and eval mode
    model.to(device)
    model.eval()

    # run
    with torch.no_grad():
        output = model(batch_images)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bboxes_xyxy = [o.bbox for o in output]
    labels = [o.get_field("labels") for o in output]
    scores = [o.get_field("scores") for o in output]

    return bboxes_xyxy, labels, scores
