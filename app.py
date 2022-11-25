from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import torchvision.transforms as transforms

from os2d.config import cfg
from os2d.structures.bounding_box import filter_bbox, convert_xyxy_bbox_to_relative_coords
from os2d.engine.optimization import create_optimizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.modeling.model import build_os2d_from_config
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, setup_logger, decode_base64_to_image, get_image_size_after_resize_preserving_aspect_ratio

class ImageRequest(BaseModel):
    content: str

class QueryImageResquest(BaseModel):
    image: ImageRequest
    query: List[ImageRequest]

def preprocess_image(image, transform_image, target_size, cuda=True):
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=image.size[1],
                                                               w=image.size[0],
                                                               target_size=target_size)
    image = image.resize((w, h))
    image = transform_image(image)
    if cuda:
        image = image.cuda()
    return image

def init_logger(cfg):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("OS2D", output_dir if cfg.output.save_log_to_file else None)

model_path = "models/os2d_v2-train.pth"
init_logger(cfg)

app = FastAPI()

cfg.visualization.eval.max_detections = 30
cfg.visualization.eval.score_threshold = 0.45

@app.post('/detect-all-instances')
def query_image(request: QueryImageResquest):
    # set this to use faster convolutions
    cfg.is_cuda = torch.cuda.is_available()
    
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)


    # Model
    cfg.init.model = model_path
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

    # Optimizer
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)

    # load the dataset
    input_image = decode_base64_to_image(request.image.content)
    query_image = [decode_base64_to_image(image.content) for image in request.query]
    class_ids = [0 for _ in range(len(query_image))]

    transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])
    input_processed = preprocess_image(input_image, transform_image, 1500, cfg.is_cuda).unsqueeze(0)
    input_h, input_w = input_processed.size()[-2:]
    
    query_processed = [preprocess_image(image, transform_image, cfg.model.class_image_size, cfg.is_cuda) for image in query_image]

    with torch.no_grad():
        loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_processed, class_images=query_processed)
    

    image_loc_scores_pyramid = [loc_prediction_batch[0]]
    image_class_scores_pyramid = [class_prediction_batch[0]]
    img_size_pyramid = [FeatureMapSize(img=input_processed)]
    transform_corners_pyramid = [transform_corners_batch[0]]

    boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                    img_size_pyramid, class_ids,
                                    nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                    nms_score_threshold=cfg.eval.nms_score_threshold,
                                    transform_corners_pyramid=transform_corners_pyramid)

    # remove some fields to lighten visualization                                       
    boxes.remove_field("default_boxes")

    scores, boxes_coords = filter_bbox(boxes, cfg.visualization.eval.score_threshold, cfg.visualization.eval.max_detections)
    boxes_coords = [convert_xyxy_bbox_to_relative_coords(box, im_height=input_h, im_width=input_w) for box in boxes_coords.tolist()]
    return JSONResponse(content={'scores': scores.tolist(), 'bboxes': boxes_coords})