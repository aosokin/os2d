import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_coder import Os2dBoxCoder, BoxGridGenerator
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList, cat_boxlist


def build_os2d_head_creator(do_simple_affine, is_cuda, use_inverse_geom_model, feature_map_stride, feature_map_receptive_field):
    aligner = Os2dAlignment(do_simple_affine, is_cuda, use_inverse_geom_model)
    head_creator = Os2dHeadCreator(aligner, feature_map_stride, feature_map_receptive_field)
    return head_creator


def convert_box_coordinates_local_to_global(resampling_grids, default_boxes_xyxy):
    # get box transformations:
    # x_global = (x_2 - x_1) / 2 * x_local + (x_1 + x_2) / 2 = x_A * x_local + x_B
    # y_global = (y_2 - y_1) / 2 * y_local + (y_1 + y_2) / 2 = y_A * y_local + y_B
    box_transforms_x_A = (default_boxes_xyxy.narrow(-1, 2, 1) - default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_x_B = (default_boxes_xyxy.narrow(-1, 2, 1) + default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_y_A = (default_boxes_xyxy.narrow(-1, 3, 1) - default_boxes_xyxy.narrow(-1, 1, 1)) / 2
    box_transforms_y_B = (default_boxes_xyxy.narrow(-1, 3, 1) + default_boxes_xyxy.narrow(-1, 1, 1)) / 2

    resampling_grids_size = [-1] * resampling_grids.dim()
    resampling_grids_size[-2] = resampling_grids.size(-2)
    resampling_grids_size[-3] = resampling_grids.size(-3)
    add_dims = lambda x: x.unsqueeze(-2).unsqueeze(-3).expand(resampling_grids_size)
    # convert to the original coordinates
    b_x_A = add_dims(box_transforms_x_A)
    b_x_B = add_dims(box_transforms_x_B)
    b_y_A = add_dims(box_transforms_y_A)
    b_y_B = add_dims(box_transforms_y_B)
    resampling_grids_x = resampling_grids.narrow(-1, 0, 1) * b_x_A + b_x_B
    resampling_grids_y = resampling_grids.narrow(-1, 1, 1) * b_y_A + b_y_B
    resampling_grids_global = torch.cat([resampling_grids_x, resampling_grids_y], -1)

    return resampling_grids_global


class Os2dAlignment(nn.Module):
    """This class contains all the operations related to the transfomation computation.
    If adding new transformation type, only this class should be changed.
    """
    def __init__(self, do_simple_affine, is_cuda, use_inverse_geom_model):
        super(Os2dAlignment, self).__init__()

        self.model_type = "affine" if not do_simple_affine else "simple_affine" # "affine" or "simple_affine"
        self.use_inverse_geom_model = use_inverse_geom_model

        # create the parameter regression network
        if self.model_type == "affine":
            transform_net_output_dim = 6
        elif self.model_type == "simple_affine":
            transform_net_output_dim = 4
        else:
            raise(RuntimeError("Unknown transformation model \"{0}\"".format(self.model_type)))

        # all these numbers are semantically different, but are set to 15 due to the details in the model architecture
        # these number have to be compatible with the network regressing transformation parameters
        # following the weakalign code, we use 15 here
        # all the sizes are in (H, W) format
        # NOTE: tenchically the code should work with non-square grids, but this was never tested, so expect bugs
        self.out_grid_size = FeatureMapSize(w=15, h=15)
        self.reference_feature_map_size = FeatureMapSize(w=15, h=15)
        self.network_stride = FeatureMapSize(w=1, h=1)
        self.network_receptive_field = FeatureMapSize(w=15, h=15)

        self.input_feature_dim = self.reference_feature_map_size.w * self.reference_feature_map_size.h
        self.parameter_regressor = TransformationNet(output_dim=transform_net_output_dim,
                                                     use_cuda=is_cuda,
                                                     normalization='batchnorm', # if not self.use_group_norm else 'groupnorm',
                                                     kernel_sizes=[7, 5],
                                                     channels=[128, 64],
                                                     input_feature_dim=self.input_feature_dim)



    def prepare_transform_parameters_for_grid_sampler(self, transform_parameters):
        """Function to standardize the affine transformation models:
         - either full of simplified transformation based on self self.model_type (defined in self.__init__)
         - use invertion or not based on self.use_inverse_geom_model (defined in self.__init__)
        Prepares transform parameters to be used with apply_affine_transform_to_grid
        Args:
            transform_parameters (Tensor[float], size = batch_size x num_params x h^A x w^A) - contains the transformation parameters for each image-class pair and for each spatial location in the image
            Here the batch size equals the product of the image batch size b^A and class batch size b^C
            The number of parameters num_params equals 6 for the full affine transform, and 4 for the simlified version (translation and scaling only)

        Returns:
             transform_parameters (Tensor[float], size = (batch_size * h^A * w^A) x 2, 3) - contains the tranformation parameters prepared for apply_affine_transform_to_grid
        """
        num_params = transform_parameters.size(1)
        transform_parameters = transform_parameters.transpose(0, 1)  # num_params x batch_size x image_height x image_width
        transform_parameters = transform_parameters.contiguous().view(num_params, -1) # num_params x -1

        if self.model_type == "affine":
            assert num_params == 6, 'Affine tranformation parameter vector has to be of dimension 6, have {0} instead'.format(num_params)
            transform_parameters = transform_parameters.transpose(0, 1).view(-1, 2, 3) # -1, 2, 3 - shape for apply_affine_transform_to_grid function
        elif self.model_type == "simple_affine":
            assert num_params == 4, 'Simplified affine tranformation parameter vector has to be of dimension 4, have {0} instead'.format(num_params)
            zeros_to_fill_blanks = torch.zeros_like(transform_parameters[0])
            transform_parameters = torch.stack( [transform_parameters[0], zeros_to_fill_blanks, transform_parameters[1],
                                                 zeros_to_fill_blanks, transform_parameters[2], transform_parameters[3]] ,dim=1)
            transform_parameters = transform_parameters.view(-1, 2, 3)
            # -1, 2, 3 - shape for apply_affine_transform_to_grid function
        else:
            raise(RuntimeError("Unknown transformation model \"{0}\"".format(self.model_type)))

        if self.use_inverse_geom_model:
            assert self.model_type in ["affine", "simple_affine"], "Invertion of the transformation is implemented only for the affine transfomations"
            assert transform_parameters.size(-2) == 2 and transform_parameters.size(-1) == 3, "transform_parameters should be of size ? x 2 x 3 to interpret them ass affine matrix, have {0} instead".format(transform_parameters.size())
            grid_batch_size = transform_parameters.size(0)

            # # slow code:
            # lower_row = torch.tensor([0,0,1], device=transform_parameters.device, dtype=transform_parameters.dtype)
            # lower_row = torch.stack([lower_row.unsqueeze(0)] * grid_batch_size, dim=0)
            # faster version
            lower_row = torch.zeros(grid_batch_size, 1, 3, device=transform_parameters.device, dtype=transform_parameters.dtype)
            lower_row[:, :, 2] = 1

            full_matrices = torch.cat( [transform_parameters, lower_row], dim=1 )

            def robust_inverse(batchedTensor):
                try:
                    inv = torch.inverse(batchedTensor)
                except:
                    n = batchedTensor.size(1)
                    batchedTensor_reg = batchedTensor.clone().contiguous()
                    for i in range(n):
                        batchedTensor_reg[:,i,i] = batchedTensor_reg[:,i,i] + (1e-5)
                    inv = torch.inverse(batchedTensor_reg)
                return inv

            def batched_inverse(batchedTensor):
                """
                A workaround of a bug in the pytorch backend from here:
                https://github.com/pytorch/pytorch/issues/13276
                """
                if batchedTensor.shape[0] >= 256 * 256 - 1:
                    temp = []
                    for t in torch.split(batchedTensor, 256 * 256 - 1):
                        temp.append(robust_inverse(t))
                    return torch.cat(temp)
                else:
                    return robust_inverse(batchedTensor)

            inverted = batched_inverse(full_matrices)
            transform_parameters = inverted[:,:2,:]
            transform_parameters = transform_parameters.contiguous()

        return transform_parameters

    def forward(self, corr_maps):
        """
        Args:
            corr_maps (Tensor[float]): a batch_size x num_features x h^A x w^A tensor containing the transformation parameters for each image-class pair and for each spatial location in the image
            Here the batch size batch_size equals the product of the image batch size b^A and class batch size b^C
            The number of channels num_features should be compatible with the created feature regression network (equals 225 for the weakalign models).

        Returns:
            resampling_grids_local_coord (Tensor[float]): a batch_size x h^A x w^A x out_grid_height x out_grid_width x 2 tensor
            The tensor represents a grid of points under the computed transformations for each batch element and each spatial location.
            Each point has two coordinates: x \in [-1, 1] and y \in  [-1,1].
            CAUTION! each point is in the coordinate system local to the corresponding spatial location
        """

        batch_size = corr_maps.size(0)
        fm_height = corr_maps.size(-2)
        fm_width = corr_maps.size(-1)
        assert corr_maps.size(1) == self.input_feature_dim, "The dimension 1 of corr_maps={0} should be equal to self.input_feature_dim={1}".format(corr_maps.size(1), self.input_feature_dim)

        # apply the feature regression network (initial ReLU + normalization is inside)
        transform_parameters = self.parameter_regressor(corr_maps)  # batch_size x num_params x image_height x image_width

        # process transform parameters (convert the full affine and invert if needed)
        transform_parameters = self.prepare_transform_parameters_for_grid_sampler(transform_parameters)

        # compute the positions of the grid points under the transformations
        # this is an analogue of AffineGridGenV2 from
        # https://github.com/ignacio-rocco/weakalign/blob/dd0892af1e05df1765f8a729644a33ed75ee657e/geotnf/transformation.py
        # Note that it is important to have non-default align_corners=True, otherwise the results differ
        resampling_grids_local_coord = F.affine_grid(transform_parameters, torch.Size((transform_parameters.size(0), 1, self.out_grid_size.h, self.out_grid_size.w)), align_corners=True)
        # size takes batch_size, num_channels (ignored), grid height, grid width; both height and width are in the range [-1, 1]
        # coordinates are in the local box coordinate system

        # CAUTION! now we have all the points in the local to each spatial location coordinate system
        assert resampling_grids_local_coord.ndimension() == 4 and resampling_grids_local_coord.size(-1) == 2 and resampling_grids_local_coord.size(-2) == self.out_grid_size.w and resampling_grids_local_coord.size(-3) == self.out_grid_size.h, "resampling_grids_local_coord should be of size batch_size x out_grid_width x out_grid_height x 2, but have {0}".format(resampling_grids_local_coord.size())

        resampling_grids_local_coord = resampling_grids_local_coord.view(batch_size, fm_height, fm_width, self.out_grid_size.h, self.out_grid_size.w, 2)

        return resampling_grids_local_coord


def spatial_norm(feature_mask):
    mask_size = feature_mask.size()
    feature_mask = feature_mask.view(mask_size[0], mask_size[1], -1)
    feature_mask = feature_mask / (feature_mask.sum(dim=2, keepdim=True))
    feature_mask = feature_mask.view(mask_size)
    return feature_mask


class Os2dHeadCreator(nn.Module):
    """Os2dHeadCreator creates specific instances of Os2dHead that contain features extracted from lavel/query images

    Note: the Os2dHeadCreator objects should be a submodule of the Os2dModel object as it has trainable parameters:
        TransformNet in self.aligner, the Os2dHead objects should not be submodules of Os2dModel.
        At the same time, the forward method is implemented only in Os2dHead, but not in Os2dHeadCreator
    """
    def __init__(self, aligner, feature_map_stride, feature_map_receptive_field):
        super(Os2dHeadCreator, self).__init__()
        # create the alignment module
        self.aligner = aligner

        rec_field, stride = self.get_rec_field_and_stride_after_concat_nets(feature_map_receptive_field, feature_map_stride,
                                                                             self.aligner.network_receptive_field, self.aligner.network_stride)
        self.box_grid_generator_image_level = BoxGridGenerator(box_size=rec_field, box_stride=stride)
        self.box_grid_generator_feature_map_level = BoxGridGenerator(box_size=self.aligner.network_receptive_field,
                                                                     box_stride=self.aligner.network_stride)

    @staticmethod
    def get_rec_field_and_stride_after_concat_nets(receptive_field_netA, stride_netA,
                                                   receptive_field_netB, stride_netB):
        """We are concatenating the two networks  net(x) = netB(netA(x)), both with strides and receptive fields.
        This functions computes the stride and receptive field of the combination
        """
        if isinstance(receptive_field_netA, FeatureMapSize):
            assert isinstance(stride_netA, FeatureMapSize) and isinstance(receptive_field_netB, FeatureMapSize) and isinstance(stride_netB, FeatureMapSize), "All inputs should be either of type FeatureMapSize or int"
            rec_field_w, stride_w = Os2dHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.w, stride_netA.w,
                                                                                               receptive_field_netB.w, stride_netB.w)
            rec_field_h, stride_h = Os2dHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.h, stride_netA.h,
                                                                                               receptive_field_netB.h, stride_netB.h)
            return FeatureMapSize(w=rec_field_w, h=rec_field_h), FeatureMapSize(w=stride_w, h=stride_h)

        rec_field = stride_netA * (receptive_field_netB - 1) + receptive_field_netA
        stride = stride_netA * stride_netB
        return rec_field, stride

    @staticmethod
    def resize_feature_maps_to_reference_size(ref_size, feature_maps):
        feature_maps_ref_size = []
        for fm in feature_maps:
            assert fm.size(0) == 1, "Can process only batches of size 1, but have {0}".format(fm.size(0))
            num_feature_channels = fm.size(1)

            # resample class features
            identity = torch.tensor([[1, 0, 0], [0, 1, 0]], device=fm.device, dtype=fm.dtype)
            grid_size = torch.Size([1,
                                    num_feature_channels,
                                    ref_size.h,
                                    ref_size.w])
            resampling_grid = F.affine_grid(identity.unsqueeze(0), grid_size, align_corners=True)
            fm_ref_size = F.grid_sample(fm, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            feature_maps_ref_size.append(fm_ref_size)

        feature_maps_ref_size = torch.cat(feature_maps_ref_size, dim=0)
        return feature_maps_ref_size

    def create_os2d_head(self, class_feature_maps):
        # convert all the feature maps to the standard size
        reference_feature_map_size = self.aligner.reference_feature_map_size
        class_feature_maps_ref_size = self.resize_feature_maps_to_reference_size(reference_feature_map_size, class_feature_maps)
        return Os2dHead(class_feature_maps_ref_size,
                        self.aligner,
                        self.box_grid_generator_image_level,
                        self.box_grid_generator_feature_map_level)


class Os2dHead(nn.Module):
    """This class computes the recognition and localization scores for a batch of input feature maps and a batch of class feature maps.
    The class feature maps should be fed into the constructor that stores references to them inside.
    The input feature maps should be fed into the forward method.
    Instances of this class are supposed to be created by Os2dHeadCreator.create_os2d_head with passing in the class_feature_maps
    """
    def __init__(self, class_feature_maps, aligner,
                       box_grid_generator_image_level,
                       box_grid_generator_feature_map_level,
                       pool_border_width=2):
        super(Os2dHead, self).__init__()

        # initialize class feature maps
        self.class_feature_maps = class_feature_maps
        self.class_batch_size = self.class_feature_maps.size(0)

        # class to generate box grids in the image plane that correspond to positions in the feature map
        self.box_grid_generator_image_level = box_grid_generator_image_level
        # class to generate box grids in the feature map plane
        self.box_grid_generator_feature_map_level = box_grid_generator_feature_map_level

        # class feature maps have to be normalized
        self.class_feature_maps = normalize_feature_map_L2(self.class_feature_maps, 1e-5)

        # create a mask for pooling activations - for now just block few pixels at the edges
        self.class_pool_mask = torch.zeros( (self.class_feature_maps.size(0), 1,
                                             self.class_feature_maps.size(2), self.class_feature_maps.size(3)), # batch_size x 1 x H x W
                                             dtype=torch.float, device=self.class_feature_maps.device)
        self.class_pool_mask[:, :,
                             pool_border_width : self.class_pool_mask.size(-2) - pool_border_width,
                             pool_border_width : self.class_pool_mask.size(-1) - pool_border_width] = 1
        self.class_pool_mask = spatial_norm(self.class_pool_mask)

        # create the alignment module
        self.aligner = aligner


    def forward(self, feature_maps):
        """
        Args:
            feature_maps (Tensor[float], size b^A x d x h^A x w^A) - contains the feature map of the input image
            b^A - batch size
            d - feature dimensionality
            h^A - height of the feature map
            w^A - width of the feature map
​
        Returns:
                # here b^C is the class batch size, i.e., the number of class images contained in self.class_batch_size passed when creating this object
            output_localization (Tensor[float], size b^A x b^C x 4 x h^A x w^A) - the localization output w.r.t. the standard box encoding - computed by DetectionBoxCoder.build_loc_targets
            output_recognition (Tensor[float], size size b^A x b^C x 1 x h^A x w^A) - the recognition output for each of the classes:
                in the [-1, 1] segment, the higher the better match to the class
            output_recognition_transform_detached (Tensor[float], size b^A x b^C x 1 x h^A x w^A) - same as output_recognition,
                but with the computational graph detached from the transformation (for backward  that does not update
                the transofrmation - intended for the negatives)
            corner_coordinates (Tensor[float], size size b^A x b^C x 8 x h^A x w^A) - the corners of the default boxes after
                the transofrmation, datached from the computational graph, for visualisation only
        """
        # get dims
        batch_size = feature_maps.size(0)
        feature_dim = feature_maps.size(1)
        image_fm_size = FeatureMapSize(img=feature_maps)
        class_fm_size = FeatureMapSize(img=self.class_feature_maps)
        feature_dim_for_regression = class_fm_size.h * class_fm_size.w

        class_feature_dim = self.class_feature_maps.size(1)
        assert feature_dim == class_feature_dim, "Feature dimensionality of input={0} and class={1} feature maps has to equal".format(feature_dim, class_feature_dim)

        # L2-normalize the feature map
        feature_maps = normalize_feature_map_L2(feature_maps, 1e-5)

        # get correlations all to all
        corr_maps = torch.einsum( "bfhw,afxy->abwhxy", self.class_feature_maps, feature_maps )
        # need to try to optimize this with opt_einsum: https://optimized-einsum.readthedocs.io/en/latest/
        # CAUTION: note the switch of dimensions hw to wh. This is done for compatability with the FeatureCorrelation class by Ignacio Rocco https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py (to be able to load their models)

        # reshape to have the correlation map of dimensions similar to the standard tensor for image feature maps
        corr_maps = corr_maps.contiguous().view(batch_size * self.class_batch_size,
                                                feature_dim_for_regression,
                                                image_fm_size.h,
                                                image_fm_size.w)

        # compute the grids to resample corr maps
        resampling_grids_local_coord = self.aligner(corr_maps)

        # build classifications outputs
        cor_maps_for_recognition = corr_maps.contiguous().view(batch_size,
                                                       self.class_batch_size,
                                                       feature_dim_for_regression,
                                                       image_fm_size.h,
                                                       image_fm_size.w)
        resampling_grids_local_coord = resampling_grids_local_coord.contiguous().view(batch_size,
                                                                                      self.class_batch_size,
                                                                                      image_fm_size.h,
                                                                                      image_fm_size.w,
                                                                                      self.aligner.out_grid_size.h,
                                                                                      self.aligner.out_grid_size.w,
                                                                                      2)

        # need to recompute resampling_grids to [-1, 1] coordinates w.r.t. the feature maps to sample points with F.grid_sample
        # first get the list of boxes that corresponds to the receptive fields of the parameter regression network: box sizes are the receptive field sizes, stride is the network stride
        default_boxes_xyxy_wrt_fm = self.box_grid_generator_feature_map_level.create_strided_boxes_columnfirst(fm_size=image_fm_size)

        default_boxes_xyxy_wrt_fm = default_boxes_xyxy_wrt_fm.view(1, 1, image_fm_size.h, image_fm_size.w, 4)
        # 1 (to broadcast to batch_size) x 1 (to broadcast to class batch_size) x  box_grid_height x box_grid_width x 4
        default_boxes_xyxy_wrt_fm = default_boxes_xyxy_wrt_fm.to(resampling_grids_local_coord.device)
        resampling_grids_fm_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_fm)

        # covert to coordinates normalized to [-1, 1] (to be compatible with torch.nn.functional.grid_sample)
        resampling_grids_fm_coord_x = resampling_grids_fm_coord.narrow(-1,0,1)
        resampling_grids_fm_coord_y = resampling_grids_fm_coord.narrow(-1,1,1)
        resampling_grids_fm_coord_unit = torch.cat( [resampling_grids_fm_coord_x / (image_fm_size.w - 1) * 2 - 1,
            resampling_grids_fm_coord_y / (image_fm_size.h - 1) * 2 - 1], dim=-1 )
        # clamp to fit the image plane
        resampling_grids_fm_coord_unit = resampling_grids_fm_coord_unit.clamp(-1, 1)

        # extract and pool matches
        # # slower code:
        # output_recognition = self.resample_of_correlation_map_simple(cor_maps_for_recognition,
        #                                                          resampling_grids_fm_coord_unit,
        #                                                          self.class_pool_mask)

        # we use faster, but somewhat more obscure version
        output_recognition = self.resample_of_correlation_map_fast(cor_maps_for_recognition,
                                                             resampling_grids_fm_coord_unit,
                                                             self.class_pool_mask)
        if output_recognition.requires_grad:
            output_recognition_transform_detached = self.resample_of_correlation_map_fast(cor_maps_for_recognition,
                                                                                      resampling_grids_fm_coord_unit.detach(),
                                                                                      self.class_pool_mask)
        else:
            # Optimization to make eval faster
            output_recognition_transform_detached = output_recognition

        # build localization targets
        default_boxes_xyxy_wrt_image = self.box_grid_generator_image_level.create_strided_boxes_columnfirst(fm_size=image_fm_size)

        default_boxes_xyxy_wrt_image = default_boxes_xyxy_wrt_image.view(1, 1, image_fm_size.h, image_fm_size.w, 4)
        # 1 (to broadcast to batch_size) x 1 (to broadcast to class batch_size) x  box_grid_height x box_grid_width x 4
        default_boxes_xyxy_wrt_image = default_boxes_xyxy_wrt_image.to(resampling_grids_local_coord.device)
        resampling_grids_image_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_image)


        num_pooled_points = self.aligner.out_grid_size.w * self.aligner.out_grid_size.h
        resampling_grids_x = resampling_grids_image_coord.narrow(-1, 0, 1).contiguous().view(-1, num_pooled_points)
        resampling_grids_y = resampling_grids_image_coord.narrow(-1, 1, 1).contiguous().view(-1, num_pooled_points)
        class_boxes_xyxy = torch.stack([resampling_grids_x.min(dim=1)[0],
                                        resampling_grids_y.min(dim=1)[0],
                                        resampling_grids_x.max(dim=1)[0],
                                        resampling_grids_y.max(dim=1)[0]], 1)

        # extract rectangle borders to draw complete boxes
        corner_coordinates = resampling_grids_image_coord[:,:,:,:,[0,-1]][:,:,:,:,:,[0,-1]] # only the corners
        corner_coordinates = corner_coordinates.detach_()
        corner_coordinates = corner_coordinates.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 8) # batch_size x label_batch_size x fm_height x fm_width x 8
        corner_coordinates = corner_coordinates.transpose(3, 4).transpose(2, 3)  # batch_size x label_batch_size x 5 x fm_height x fm_width

        class_boxes = BoxList(class_boxes_xyxy.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_wrt_image = BoxList(default_boxes_xyxy_wrt_image.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_with_image_batches = cat_boxlist([default_boxes_wrt_image] * batch_size * self.class_batch_size)

        output_localization = Os2dBoxCoder.build_loc_targets(class_boxes, default_boxes_with_image_batches) # num_boxes x 4
        output_localization = output_localization.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 4)  # batch_size x label_batch_size x fm_height x fm_width x 4
        output_localization = output_localization.transpose(3, 4).transpose(2, 3)  # batch_size x label_batch_size x 4 x fm_height x fm_width

        return output_localization, output_recognition, output_recognition_transform_detached, corner_coordinates


    @staticmethod
    def resample_of_correlation_map_fast(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        """This function resamples the correlation tensor according to the grids of points representing the transformations produces by the transformation network.
        This is a more efficient version of resample_of_correlation_map_simple
        Args:
            corr_maps (Tensor[float], size=batch_size x class_batch_size x (h^T*w^T) x h^A x w^A):
                This tensor contains correlations between of features of the input and class feature maps.
                This function resamples this tensor.
                CAUTION: this tensor shows be viewed to batch_size x class_batch_size x w^T x h^T x h^A x w^A (note the switch of w^T and h^T dimensions)
                This happens to be able to load models of the weakalign repo
            resampling_grids_grid_coord (Tensor[float], size=batch_size x class_batch_size x h^A x w^A x h^T x w^T x 2):
                This tensor contains non-integer coordinates of the points that show where we need to resample
            class_pool_mask (Tensor[float]): size=class_batch_size x 1 x h^T x w^T
                This tensor contains the mask, by which the resampled correlations are multiplied before final average pooling.
                It masks out the border features of the class feature maps.

        Returns:
            matches_pooled (Tensor[float]): size=batch_size x class_batch_size x x 1 x h^A x w^A

        Time comparison resample_of_correlation_map_simple vs resample_of_correlation_map_fast:
            for 2 images, 11 labels, train_patch_width 400, train_patch_height 600 (fm width = 25, fm height = 38)
                CPU time simple: 0.14s
                CPU time fast: 0.11s
                GPU=Geforce GTX 1080Ti
                GPU time simple: 0.010s
                GPU time fast: 0.006s
        """
        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), 'the number of channels in the correlation map = {0} should match the size of the resampling grid = {1}'.format(corr_maps.size(2), template_fm_size)

        # memory efficient computation will be done by merging the Y coordinate
        # and the index of the channel in corr_map into one single float

        # merge the two dimensions together
        corr_map_merged_y_and_id_in_corr_map = corr_maps.contiguous().view(batch_size * class_batch_size,
            1, -1, image_fm_size.w)

        # note the weird order of coordinates - related to the transposed coordinates in the Ignacio's network
        y_grid, x_grid = torch.meshgrid( torch.arange(template_fm_size.h), torch.arange(template_fm_size.w) )
        index_in_corr_map = y_grid + x_grid * template_fm_size.h

        # clamp to strict [-1, 1]
        # convert to torch.double to get more accuracy
        resampling_grids_grid_coord_ = resampling_grids_grid_coord.clamp(-1, 1).to(dtype=torch.double)
        resampling_grids_grid_coord_x_ = resampling_grids_grid_coord_.narrow(-1,0,1)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_.narrow(-1,1,1)
        # adjust the y coordinate to take into account the index in the corr_map:
        # convert from [-1, 1] to [0, image_fm_size[0]]
        resampling_grids_grid_coord_y_ = (resampling_grids_grid_coord_y_ + 1) / 2 * (image_fm_size.h - 1)
        # merge with the index in corr map [0]
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view( [-1] + list(index_in_corr_map.size()) )
        index_in_corr_map = index_in_corr_map.unsqueeze(0)
        index_in_corr_map = index_in_corr_map.to(device=resampling_grids_grid_coord_.device,
                                                 dtype=resampling_grids_grid_coord_.dtype)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ + index_in_corr_map * image_fm_size.h
        # convert back to [-1, -1]
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ / (image_fm_size.h * template_fm_size.h * template_fm_size.w - 1) * 2 - 1
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view_as(resampling_grids_grid_coord_x_)
        resampling_grids_grid_coord_merged_y_and_id_in_corr_map = torch.cat([resampling_grids_grid_coord_x_, resampling_grids_grid_coord_y_], dim=-1)

        # flatten the resampling grid
        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d = \
            resampling_grids_grid_coord_merged_y_and_id_in_corr_map.view(batch_size * class_batch_size, -1, 1, 2)
        # extract the required points
        matches_all_channels = F.grid_sample(corr_map_merged_y_and_id_in_corr_map.to(dtype=torch.double),
                                        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d,
                                        mode="bilinear", padding_mode='border', align_corners=True)

        matches_all_channels = matches_all_channels.view(batch_size, class_batch_size, 1,
                                                image_fm_size.h * image_fm_size.w,
                                                template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels.to(dtype=torch.float)

        # combine extracted matches using the average pooling w.r.t. the mask of active points defined by class_pool_mask)
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask

        matches_pooled = matches_all_channels.sum(4)
        matches_pooled = matches_pooled.view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
        return matches_pooled

    @staticmethod
    def resample_of_correlation_map_simple(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        """This function resamples the correlation tensor according to the grids of points representing the transformations produces by the transformation network.
        This function is left hear for understanding, use resample_of_correlation_map_fast, which is faster.
        Args:
            corr_maps (Tensor[float], size=batch_size x class_batch_size x (h^T*w^T) x h^A x w^A):
                This tensor contains correlations between of features of the input and class feature maps.
                This function resamples this tensor.
                CAUTION: this tensor shows be viewed to batch_size x class_batch_size x w^T x h^T x h^A x w^A (note the switch of w^T and h^T dimensions)
                This happens to be able to load models of the weakalign repo
            resampling_grids_grid_coord (Tensor[float], size=batch_size x class_batch_size x h^A x w^A x h^T x w^T x 2):
                This tensor contains non-integer coordinates of the points that show where we need to resample
            class_pool_mask (Tensor[float]): size=class_batch_size x 1 x h^T x w^T
                This tensor contains the mask, by which the resampled correlations are multiplied before final average pooling.
                It masks out the border features of the class feature maps.

        Returns:
            matches_pooled (Tensor[float]): size=batch_size x class_batch_size x x 1 x h^A x w^A

        Time comparison resample_of_correlation_map_simple vs resample_of_correlation_map_fast:
            for 2 images, 11 labels, train_patch_width 400, train_patch_height 600 (fm width = 25, fm height = 38)
                CPU time simple: 0.14s
                CPU time fast: 0.11s
                GPU=Geforce GTX 1080Ti
                GPU time simple: 0.010s
                GPU time fast: 0.006s
        """

        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), 'the number of channels in the correlation map = {0} should match the size of the resampling grid = {1}'.format(corr_maps.size(2), template_fm_size)

        # use a single batch dimension
        corr_maps = corr_maps.view(batch_size * class_batch_size,
                                   corr_maps.size(2),
                                   image_fm_size.h,
                                   image_fm_size.w)
        resampling_grids_grid_coord = resampling_grids_grid_coord.view(batch_size * class_batch_size,
                                                                       image_fm_size.h,
                                                                       image_fm_size.w,
                                                                       template_fm_size.h,
                                                                       template_fm_size.w,
                                                                       2)

        # extract matches from all channels one by one in a loop, and then combine them (using the average pooling w.r.t. the mask of active points defined by class_pool_mask)
        matches_all_channels = []
        # the order of the loops matters
        for template_x in range(template_fm_size.w):
            for template_y in range(template_fm_size.h):
                # note the weird order of coordinates - related to the transposed coordinates in the weakalign network
                channel_id = template_x * template_fm_size.h + template_y

                channel = corr_maps[:,channel_id:channel_id+1,:,:]
                points = resampling_grids_grid_coord[:,:,:,template_y,template_x,:]

                matches_one_channel = F.grid_sample(channel, points, mode="bilinear", padding_mode='border', align_corners=True)
                matches_all_channels.append(matches_one_channel)
        matches_all_channels = torch.stack(matches_all_channels, -1)

        # start pooling: fix all dimensions explicitly mostly to be safe
        matches_all_channels = matches_all_channels.view(batch_size,
                                                         class_batch_size,
                                                         image_fm_size.h,
                                                         image_fm_size.w,
                                                         template_fm_size.h * template_fm_size.w)
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask

        matches_pooled = matches_all_channels.sum(4)
        matches_pooled = matches_pooled.view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
        return matches_pooled


def normalize_feature_map_L2(feature_maps, epsilon=1e-6):
    """Note that the code is slightly different from featureL2Norm of
    From https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py
    """
    return feature_maps / (feature_maps.norm(dim=1, keepdim=True) + epsilon)


class TransformationNet(nn.Module):
    """This class is implemented on top of the FeatureRegression class form the weakalign repo
    https://github.com/ignacio-rocco/weakalign/blob/master/model/cnn_geometric_model.py
    """
    def __init__(self, output_dim=6, use_cuda=True, normalization='batchnorm', kernel_sizes=[7,5], channels=[128,64], input_feature_dim=15*15, num_groups=16):
        super(TransformationNet, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = input_feature_dim
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=k_size//2))
            # Added padding to make this module preserve spatial size

            if normalization.lower() == 'batchnorm':
                nn_modules.append(nn.BatchNorm2d(ch_out))
            elif normalization.lower() == 'groupnorm':
                nn_modules.append(nn.GroupNorm(num_groups, ch_out))

            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Conv2d(ch_out, output_dim, kernel_size=(k_size, k_size), padding=k_size//2)

        # initialize the last layer to deliver identity transform
        if output_dim==6:
            # assert output_dim==6, "Implemented only for affine transform"
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[4] = 1
        elif output_dim==4:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[2] = 1

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, corr_maps):
        # normalization
        corr_maps_norm = normalize_feature_map_L2(F.relu(corr_maps))
        # corr_maps_norm = featureL2Norm(F.relu(corr_maps))

        # apply the network
        transform_params = self.linear(self.conv(corr_maps_norm))
        return transform_params

    def freeze_bn(self):
        # Freeze BatchNorm layers
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
