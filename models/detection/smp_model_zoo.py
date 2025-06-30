import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any
import cv2
from shapely.geometry import Polygon
import pyclipper
import segmentation_models_pytorch as smp

from doctane.utils.dl_utils import _bf16_to_float32
from doctane.models.detection.detection_postprocessor import DetectionPostProcessor 
from doctane.models.detection.smp_models import SegmentationModel



def seg_unet_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnet18 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnet34 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnet50 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnet101 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnet152 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn68 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn68b encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn92 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn98 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn107 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with dpn131 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg11 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg13 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg16 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg19 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with senet154 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with densenet121 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with densenet169 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with densenet201 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with densenet161 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with xception encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b0 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b1 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b3 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mit_b5 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unet_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Unet with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="Unet",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnet18 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnet34 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnet50 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnet101 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnet152 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn68 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn68b encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn92 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn98 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn107 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with dpn131 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg11 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg13 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg16 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg19 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with senet154 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with densenet121 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with densenet169 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with densenet201 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with densenet161 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with xception encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b0 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b1 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b3 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mit_b5 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_unetplusplus_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UnetPlusPlus with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="UnetPlusPlus",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnet18 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnet34 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnet50 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnet101 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnet152 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn68 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn68b encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn92 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn98 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn107 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with dpn131 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg11 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg13 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg16 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg19 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with senet154 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with densenet121 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with densenet169 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with densenet201 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with densenet161 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with xception encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b0 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b1 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b3 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mit_b5 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_manet_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """MAnet with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="MAnet",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnet18 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnet34 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnet50 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnet101 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnet152 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn68 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn68b encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn92 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn98 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn107 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with dpn131 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg11 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg13 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg16 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg19 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with senet154 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with densenet121 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with densenet169 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with densenet201 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with densenet161 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with xception encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b0 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b1 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b3 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mit_b5 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_linknet_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Linknet with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="Linknet",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnet18 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnet34 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnet50 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnet101 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnet152 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn68 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn68b encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn92 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn98 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn107 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with dpn131 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg11 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg13 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg16 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg19 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with senet154 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with densenet121 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with densenet169 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with densenet201 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with densenet161 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with xception encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b0 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b1 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b3 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mit_b5 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_fpn_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """FPN with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="FPN",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnet18 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnet34 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnet50 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnet101 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnet152 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn68 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn68b encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn92 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn98 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn107 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with dpn131 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg11 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg13 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg16 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg19 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with senet154 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with densenet121 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with densenet169 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with densenet201 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with densenet161 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with xception encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b0 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b1 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b3 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mit_b5 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pspnet_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PSPNet with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="PSPNet",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnet18 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnet34 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnet50 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnet101 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnet152 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn68 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn68b encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn92 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn98 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn107 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with dpn131 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg11 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg13 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg16 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg19 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with senet154 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with densenet121 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with densenet169 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with densenet201 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with densenet161 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with xception encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b0 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b1 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b3 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mit_b5 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_pan_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """PAN with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="PAN",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnet18 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnet34 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnet50 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnet101 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnet152 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn68 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn68b encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn92 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn98 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn107 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with dpn131 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg11 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg13 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg16 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg19 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with senet154 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with densenet121 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with densenet169 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with densenet201 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with densenet161 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with xception encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mit_b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3 with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnet18 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnet34 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnet50 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnet101 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnet152 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn68 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn68b encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn92 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn98 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn107 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with dpn131 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg11 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg13 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg16 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg19 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with senet154 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with densenet121 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with densenet169 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with densenet201 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with densenet161 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with xception encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mit_b5 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_deeplabv3plus_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DeepLabV3Plus with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="DeepLabV3Plus",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnet18 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnet34 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnet50 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnet101 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnet152 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn68 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn68b encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn92 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn98 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn107 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with dpn131 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg11 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg13 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg16 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg19 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with senet154 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with densenet121 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with densenet169 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with densenet201 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with densenet161 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with xception encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b0 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b1 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b3 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mit_b5 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_upernet_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """UPerNet with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="UPerNet",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnet18 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnet34 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnet50 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnet101 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnet152 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn68 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn68b encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn92 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn98 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn107 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with dpn131 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg11 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg13 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg16 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg19 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with senet154 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with densenet121 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with densenet169 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with densenet201 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with densenet161 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with xception encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b0 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b1 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b3 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mit_b5 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_segformer_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """Segformer with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="Segformer",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnet18 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnet34 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnet50 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnet101 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnet152 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext101_32x8d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext101_32x8d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext101_32x8d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext101_32x16d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext101_32x16d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext101_32x16d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext101_32x32d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext101_32x32d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext101_32x32d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_resnext101_32x48d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with resnext101_32x48d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="resnext101_32x48d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn68(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn68 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn68",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn68b(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn68b encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn68b",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn92(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn92 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn92",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn98(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn98 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn98",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn107(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn107 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn107",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_dpn131(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with dpn131 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="dpn131",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg11(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg11 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg11",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg11_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg11_bn encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg11_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg13(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg13 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg13",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg13_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg13_bn encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg13_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg16(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg16 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg16",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg16_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg16_bn encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg16_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg19(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg19 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg19",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_vgg19_bn(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with vgg19_bn encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="vgg19_bn",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_senet154(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with senet154 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="senet154",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_se_resnet50(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with se_resnet50 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="se_resnet50",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_se_resnet101(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with se_resnet101 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="se_resnet101",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_se_resnet152(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with se_resnet152 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="se_resnet152",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_se_resnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with se_resnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_se_resnext101_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with se_resnext101_32x4d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="se_resnext101_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_densenet121(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with densenet121 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="densenet121",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_densenet169(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with densenet169 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="densenet169",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_densenet201(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with densenet201 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="densenet201",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_densenet161(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with densenet161 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="densenet161",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_inceptionresnetv2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with inceptionresnetv2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="inceptionresnetv2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_inceptionv4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with inceptionv4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="inceptionv4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobilenet_v2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobilenet_v2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_xception(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with xception encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="xception",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b0 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b1 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b3 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b5 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b6(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b6 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b6",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b7(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b7 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b7",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_b8(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-b8 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-b8",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_efficientnet_l2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-efficientnet-l2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-efficientnet-l2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_tf_efficientnet_lite0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-tf_efficientnet_lite0 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-tf_efficientnet_lite0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_tf_efficientnet_lite1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-tf_efficientnet_lite1 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-tf_efficientnet_lite1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_tf_efficientnet_lite2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-tf_efficientnet_lite2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-tf_efficientnet_lite2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_tf_efficientnet_lite3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-tf_efficientnet_lite3 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-tf_efficientnet_lite3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_tf_efficientnet_lite4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-tf_efficientnet_lite4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-tf_efficientnet_lite4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_skresnet18(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-skresnet18 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-skresnet18",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_skresnet34(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-skresnet34 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-skresnet34",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_timm_skresnext50_32x4d(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with timm-skresnext50_32x4d encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="timm-skresnext50_32x4d",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b0 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b1 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b3 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mit_b5(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mit_b5 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mit_b5",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobileone_s0(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobileone_s0 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobileone_s0",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobileone_s1(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobileone_s1 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobileone_s1",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobileone_s2(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobileone_s2 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobileone_s2",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobileone_s3(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobileone_s3 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobileone_s3",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )

def seg_dpt_mobileone_s4(pretrained: bool = False, **kwargs) -> SegmentationModel:
    """DPT with mobileone_s4 encoder."""
    return SegmentationModel(
        model_name="DPT",
        encoder_name="mobileone_s4",
        encoder_weights="imagenet" if pretrained else None,
        **kwargs
    )
