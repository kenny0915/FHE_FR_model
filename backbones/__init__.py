from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def iresnet18_no_relu(pretrained=False, progress=True, **kwargs):
    from .iresnet_no_relu import iresnet18 as _iresnet18_no_relu
    return _iresnet18_no_relu(pretrained=pretrained, progress=progress, **kwargs)


def iresnet34_no_relu(pretrained=False, progress=True, **kwargs):
    from .iresnet_no_relu import iresnet34 as _iresnet34_no_relu
    return _iresnet34_no_relu(pretrained=pretrained, progress=progress, **kwargs)


def iresnet50_no_relu(pretrained=False, progress=True, **kwargs):
    from .iresnet_no_relu import iresnet50 as _iresnet50_no_relu
    return _iresnet50_no_relu(pretrained=pretrained, progress=progress, **kwargs)


def iresnet100_no_relu(pretrained=False, progress=True, **kwargs):
    from .iresnet_no_relu import iresnet100 as _iresnet100_no_relu
    return _iresnet100_no_relu(pretrained=pretrained, progress=progress, **kwargs)


def iresnet200_no_relu(pretrained=False, progress=True, **kwargs):
    from .iresnet_no_relu import iresnet200 as _iresnet200_no_relu
    return _iresnet200_no_relu(pretrained=pretrained, progress=progress, **kwargs)


def get_iresnet_quadratic(depth, pretrained=False, progress=True, **kwargs):
    from . import iresnet_quadratic
    factory = {
        18: iresnet_quadratic.iresnet18,
        34: iresnet_quadratic.iresnet34,
        50: iresnet_quadratic.iresnet50,
        100: iresnet_quadratic.iresnet100,
        200: iresnet_quadratic.iresnet200,
    }[depth]
    return factory(pretrained=pretrained, progress=progress, **kwargs)


def get_mbf_no_relu(fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
    from .mobilefacenet_no_relu import get_mbf as _get_mbf_no_relu
    return _get_mbf_no_relu(
        fp16=fp16,
        num_features=num_features,
        blocks=blocks,
        scale=scale,
    )


def get_mbf_large_no_relu(fp16=False, num_features=512, blocks=(2, 8, 12, 4), scale=4):
    from .mobilefacenet_no_relu import get_mbf_large as _get_mbf_large_no_relu
    return _get_mbf_large_no_relu(
        fp16=fp16,
        num_features=num_features,
        blocks=blocks,
        scale=scale,
    )


def get_model(name, **kwargs):
    # no-ReLU / FHE-friendly CryptoFace polynomial variants
    if name in ("r18_no_relu",):
        return iresnet18_no_relu(False, **kwargs)
    elif name in ("r34_no_relu",):
        return iresnet34_no_relu(False, **kwargs)
    elif name in ("r50_no_relu",):
        return iresnet50_no_relu(False, **kwargs)
    elif name in ("r100_no_relu",):
        return iresnet100_no_relu(False, **kwargs)
    elif name in ("r200_no_relu",):
        return iresnet200_no_relu(False, **kwargs)
    elif name in ("r18_quadratic",):
        return get_iresnet_quadratic(18, False, **kwargs)
    elif name in ("r34_quadratic",):
        return get_iresnet_quadratic(34, False, **kwargs)
    elif name in ("r50_quadratic",):
        return get_iresnet_quadratic(50, False, **kwargs)
    elif name in ("r100_quadratic",):
        return get_iresnet_quadratic(100, False, **kwargs)
    elif name in ("r200_quadratic",):
        return get_iresnet_quadratic(200, False, **kwargs)
    elif name in ("mbf_no_relu",):
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        blocks = kwargs.get("blocks", (1, 4, 6, 2))
        scale = kwargs.get("scale", 2)
        return get_mbf_no_relu(
            fp16=fp16,
            num_features=num_features,
            blocks=blocks,
            scale=scale,
        )
    elif name in ("mbf_large_no_relu",):
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        blocks = kwargs.get("blocks", (2, 8, 12, 4))
        scale = kwargs.get("scale", 4)
        return get_mbf_large_no_relu(
            fp16=fp16,
            num_features=num_features,
            blocks=blocks,
            scale=scale,
        )

    # editable local copy of the default IResNet implementation
    if name == "custom_r18":
        from .iresnet_custom import iresnet18 as custom_iresnet18
        return custom_iresnet18(False, **kwargs)
    elif name == "custom_r34":
        from .iresnet_custom import iresnet34 as custom_iresnet34
        return custom_iresnet34(False, **kwargs)
    elif name == "custom_r50":
        from .iresnet_custom import iresnet50 as custom_iresnet50
        return custom_iresnet50(False, **kwargs)
    elif name == "custom_r100":
        from .iresnet_custom import iresnet100 as custom_iresnet100
        return custom_iresnet100(False, **kwargs)
    elif name == "custom_r200":
        from .iresnet_custom import iresnet200 as custom_iresnet200
        return custom_iresnet200(False, **kwargs)

    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "patch_cnn":
        from .patch_cnn import patch_cnn
        return patch_cnn(**kwargs)

    elif name in (
        "poolformer_s12",
        "poolformer_s24",
        "poolformer_s24_mlp2",
        "poolformer_s36",
        "poolformer_m36",
        "poolformer_m48",
    ):
        from .poolformer import (
            poolformer_s12,
            poolformer_s24,
            poolformer_s24_mlp2,
            poolformer_s36,
            poolformer_m36,
            poolformer_m48,
        )
        poolformer_factory = {
            "poolformer_s12": poolformer_s12,
            "poolformer_s24": poolformer_s24,
            "poolformer_s24_mlp2": poolformer_s24_mlp2,
            "poolformer_s36": poolformer_s36,
            "poolformer_m36": poolformer_m36,
            "poolformer_m48": poolformer_m48,
        }[name]
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return poolformer_factory(
            pretrained=False,
            num_classes=num_features,
            face_embedding=True,
            fp16=fp16,
        )

    elif name in (
        "poolformer_no_ln_s12",
        "poolformer_no_ln_s24",
        "poolformer_no_ln_s36",
        "poolformer_no_ln_m36",
        "poolformer_no_ln_m48",
    ):
        from .poolformer_no_ln import (
            poolformer_s12,
            poolformer_s24,
            poolformer_s36,
            poolformer_m36,
            poolformer_m48,
        )
        poolformer_factory = {
            "poolformer_no_ln_s12": poolformer_s12,
            "poolformer_no_ln_s24": poolformer_s24,
            "poolformer_no_ln_s36": poolformer_s36,
            "poolformer_no_ln_m36": poolformer_m36,
            "poolformer_no_ln_m48": poolformer_m48,
        }[name]
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return poolformer_factory(
            pretrained=False,
            num_classes=num_features,
            face_embedding=True,
            fp16=fp16,
        )
    
    elif name in (
        "poolformer_no_ln_no_gelu_s12",
        "poolformer_no_ln_no_gelu_s24",
        "poolformer_no_ln_no_gelu_s36",
        "poolformer_no_ln_no_gelu_m36",
        "poolformer_no_ln_no_gelu_m48",
    ):
        from .poolformer_no_ln_no_gelu import (
            poolformer_s12,
            poolformer_s24,
            poolformer_s24_mlp2,
            poolformer_s36,
            poolformer_m36,
            poolformer_m48,
        )
        poolformer_factory = {
            "poolformer_no_ln_no_gelu_s12": poolformer_s12,
            "poolformer_no_ln_no_gelu_s24": poolformer_s24,
            "poolformer_no_ln_no_gelu_s36": poolformer_s36,
            "poolformer_no_ln_no_gelu_m36": poolformer_m36,
            "poolformer_no_ln_no_gelu_m48": poolformer_m48,
        }[name]
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return poolformer_factory(
            pretrained=False,
            num_classes=num_features,
            face_embedding=True,
            fp16=fp16,
        )

    elif name in (
        "poolformer_no_ln_x2_act_s12",
        "poolformer_no_ln_x2_act_s24",
        "poolformer_no_ln_x2_act_s24_mlp2",
        "poolformer_no_ln_x2_act_s36",
        "poolformer_no_ln_x2_act_m36",
        "poolformer_no_ln_x2_act_m48",
    ):
        from .poolformer_no_ln_x2_act import (
            poolformer_s12,
            poolformer_s24,
            poolformer_s24_mlp2,
            poolformer_s36,
            poolformer_m36,
            poolformer_m48,
        )
        poolformer_factory = {
            "poolformer_no_ln_x2_act_s12": poolformer_s12,
            "poolformer_no_ln_x2_act_s24": poolformer_s24,
            "poolformer_no_ln_x2_act_s24_mlp2": poolformer_s24_mlp2,
            "poolformer_no_ln_x2_act_s36": poolformer_s36,
            "poolformer_no_ln_x2_act_m36": poolformer_m36,
            "poolformer_no_ln_x2_act_m48": poolformer_m48,
        }[name]
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return poolformer_factory(
            pretrained=False,
            num_classes=num_features,
            face_embedding=True,
            fp16=fp16,
            gate_range_limit=kwargs.get("gate_range_limit", 6.0),
            gate_stats_sample_size=kwargs.get("gate_stats_sample_size", 16384),
            gate_compute_fp32=kwargs.get("gate_compute_fp32", True),
            gate_fail_on_nonfinite=kwargs.get("gate_fail_on_nonfinite", True),
            gate_initial_blend=kwargs.get("gate_initial_blend", 0.0),
        )

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
        
    elif name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True)

    else:
        raise ValueError(f"Unknown model name: {name}")
