import torch
from timm.models.registry import register_model
from nat import NAT

model_urls = {
    # ImageNet-1K
    "dinat_mini_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_mini_in1k_224.pth",
    "dinat_tiny_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth",
    "dinat_small_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_small_in1k_224.pth",
    "dinat_base_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth",
    "dinat_large_1k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_224.pth",
    "dinat_large_1k_384": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384.pth",
    "dinat_large_1k_384_11x11": "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth",
    # ImageNet-22K
    "dinat_large_21k": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth",
    "dinat_large_21k_11x11": "https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224_11x11interp.pth",
    # 11x11 contains the same weights as the original, except for RPB which is interpolated using a bicubic interpolation.
    # Swin uses the same interpolation when changing window sizes.
}

class DiNAT(NAT):
    """
    DiNAT is NAT with dilations.
    It's that simple!
    """

    pass

@register_model
def dinat_base(pretrained=False, **kwargs):
    print("---------USE MY MODEL")
    print("*"*60)
    model = DiNAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=7,
        dilations=[
            [1, 8, 1],
            [1, 4, 1, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        **kwargs
    )
    if pretrained:
        url = model_urls["dinat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model