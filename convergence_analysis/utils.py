

def get_models(args):
    if args.modality == 'vision':
        if args.model_family == 'vit':
            vision_model_names = [
                    "vit_tiny_patch16_224.augreg_in21k",
                    "vit_small_patch16_224.augreg_in21k",
                    "vit_base_patch16_224.augreg_in21k",
                    "vit_large_patch16_224.augreg_in21k",
                    # different type of models
                    "vit_base_patch16_clip_224.laion2b_ft_in12k",
                    "vit_large_patch14_clip_224.laion2b_ft_in12k",
                    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
                ]
        elif args.model_family=='vit_mae':
            vision_model_names = [
                "vit_base_patch16_224.mae",
                "vit_large_patch16_224.mae",
                "vit_huge_patch14_224.mae",
            ]
        elif args.model_family == 'vit_dinov2':
            vision_model_names = [
                "vit_small_patch14_reg4_dinov2.lvd142m",
                "vit_base_patch14_reg4_dinov2.lvd142m",
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_giant_patch14_reg4_dinov2.lvd142m",
            ]
        elif args.model_family == 'vit_clip':
            vision_model_names = [
                "vit_base_patch16_clip_224.laion2b",
                "vit_large_patch14_clip_224.laion2b",
                "vit_huge_patch14_clip_224.laion2b"
            ]
        elif args.model_family == 'resnet':
            vision_model_names = [
                'resnet18.a1_in1k',
                'resnet34.a1_in1k',
                'resnet50.a1_in1k',
                'resnet101.a1_in1k',
                'resnet152.a1_in1k',
            ]
        elif args.model_family == 'efficientnet':
            vision_model_names = [
                'efficientnet_b0.ra4_e3600_r224_in1k',
                'efficientnet_b1.ft_in1k',
                'efficientnet_b2.ra_in1k',
                'efficientnet_b3.ra2_in1k',
                'efficientnet_b4.ra2_in1k',
                'efficientnet_b5.sw_in12k'
            ]
        elif args.model_family == 'convnext':
            vision_model_names = [
                'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                'convnextv2_base.fcmae_ft_in22k_in1k_384',
                'convnextv2_large.fcmae_ft_in22k_in1k_384',
                'convnextv2_huge.fcmae_ft_in22k_in1k_384',
            ]
        else:
            raise ValueError(f"Unknown Vision model family {args.model_family}")

        return vision_model_names
        
    elif args.modality == 'language':
        if args.model_family == 'bloomz':
            language_model_names = [
                "bigscience/bloomz-560m",
                "bigscience/bloomz-1b1",
                "bigscience/bloomz-1b7",
                "bigscience/bloomz-3b",
                "bigscience/bloomz-7b1",
            ]
        elif args.model_family == 'open_llama':
            language_model_names = [
                "openlm-research/open_llama_3b",
                "openlm-research/open_llama_7b",
                "openlm-research/open_llama_13b",
            ]
        elif args.model_family == 'huggyllama':
            language_model_names = [
                "huggyllama/llama-7b",
                "huggyllama/llama-13b",
                "huggyllama/llama-30b",
                # "huggyllama/llama-65b", # too big
            ]
        else:
            raise ValueError(f"Unknown LLM model family {args.model_family}")

        return language_model_names
    
    raise NotImplementedError("Wrong modality")