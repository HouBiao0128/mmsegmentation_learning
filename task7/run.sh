python tools/deploy.py \
        configs/mmseg/segmentation_onnxruntime_dynamic.py \
        ../mmsegmentation/Zihao-Configs/ZihaoDataset_Segformer_20230818.py \
        ../mmsegmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
        ../mmsegmentation/data/watermelon_test1.jpg \
        --work-dir mmseg2onnx_segformer \
        --dump-info