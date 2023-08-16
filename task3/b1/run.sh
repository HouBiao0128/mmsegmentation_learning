# python demo/image_demo.py  data/street_uk.jpeg  configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py         checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth         --out-file outputs/B1_uk_segformer.jpg         --device cuda:0         --opacity 0.5

python demo/image_demo.py  data/bedroom.jpg  configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py         checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth         --out-file outputs/B1_uk_segformer_bedroom.jpg         --device cuda:0         --opacity 0.5

