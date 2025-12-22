config = {
    "batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "model_name": "teaformer2", #teaformer, teaformer2
    "model_mode": "tiny-lcaf",  # t1: dynamic, concat, cnn, vit, bottleneck, bottleneck_freeze
                             # t2: lcaf , mobilenet (mobilenetv3), eformer (efficientformer-li), mobilevit, cnn, vit , tiny-lcaf
    "activation": "gelu",  # "softplus", "gelu", "leaky_relu", "relu" 등 원하는 활성화 함수 선택
    "num_classes": 7,
    "image_size": 224,
    "decay_weight": 0.0001,
    "resume": False,
    "checkpoint_path": "/data1/seyong/metaverse/tealeaf2/outputs/checkpoints/latest_checkpoint.pth"
}
