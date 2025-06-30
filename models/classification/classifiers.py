import torch.nn as nn
from torchvision import models

def get_model(name, num_classes=3):
    name = name.lower()
    
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet101":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    elif name in ["vit_s", "vit_b"]:
        # ViT from torchvision >= 0.13
        vit = models.__dict__[name](pretrained=True)
        vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes)
        model = vit

    # ❗ Custom/non-torchvision models below — must be defined/imported elsewhere
    else:
        raise ValueError(f"Unknown model: {name}")

    return model
