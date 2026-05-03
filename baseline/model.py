import torch
import torch.nn as nn
from torchvision import models

class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        # drop avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.head = nn.Sequential(
            # need to change this hardcoded 512 later if backbone changes
            nn.Conv2d(512, 1, kernel_size=1),
            # hope bilinear is better than nearest here
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        
        return output

if __name__ == "__main__":
    model = SaliencyModel()
    test_data = torch.randn(1, 3, 224, 224)
    result = model(test_data)
    print("Result shape:", result.shape)
