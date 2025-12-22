# 경량화하면서 성능을 잃지않는 bottleneck , bottleneck_freeze 모드 추가
import torch
import torch.nn as nn
import torchvision
import timm

def get_activation_fn(name="relu"):
    """
    입력된 이름에 따라 활성화 함수를 반환합니다.
      - "softplus": Softplus 함수 (부드러운 활성화)
      - "gelu": GELU 함수
      - "leaky_relu": LeakyReLU (negative_slope=0.1)
      - 그 외 또는 "relu": ReLU 함수
    """
    name = name.lower()
    if name == "softplus":
        return nn.Softplus()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    else:
        return nn.ReLU()

class DynamicFusion(nn.Module):
    """
    CNN과 ViT로부터 얻은 feature를 learnable gating mechanism을 통해 융합하는 모듈.
    CNN branch의 feature (B, 512)와 ViT branch의 feature (B, 768)를 각각 투영한 후, 
    두 feature의 결합 정보를 기반으로 gating vector를 생성하여 동적 융합을 수행.
    """
    def __init__(self, cnn_dim=512, vit_dim=768, fuse_dim=512, activation="relu"):
        super(DynamicFusion, self).__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fuse_dim)
        self.vit_proj = nn.Linear(vit_dim, fuse_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, fuse_dim),
            get_activation_fn(activation),
            nn.Linear(fuse_dim, fuse_dim),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_feat, vit_feat):
        cnn_feat_proj = self.cnn_proj(cnn_feat)   # (B, fuse_dim)
        vit_feat_proj = self.vit_proj(vit_feat)     # (B, fuse_dim)
        fusion_input = torch.cat([cnn_feat, vit_feat], dim=1)  # (B, 512+768)
        gate = self.fusion_mlp(fusion_input)  # (B, fuse_dim) – 값 범위 0~1
        fused = gate * cnn_feat_proj + (1 - gate) * vit_feat_proj
        return fused

class TeaFormer(nn.Module):
    def __init__(self, num_classes=7, mode="dynamic", fuse_dim=512, activation="relu"):
        """
        mode 옵션:
            - "dynamic": 제안하는 Dynamic Fusion 모듈 사용 (CNN + ViT)
            - "concat": 단순 Concatenation (CNN + ViT)
            - "cnn": CNN 단독 (ResNet18)
            - "vit": ViT 단독 (ViT-B/16)
            - "bottleneck": 파라미터 수 절감을 위한 Bottleneck Fusion 모드 (Residual Connection 추가)
            - "bottleneck_freeze": bottleneck 모드와 동일한 구조이나, CNN 백본을 freeze하여 trainable 파라미터 수를 더욱 줄임.
        activation: "relu", "softplus", "gelu", "leaky_relu" 등 원하는 활성화 함수 선택
        """
        super(TeaFormer, self).__init__()
        self.mode = mode
        self.cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()

        if self.mode == "dynamic":
            self.fusion = DynamicFusion(cnn_dim=512, vit_dim=768, fuse_dim=fuse_dim, activation=activation)
            self.classifier = nn.Sequential(
                nn.Linear(fuse_dim, fuse_dim // 2),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(fuse_dim // 2, num_classes)
            )
        elif self.mode == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(512 + 768, fuse_dim),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(fuse_dim, num_classes)
            )
        elif self.mode == "cnn":
            self.classifier = nn.Sequential(
                nn.Linear(512, fuse_dim // 2),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(fuse_dim // 2, num_classes)
            )
        elif self.mode == "vit":
            self.classifier = nn.Sequential(
                nn.Linear(768, fuse_dim // 2),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(fuse_dim // 2, num_classes)
            )
        elif self.mode == "bottleneck":
            # 투영 차원을 더 낮게 줄여 경량화: 512 -> 128, 768 -> 128
            self.cnn_proj_bottle = nn.Linear(512, 128)
            self.vit_proj_bottle = nn.Linear(768, 128)
            # 경량 분류기: 128 -> 64 -> num_classes
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        elif self.mode == "bottleneck_freeze":
            # 새로운 freeze 모드: 기존 bottleneck 구조와 동일하되, CNN 백본의 파라미터를 freeze
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn_proj_bottle = nn.Linear(512, 128)
            self.vit_proj_bottle = nn.Linear(768, 128)
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                get_activation_fn(activation),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
            
    def forward(self, x):
        if self.mode == "cnn":
            feat = self.cnn(x)
            out = self.classifier(feat)
        elif self.mode == "vit":
            feat = self.vit(x)
            out = self.classifier(feat)
        elif self.mode == "concat":
            feat_cnn = self.cnn(x)
            feat_vit = self.vit(x)
            fused = torch.cat([feat_cnn, feat_vit], dim=1)
            out = self.classifier(fused)
        elif self.mode == "dynamic":
            feat_cnn = self.cnn(x)
            feat_vit = self.vit(x)
            fused = self.fusion(feat_cnn, feat_vit)
            out = self.classifier(fused)
        elif self.mode == "bottleneck":
            feat_cnn = self.cnn(x)   # shape: (B, 512)
            feat_vit = self.vit(x)   # shape: (B, 768)
            proj_cnn = self.cnn_proj_bottle(feat_cnn)  # shape: (B, 128)
            proj_vit = self.vit_proj_bottle(feat_vit)  # shape: (B, 128)
            # Residual Connection 추가: CNN 투영 feature의 10%를 덧셈하여 정보 보완
            fused = proj_cnn + proj_vit + 0.1 * proj_cnn  
            out = self.classifier(fused)
        elif self.mode == "bottleneck_freeze":
            feat_cnn = self.cnn(x)   # CNN 백본은 freeze되었음
            feat_vit = self.vit(x)
            proj_cnn = self.cnn_proj_bottle(feat_cnn)  # shape: (B, 128)
            proj_vit = self.vit_proj_bottle(feat_vit)  # shape: (B, 128)
            fused = proj_cnn + proj_vit + 0.1 * proj_cnn  
            out = self.classifier(fused)
        return out

if __name__ == "__main__":
    # 간단한 테스트: 각 모드에서 임의 입력에 대해 출력 shape을 확인합니다.
    for mode in ["dynamic", "concat", "cnn", "vit", "bottleneck", "bottleneck_freeze"]:
        model = TeaFormer(num_classes=7, mode=mode, activation="softplus")
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model(dummy_input)
        print(f"Mode: {mode}, Output shape: {output.shape}")
