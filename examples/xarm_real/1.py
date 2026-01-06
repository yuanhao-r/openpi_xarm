import torch
import torchvision.models as models

# 加载模型
alexnet = models.alexnet(pretrained=False)
resnet18 = models.resnet18(pretrained=False)

# print("--- AlexNet 结构摘要 ---")
# # 注意它的 classifier 部分，参数量巨大
# print(alexnet.classifier)

# print("\n--- ResNet-18 结构摘要 ---")
# # 注意它的基本单元是 Sequential 组成的 BasicBlock
# print(resnet18.layer1[0])

def benchmark_model(model, name):
    # 模拟一张 224x224 的 RGB 图片
    input_data = torch.randn(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(input_data)
        
    print(f"{name} 输出张量形状: {output.shape}")

benchmark_model(alexnet, "AlexNet")
benchmark_model(resnet18, "ResNet18")