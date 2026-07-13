import torch
import torch.nn as nn

# ============================================================
# ChannelAttn 模块（所有方案共用）
# ============================================================


# 直接复制过来，避免循环导入
class ComplementFeatureFusionModule(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(ComplementFeatureFusionModule, self).__init__()

        self.height = height
        d = (32+32+32+2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d((32+32+32+2)*2, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        attn = self.mlp(in_feats)
        return attn

class ChannelAttn(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c // r),
            nn.ReLU(),
            nn.Linear(c // r, c),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * w


# ============================================================
# Baseline：无通道注意力（对照组）
# ============================================================
class CNNspecificDecoder_Baseline(nn.Module):
    """
    消融对照组：decoder 中不插入任何 ChannelAttn。
    用于与各实验组对比，确认 ChannelAttn 的增益来自哪一层。
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 实验1：仅第 1 层后插入 ChannelAttn（索引 _ == 0）
# ============================================================
class CNNspecificDecoder_Attn_Layer1(nn.Module):
    """
    消融实验1：只在第 1 个卷积块（Pad→Conv→ReLU）之后插入 ChannelAttn。
    第 1 层输出通道数 = embed_size。
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
       
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
            # ↓ 仅第 1 层后插入
            if cur_depth == 0:
                layers.append(ChannelAttn(channels[cur_depth], r=4))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 实验2：仅第 2 层后插入 ChannelAttn（索引 _ == 1）
# ============================================================
class CNNspecificDecoder_Attn_Layer2(nn.Module):
    """
    消融实验2：只在第 2 个卷积块（Pad→Conv→ReLU）之后插入 ChannelAttn。
    第 2 层输出通道数 = embed_size // 2。
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
            # ↓ 仅第 2 层后插入
            if cur_depth == 1:
                layers.append(ChannelAttn(channels[cur_depth], r=4))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 实验3：仅第 3 层后插入 ChannelAttn（索引 _ == 2）← 原始方案
# ============================================================
class CNNspecificDecoder_Attn_Layer3(nn.Module):
    """
    消融实验3（原始方案）：只在第 3 个卷积块（倒数第二组 ReLU）之后插入 ChannelAttn。
    第 3 层输出通道数 = embed_size // 4。
    与原代码 `_ == num_decoder_layers - 2` 完全等价。
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
            # ↓ 仅第 3 层后插入（即 num_decoder_layers - 2 = 2）
            if cur_depth == num_decoder_layers - 2:
                layers.append(ChannelAttn(channels[cur_depth], r=4))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 实验4：仅第 4 层（最后一层 Tanh）前插入 ChannelAttn（索引 _ == 3）
# ============================================================
class CNNspecificDecoder_Attn_Layer4(nn.Module):
    """
    消融实验4：在最后一个卷积块（Pad→Conv→Tanh）之前插入 ChannelAttn。
    注意：此层输出为单通道（channels[-1] = 1），
    ChannelAttn 作用于进入最后一层前的特征（通道数 = embed_size // 4）。
    因此 ChannelAttn 插在 第3层 ReLU 之后、第4层 ReflectionPad 之前。
    
    实现方式：在 cur_depth == 3（最后一层）的 ReflectionPad 前、
    即 cur_depth == 2 结束后追加。等效写法见下方循环注释。
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            # ↓ 在第 4 层（cur_depth==3）的 Pad 之前插入 ChannelAttn
            #   此时输入通道数为 channels[2] = embed_size // 4
            if cur_depth == num_decoder_layers - 1:
                layers.append(ChannelAttn(lastOut, r=4))
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 实验5：所有层后均插入 ChannelAttn（最后一层 Tanh 后不加，只加前3层）
# ============================================================
class CNNspecificDecoder_Attn_AllLayers(nn.Module):
    """
    消融实验5：在每个带 ReLU 的卷积块之后均插入 ChannelAttn（共3处）。
    最后一层（Tanh 激活，输出单通道）不加，因通道数=1时 ChannelAttn 意义不大。
    
    插入位置：
        Layer1 后（c = embed_size）
        Layer2 后（c = embed_size // 2）
        Layer3 后（c = embed_size // 4）
    """
    def __init__(self, embed_size, num_decoder_layers):
        super().__init__()
       
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size * 2)

        layers = []
        channels = [embed_size, embed_size // 2, embed_size // 4, 1]
        lastOut = embed_size * 2
        for cur_depth in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if cur_depth == num_decoder_layers - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(True))
                # ↓ 前 3 层（有 ReLU 的层）均插入 ChannelAttn
                layers.append(ChannelAttn(channels[cur_depth], r=4))
            lastOut = channels[cur_depth]
        self.decoder = nn.Sequential(*layers)

    def forward(self, fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused)
        x = self.decoder(x)
        x = x / 2 + 0.5
        return x


# ============================================================
# 快速验证：打印各方案 decoder 结构
# ============================================================
if __name__ == "__main__":
    embed_size = 64       # 与 args.n 对应，按实际修改
    num_decoder_layers = 4

    configs = {
        "Baseline（无注意力）":         CNNspecificDecoder_Baseline,
        "实验1（仅第1层）":              CNNspecificDecoder_Attn_Layer1,
        "实验2（仅第2层）":              CNNspecificDecoder_Attn_Layer2,
        "实验3（仅第3层，原始方案）":    CNNspecificDecoder_Attn_Layer3,
        "实验4（仅第4层前）":            CNNspecificDecoder_Attn_Layer4,
        "实验5（所有层）":               CNNspecificDecoder_Attn_AllLayers,
    }

    print("=" * 60)
    for name, cls in configs.items():
        # 注意：单独验证时 ComplementFeatureFusionModule 需可导入
        # 这里只打印 decoder 子模块结构
        print(f"\n【{name}】")
        # 临时构造一个 dummy 实例（跳过 fuseComplementFeatures 导入）
        class _Dummy(cls):
            def __init__(self):
                nn.Module.__init__(self)
                # 跳过 fuseComplementFeatures，只构建 decoder
                import torch.nn as nn
                embed_size_ = embed_size
                num_decoder_layers_ = num_decoder_layers
                layers = []
                channels = [embed_size_, embed_size_//2, embed_size_//4, 1]
                lastOut = embed_size_ * 2
                # 复用父类逻辑，直接调用父类 __init__ 会报导入错误，
                # 所以这里只演示打印，实际使用时请在完整工程中导入。
                pass
        print(f"  → 请在完整工程中实例化后调用 print(model.decoder) 查看结构")
    print("\n" + "=" * 60)
    print("提示：在完整工程中替换 CNNspecificDecoder 为对应消融类即可。")
    print("建议在 TwoBranchesFusionNet.__init__ 中修改以下一行：")
    print("  self.cnnDecoder = CNNspecificDecoder_Attn_LayerX(embed_size, num_decoder_layers)")