import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import cv2
import math
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    from SGCAFusion_model_visual import TwoBranchesFusionNet
except ImportError:
    from SGCAFusion_model import TwoBranchesFusionNet

from args import Args as args


parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    type=str,
    default='/root/model/MTFusion_net_epoch_2_twoBranches.model',
    help='fusion network weight'
)

parser.add_argument('--test_ir_root', type=str, required=True, help='the test IR images root')
parser.add_argument('--IR_IS_RGB', action='store_true', help='the IR input is stored in RGB format')

parser.add_argument('--test_vis_root', type=str, required=True, help='the test VIS images root')
parser.add_argument('--VIS_IS_RGB', action='store_true', help='the VIS input is stored in RGB format')

parser.add_argument(
    '--save_path',
    type=str,
    default='SGCAFusion-main/outputs/',
    help='the fusion results and feature maps will be saved here'
)

parser.add_argument(
    '--task',
    type=str,
    default='IVIF',
    choices=['IVIF', 'MFIF'],
    help='task type. IVIF uses trainingTag=1, MFIF uses trainingTag=2'
)

parser.add_argument(
    '--trainingTag',
    type=int,
    default=None,
    choices=[1, 2],
    help='optional manual override: 1 for IVIF, 2 for MFIF'
)

parser.add_argument(
    '--max_feature_channels',
    type=int,
    default=16,
    help='number of single-channel feature maps to save for each module'
)

parser.add_argument(
    '--save_channel_grid',
    action='store_true',
    help='also save a grid of several individual channels for each feature tensor'
)

opt = parser.parse_args()


def get_training_tag(opt):
    if opt.trainingTag is not None:
        return opt.trainingTag
    return 1 if opt.task.upper() == 'IVIF' else 2


def load_model(model_path_twoBranches):
    model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)

    checkpoint = torch.load(model_path_twoBranches, map_location='cpu')
    model.load_state_dict(checkpoint)

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    total = sum([param.nelement() for param in model.parameters()])
    print('Number    of    parameter: {:4f}M'.format(total / 1e6))

    model.eval()
    if args.cuda:
        model.cuda()

    return model


def normalize_feature_map(x):
    """
    Normalize feature map to [0, 1].
    """
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def get_last_feature_tensor(features):
    """
    Extract the last valid Tensor from Tensor / list / tuple / dict.
    This is used when the network returns multi-level features.
    """

    if isinstance(features, torch.Tensor):
        return features

    elif isinstance(features, (list, tuple)):
        for item in reversed(features):
            if isinstance(item, torch.Tensor):
                return item
            elif isinstance(item, (list, tuple, dict)):
                tensor = get_last_feature_tensor(item)
                if tensor is not None:
                    return tensor

    elif isinstance(features, dict):
        values = list(features.values())
        for item in reversed(values):
            if isinstance(item, torch.Tensor):
                return item
            elif isinstance(item, (list, tuple, dict)):
                tensor = get_last_feature_tensor(item)
                if tensor is not None:
                    return tensor

    return None


def save_single_feature_map(features, save_root, img_name, tag):
    """
    Save one real single-channel grayscale mean activation map.
    Return the normalized activation map for three-panel visualization.
    """

    os.makedirs(save_root, exist_ok=True)

    feat = get_last_feature_tensor(features)

    if feat is None:
        print(f"[Warning] No valid feature tensor for {tag}")
        return None

    feat = feat.detach().float().cpu()

    # [B, C, H, W] -> [C, H, W]
    if feat.ndim == 4:
        feat = feat[0]
    elif feat.ndim == 3:
        pass
    else:
        print(f"[Warning] Unsupported feature shape for {tag}: {feat.shape}")
        return None

    # Mean activation map: [C, H, W] -> [H, W]
    activation_map = torch.mean(torch.abs(feat), dim=0)

    # Normalize to [0, 1]
    activation_map = normalize_feature_map(activation_map).numpy()

    # Save as real 8-bit single-channel grayscale image
    activation_map_uint8 = (activation_map * 255).astype(np.uint8)

    img_base_name = os.path.splitext(img_name)[0]

    save_path = os.path.join(
        save_root,
        f"{img_base_name}_{tag}.png"
    )

    Image.fromarray(activation_map_uint8, mode="L").save(save_path)

    print(f"Saved single-channel feature map: {save_path}")

    # Return [0, 1] map for three-panel figure
    return activation_map

def normalize_feature_map(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def get_tensor_from_feature(features):
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, (list, tuple)):
        for item in reversed(features):
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(features, dict):
        for item in reversed(list(features.values())):
            if isinstance(item, torch.Tensor):
                return item
    return None


def feature_to_activation_map(features):
    """
    Convert BxCxHxW feature tensor to one 2D mean-activation map.
    This is suitable for paper visualisation.
    """
    feat = get_tensor_from_feature(features)
    if feat is None:
        return None

    feat = feat.detach().float().cpu()

    if feat.ndim == 4:
        feat = feat[0]  # CxHxW
    elif feat.ndim == 3:
        pass
    else:
        print(f'[Warning] Unsupported feature shape: {feat.shape}')
        return None

    activation_map = torch.mean(torch.abs(feat), dim=0)
    activation_map = normalize_feature_map(activation_map).numpy()

    return activation_map


def save_single_feature_map(features, save_root, img_name, tag):
    """
    Save one single-channel grayscale mean activation map.
    This saves a real 8-bit single-channel image: H x W.
    """

    os.makedirs(save_root, exist_ok=True)

    feat = get_last_feature_tensor(features)

    if feat is None:
        print(f"[Warning] No valid feature tensor for {tag}")
        return

    feat = feat.detach().float().cpu()

    # [B, C, H, W] -> [C, H, W]
    if feat.ndim == 4:
        feat = feat[0]
    elif feat.ndim == 3:
        pass
    else:
        print(f"[Warning] Unsupported feature shape for {tag}: {feat.shape}")
        return

    # 对所有通道取平均，得到单通道响应图 [H, W]
    activation_map = torch.mean(torch.abs(feat), dim=0)

    # 归一化到 [0, 1]
    activation_map = normalize_feature_map(activation_map).numpy()

    # 转成真正的 8-bit 单通道灰度图 [0, 255]
    activation_map = (activation_map * 255).astype(np.uint8)

    img_base_name = os.path.splitext(img_name)[0]

    save_path = os.path.join(
        save_root,
        f"{img_base_name}_{tag}.png"
    )

    # mode="L" 表示单通道灰度图
    Image.fromarray(activation_map, mode="L").save(save_path)

    print(f"Saved single-channel feature map: {save_path}")


def save_channel_grid(features, save_root, img_name, tag, max_channels=16):
    os.makedirs(save_root, exist_ok=True)

    feat = get_tensor_from_feature(features)
    if feat is None:
        return

    feat = feat.detach().float().cpu()

    if feat.ndim == 4:
        feat = feat[0]
    elif feat.ndim == 3:
        pass
    else:
        return

    C, H, W = feat.shape
    channel_num = min(C, max_channels)
    cols = int(math.ceil(math.sqrt(channel_num)))
    rows = int(math.ceil(channel_num / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis('off')
        if i < channel_num:
            fmap = normalize_feature_map(feat[i]).numpy()
            axes[i].imshow(fmap, cmap='gray')
            axes[i].set_title(f'C{i}', fontsize=8)

    img_base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(save_root, f'{img_base_name}_{tag}_channels.png')

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved channel grid: {save_path}')


def save_three_feature_panel(shared_map, ivif_map, mfif_map, save_root, img_name):
    """
    Save a single three-column figure:
    shared encoder | IVIF branch | MFIF branch
    """
    os.makedirs(save_root, exist_ok=True)

    if shared_map is None or ivif_map is None or mfif_map is None:
        return

    img_base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(save_root, f'{img_base_name}_three_feature_maps.png')

    titles = ['Shared encoder', 'IVIF branch', 'MFIF branch']
    maps = [shared_map, ivif_map, mfif_map]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, title, fmap in zip(axes, titles, maps):
        ax.imshow(fmap, cmap='jet')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved three-feature panel: {save_path}')


def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]

    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel


def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image


def fuse_cb_cr(Cb1, Cr1, Cb2, Cr2):
    H, W = Cb1.shape
    Cb = np.ones((H, W), dtype=np.float32)
    Cr = np.ones((H, W), dtype=np.float32)

    for k in range(H):
        for n in range(W):
            if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                Cb[k, n] = 128
            else:
                middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                Cb[k, n] = middle_1 / (middle_2 + 1e-8)

            if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                Cr[k, n] = 128
            else:
                middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                Cr[k, n] = middle_3 / (middle_4 + 1e-8)

    return Cb, Cr


def run(model, ir_test_batch, vis_test_batch, output_path, img_name, training_tag):
    img_ir = Variable(ir_test_batch, requires_grad=False)
    img_vi = Variable(vis_test_batch, requires_grad=False)

    # 1. Shared encoder feature
    fea_com = model.forward_encoder(img_ir, img_vi)

    # 2. Two branch features
    # fea_fused is the selected task output used by the decoder.
    # fea_ivif and fea_mfif are saved for visualisation.
    fea_fused, fea_ivif, fea_mfif = model.forward_MultiTask_branch(
        fea_com_ivif=fea_com,
        fea_com_mfif=fea_com,
        trainingTag=training_tag,
        return_features=True
    )

    feature_root = os.path.join(output_path, 'feature_maps')
    single_root = os.path.join(feature_root, 'single_maps')
    panel_root = os.path.join(feature_root, 'three_panel')

    shared_map = save_single_feature_map(
        fea_com, single_root, img_name, tag='shared_encoder'
    )
    ivif_map = save_single_feature_map(
        fea_ivif, single_root, img_name, tag='ivif_branch'
    )
    mfif_map = save_single_feature_map(
        fea_mfif, single_root, img_name, tag='mfif_branch'
    )

    save_three_feature_panel(shared_map, ivif_map, mfif_map, panel_root, img_name)

    if opt.save_channel_grid:
        grid_root = os.path.join(feature_root, 'channel_grids')
        save_channel_grid(fea_com, grid_root, img_name, 'shared_encoder', opt.max_feature_channels)
        save_channel_grid(fea_ivif, grid_root, img_name, 'ivif_branch', opt.max_feature_channels)
        save_channel_grid(fea_mfif, grid_root, img_name, 'mfif_branch', opt.max_feature_channels)

    # 3. Decoder output
    out_y_or_gray = model.forward_mixed_decoder(fea_com, fea_fused)

    out_y_or_gray = out_y_or_gray[0, 0, :, :].detach().cpu().numpy()
    out_y_or_gray = np.clip(out_y_or_gray * 255.0, 0, 255)

    return out_y_or_gray


def is_image_file(filename):
    suffix = os.path.splitext(filename)[1].lower()
    return suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


def main():
    model_path_twoBranches = opt.checkpoint
    output_path = opt.save_path
    os.makedirs(output_path, exist_ok=True)

    training_tag = get_training_tag(opt)
    print(f'Task: {opt.task}, trainingTag: {training_tag}')

    with torch.no_grad():
        model = load_model(model_path_twoBranches)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        ir_path_root = opt.test_ir_root
        vis_path_root = opt.test_vis_root

        names = sorted([name for name in os.listdir(ir_path_root) if is_image_file(name)])

        for fileName in names:
            ir_path = os.path.join(ir_path_root, fileName)
            vis_path = os.path.join(vis_path_root, fileName)

            if not os.path.exists(vis_path):
                print(f'[Warning] VIS image not found, skip: {vis_path}')
                continue

            # IR input
            if opt.IR_IS_RGB:
                ir_img_pil = Image.open(ir_path).convert('RGB')
                ir_img, ir_img_cb, ir_img_cr = rgb_to_ycbcr(ir_img_pil)
                ir_img = ir_img.astype(np.uint8)
            else:
                ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

            # VIS input
            if opt.VIS_IS_RGB:
                vis_img_pil = Image.open(vis_path).convert('RGB')
                vis_img, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vis_img_pil)
                vis_img = vis_img.astype(np.uint8)
            else:
                vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

            if ir_img is None:
                print(f'[Warning] Failed to read IR image, skip: {ir_path}')
                continue
            if vis_img is None:
                print(f'[Warning] Failed to read VIS image, skip: {vis_path}')
                continue

            # If at least one input is RGB, use Cb/Cr to reconstruct the final color image.
            if opt.IR_IS_RGB or opt.VIS_IS_RGB:
                if opt.IR_IS_RGB and opt.VIS_IS_RGB:
                    vi_img_cb, vi_img_cr = fuse_cb_cr(vi_img_cb, vi_img_cr, ir_img_cb, ir_img_cr)
                elif opt.IR_IS_RGB:
                    vi_img_cb = ir_img_cb
                    vi_img_cr = ir_img_cr
                # else: keep VIS Cb and Cr.

            ir_img = transform(ir_img)
            vis_img = transform(vis_img)

            if args.cuda:
                ir_img = ir_img.cuda()
                vis_img = vis_img.cuda()

            ir_test_batch = ir_img.unsqueeze(0).to(torch.float32)
            vis_test_batch = vis_img.unsqueeze(0).to(torch.float32)

            fused_y_or_gray = run(
                model,
                ir_test_batch,
                vis_test_batch,
                output_path,
                fileName,
                training_tag
            )

            outputFuse_path = os.path.join(output_path, fileName)

            if opt.IR_IS_RGB or opt.VIS_IS_RGB:
                fuseImage = ycbcr_to_rgb(fused_y_or_gray, vi_img_cb, vi_img_cr)
                fuseImage.save(outputFuse_path)
            else:
                fused_y_or_gray = fused_y_or_gray.astype(np.uint8)
                cv2.imwrite(outputFuse_path, fused_y_or_gray)

            print('Image -> ' + fileName + ' Done......')


if __name__ == '__main__':
    main()
