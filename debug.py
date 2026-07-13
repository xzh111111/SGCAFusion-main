python -c "
import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, '/root/SGCAFusion-main')
from SGCAFusion_model import TwoBranchesFusionNet
from args import Args as args

model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)
model.load_state_dict(torch.load('/root/model/MTFusion_net_epoch_2_twoBranches.model'))
model.eval()
model.cuda()

ir  = torch.randn(1,1,480,640).cuda()
vis = torch.randn(1,1,480,640).cuda()

fea_com   = model.forward_encoder(ir, vis)
fea_fused = model.forward_MultiTask_branch(fea_com_ivif=fea_com, fea_com_mfif=fea_com)

print('fea_com type:', type(fea_com))
if isinstance(fea_com, (list,tuple)):
    for i,f in enumerate(fea_com): print(f'  fea_com[{i}] shape:', f.shape)
else:
    print('  fea_com shape:', fea_com.shape)

print('fea_fused type:', type(fea_fused))
if isinstance(fea_fused, (list,tuple)):
    for i,f in enumerate(fea_fused): print(f'  fea_fused[{i}] shape:', f.shape)
else:
    print('  fea_fused shape:', fea_fused.shape)
"