import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model import FrozenSAMFeatureExtractor, DualCueHeads, seg_loss, structural_boundary_loss

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dice_score(prob, target, threshold=0.5, eps=1e-6):
    pred = (prob >= threshold).float()
    inter = (pred*target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter+eps)/(denom+eps)).mean().item()

@torch.no_grad()
def validate(ext, heads, loader, device, loss_size):
    heads.eval()
    sem_ds=[]; str_ds=[]; avg_ds=[]
    for b in loader:
        image=b["image"].to(device); gt=b["mask"].to(device)
        mid, deep = ext(image)
        sem_l, str_l = heads(mid, deep)
        sem_l = F.interpolate(sem_l,(loss_size,loss_size),mode="bilinear",align_corners=False)
        str_l = F.interpolate(str_l,(loss_size,loss_size),mode="bilinear",align_corners=False)
        gt = F.interpolate(gt,(loss_size,loss_size),mode="nearest")
        psem=torch.sigmoid(sem_l); pstr=torch.sigmoid(str_l); pavg=.5*(psem+pstr)
        sem_ds.append(dice_score(psem,gt)); str_ds.append(dice_score(pstr,gt)); avg_ds.append(dice_score(pavg,gt))
    return {"semantic_dice":float(np.mean(sem_ds)),
            "structural_dice":float(np.mean(str_ds)),
            "naive_avg_dice":float(np.mean(avg_ds))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_root",required=True)
    ap.add_argument("--val_root",required=True)
    ap.add_argument("--sam_checkpoint",required=True)
    ap.add_argument("--model_type",default="vit_h",choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--mid_block",type=int,default=15)
    ap.add_argument("--work_dir",default="workdir/crss_sanity")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=1)
    ap.add_argument("--accumulation_steps",type=int,default=4)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--weight_decay",type=float,default=1e-4)
    ap.add_argument("--num_workers",type=int,default=4)
    ap.add_argument("--image_size",type=int,default=1024)
    ap.add_argument("--loss_size",type=int,default=256)
    ap.add_argument("--lambda_sem_gt",type=float,default=.5)
    ap.add_argument("--lambda_distill",type=float,default=1.0)
    ap.add_argument("--lambda_str_gt",type=float,default=1.0)
    ap.add_argument("--lambda_boundary",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()

    seed_all(args.seed)
    work=Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)
    tr=KvasirTeacherDataset(args.train_root,args.image_size,augment=True)
    va=KvasirTeacherDataset(args.val_root,args.image_size,augment=False)
    trl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    val=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

    sam=sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(args.device)
    ext=FrozenSAMFeatureExtractor(sam,args.mid_block).to(args.device)
    heads=DualCueHeads(ext.mid_channels,ext.deep_channels).to(args.device)
    opt=torch.optim.AdamW(heads.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best=-1.0; history=[]
    for epoch in range(1,args.epochs+1):
        heads.train(); opt.zero_grad(set_to_none=True); losses=[]
        pbar=tqdm(trl,desc=f"epoch {epoch}/{args.epochs}")
        for step,b in enumerate(pbar,1):
            image=b["image"].to(args.device); gt=b["mask"].to(args.device); teacher=b["teacher"].to(args.device)
            with torch.no_grad():
                mid,deep=ext(image)
            sem_l,str_l=heads(mid,deep)
            sem_l=F.interpolate(sem_l,(args.loss_size,args.loss_size),mode="bilinear",align_corners=False)
            str_l=F.interpolate(str_l,(args.loss_size,args.loss_size),mode="bilinear",align_corners=False)
            gt_s=F.interpolate(gt,(args.loss_size,args.loss_size),mode="nearest")
            teacher_s=F.interpolate(teacher,(args.loss_size,args.loss_size),mode="bilinear",align_corners=False).clamp(0,1)

            l_sem_gt=seg_loss(sem_l,gt_s)
            l_dist=F.smooth_l1_loss(torch.sigmoid(sem_l),teacher_s)
            l_str_gt=seg_loss(str_l,gt_s)
            l_bnd=structural_boundary_loss(str_l,gt_s)
            loss=args.lambda_sem_gt*l_sem_gt+args.lambda_distill*l_dist+args.lambda_str_gt*l_str_gt+args.lambda_boundary*l_bnd
            (loss/args.accumulation_steps).backward()
            if step%args.accumulation_steps==0 or step==len(trl):
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item()); pbar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        m=validate(ext,heads,val,args.device,args.loss_size)
        m["epoch"]=epoch; m["train_loss"]=float(np.mean(losses)); history.append(m)
        print(json.dumps(m,indent=2))
        score=.5*(m["semantic_dice"]+m["structural_dice"])
        ck={"heads":heads.state_dict(),"args":vars(args),"epoch":epoch,"metrics":m}
        torch.save(ck,work/"latest.pth")
        if score>best:
            best=score; torch.save(ck,work/"best.pth"); print("[best]",best)
        (work/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
