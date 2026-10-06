import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry
from crss_dataset import KvasirTeacherDataset
from crss_model import FrozenSAMFeatureExtractor, DualCueHeads

def dice_binary(prob,gt,threshold=.5,eps=1e-6):
    pred=prob>=threshold; gt=gt>=.5
    inter=(pred&gt).sum().item(); denom=pred.sum().item()+gt.sum().item()
    return (2*inter+eps)/(denom+eps)

def masked_corr(a,b,mask):
    a=a[mask].float(); b=b[mask].float()
    if a.numel()<10: return float("nan")
    a=a-a.mean(); b=b-b.mean()
    den=torch.sqrt((a*a).sum()*(b*b).sum()).clamp_min(1e-8)
    return float(((a*b).sum()/den).item())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",required=True)
    ap.add_argument("--sam_checkpoint",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--model_type",default="vit_h",choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--mid_block",type=int,default=15)
    ap.add_argument("--eval_size",type=int,default=256)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--output",default="workdir/crss_sanity/sanity_metrics.json")
    args=ap.parse_args()

    ds=KvasirTeacherDataset(args.data_root,1024,False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
    sam=sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(args.device)
    ext=FrozenSAMFeatureExtractor(sam,args.mid_block).to(args.device)
    heads=DualCueHeads(ext.mid_channels,ext.deep_channels).to(args.device)
    ck=torch.load(args.checkpoint,map_location="cpu"); heads.load_state_dict(ck["heads"]); heads.eval()

    semd=[]; strd=[]; avgd=[]; orad=[]; corrs=[]
    semwin=strwin=ties=0
    all_conf=[]; all_err=[]

    with torch.no_grad():
        for b in tqdm(loader,desc="sanity-eval"):
            image=b["image"].to(args.device); gt=b["mask"].to(args.device)
            mid,deep=ext(image); sl,tl=heads(mid,deep)
            sl=F.interpolate(sl,(args.eval_size,args.eval_size),mode="bilinear",align_corners=False)
            tl=F.interpolate(tl,(args.eval_size,args.eval_size),mode="bilinear",align_corners=False)
            gt=F.interpolate(gt,(args.eval_size,args.eval_size),mode="nearest")
            ps=torch.sigmoid(sl); pt=torch.sigmoid(tl); pa=.5*(ps+pt)
            es=(ps-gt).abs(); et=(pt-gt).abs()
            choose_s=es<et; choose_t=et<es; tie=(es-et).abs()<=.02
            po=torch.where(choose_s,ps,pt)

            semd.append(dice_binary(ps,gt)); strd.append(dice_binary(pt,gt))
            avgd.append(dice_binary(pa,gt)); orad.append(dice_binary(po,gt))
            informative=(gt>.5)|(ps>.1)|(pt>.1)
            corrs.append(masked_corr(ps,pt,informative))

            conf=(ps-pt).abs(); cm=conf>.10
            semwin+=int((choose_s&cm&~tie).sum().item())
            strwin+=int((choose_t&cm&~tie).sum().item())
            ties+=int((tie&cm).sum().item())

            err=((ps>=.5)!=(gt>=.5)) | ((pt>=.5)!=(gt>=.5))
            all_conf.append(conf.cpu().flatten()); all_err.append(err.cpu().flatten().float())

    conf=torch.cat(all_conf); err=torch.cat(all_err)
    total=semwin+strwin+ties
    result={
        "n_images":len(ds),
        "semantic_dice":float(np.mean(semd)),
        "structural_dice":float(np.mean(strd)),
        "naive_average_dice":float(np.mean(avgd)),
        "oracle_dice":float(np.mean(orad)),
        "oracle_gain_vs_best_head":float(np.mean(orad)-max(np.mean(semd),np.mean(strd))),
        "prediction_corr_informative_pixels":float(np.nanmean(corrs)),
        "conflict_semantic_win_rate":semwin/total if total else None,
        "conflict_structural_win_rate":strwin/total if total else None,
        "conflict_tie_rate":ties/total if total else None,
        "overall_error_rate":float(err.mean().item())
    }
    order=torch.argsort(conf,descending=True); n=conf.numel(); total_err=err.sum().item()
    for r in (.1,.2,.3):
        k=max(1,int(n*r)); idx=order[:k]
        result[f"top{int(r*100)}_difficulty_error_rate"]=float(err[idx].mean().item())
        result[f"top{int(r*100)}_difficulty_error_capture"]=float(err[idx].sum().item()/total_err) if total_err>0 else 0.0

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    print("saved:",out)

if __name__=="__main__":
    main()
