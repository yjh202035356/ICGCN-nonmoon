import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crss_dataset
from crss_dataset import KvasirTeacherDataset
from scripts.prepare_polyp1450 import paired, check_pairs
import segment_anything
import train_crss_lora
import train_resolver_v2
from polyp_checkpoint import build_backbone
import export_polyp_benchmark

torch.set_num_threads(1)

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.norm1=nn.LayerNorm(8)
        self.attn=nn.Module(); self.attn.qkv=nn.Linear(8,24)
    def forward(self, x):
        q,k,v=self.attn.qkv(self.norm1(x)).chunk(3,-1)
        return x+0.1*(q+v)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.patch=nn.Conv2d(3,8,16,stride=256)
        self.blocks=nn.ModuleList([Block(),Block()]); self.neck=nn.Sequential(nn.Conv2d(8,8,1))
    def forward(self,x):
        x=self.patch(x).permute(0,2,3,1)
        for block in self.blocks: x=block(x)
        return self.neck(x.permute(0,3,1,2))

class TinySAM(nn.Module):
    def __init__(self): super().__init__(); self.image_encoder=Encoder()
    def preprocess(self,x): return x/255

def factory(checkpoint):
    assert checkpoint=='fake_sam_checkpoint'
    with torch.random.fork_rng():
        torch.manual_seed(567); return TinySAM()

segment_anything.sam_model_registry['vit_h']=factory

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.train=self.root/'train'
        for folder in ['images','labels','attribution_map']: (self.train/folder).mkdir(parents=True)
        for i,(h,w) in enumerate([(18,12),(12,18)]):
            image=np.arange(h*w*3,dtype=np.uint8).reshape(h,w,3)
            mask=np.zeros((h,w),np.uint8); mask[h//4:h*3//4,w//4:w*3//4]=255
            teacher=np.linspace(0,255,h*w,dtype=np.uint8).reshape(h,w)
            for folder,array in [('images',image),('labels',mask),('attribution_map',teacher)]: Image.fromarray(array).save(self.train/folder/f'case{i}.png')
    def tearDown(self): self.temp.cleanup()
    def call(self,module,args):
        previous=sys.argv; sys.argv=[module.__name__]+args
        try:
            with contextlib.redirect_stdout(io.StringIO()): module.main()
        finally: sys.argv=previous
    def test_strict_data_and_teacher_free_path(self):
        self.assertEqual(len(KvasirTeacherDataset(self.train)),2)
        (self.train/'attribution_map'/'case1.png').unlink()
        with self.assertRaisesRegex(RuntimeError,'Incomplete attribution_map'): KvasirTeacherDataset(self.train)
        dataset=KvasirTeacherDataset(self.train,require_teacher=False)
        self.assertNotIn('teacher', dataset[0]); self.assertEqual(dataset[0]['image'].shape,(3,1024,1024))
        with self.assertRaisesRegex(RuntimeError,'Incomplete teacher'): check_pairs(self.train,2,teacher=True)
        self.assertEqual(len(check_pairs(self.train,2)),2)
    def test_duplicate_dimensions_and_constant_teacher(self):
        Image.new('RGB',(12,18)).save(self.train/'images'/'case0.jpg')
        with self.assertRaisesRegex(RuntimeError,'Duplicate stem'): KvasirTeacherDataset(self.train)
        (self.train/'images'/'case0.jpg').unlink()
        Image.new('L',(1,1)).save(self.train/'attribution_map'/'case0.png')
        with self.assertRaisesRegex(RuntimeError,'dimension mismatch'): KvasirTeacherDataset(self.train)[0]
        with self.assertRaisesRegex(RuntimeError,'constant teacher'): check_pairs(self.train,2,teacher=True)
    def test_fixed_schedule_training_lora_restore_resolver_and_export(self):
        heads=self.root/'heads'; resolver=self.root/'resolver'
        self.call(train_crss_lora,['--train_root',str(self.train),'--sam_checkpoint','fake_sam_checkpoint','--fixed_schedule','--expected_train_count','2','--work_dir',str(heads),'--epochs','1','--mid_block','0','--lora_start_block','1','--num_workers','0','--max_train_batches','2','--accumulation_steps','2','--teacher_refine_v2','--device','cpu'])
        hpath=heads/'latest.pth'; hck=torch.load(hpath,map_location='cpu')
        self.assertFalse((heads/'best.pth').exists()); self.assertEqual(hck['metrics']['train_batches'],2)
        self.assertIsNone(hck['score']); self.assertEqual(hck['metrics']['selection'],'fixed_final_epoch')
        self.assertTrue(any(value.abs().sum()>0 for name,value in hck['lora'].items() if '_B.' in name))
        ext,heads_model,_=build_backbone(factory('fake_sam_checkpoint'),hpath,'cpu')
        params=dict(ext.named_parameters())
        for name,value in hck['lora'].items(): self.assertTrue(torch.equal(params[name],value))
        self.assertFalse(any(p.requires_grad for p in ext.parameters()))
        incomplete=dict(hck); incomplete['lora']=dict(hck['lora']); incomplete['lora'].pop(next(iter(incomplete['lora'])))
        bad=self.root/'bad.pth'; torch.save(incomplete,bad)
        with self.assertRaisesRegex(RuntimeError,'exactly all required'): build_backbone(factory('fake_sam_checkpoint'),bad,'cpu')
        # Resolver must work after removing teacher maps entirely.
        for path in (self.train/'attribution_map').iterdir(): path.unlink()
        self.call(train_resolver_v2,['--train_root',str(self.train),'--sam_checkpoint','fake_sam_checkpoint','--heads_checkpoint',str(hpath),'--fixed_schedule','--expected_train_count','2','--work_dir',str(resolver),'--epochs','1','--num_workers','0','--max_train_batches','1','--accumulation_steps','2','--device','cpu'])
        rpath=resolver/'latest.pth'; rck=torch.load(rpath,map_location='cpu')
        self.assertFalse((resolver/'best.pth').exists()); self.assertEqual(rck['metrics']['train_batches'],1)
        self.assertIn('backbone_sha256',rck)
        testroot=self.root/'test'; testset=testroot/'Kvasir'
        for folder,origin in [('images','images'),('masks','labels')]:
            (testset/folder).mkdir(parents=True)
            for p in (self.train/origin).iterdir(): (testset/folder/p.name).write_bytes(p.read_bytes())
        previous_counts=export_polyp_benchmark.TEST_COUNTS; export_polyp_benchmark.TEST_COUNTS={'Kvasir':2}
        predictions=self.root/'predictions'
        try:
            self.call(export_polyp_benchmark,['--test-root',str(testroot),'--sam_checkpoint','fake_sam_checkpoint','--heads_checkpoint',str(hpath),'--resolver_checkpoint',str(rpath),'--output',str(predictions),'--device','cpu','--max-images','2'])
            report=json.loads((predictions/'export_manifest.json').read_text()); self.assertEqual(report['status'],'smoke_complete'); self.assertEqual(report['counts']['Kvasir'],2)
            for stem in ['case0','case1']:
                with Image.open(self.train/'images'/f'{stem}.png') as image: size=image.size
                for method in ['semantic','structural','naive_average','resolver_v2']:
                    with Image.open(predictions/method/'Kvasir'/f'{stem}.png') as image: self.assertEqual(image.size,size)
                with np.load(predictions/'raw_probabilities'/'Kvasir'/f'{stem}.npz') as maps:
                    self.assertEqual(set(maps.files),{'semantic','structural','naive_average','resolver_v2'})
                    for array in maps.values(): self.assertTrue(np.isfinite(array).all()); self.assertGreaterEqual(array.min(),0); self.assertLessEqual(array.max(),1)
        finally: export_polyp_benchmark.TEST_COUNTS=previous_counts
    def test_test_data_rejected_during_fixed_training(self):
        for module in [train_crss_lora,train_resolver_v2]:
            args=['--train_root',str(self.train),'--val_root','forbidden_test_path','--sam_checkpoint','fake_sam_checkpoint','--fixed_schedule']
            if module is train_resolver_v2: args+=['--heads_checkpoint','not_opened.pth']
            with self.assertRaises(SystemExit) as error: self.call(module,args)
            self.assertEqual(error.exception.code,2)

    def test_raw_probability_metric_is_distinct_from_normalized_png(self):
        import evaluate_polyp_masks as evaluator
        evaluator.DATASETS={'Kvasir':2}
        pred_root=self.root/'pred'; gt_root=self.root/'gt'; raw_root=self.root/'raw'
        for folder in [pred_root/'Kvasir',gt_root/'Kvasir'/'masks',raw_root/'Kvasir']: folder.mkdir(parents=True)
        for path in (self.train/'labels').iterdir():
            values=np.asarray(Image.open(path))
            Image.fromarray(values).save(pred_root/'Kvasir'/path.name)
            Image.fromarray(values).save(gt_root/'Kvasir'/'masks'/path.name)
            np.savez(raw_root/'Kvasir'/f'{path.stem}.npz',semantic=(0.1+0.1*(values>127)).astype(np.float32))
        output=self.root/'measured'
        with contextlib.redirect_stdout(io.StringIO()): evaluator.evaluate(pred_root,gt_root,'synthetic','local_inference',output,raw_root,'semantic')
        row=json.loads(output.with_suffix('.json').read_text())['results'][0]
        self.assertEqual(row['stored_map_dice_at_0.5'],1)
        self.assertEqual(row['raw_probability_dice_at_0_5'],0)
        self.assertGreater(row['threshold_mean_dice'],0.99)

    def test_fixed_teacher_iba_and_lossless_resume(self):
        # Actual IBA code, small deterministic teacher; no pretrained CLIP claim.
        import torch.nn.functional as F
        class Layer(nn.Module):
            def __init__(self): super().__init__(); self.linear=nn.Linear(8,8)
            def forward(self,x,**kwargs): return (self.linear(x).tanh(),)
        class Vision(nn.Module):
            def __init__(self):
                super().__init__(); self.embedding=nn.Linear(3,8); self.encoder=nn.Module(); self.encoder.layers=nn.ModuleList([Layer()])
            def forward(self,x,**kwargs):
                patches=F.adaptive_avg_pool2d(x,(2,2)).flatten(2).transpose(1,2)
                embedded=self.embedding(patches); embedded=torch.cat([embedded.mean(1,keepdim=True),embedded],1)
                encoded=self.encoder.layers[0](embedded)[0]
                return {'hidden_states':(embedded,encoded),'attentions':(torch.ones(1,1,5,5),),'last_hidden_state':encoded}
        class Teacher(nn.Module):
            calls=0
            def __init__(self): super().__init__(); self.vision_model=Vision()
            def get_text_features(self,ids): return torch.arange(1,9,dtype=torch.float32).unsqueeze(0)
            def get_image_features(self,pixels):
                Teacher.calls+=1
                return self.vision_model(pixels)['last_hidden_state'].mean(1)
        class ModelFactory:
            latest=None
            @staticmethod
            def from_pretrained(*args,**kwargs):
                ModelFactory.latest=Teacher(); return ModelFactory.latest
        class Processor:
            @staticmethod
            def from_pretrained(*args,**kwargs): return Processor()
            def __call__(self,images,**kwargs):
                array=np.array(images,dtype=np.float32)/255
                return {'pixel_values':torch.from_numpy(array).permute(2,0,1).unsqueeze(0)}
        class Tokenizer:
            @staticmethod
            def from_pretrained(*args,**kwargs): return Tokenizer()
            def encode(self,*args,**kwargs): return [1,2,3]
        stub=types.ModuleType('transformers')
        for name,value in [('AutoModel',ModelFactory),('AutoProcessor',Processor),('AutoTokenizer',Tokenizer),('CLIPModel',ModelFactory),('CLIPProcessor',Processor),('CLIPTokenizerFast',Tokenizer)]: setattr(stub,name,value)
        sys.modules['transformers']=stub
        from scripts import generate_polyp_teacher as generator
        model_path=self.root/'teacher_model'; model_path.mkdir(); (model_path/'config.json').write_text('{}')
        output=self.root/'teacher_outputs'
        args=['--input-path',str(self.train/'images'),'--output-path',str(output),'--model-path',str(model_path),'--max-images','2','--vlayer','0','--device','cpu']
        self.call(generator,args)
        manifest=json.loads((output/'teacher_generation.json').read_text())
        self.assertEqual(manifest['status'],'complete'); self.assertEqual(len(manifest['completed']),2)
        self.assertFalse(any(parameter.grad is not None for parameter in ModelFactory.latest.parameters()))
        for path in (self.train/'images').iterdir():
            with Image.open(path) as original, Image.open(output/path.name) as heatmap:
                self.assertEqual(original.size,heatmap.size); self.assertEqual(heatmap.mode,'L')
        calls=Teacher.calls; self.call(generator,args); self.assertEqual(Teacher.calls,calls)
        with self.assertRaises(SystemExit): self.call(generator,args+['--text','other_text'])

if __name__=='__main__': unittest.main(verbosity=2)
