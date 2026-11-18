"""CPU mathematical/lifecycle checks; no downloaded or private model weights."""
import copy,json
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image
from parseq_ocr import ppocr_corrected_training as c

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,4,1);self.bn=torch.nn.BatchNorm2d(4);self.fc=torch.nn.Linear(4,96)
    def forward(self,x):
        x=self.bn(self.conv(x)).mean(2);x=torch.nn.functional.adaptive_avg_pool1d(x,40).transpose(1,2);return self.fc(x)

def model():
    torch.manual_seed(3);m=Toy();c.install_bn_layout(m);return m

def test_literal_codec_ctc_repeat_and_profile():
    for text in [' 98.20 DH ','dh','11.00']:
        ids,lengths=c.targets([text],'cpu');assert lengths.item()==len(text);assert ''.join(c.VOCAB[i]for i in ids)==text
    with pytest.raises(ValueError,match='40frames'):c.targets(['1'*21],'cpu')
    with pytest.raises(ValueError,match='unsupported'):c.targets(['€1'],'cpu')
    cfg={'height':128,'width':320,'initial_codec':'ppocr_ascii96_v1','initial_checkpoint':'x','initial_checkpoint_sha256':'0'*64}
    c.validate_config(cfg)
    for key,value in [('height',48),('width',128),('initial_codec','other')]:
        with pytest.raises(ValueError):c.validate_config({**cfg,key:value})

def test_ctc_sample_denominator_full_vs_micro_and_bn_gradient_layout():
    torch.set_num_threads(2);a=model().eval();b=copy.deepcopy(a).eval();x=torch.randn(2,3,128,320);labels=[' 11.00 DH ','Ab']
    full,_=c.ctc_sum(a,x,labels,2);full.backward();micro=0
    for i,s in enumerate(labels):loss,_=c.ctc_sum(b,x[i:i+1],[s],2);loss.backward();micro+=float(loss.detach())
    assert abs(full.item()-micro)<2e-4
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa.grad,pb.grad,atol=2e-4,rtol=1e-4)
    g=torch.randn(4,3).t();fixed=c.contiguous_gradient(g);assert fixed.is_contiguous()and torch.equal(g,fixed)and fixed.dtype==g.dtype

def test_two_real_helper_updates_reload_and_public_inference_loader(tmp_path,monkeypatch):
    torch.set_num_threads(2);m=model();opt=c.optimizer(m);before=c.tree(dict(m.named_buffers()));x=torch.randn(2,3,128,320)
    for _ in range(2):
        c.train_mode(m);assert m.training and not m.bn.training;opt.zero_grad(set_to_none=True)
        for i,s in enumerate([' 11.00 DH ','Ab']):loss,_=c.ctc_sum(m,x[i:i+1],[s],2);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
    assert c.tree(dict(m.named_buffers()))==before and all(int(v['step'])==2 for v in opt.state.values())
    cfg={'height':128,'width':320,'device':'cpu','precision':'fp32','pretrained_dir':str(tmp_path),'recipe':c.RECIPE}
    path=tmp_path/'model.pt';c.save_checkpoint(path,m,opt,cfg,{'steps':2});fresh=model();o=c.optimizer(fresh);c.restore_checkpoint(path,fresh,o);assert c.tree(fresh.state_dict())==c.tree(m.state_dict())
    from parseq_ocr import ppocr_training as legacy
    # Exercise the real public native/export loader with a small architecture factory.
    monkeypatch.setattr(legacy,'create_model',lambda cfg:Toy());monkeypatch.setattr(legacy,'restore_bn',lambda m:None)
    loaded,_=legacy.load_model(path,'cpu');assert c.tree(loaded.state_dict())==c.tree(m.state_dict())
    assert torch.equal(loaded(x),m.eval()(x))
    # Public export delegates to the exact same load_model used above.
    import inspect
    from parseq_ocr import ppocr_export
    assert 'from .ppocr_training import load_model' in inspect.getsource(ppocr_export.export_onnx)

def test_shared_augmentation_deterministic_and_manifest_order(tmp_path):
    p=tmp_path/'x.png';Image.fromarray(np.arange(60*120*3,dtype=np.uint8).reshape(60,120,3)).save(p);r={'path':str(p)}
    assert torch.equal(c.pixels(r,20271115),c.pixels(r,20271115))
    manifest=tmp_path/'fit.jsonl';manifest.write_text(json.dumps({'image_path':'x.png','text':' A '})+'\n'+json.dumps({'image_path':'x.png','text':'b'})+'\n')
    rows=c.load_rows({'train_manifest':manifest},'train');assert [r['label']for r in rows]==[' A ','b']and rows[0]['path']==str(p)


def test_actual_trainer_two_update_lifecycle_on_fictional_data(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    for split,n in [('train',32),('val',2)]:
        folder=tmp_path/'data'/split;folder.mkdir(parents=True)
        for i in range(n):
            rng=np.random.default_rng(i+(100 if split=='val'else 0));Image.fromarray(rng.integers(0,256,(20,40,3),dtype=np.uint8)).save(folder/f'{i:03}.png');(folder/f'{i:03}.txt').write_text('11.00 DH')
    initial=tmp_path/'initial.pt';torch.save(model().state_dict(),initial)
    cfg={'recipe':c.RECIPE,'height':128,'width':320,'epochs':20,'batch_size':16,'microbatch':1,'initial_codec':'ppocr_ascii96_v1','initial_checkpoint':initial,'initial_checkpoint_sha256':c.sha(initial),'dataset_root':tmp_path/'data','output_dir':tmp_path/'out','pretrained_dir':tmp_path/'unused','precision':'fp32','device':'cpu','normalization':'none'}
    monkeypatch.setattr(c,'load_initial',lambda cfg:model())
    result=c.train(cfg,smoke_steps=2);assert result['pass']and result['steps']==2 and result['exposures']==32 and result['selected_epoch']==1
    cp=torch.load(result['checkpoint'],weights_only=True);assert all(int(v['step'])==2 for v in cp['optimizer']['state'].values())
    history=json.loads((tmp_path/'out/smoke/history.json').read_text());assert history[0]['selection_eligible']is False and history[1]['selection_eligible']is True
