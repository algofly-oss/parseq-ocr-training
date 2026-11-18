import json
from pathlib import Path
from PIL import Image
import pytest
from parseq_ocr import ppocr_precursor as p
from parseq_ocr import ppocr_training as t


def rows(tmp_path):
    out=[]
    for i in range(2):
        path=tmp_path/f'{i}.png';Image.new('RGB',(16,16),(i*100,20,30)).save(path)
        out.append(dict(path=str(path),image=path.name,label=' Ab '))
    return out


def test_selection_only_never_refits(tmp_path,monkeypatch):
    fit,dev=rows(tmp_path)
    calls=[]
    def fake(cfg,a,b,epochs,folder):
        calls.append((a,b));folder.mkdir(parents=True);(folder/'model.pt').write_bytes(b'model');return 1
    monkeypatch.setattr(t,'_fit',fake)
    cfg=dict(output_dir=tmp_path/'out',epochs=30,batch_size=16,refit=False)
    checkpoint=t.train(cfg,[fit],[dev])
    assert checkpoint==tmp_path/'out/selection/model.pt'
    assert calls==[([fit],[dev])]
    assert not (tmp_path/'out/full_fit').exists()


def test_handoff_binds_labels_pixels_order_and_initializer(tmp_path):
    rr=rows(tmp_path);cp=tmp_path/'model.pt'
    import torch
    policy=dict(recipe='selection_only_precursor_v1',refit=False)
    torch.save(dict(full_fit=False,recover_bn=True,epoch=1,config=policy),cp)
    value=dict(schema=1,refit=False,source_sha256=p.source_binding(),checkpoint_sha256=p.sha(cp),fit_membership_sha256=p.membership(rr[:1]),dev_membership_sha256=p.membership(rr[1:]))
    value['training_config']=policy;value['binding_sha256']=p.receipt_digest(value)
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(value))
    cfg=dict(precursor_receipt=receipt,initial_checkpoint=cp,initial_checkpoint_sha256=p.sha(cp))
    p.validate_handoff(cfg,rr[:1],rr[1:])
    with pytest.raises(ValueError,match='changed'):p.validate_handoff(cfg,[{**rr[0],'label':'Ab'}],rr[1:])
    Image.new('RGB',(16,16),'red').save(rr[0]['path'])
    with pytest.raises(ValueError,match='changed'):p.validate_handoff(cfg,rr[:1],rr[1:])
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):p.validate_handoff(cfg,rr[:1],rr[1:])


def test_legacy_default_still_refits(tmp_path,monkeypatch):
    fit,dev=rows(tmp_path);calls=[]
    def fake(cfg,a,b,epochs,folder):
        calls.append((a,b));return 1
    monkeypatch.setattr(t,'_fit',fake)
    path=t.train(dict(output_dir=tmp_path/'out',epochs=1,batch_size=2),[fit],[dev])
    assert path==tmp_path/'out/full_fit/model.pt'
    assert calls==[([fit],[dev]),([fit,dev],[])]


@pytest.mark.parametrize('override',[{'height':64},{'normalization':'upper'},{'epochs':1},{'device':'cpu','precision':'bf16'}])
def test_precursor_rejects_incompatible_recipe_before_model(tmp_path,override):
    cfg=dict(height=128,width=320,normalization='none',device='cpu',precision='fp32',output_dir=tmp_path/'out',**{})
    cfg.update(override)
    with pytest.raises(ValueError):p.train(cfg)


def test_wrapper_receipt_and_corrected_handoff(tmp_path,monkeypatch):
    import torch
    from parseq_ocr import ppocr_models as m
    rr=rows(tmp_path);manifests=[]
    for i,row in enumerate(rr):
        path=tmp_path/f'{i}.jsonl';path.write_text(json.dumps(dict(image_path=row['path'],text=row['label']))+'\n');manifests.append(path)
    def prepare(cache):
        cache.mkdir()
        for name in ['model.onnx','trainable.onnx']:(cache/name).write_bytes(name.encode())
    monkeypatch.setattr(m,'prepare_pretrained',prepare);monkeypatch.setattr(m,'FILES',{})
    import sys,types
    def tiny(name,folder):
        folder.mkdir();path=folder/'tiny.pt';path.write_bytes(b'fixture');return path
    monkeypatch.setitem(sys.modules,'parseq_ocr.models',types.SimpleNamespace(download_pretrained=tiny))
    def fit(cfg,fit,dev):
        assert cfg['refit'] is False and cfg['epochs']==30 and cfg['batch_size']==16
        assert fit[0]['path']==rr[0]['path'] and dev[0]['path']==rr[1]['path']
        dest=Path(cfg['output_dir'])/'selection/model.pt';dest.parent.mkdir(parents=True)
        torch.save(dict(full_fit=False,recover_bn=True,epoch=1,config={k:str(v)if isinstance(v,Path)else v for k,v in cfg.items()}),dest)
        return dest
    monkeypatch.setattr(t,'train',fit)
    cfg=dict(height=128,width=320,normalization='none',device='cpu',precision='fp32',output_dir=tmp_path/'out',pretrained_dir=tmp_path/'cache',train_manifest=manifests[0],validation_manifest=manifests[1])
    result=p.train(cfg)
    p.validate_handoff({**result,'initial_checkpoint':result['checkpoint'],'initial_checkpoint_sha256':result['checkpoint_sha256']},rr[:1],rr[1:])
    assert not (tmp_path/'out/full_fit').exists()
    with pytest.raises(ValueError,match='Fresh'):p.train(cfg)
