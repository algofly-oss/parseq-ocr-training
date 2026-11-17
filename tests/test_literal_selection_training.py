"""Exercise actual PARSeq selector/resume with a tiny CPU fake, no weights."""
import types
import pytest
import torch
from parseq_ocr import training
from parseq_ocr.metrics import score_pairs, EXACT_POLICY
class Fake(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=torch.nn.Linear(1,1);self.tokenizer=types.SimpleNamespace(_stoi={'A':0,'B':1})
 def training_step(self,batch,index):return self.model.weight.square().mean()
def setup(monkeypatch,scores):
 monkeypatch.setattr(training,'create_system',lambda _:Fake())
 monkeypatch.setattr(training,'make_loader',lambda *args,**kwargs:[(torch.ones(1,1),['A'],['a.png'])])
 it=iter(scores)
 def evaluate(*args):
  result=score_pairs(['A','B'],['A','B' if next(it)==2 else 'X']);result.update(cer=0,numeric_correct=999);return result
 monkeypatch.setattr(training,'evaluate_records',evaluate)
 return dict(seed=1,precision='fp32',device='cpu',normalization='none',learning_rate=.001,weight_decay=0,lr_schedule='constant')
def test_actual_fit_exact_ties_and_current_resume(monkeypatch,tmp_path):
 cfg=setup(monkeypatch,[1,1]);summary=training._fit([{'label':'A','image':'a'}],[{'label':'B','image':'b'}],cfg,tmp_path,2,False)
 assert summary['selected_epoch']==1 and summary['selection_policy']==EXACT_POLICY
 setup(monkeypatch,[2]);summary=training._fit([{'label':'A','image':'a'}],[{'label':'B','image':'b'}],cfg,tmp_path,3,True)
 assert summary['selected_epoch']==3

def test_legacy_resume_rejected_before_model_allocation(monkeypatch,tmp_path):
 torch.save({'best':[1,0]},tmp_path/'last.pt');monkeypatch.setattr(training,'create_system',lambda _:pytest.fail('Model must not be allocated'))
 with pytest.raises(ValueError,match='legacy/incompatible'):
  training._fit([],[],{},tmp_path,1,True)
