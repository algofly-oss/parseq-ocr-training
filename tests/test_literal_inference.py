"""Raw output compatibility, including legacy annotation metadata; CPU fakes only."""
from pathlib import Path
import ast,types,math
import numpy as np
import torch
from PIL import Image
from parseq_ocr import ppocr_training
from parseq_ocr.ppocr_models import VOCAB
from parseq_ocr.inference import TorchRecognizer

def test_parseq_legacy_metadata_does_not_change_predictions():
 reader=TorchRecognizer.__new__(TorchRecognizer);reader.config={'normalization':'upper_no_whitespace','device':'cpu','precision':'fp32','image_size':[4,8]}
 class System:
  tokenizer=types.SimpleNamespace(decode=lambda _: ([' aB 10.50 '],None))
  def __call__(self,x):return torch.zeros(1,1,2)
 reader.system=System()
 assert reader.infer([Image.new('RGB',(8,4))])==[' aB 10.50 ']

def test_pp_torch_infer_and_evaluate_preserve_edge_spaces(monkeypatch):
 ids=torch.tensor([[VOCAB.index(' '),VOCAB.index('A'),VOCAB.index(' ')]])
 class Model(torch.nn.Module):
  def forward(self,x):return torch.nn.functional.one_hot(ids,len(VOCAB)).float()
 reader=ppocr_training.TorchRecognizer.__new__(ppocr_training.TorchRecognizer);reader.model=Model();reader.config={'device':'cpu','height':128,'precision':'fp32'}
 monkeypatch.setattr(ppocr_training,'preprocess',lambda *a:np.zeros((3,2,2),np.float32))
 assert reader.infer([object()])==[' A ']
 monkeypatch.setattr(ppocr_training,'loader',lambda *a:[(torch.zeros(1,3,2,2),['A'],['x'])])
 assert ppocr_training.evaluate_model(Model(),[{'path':'x','label':'A'}],reader.config)['correct']==0

def test_actual_pp_trt_method_preserves_spaces_without_gpu():
 # Execute the actual method AST with fake driver calls; no GPU/library constructor.
 p=Path(__file__).resolve().parents[1]/'parseq_ocr/ppocr_runtime.py';tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PPTRTManaged');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='infer');fn.decorator_list=[]
 rt=types.SimpleNamespace(cudaGraphLaunch=lambda *a:None,cudaStream_t=lambda x:x)
 import cv2
 env={'np':np,'Image':Image,'math':math,'cv2':cv2,'rt':rt,'checked':lambda x:None};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),env)
 fake=types.SimpleNamespace(max_batch=1,height=2,width=2,meta={'resize':'stretch','vocabulary':VOCAB},h_input=np.zeros((1,3,2,2),np.float32),h_output=np.array([[VOCAB.index(' '),VOCAB.index('A'),VOCAB.index(' ')]]),cuda_context=types.SimpleNamespace(push=lambda:None,pop=lambda:None),graphs={1:(None,None,0)},stream=types.SimpleNamespace(handle=0,synchronize=lambda:None))
 assert env['infer'](fake,[np.zeros((2,2,3),np.uint8)])==[' A ']
