"""Bundled demo data must remain synthetic, separate, and safe to regenerate."""
from pathlib import Path
import subprocess,sys,json,ast
import pytest
from parseq_ocr.data_io import scan_pairs
from parseq_ocr.data import split_holdout,audit_splits
from parseq_ocr.config import load_config
ROOT=Path(__file__).resolve().parents[1]

def test_dummy_generator_is_idempotent_and_rejects_user_data(tmp_path):
 output=tmp_path/'demo';command=[sys.executable,str(ROOT/'tools/make_dummy_data.py'),'--output',str(output)]
 subprocess.run(command,check=True,capture_output=True);manifest=(output/'DUMMY_DATA.json').read_bytes();subprocess.run(command,check=True,capture_output=True);assert (output/'DUMMY_DATA.json').read_bytes()==manifest
 for task in ['plates','prices']:
  train=scan_pairs(output/task/'train','none');val=scan_pairs(output/task/'val','none');assert (len(train),len(val))==(16,4);assert not audit_splits(train,val)['pixel_duplicates_across_splits'];fit,held=split_holdout(train,.2,15001);assert not {r['label'] for r in fit}&{r['label'] for r in held}
 user=tmp_path/'user';user.mkdir();(user/'keep.txt').write_text('user data');result=subprocess.run(command[:-1]+[str(user)],capture_output=True);assert result.returncode!=0 and (user/'keep.txt').read_text()=='user data'

def test_dummy_presets_and_main_notebooks_are_self_contained():
 for name in ['base','tiny','fast_tiny']:
  c=load_config(ROOT/f'configs/dummy_{name}.yaml');assert c['device']=='cpu' and c['precision']=='fp32';assert c['dataset_root']==ROOT/'dummy_data/plates';assert c['output_dir']==ROOT/f'artifacts/dummy_{name}'
 for number in range(8):
  p=next((ROOT/'notebooks').glob(str(number)+'_*.ipynb'));nb=json.loads(p.read_text())
  for cell in nb['cells']:
   if cell['cell_type']=='code':ast.parse(''.join(cell['source']));assert cell['outputs']==[] and cell['execution_count'] is None
