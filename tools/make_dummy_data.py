"""Create small image/TXT workflow fixtures, never overwrite an existing dataset."""
from pathlib import Path
import argparse,json,hashlib,io
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dummy_data');a=p.parse_args();out=a.output.resolve()
 if out.exists():
  manifest=out/'DUMMY_DATA.json'
  if not manifest.exists():raise FileExistsError(f'{out} already exists; refusing to alter it')
  old=json.loads(manifest.read_text())
  for name,digest in old['files'].items():
   if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Dummy fixture changed: {name}')
  print(out);return
 out.mkdir(parents=True);files={}
 for task in ['plates','prices']:
  for split,count in [('train',16),('val',4)]:
   folder=out/task/split;folder.mkdir(parents=True)
   for i in range(count):
    n=i+(100 if split=='val' else 0)
    text=f'AB{n:02d}CD{n+123:03d}' if task=='plates' else f'{n+7}.{(n*13+25)%100:02d}'
    image=Image.new('RGB',(320,96),(245,240,210) if task=='plates' else (250,235,90));draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('DejaVuSans.ttf',34)
    except OSError:font=ImageFont.load_default()
    box=draw.textbbox((0,0),text,font=font);draw.text(((320-box[2])/2,(96-(box[3]-box[1]))/2-box[1]),text,font=font,fill=(10,10,10));image.save(folder/f'{i:03d}.png');(folder/f'{i:03d}.txt').write_text(text+'\n',encoding='utf8')
    for f in [folder/f'{i:03d}.png',folder/f'{i:03d}.txt']:files[str(f.relative_to(out))]=hashlib.sha256(f.read_bytes()).hexdigest()
 (out/'DUMMY_DATA.json').write_text(json.dumps({'purpose':'Synthetic smoke/workflow fixtures only; not OCR benchmark or accuracy evidence','counts_per_task':{'train':16,'val':4},'annotation':'Only same-stem UTF8 expected-text files; no boxes','files':files},indent=2)+'\n');print(out)
if __name__=='__main__':main()
