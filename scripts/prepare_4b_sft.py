"""Prepare reviewed JSONL conversations for assistant-only supervised fine-tuning."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def conversations(path):
    seen=set()
    with path.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            try: item=json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"Invalid JSONL line {line_no}") from e
            msgs=item.get("messages") if isinstance(item,dict) else None
            if not isinstance(msgs,list) or not msgs: continue
            clean=[]
            for m in msgs:
                if not isinstance(m,dict) or m.get("role") not in {"system","user","assistant"} or not isinstance(m.get("content"),str) or not m["content"].strip():
                    clean=[]; break
                clean.append({"role":m["role"],"content":m["content"].strip()})
            if not clean or not any(m["role"]=="assistant" for m in clean): continue
            raw=json.dumps(clean,ensure_ascii=False,sort_keys=True)
            key=hashlib.sha256(raw.encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            yield key,clean

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,default=Path("data/4b/tokenizer.json"))
    p.add_argument("--output-dir",type=Path,default=Path("data/4b-sft"))
    p.add_argument("--block-size",type=int,default=2048)
    p.add_argument("--validation-percent",type=int,default=5)
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    if not 1<=a.validation_percent<=30: p.error("validation-percent must be 1..30")
    tok=Tokenizer.from_file(str(a.tokenizer))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    names=["train-inputs.bin","train-labels.bin","validation-inputs.bin","validation-labels.bin","manifest.json"]
    existing=[a.output_dir/n for n in names if (a.output_dir/n).exists()]
    if existing and not a.overwrite: raise FileExistsError("Output exists; pass --overwrite")
    for f in existing:f.unlink()
    files={k:a.output_dir/v for k,v in {"train_inputs":"train-inputs.bin","train_labels":"train-labels.bin","validation_inputs":"validation-inputs.bin","validation_labels":"validation-labels.bin"}.items()}
    counts={k:0 for k in files}; nconv=0
    with files["train_inputs"].open("wb") as ti, files["train_labels"].open("wb") as tl, files["validation_inputs"].open("wb") as vi, files["validation_labels"].open("wb") as vl:
        handles={"train_inputs":ti,"train_labels":tl,"validation_inputs":vi,"validation_labels":vl}
        for digest,msgs in conversations(a.input):
            nconv+=1
            split="validation" if int(digest[:8],16)%100<a.validation_percent else "train"
            ids=[]; mask=[]
            for m in msgs:
                role=m["role"]
                part=tok.encode(f"<|{role}|>\n").ids+tok.encode(m["content"]).ids+tok.encode("\n").ids
                ids.extend(part); mask.extend([role=="assistant"]*len(part))
            end=tok.encode("<|endofturn|>\n").ids
            ids.extend(end); mask.extend([True]*len(end))
            # Each sample predicts the next token; only assistant target tokens count toward loss.
            for start in range(0,len(ids)-1,a.block_size):
                seq=ids[start:start+a.block_size+1]; msk=mask[start:start+a.block_size+1]
                if len(seq)<2 or not any(msk[1:]): continue
                x=np.zeros(a.block_size+1,dtype=np.uint16); y=np.full(a.block_size+1,-100,dtype=np.int32)
                x[:len(seq)]=seq; y[:len(seq)]=np.where(msk,np.asarray(seq,dtype=np.int32),-100)
                handles[split+"_inputs"].write(x.tobytes()); handles[split+"_labels"].write(y.tobytes())
                counts[split+"_inputs"]+=1; counts[split+"_labels"]+=1
            if nconv%10000==0: print(f"Processed {nconv:,} unique conversations",flush=True)
    if not counts["train_inputs"] or not counts["validation_inputs"]:
        raise RuntimeError("Need non-empty train and validation shards; provide more reviewed conversations.")
    manifest={"format":"neuronforge-sft-v1","block_size":a.block_size,"unique_conversations":nconv,
      "train_examples":counts["train_inputs"],"validation_examples":counts["validation_inputs"],
      "assistant_only_loss":True,"deduplication":"exact normalized conversation SHA-256",
      "files":{k:v.name for k,v in files.items()}}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=="__main__": main()
