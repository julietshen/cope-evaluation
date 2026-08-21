import json, glob, re, csv, os, urllib.request, urllib.parse, time
sig=re.compile(r"(t\.me/|whats ?app|telegram|airdrop|gift ?card|crypto|bitcoin|\$\s?\d{2,}|giveaway|claim|free \$|bit\.ly|forex|signals|invest|guaranteed|winner|prize|deposit)",re.I)
cands=[];seen=set()
for f in glob.glob("*.jsonl"):
    for line in open(f):
        try: r=json.loads(line)
        except: continue
        if r.get("has_images") and sig.search(r.get("text","") or ""):
            if r["uri"] in seen: continue
            seen.add(r["uri"]); cands.append(r)
    if len(cands)>=220: break
print("candidates:",len(cands),flush=True)

def getposts(uris):
    q="&".join("uris="+urllib.parse.quote(u,safe="") for u in uris)
    url="https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?"+q
    req=urllib.request.Request(url,headers={"User-Agent":"vibecheck-eval"})
    return json.load(urllib.request.urlopen(req,timeout=25))

outdir="/Users/julietshen/ROOST/vibecheck/eval/scam_eval/scam_images"
os.makedirs(outdir,exist_ok=True)
manifest=[]; idx=0
for i in range(0,len(cands),25):
    batch=cands[i:i+25]
    try: data=getposts([r["uri"] for r in batch])
    except Exception as e: print("batch err",e); continue
    posts={p["uri"]:p for p in data.get("posts",[])}
    for r in batch:
        p=posts.get(r["uri"])
        if not p: continue
        imgs=(p.get("embed") or {}).get("images") or []
        if not imgs: continue
        u=imgs[0].get("fullsize")
        if not u: continue
        idx+=1; fid=f"scamimg{idx:03d}"; path=f"{outdir}/{fid}.jpg"
        try:
            req=urllib.request.Request(u,headers={"User-Agent":"vibecheck-eval"})
            b=urllib.request.urlopen(req,timeout=25).read()
            if len(b)<1500: continue
            open(path,"wb").write(b)
        except Exception: continue
        manifest.append({"id":fid,"content":path,"ground_truth":"","post_text":r.get("text","")[:300],"alt":(imgs[0].get("alt") or "")[:120],"img_url":u})
    print(f"downloaded {len(manifest)}",flush=True)
    time.sleep(0.5)
    if len(manifest)>=90: break
with open("/Users/julietshen/ROOST/vibecheck/eval/scam_eval/image_set_for_labeling.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["id","content","ground_truth","post_text","alt","img_url"]); w.writeheader(); w.writerows(manifest)
print("DONE images:",len(manifest),flush=True)
