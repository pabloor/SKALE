import json,urllib.request,urllib.parse,re,sys
S=sys.argv[1]
key=re.compile(r'(?<![A-Za-z#])([A-G])\s?(#|♯|b|♭|-?sharp|-?flat)?\s?[- ]?(minor|major|min|maj)\b(?![a-z])|(?<![A-Za-z])([A-G])(#|b)?m(?![a-z])',re.I)
exclude={l.split('\t')[2] for l in open(S+'/set2/sources.tsv')}
cands={}
terms=['bpm','beat','loop','instrumental','song','track','jam','riff','guitar','piano','synth','bass','drum','pad','ambient','melody','chill','lofi','rock','pop','metal','trap','house','techno','blues','funk','jazz','cinematic','acoustic','demo']
for t in terms:
  for page in (1,2,3):
    q=f'mediatype:audio AND title:({t}) AND (title:minor OR title:major OR title:min OR title:maj) AND (licenseurl:*creativecommons*)'
    url='https://archive.org/advancedsearch.php?'+urllib.parse.urlencode({'q':q,'fl[]':['identifier','title','licenseurl','creator'],'rows':100,'page':page,'output':'json'},doseq=True)
    try: docs=json.load(urllib.request.urlopen(url,timeout=40))['response']['docs']
    except Exception as e: break
    if not docs: break
    for d in docs:
        ti=d.get('title','')
        if isinstance(ti,list): ti=ti[0]
        m=key.search(ti)
        if not m or d['identifier'] in exclude: continue
        lic=d.get('licenseurl','')
        if isinstance(lic,list): lic=lic[0]
        cands[d['identifier']]=(ti,m.group(0),lic.split('.org/')[-1],d.get('creator'))
print(len(cands))
json.dump(cands,open(S+'/cands6.json','w'))
import collections
for i,(k,v) in enumerate(list(cands.items())[:25]): print(k,'|',v[0][:60],'|',v[1],'|',v[2])
