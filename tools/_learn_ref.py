import numpy as np,json,sys,collections
from scipy.optimize import minimize
sys.path.insert(0,'/tmp/claude-0/-home-user-Medidores/6d6df73b-c5ba-5cd2-8c04-4d1b46822468/scratchpad')
from exp import D,TM,Tm,pear,rot,label,report,baseline
MAJ={0,2,3}; MIN={1,4}
def chordshare(r):
    mj=np.zeros(12); mn=np.zeros(12); ot=np.zeros(12); tot=sum(c[2] for c in r['chords']) or 1
    for root,typ,sec in r['chords']:
        (mj if typ in MAJ else mn if typ in MIN else ot)[root]+=sec/tot
    return mj,mn,ot
def feats(r,use):
    c=np.array(r['chroma'])*12; b=np.array(r['bass'])*12; e=np.array(r['ending'])*12
    mj,mn,ot=chordshare(r)
    F=np.zeros((24,0)); cols=[]
    out=[]
    for t in range(12):
        for m,prof in ((0,TM),(1,Tm)):
            v=[]
            if 'temp' in use: v.append([pear(c,np.roll(prof,t))])
            if 'chroma' in use: v.append(rot(c,t))
            if 'bass' in use: v.append(rot(b,t))
            if 'end' in use: v.append(rot(e,t))
            if 'chord' in use: v+= [rot(mj,t)*3,rot(mn,t)*3,rot(ot,t)*3]
            out.append(np.concatenate(v))
    return np.array(out)               # (24, d)
def fit(X,y,w,lam):
    n,K,d=X.shape
    def f(theta):
        W=theta.reshape(2,d)                          # un vector por modo
        S=np.einsum('nkd,kd->nk',X,W[np.arange(24)%2])
        S=S-S.max(1,keepdims=True); P=np.exp(S); P/=P.sum(1,keepdims=True)
        nll=-(w*np.log(P[np.arange(n),y]+1e-12)).sum()+lam*(theta**2).sum()
        G=P.copy(); G[np.arange(n),y]-=1; G*=w[:,None]            # (n,K)
        gW=np.zeros((2,d))
        for m in (0,1):
            idx=np.arange(m,24,2); gW[m]=np.einsum('nk,nkd->d',G[:,idx],X[:,idx])
        return nll,(gW.ravel()+2*lam*theta)
    th=minimize(f,np.zeros(2*d),jac=True,method='L-BFGS-B',options={'maxiter':300}).x
    return th.reshape(2,d)
def predict(X,W): return np.argmax(np.einsum('nkd,kd->nk',X,W[np.arange(24)%2]),1)
def logo(use,lam,verbose=True,name=None):
    X=np.array([feats(r,use) for r in D]); y=np.array([label(r) for r in D]); grp=np.array([r['group'] for r in D])
    size=collections.Counter(grp); w=np.array([1.0/size[g] for g in grp]); w*=len(D)/w.sum()/len(size)
    pred=np.zeros(len(D),int)
    for g in set(grp):
        tr=grp!=g; W=fit(X[tr],y[tr],w[tr],lam); pred[~tr]=predict(X[~tr],W)
    report(name or f"LOGO {'+'.join(use)} lam={lam}",pred); return pred
if __name__=='__main__':
    report('base actual',[int(np.argmax(baseline(r))) for r in D])
    for use in (['temp','end','bass'],['chroma','end','bass'],['chroma','end','bass','chord'],['temp','chroma','end','bass','chord']):
        for lam in (0.01,0.1,1):
            logo(use,lam)
