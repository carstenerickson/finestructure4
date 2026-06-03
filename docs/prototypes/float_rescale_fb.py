# Float-Alphamat-via-linear-rescale feasibility prototype.
# Companion to docs/float-alphamat-linear-rescale.md (§4).
# Faithful NumPy forward-backward incl. the ChromoPainter chunk-count / expected-chunk-length
# formulas, run in 4 precision variants + an offset-stress sweep. Shows: rescale-float32 is
# exact through the chunk formulas at any cumulative-normalizer magnitude; the naive double->float
# Alphamat drop-in degrades with that magnitude. No deps beyond numpy. Run: python3 float_rescale_fb.py
import numpy as np
rng = np.random.default_rng(7)
N, T = 600, 80000
mu = 1e-3
# donor haplotypes, recipient as a long-chunk mosaic (realistic)
H = rng.integers(0,2,size=(T,N)).astype(np.float64)
r = np.empty(T); d=int(rng.integers(N))
for t in range(T):
    if rng.random()<1e-3: d=int(rng.integers(N))
    r[t]=H[t,d] if rng.random()>mu else 1-H[t,d]
e = np.where(H==r[:,None], 1-mu, mu)                 # emission [t][i]
copy_prob = rng.dirichlet(np.ones(N)*0.5)            # donor prior
c = rng.uniform(2e-6, 4e-5, T)                       # per-locus jump prob (tiny -> large log-offset)
pos = np.cumsum(rng.uniform(20,200,T)); lam = rng.uniform(1e-9,5e-9,T); delta=1.0

# ---- forward (log-space double, exact code form) -> Alphamat + per-locus Alphasum ----
Alphamat = np.empty((T,N))
Alphamat[0] = np.log(copy_prob*e[0])
Asum = np.log(np.sum(np.exp(Alphamat[0])*c[0]))
Asv = np.empty(T); Asv[0]=Asum
for t in range(1,T):
    large=-Asum
    prev=np.exp(Alphamat[t-1]+large)
    Anew=e[t]*copy_prob + e[t]*(1-c[t-1])*prev
    Alphamat[t]=np.log(Anew)-large
    Snew=np.sum(Anew*(c[t] if t<T-1 else 1.0))
    Asum=np.log(Snew)-large; Asv[t]=Asum
Asum_final=Asum
print(f"loci={T} donors={N};  |Alphasum| reaches {abs(Asum_final):,.0f}")

# normalized forward ahat (what rescale stores) + reconstruct Alphamat from float32 ahat
ahat = np.exp(Alphamat - Asv[:,None])                # = exp(Alphamat - logScale), O(1e5)-ish, RELATIVE-precision friendly
def alphamat_from_ahat(ahat_store):
    return np.log(ahat_store.astype(np.float64)) + Asv[:,None]

# ---- backward + EXACT chunk formulas (Beta stays double in all variants) ----
def backward(A):
    Bprev=np.zeros(N)
    Bsum=np.log(np.sum(c[T-2]*copy_prob*e[T-1]*np.exp(Bprev)))
    ccc=np.zeros(N); ecl=np.zeros(N); cpnS=np.zeros(N)
    for L in range(T-2,-1,-1):
        large=-Bsum; Op=e[L]; OpP=e[L+1]
        Bcur=np.log(np.exp(Bsum+large)+(1-c[L])*OpP*np.exp(Bprev+large))-large
        e_lp1_bp=np.exp(A[L+1]+Bprev-Asum_final)
        e_l_bp  =np.exp(A[L]+Bprev-Asum_final)
        e_l_bc  =np.exp(A[L]+Bcur-Asum_final)
        if L==0: cpnS=np.exp(A[0]+Bcur-Asum_final)
        cc=e_lp1_bp - e_l_bp*OpP*(1-c[L])
        ccc+=cc
        f=1-c[L]+c[L]*copy_prob
        t_ii=e_l_bp*OpP*f
        t_toi=e_lp1_bp - e_l_bp*OpP*f
        t_fri=e_l_bc - e_l_bp*OpP*f
        if lam[L]>=0:
            ecl+=100*(pos[L+1]-pos[L])*delta*lam[L]*(1.0*t_ii + 0.5*(t_toi+t_fri))
        Bprev=Bcur
        if L>0:
            Bsum=np.log(np.sum(c[L-1]*copy_prob*Op*np.exp(Bcur+large)))-large
    ccc+=cpnS
    return ccc, ecl

ref_cc, ref_cl = backward(Alphamat)
def cmp(name, A):
    cc,cl=backward(A); top=np.argsort(ref_cl)[-40:]
    md_cl=np.max(np.abs(cl[top]-ref_cl[top])/ref_cl[top])*100
    md_cc=np.max(np.abs(cc[top]-ref_cc[top])/np.abs(ref_cc[top]))*100
    print(f"  {name:34s} chunkLEN max%Δ={md_cl:9.4f}%   chunkCNT max%Δ={md_cc:9.4f}%")
print("  per-donor (top-40 by chunk length) vs double/log-space reference:")
cmp("double log-space (ref)", Alphamat)
cmp("rescale ahat, DOUBLE", alphamat_from_ahat(ahat.astype(np.float64)))
cmp("rescale ahat, FLOAT32", alphamat_from_ahat(ahat.astype(np.float32)))
cmp("drop-in Alphamat FLOAT32", Alphamat.astype(np.float32).astype(np.float64))

# ---- controlled offset stress: add constant K to Alphamat & Alphasum (chunk results invariant
#      in exact arithmetic; only float STORAGE precision is stressed -> simulates real |Sigma|) ----
def backward_K(A, asum_final):
    global Asum_final
    save=Asum_final; Asum_final=asum_final
    out=backward(A); Asum_final=save; return out
print("\n  offset stress (add K to the cumulative normalizer; ref invariant):")
print(f"  {'K':>9} {'drop-in f32 chunkLEN%Δ':>26} {'rescale f32 chunkLEN%Δ':>26}")
for K in [0,1e5,1e6,2e6]:
    Ad=(Alphamat+K).astype(np.float32).astype(np.float64)
    Ar=np.log(ahat.astype(np.float32).astype(np.float64))+Asv[:,None]+K
    top=np.argsort(ref_cl)[-40:]
    _,cld=backward_K(Ad,Asum_final+K); _,clr=backward_K(Ar,Asum_final+K)
    ed=np.max(np.abs(cld[top]-ref_cl[top])/ref_cl[top])*100
    er=np.max(np.abs(clr[top]-ref_cl[top])/ref_cl[top])*100
    print(f"  {K:>9.0f} {ed:>24.4f}% {er:>24.6f}%")
