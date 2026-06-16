/* cpfold.c - standalone C port of the validated ChromoPainter block-fold engine.
 *
 * Implements BOTH the dense O(N*K) forward-backward-chunkcount and the folded
 * O(N*U) engine (a direct port of cpfold/engine_fold.py:fold_ONU), times both,
 * and checks fold==dense (a PORT-DEFECT check - exactness is already proven in
 * Python) and dense==python-reference. The open question this answers is the one
 * numpy could not: does the fold win in WALL-CLOCK in real C?
 *
 * Input: the flat binary written by prep_cdata.py.
 * Build: cc -O3 -o cpfold cpfold.c -lm
 * Run:   ./cpfold win/cdata2000.bin <blocksize>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define SMALL_NUM 1e-20

static int K, N, npop, nrecip;
static double rhobar, mut, copyprob;
static uint8_t *donors;     /* [K*N] */
static uint8_t *recip;      /* [nrecip*N] */
static double *pos, *lam;   /* [N] */
static int *pop_vec;        /* [K] */
static double *ref_cc;      /* [npop] */
static double *T;           /* [N-1] */

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* emission e(donor i, recipient allele r) */
static inline double emis(int r, int d){
    if(r==9) return 1.0;
    if(r==8) return (r==d)?(1-SMALL_NUM):SMALL_NUM;
    return (r==d)?(1-mut):mut;
}

/* fill E[N*K] for recipient hap rr */
static void fill_E(double *E, int rr){
    const uint8_t *rh = recip + (size_t)rr*N;
    for(int l=0;l<N;l++){
        int r = rh[l];
        for(int i=0;i<K;i++) E[(size_t)l*K+i] = emis(r, donors[(size_t)l*K+i]);
    }
}

/* ---------------- DENSE engine (port of dense_forward/backward/chunkcount) ---- */
static void dense_cc(const double *E, double *cc /*[K], zeroed by caller*/,
                     double *a, double *c, double *As, double *Bs){
    /* forward */
    double s=0.0;
    for(int i=0;i<K;i++){ a[i]=copyprob*E[i]; s+=a[i]*T[0]; }
    As[0]=log(s);
    for(int l=1;l<N;l++){
        double rlm1 = (l>=2)? exp(As[l-2]-As[l-1]) : exp(-As[0]);
        double sm=0.0; const double *El=E+(size_t)l*K; double *al=a+(size_t)l*K, *alm=a+(size_t)(l-1)*K;
        double sp=(1-T[l-1])*rlm1;
        for(int i=0;i<K;i++){ al[i]=El[i]*copyprob + El[i]*sp*alm[i]; sm += al[i]*((l<N-1)?T[l]:1.0); }
        As[l]=As[l-1]+log(sm);
    }
    /* backward */
    double bi=0.0; { const double *EN=E+(size_t)(N-1)*K; double *cN=c+(size_t)(N-1)*K;
        for(int i=0;i<K;i++){ cN[i]=1.0; bi += T[N-2]*copyprob*EN[i]; } }
    Bs[N-1]=log(bi);
    for(int l=N-2;l>=0;l--){
        double rb = (l+2<=N-1)? exp(Bs[l+2]-Bs[l+1]) : exp(-Bs[N-1]);
        const double *Ep=E+(size_t)(l+1)*K; double *cl=c+(size_t)l*K, *clp=c+(size_t)(l+1)*K;
        double f=(1-T[l])*rb;
        for(int i=0;i<K;i++) cl[i]=1.0 + f*Ep[i]*clp[i];
        if(l>0){ double sb=0.0; const double *El=E+(size_t)l*K;
            for(int i=0;i<K;i++) sb += T[l-1]*copyprob*El[i]*cl[i];
            Bs[l]=Bs[l+1]+log(sb); }
    }
    /* chunkcount */
    double Asf=As[N-1];
    for(int l=0;l<N-1;l++){
        double BsR=(l+2<=N-1)?Bs[l+2]:0.0, Asm1=(l>=1)?As[l-1]:0.0;
        double KF1=exp(As[l]+BsR-Asf), KF=exp(Asm1+BsR-Asf), om=(1-T[l]);
        const double *alp=a+(size_t)(l+1)*K, *al=a+(size_t)l*K, *clp=c+(size_t)(l+1)*K, *Ep=E+(size_t)(l+1)*K;
        for(int i=0;i<K;i++) cc[i] += alp[i]*clp[i]*KF1 - al[i]*clp[i]*KF*Ep[i]*om;
    }
    { double Asf2=Asf, k0=exp(Bs[1]-Asf2); const double *a0=a, *c0=c;
        for(int i=0;i<K;i++) cc[i] += a0[i]*c0[i]*k0; }
}

/* ---------------- grouping by (pop, block-substring): O(N*K) hash ------------ */

/* ---------------- FOLDED engine (port of fold_ONU) -------------------------- */
typedef struct { int s,e,U; } Block;
typedef struct {           /* panel-fixed, target-independent: build ONCE */
    int B, nb, Umax;
    Block *blk;            /* [nb] */
    int *gidB;             /* [nb*K] gid per (block,donor) */
    int *sizeg,*repg,*popg;/* [nb*Umax] */
    /* L2 boundary join: panel-fixed (g_b,g_{b+1}) contingency per boundary b */
    int *cellB;           /* [nb*K] cell id of donor i at boundary b (block b->b+1) */
    int *celloff;         /* [nb+1] prefix sum of cell counts per boundary */
    int *cellGb,*cellG2;  /* [total cells] the (g_b,g_{b+1}) pair of each cell */
    int maxcell;          /* max cells over boundaries (sizes the Mc/Mce scratch) */
} Groups;

/* Build the panel-fixed boundary contingency: for each boundary b (block b -> b+1)
   enumerate the occupied (g_b, g_{b+1}) cells and tag every donor with its cell id.
   The L2 boundary chunkcount then folds over these cells instead of per-donor. */
static void build_contingency(Groups *G){
    int nb=G->nb, Umax=G->Umax; int *gidB=G->gidB;
    G->celloff=malloc(sizeof(int)*(size_t)(nb+1));
    G->cellB=malloc(sizeof(int)*(size_t)nb*K);
    int *cmap=malloc(sizeof(int)*(size_t)Umax*Umax);            /* dense (g_b,g2)->cellid, gen-stamped */
    int *cstamp=calloc((size_t)Umax*Umax,sizeof(int)); int gen=0;
    int capcell=nb*8+16, total=0, maxcell=0;
    int *cGb=malloc(sizeof(int)*capcell), *cG2=malloc(sizeof(int)*capcell);
    G->celloff[0]=0;
    for(int b=0;b<nb-1;b++){
        gen++; int nc=0; int *gA=gidB+(size_t)b*K, *gB=gidB+(size_t)(b+1)*K; int *cb=G->cellB+(size_t)b*K;
        for(int i=0;i<K;i++){ int key=gA[i]*Umax+gB[i];
            if(cstamp[key]!=gen){ cstamp[key]=gen; cmap[key]=nc;
                if(total+nc>=capcell){ capcell*=2; cGb=realloc(cGb,sizeof(int)*capcell); cG2=realloc(cG2,sizeof(int)*capcell); }
                cGb[total+nc]=gA[i]; cG2[total+nc]=gB[i]; nc++; }
            cb[i]=cmap[key]; }
        total+=nc; G->celloff[b+1]=total; if(nc>maxcell)maxcell=nc;
    }
    G->celloff[nb]=total;
    G->cellGb=cGb; G->cellG2=cG2; G->maxcell=maxcell<1?1:maxcell;
    free(cmap); free(cstamp);
}

static Groups build_groups(int B){
    Groups G; G.B=B;
    int maxnb=N/2+2; G.blk=malloc(sizeof(Block)*maxnb); int nb=0;
    for(int s=0;s<N;s+=B){ int e=s+B; if(e>N)e=N; G.blk[nb].s=s; G.blk[nb].e=e; nb++; }
    if(nb>=2 && (G.blk[nb-1].e-G.blk[nb-1].s)<2){ G.blk[nb-2].e=G.blk[nb-1].e; nb--; }
    G.nb=nb;
    G.gidB=malloc(sizeof(int)*(size_t)nb*K);
    /* O(N*K) grouping by hash on packed (alleles 2bit + pop) keys - no sort, no
       O(B) comparator, no log K. Open-addressing table reused per block via a gen
       stamp. Key = ceil(2B/64) allele words (SUBSTRING-ONLY: pop is split at the
       chunkcount scatter, so the forward/backward state folds over fewer groups). */
    int W = (2*B + 63)/64; if(W<1) W=1;
    int cap=1; while(cap < 4*K) cap<<=1;
    uint64_t *htkey=malloc((size_t)cap*W*sizeof(uint64_t));
    int *htgid=malloc(sizeof(int)*cap), *htstamp=calloc(cap,sizeof(int));
    /* first pass: gid assignment + Umax + per-block U */
    int *Ublk=malloc(sizeof(int)*nb), Umax=0, *repTmp=malloc(sizeof(int)*K), gen=0;
    uint64_t *key=malloc((size_t)W*sizeof(uint64_t));   /* W=ceil(2B/64) words; dynamic (B may exceed 512) */
    for(int b=0;b<nb;b++){
        int sb=G.blk[b].s, eb=G.blk[b].e; int *gb=G.gidB+(size_t)b*K; gen++;
        int U=0;
        for(int i=0;i<K;i++){
            for(int w=0;w<W;w++) key[w]=0;
            for(int l=sb;l<eb;l++){ int bp=2*(l-sb); int al=donors[(size_t)l*K+i]; int code=al<2?al:(al==8?2:3);
                key[bp>>6]|=(uint64_t)code<<(bp&63); }  /* 4-way code: 0,1,8,9 -> 0,1,2,3 (keep missing alleles DISTINCT) */
            uint64_t h=1469598103934665603ULL; for(int w=0;w<W;w++) h=(h^key[w])*1099511628211ULL;
            int slot=h&(cap-1);
            for(;;){
                if(htstamp[slot]!=gen){ htstamp[slot]=gen; memcpy(htkey+(size_t)slot*W,key,W*sizeof(uint64_t));
                    htgid[slot]=U; repTmp[U]=i; gb[i]=U; U++; break; }
                if(memcmp(htkey+(size_t)slot*W,key,W*sizeof(uint64_t))==0){ gb[i]=htgid[slot]; break; }
                slot=(slot+1)&(cap-1);
            }
        }
        Ublk[b]=U; G.blk[b].U=U; if(U>Umax)Umax=U;
    }
    G.Umax=Umax;
    G.sizeg=malloc(sizeof(int)*(size_t)nb*Umax); G.repg=malloc(sizeof(int)*(size_t)nb*Umax);
    G.popg=malloc(sizeof(int)*(size_t)nb*Umax);
    for(int b=0;b<nb;b++){ int U=G.blk[b].U; int *gb=G.gidB+(size_t)b*K;
        int *sz=G.sizeg+(size_t)b*Umax,*rp=G.repg+(size_t)b*Umax,*pg=G.popg+(size_t)b*Umax;
        for(int g=0;g<U;g++){sz[g]=0;rp[g]=-1;}
        for(int i=0;i<K;i++){int g=gb[i];sz[g]++; if(rp[g]<0){rp[g]=i;pg[g]=pop_vec[i];}}
    }
    free(htkey);free(htgid);free(htstamp);free(Ublk);free(repTmp);free(key);
    build_contingency(&G);
    return G;
}
static void free_groups(Groups *G){ free(G->blk);free(G->gidB);free(G->sizeg);free(G->repg);free(G->popg);
    free(G->cellB);free(G->celloff);free(G->cellGb);free(G->cellG2); }

/* ADAPTIVE block boundaries (PBWT-style): grow each block by incrementally
   splitting the substring grouping column-by-column; cut when U reaches Ustar.
   Blocks are long where local diversity is low (-> fewer boundaries, less boundary
   tax) and short where it is high. The incremental split IS the grouping, so this
   produces blocks + gid in one O(N*K) pass. Variable block lengths; the fold
   engine already reads blk[].s/e so it needs no change. */
static Groups build_groups_adaptive(int Ustar){
    Groups G; G.B=Ustar; G.nb=0;
    int cap_blk=64; G.blk=malloc(sizeof(Block)*cap_blk); int *gidB=NULL;
    int *gid=malloc(sizeof(int)*K), *ng=malloc(sizeof(int)*K);
    int mapsz=(Ustar+8)*4; if(mapsz<256) mapsz=256;  /* U can transiently reach ~16 (forced no-cut at b=a+1) -> ek up to 63; floor 256 for margin */
    int *mp=malloc(sizeof(int)*mapsz), *mstamp=calloc(mapsz,sizeof(int)); int gen=0, Umax=0;
    int a=0; for(int i=0;i<K;i++) gid[i]=0; int U=1; int b=0;
    while(b<N){
        gen++; int newU=0; const uint8_t *col=donors+(size_t)b*K;
        for(int i=0;i<K;i++){ int al=col[i], code=al<2?al:(al==8?2:3); int ek=gid[i]*4+code;
            if(mstamp[ek]!=gen){ mstamp[ek]=gen; mp[ek]=newU++; } ng[i]=mp[ek]; }
        if(newU>Ustar && (b-a)>=2 && b<N-1){           /* cut: block [a,b) keeps grouping `gid`; reprocess b */
            if(G.nb>=cap_blk){ cap_blk*=2; G.blk=realloc(G.blk,sizeof(Block)*cap_blk); }
            G.blk[G.nb].s=a; G.blk[G.nb].e=b; G.blk[G.nb].U=U;
            gidB=realloc(gidB,(size_t)(G.nb+1)*K*sizeof(int)); memcpy(gidB+(size_t)G.nb*K,gid,K*sizeof(int));
            if(U>Umax)Umax=U; G.nb++;
            a=b; for(int i=0;i<K;i++) gid[i]=0; U=1;     /* new block; do NOT advance b */
        } else { int *t=gid;gid=ng;ng=t; U=newU; b++; } /* accept column b */
    }
    if(G.nb>=cap_blk){ cap_blk*=2; G.blk=realloc(G.blk,sizeof(Block)*cap_blk); }
    G.blk[G.nb].s=a; G.blk[G.nb].e=N; G.blk[G.nb].U=U;
    gidB=realloc(gidB,(size_t)(G.nb+1)*K*sizeof(int)); memcpy(gidB+(size_t)G.nb*K,gid,K*sizeof(int));
    if(U>Umax)Umax=U; G.nb++;
    free(gid);free(ng);free(mp);free(mstamp);
    G.gidB=gidB; G.Umax=Umax;
    G.sizeg=malloc(sizeof(int)*(size_t)G.nb*Umax); G.repg=malloc(sizeof(int)*(size_t)G.nb*Umax);
    G.popg=malloc(sizeof(int)*(size_t)G.nb*Umax);   /* unused under substring-only, kept for free_groups */
    for(int bb=0;bb<G.nb;bb++){ int Ub=G.blk[bb].U; int *gb=gidB+(size_t)bb*K;
        int *sz=G.sizeg+(size_t)bb*Umax,*rp=G.repg+(size_t)bb*Umax;
        for(int g=0;g<Ub;g++){sz[g]=0;rp[g]=-1;}
        for(int i=0;i<K;i++){int g=gb[i];sz[g]++; if(rp[g]<0)rp[g]=i;}
    }
    build_contingency(&G);
    return G;
}

/* lazy emission: the fold needs e only at group reps (O(N*U)) + per-donor at the
   O(K)-per-block boundaries - NOT the full O(N*K) matrix the dense engine fills. */
static double fold_cc(int rr, double *ccpop /*[npop], zeroed by caller*/, Groups *Gr){
    const uint8_t *rh = recip + (size_t)rr*N;
    #define EM(L,II) emis(rh[(L)], donors[(size_t)(L)*K+(II)])
    int nb=Gr->nb, Umax=Gr->Umax; Block *blk=Gr->blk; int *gidB=Gr->gidB;
    int *sizeg=Gr->sizeg, *repg=Gr->repg, *popg=Gr->popg;
    /* TARGET 2: tight per-block layout (offset = sum len*U; no Umax waste). Only
       GF/PF + the per-group emission table Eg are stored; GB/PB are transient. */
    size_t *off=malloc(sizeof(size_t)*nb), tot=0;
    for(int b=0;b<nb;b++){ off[b]=tot; tot += (size_t)(blk[b].e-blk[b].s)*blk[b].U; }
    double *GF=malloc(tot*sizeof(double)), *PF=malloc(tot*sizeof(double)), *Eg=malloc(tot*sizeof(double));
    double *AENTRY=malloc(sizeof(double)*(size_t)nb*K), *WB=malloc(sizeof(double)*(size_t)nb*K);
    double *CEXIT=malloc(sizeof(double)*(size_t)nb*K);  /* materialized c at each block left edge */
    double *As=malloc(sizeof(double)*N), *Bs=malloc(sizeof(double)*N);
    double *SentS=malloc((size_t)nb*Umax*sizeof(double)), *SwS=malloc((size_t)nb*Umax*sizeof(double));
    double *G=malloc(Umax*sizeof(double)), *P=malloc(Umax*sizeof(double));
    double *GBp=malloc(Umax*sizeof(double)), *PBp=malloc(Umax*sizeof(double));
    double *GBc=malloc(Umax*sizeof(double)), *PBc=malloc(Umax*sizeof(double));
    /* per-(substring-group, pop) moments for the chunkcount scatter [Umax*npop] */
    double *szP=malloc((size_t)Umax*npop*sizeof(double)), *SaP=malloc((size_t)Umax*npop*sizeof(double));
    double *SwP=malloc((size_t)Umax*npop*sizeof(double)), *SawP=malloc((size_t)Umax*npop*sizeof(double));
    /* L2: per-cell moments for the boundary join-fold [maxcell*npop] */
    double *Mc=malloc((size_t)Gr->maxcell*npop*sizeof(double)), *Mce=malloc((size_t)Gr->maxcell*npop*sizeof(double));

    /* TARGET 3: precompute per-group emissions ONCE (the only gather); fwd/bwd/chunk read contiguous. */
    for(int b=0;b<nb;b++){ int sb=blk[b].s,eb=blk[b].e,U=blk[b].U; int *rp=repg+(size_t)b*Umax;
        double *base=Eg+off[b];
        for(int l=sb;l<eb;l++){ double *row=base+(size_t)(l-sb)*U; for(int g=0;g<U;g++) row[g]=EM(l, rp[g]); } }

    /* FORWARD: GF/PF (tight) + As + Sentry; materialize AENTRY (=a_{sblk-1}) */
    double *aprev=calloc(K,sizeof(double));
    for(int b=0;b<nb;b++){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K; int *sz=sizeg+(size_t)b*Umax;
        double *ent=AENTRY+(size_t)b*K, *Sx=SentS+(size_t)b*Umax;
        double *GFb=GF+off[b], *PFb=PF+off[b], *Egb=Eg+off[b];
        for(int g=0;g<U;g++) Sx[g]=0.0;
        for(int i=0;i<K;i++){ double a=(b==0)?0.0:aprev[i]; ent[i]=a; if(b>0) Sx[gb[i]]+=a; }
        for(int l=sb;l<eb;l++){
            double rlm1=(l>=2)?exp(As[l-2]-As[l-1]):((l==1)?exp(-As[0]):0.0);
            double *GFr=GFb+(size_t)(l-sb)*U, *PFr=PFb+(size_t)(l-sb)*U, *Er=Egb+(size_t)(l-sb)*U;
            double sumA=0.0;
            for(int g=0;g<U;g++){
                double eg=Er[g], gv,pv;
                if(l==sb && b==0){ gv=copyprob*eg; pv=0.0; }
                else if(l==sb){ gv=eg*copyprob; pv=eg*(1-T[l-1])*rlm1; }
                else { double fac=eg*(1-T[l-1])*rlm1; gv=eg*copyprob+fac*G[g]; pv=fac*P[g]; }
                G[g]=gv; P[g]=pv; GFr[g]=gv; PFr[g]=pv; sumA += sz[g]*gv + pv*Sx[g];
            }
            double s=(l<N-1)?sumA*T[l]:sumA; As[l]=(l>=1?As[l-1]:0.0)+log(s);
        }
        double *GFl=GFb+(size_t)(eb-1-sb)*U, *PFl=PFb+(size_t)(eb-1-sb)*U;
        for(int i=0;i<K;i++){ int g=gb[i]; aprev[i]=GFl[g]+PFl[g]*ent[i]; }
    }

    /* BACKWARD + CHUNKCOUNT FUSED (TARGET 1): GB/PB transient (cur/prev swap); the chunkcount
       increment for locus l is accumulated inside the backward sweep; boundary uses CEXIT[b+1]. */
    double Asf=As[N-1];
    for(int p=0;p<npop;p++) ccpop[p]=0.0;
    double *cexit=malloc(sizeof(double)*K); for(int i=0;i<K;i++) cexit[i]=1.0;
    { double sb0=0.0; for(int i=0;i<K;i++) sb0+=T[N-2]*copyprob*EM(N-1,i); Bs[N-1]=log(sb0); }
    for(int b=nb-1;b>=0;b--){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K;
        int *sz=sizeg+(size_t)b*Umax;
        int top=(eb-1<N-2)?eb-1:N-2;
        double *w=WB+(size_t)b*K, *Sw=SwS+(size_t)b*Umax, *ent=AENTRY+(size_t)b*K;
        double *GFb=GF+off[b], *PFb=PF+off[b], *Egb=Eg+off[b];
        for(int g=0;g<U;g++) Sw[g]=0.0;                       /* substring total (for Bs) */
        for(int gp=0;gp<U*npop;gp++){ szP[gp]=0.0; SaP[gp]=0.0; SwP[gp]=0.0; SawP[gp]=0.0; }
        for(int i=0;i<K;i++){ double wi=EM(top+1,i)*cexit[i]; w[i]=wi; int g=gb[i], p=pop_vec[i]; double ei=ent[i];
            Sw[g]+=wi; int idx=g*npop+p; szP[idx]+=1.0; SaP[idx]+=ei; SwP[idx]+=wi; SawP[idx]+=ei*wi; }
        for(int l=top;l>=sb;l--){
            double rb=(l+2<=N-1)?exp(Bs[l+2]-Bs[l+1]):exp(-Bs[N-1]);
            double *Er1=(l+1<eb)?Egb+(size_t)(l+1-sb)*U:0;
            if(l==top){ for(int g=0;g<U;g++){ GBc[g]=1.0; PBc[g]=(1-T[l])*rb; } }
            else { for(int g=0;g<U;g++){ double fac=(1-T[l])*Er1[g]*rb; GBc[g]=1.0+fac*GBp[g]; PBc[g]=fac*PBp[g]; } }
            if(l>0){ double *Er=Egb+(size_t)(l-sb)*U; double sB=0.0;
                for(int g=0;g<U;g++) sB+=Er[g]*(sz[g]*GBc[g]+PBc[g]*Sw[g]); Bs[l]=Bs[l+1]+log(T[l-1]*copyprob*sB); }
            double BsR=(l+2<=N-1)?Bs[l+2]:0.0, Asm1=(l>=1)?As[l-1]:0.0;
            double KF1=exp(As[l]+BsR-Asf), KF=exp(Asm1+BsR-Asf), om=(1-T[l]);
            double *GFr=GFb+(size_t)(l-sb)*U, *PFr=PFb+(size_t)(l-sb)*U;
            if(l < eb-1){   /* interior: GB[l+1]=GBp (prev), GF/PF stored, moments */
                double *GFr1=GFb+(size_t)(l+1-sb)*U, *PFr1=PFb+(size_t)(l+1-sb)*U;
                for(int g=0;g<U;g++){
                    double eg=Er1[g];
                    double XG=GFr1[g]*KF1 - GFr[g]*KF*eg*om, XP=PFr1[g]*KF1 - PFr[g]*KF*eg*om;
                    double GBl1,PBl1; if(l+1==N-1){GBl1=1.0;PBl1=0.0;} else {GBl1=GBp[g];PBl1=PBp[g];}
                    /* per-group coeffs computed ONCE (U_sub of them); scatter to present pops */
                    double cA=GBl1*XG, cB=GBl1*XP, cC=PBl1*XG, cD=PBl1*XP; int base=g*npop;
                    for(int p=0;p<npop;p++){ double s_=szP[base+p]; if(s_==0.0) continue;
                        ccpop[p] += cA*s_ + cB*SaP[base+p] + cC*SwP[base+p] + cD*SawP[base+p]; }
                }
            } else if(b+1<nb){   /* L2 boundary JOIN-FOLD over (g_b,g_{b+1},pop) cells.
                   Identity: ent2(i)=AENTRY[b+1](i)=a_l(i), so a_lp1=GF2[g2]+PF2[g2]*a_l with
                   a_l=GFr[g]+PFr[g]*ent. incr(i)=cx2*(cf0 + cf1*ent) folds bilinearly:
                   one O(K) moment scatter (Mc=sum cx2, Mce=sum cx2*ent) + O(cells*npop) combine. */
                int co=Gr->celloff[b], nc=Gr->celloff[b+1]-co; int *cidB=Gr->cellB+(size_t)b*K;
                double *cx2=CEXIT+(size_t)(b+1)*K;
                double *GF2=GF+off[b+1], *PF2=PF+off[b+1], *Eg2=Eg+off[b+1];  /* block b+1 row 0 */
                for(int x=0;x<nc*npop;x++){ Mc[x]=0.0; Mce[x]=0.0; }
                for(int i=0;i<K;i++){ int c=cidB[i], p=pop_vec[i]; double cv=cx2[i];
                    Mc[c*npop+p]+=cv; Mce[c*npop+p]+=cv*ent[i]; }
                int *cGb=Gr->cellGb+co, *cG2=Gr->cellG2+co;
                for(int c=0;c<nc;c++){ int g=cGb[c], g2=cG2[c];
                    double C0=KF1*GF2[g2], C1=KF1*PF2[g2]-KF*om*Eg2[g2];
                    double cf0=C0+C1*GFr[g], cf1=C1*PFr[g]; int base=c*npop;
                    for(int p=0;p<npop;p++) ccpop[p]+= cf0*Mc[base+p]+cf1*Mce[base+p]; }
            }
            { double *t; t=GBp;GBp=GBc;GBc=t; t=PBp;PBp=PBc;PBc=t; }   /* GBp now = GB[l] */
        }
        double *cxb=CEXIT+(size_t)b*K;
        for(int i=0;i<K;i++){ int g=gb[i]; double cv=GBp[g]+PBp[g]*w[i]; cexit[i]=cv; cxb[i]=cv; }
    }
    /* start term: a_0=GF[0], c_0=CEXIT[0] */
    { double k0=exp(Bs[1]-Asf); int *gb0=gidB; double *ent0=AENTRY, *cx0=CEXIT; double *GF0=GF+off[0], *PF0=PF+off[0];
      for(int i=0;i<K;i++){ int g=gb0[i]; double a0=GF0[g]+PF0[g]*ent0[i]; ccpop[pop_vec[i]] += a0*cx0[i]*k0; } }

    free(off);free(GF);free(PF);free(Eg);free(AENTRY);free(WB);free(CEXIT);free(As);free(Bs);
    free(SentS);free(SwS);free(G);free(P);free(GBp);free(PBp);free(GBc);free(PBc);
    free(szP);free(SaP);free(SwP);free(SawP);free(aprev);free(cexit);free(Mc);free(Mce);
    return 0;
}

int main(int argc, char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s cdata.bin blocksize|Ustar [reps] [ad]\n",argv[0]); return 1; }
    int B = atoi(argv[2]);
    int adaptive = (argc>4 && strcmp(argv[4],"ad")==0); /* arg5=="ad" -> PBWT adaptive blocks, B used as Ustar */
    FILE *f=fopen(argv[1],"rb"); if(!f){perror("open");return 1;}
    int hdr[4]; fread(hdr,sizeof(int),4,f); K=hdr[0];N=hdr[1];npop=hdr[2];nrecip=hdr[3];
    double par[3]; fread(par,sizeof(double),3,f); rhobar=par[0];mut=par[1];copyprob=par[2];
    { uint8_t *drm=malloc((size_t)K*N); fread(drm,1,(size_t)K*N,f);
      donors=malloc((size_t)N*K); /* transpose to locus-major [N][K] */
      for(int i=0;i<K;i++) for(int l=0;l<N;l++) donors[(size_t)l*K+i]=drm[(size_t)i*N+l];
      free(drm); }
    recip=malloc((size_t)nrecip*N); fread(recip,1,(size_t)nrecip*N,f);
    pos=malloc(sizeof(double)*N); fread(pos,sizeof(double),N,f);
    lam=malloc(sizeof(double)*N); fread(lam,sizeof(double),N,f);
    pop_vec=malloc(sizeof(int)*K); fread(pop_vec,sizeof(int),K,f);
    ref_cc=malloc(sizeof(double)*npop); fread(ref_cc,sizeof(double),npop,f);
    fclose(f);
    if(npop>16){ fprintf(stderr,"npop=%d exceeds the 16-slot per-pop arrays\n",npop); return 1; }
    if(N<2){ fprintf(stderr,"need N>=2 SNPs\n"); return 1; }
    if(!adaptive && B<2){ fprintf(stderr,"fixed blocksize must be >=2 (got %d); use adaptive mode for small Ustar\n",B); return 1; }
    T=malloc(sizeof(double)*(N-1));
    for(int l=0;l<N-1;l++) T[l]=1.0-exp(-(pos[l+1]-pos[l])*rhobar*lam[l]);

    double *E=malloc(sizeof(double)*(size_t)N*K);
    double *a=malloc(sizeof(double)*(size_t)N*K), *c=malloc(sizeof(double)*(size_t)N*K);
    double *As=malloc(sizeof(double)*N), *Bs=malloc(sizeof(double)*N), *cc=malloc(sizeof(double)*K);
    double dense_pp[16]={0}, fold_pp_tot[16]={0};
    int reps = (argc>3)? atoi(argv[3]) : 7;       /* repeat timing, take MIN (denoise) */

    Groups G = adaptive ? build_groups_adaptive(B) : build_groups(B);  /* (timed below; here for the correctness pass) */
    { long sumU=0, slots=0; int Ufold=0; for(int b=0;b<G.nb;b++){ sumU+=G.blk[b].U; slots+=(long)(G.blk[b].e-G.blk[b].s)*G.blk[b].U; if(G.blk[b].U>Ufold)Ufold=G.blk[b].U; }
      printf("  grouping[%s]: %d blocks, Umean=%.1f Umax=%d (vs K=%d -> fold ratio %.1fx), mean block len=%.1f, interior slots N*Umean=%ld vs N*K=%ld\n",
             adaptive?"adaptive":"fixed", G.nb, (double)sumU/G.nb, Ufold, K, (double)K/((double)sumU/G.nb), (double)N/G.nb, slots, (long)N*K); }

    /* correctness pass (once) */
    for(int r=0;r<nrecip;r++){ fill_E(E,r); for(int i=0;i<K;i++)cc[i]=0.0;
        dense_cc(E,cc,a,c,As,Bs); for(int i=0;i<K;i++) dense_pp[pop_vec[i]]+=cc[i]; }
    { double fp[16]; for(int r=0;r<nrecip;r++){ for(int p=0;p<npop;p++)fp[p]=0.0;
        fold_cc(r,fp,&G); for(int p=0;p<npop;p++) fold_pp_tot[p]+=fp[p]; } }

    /* timing: min over reps (dense incl E fill; fold lazy emissions; grouping once) */
    double td=1e30, tf=1e30, tg=1e30; double fp[16];
    for(int it=0; it<reps; it++){
        double t0=now_s();
        for(int r=0;r<nrecip;r++){ fill_E(E,r); dense_cc(E,cc,a,c,As,Bs); }
        double d=now_s()-t0; if(d<td)td=d;
        double tg0=now_s(); Groups Gt = adaptive ? build_groups_adaptive(B) : build_groups(B); double g=now_s()-tg0; if(g<tg)tg=g; free_groups(&Gt);
        double t1=now_s();
        for(int r=0;r<nrecip;r++){ for(int p=0;p<npop;p++)fp[p]=0.0; fold_cc(r,fp,&G); }
        double ff=now_s()-t1; if(ff<tf)tf=ff;
    }
    free_groups(&G);

    printf("cpfold  K=%d N=%d npop=%d nrecip=%d  block=%d\n",K,N,npop,nrecip,B);
    double mdr=0, mfd=0;
    for(int p=0;p<npop;p++){ double e1=fabs(dense_pp[p]-ref_cc[p])/(fabs(ref_cc[p])+1e-300);
        double e2=fabs(fold_pp_tot[p]-dense_pp[p])/(fabs(dense_pp[p])+1e-300);
        if(e1>mdr)mdr=e1; if(e2>mfd)mfd=e2; }
    printf("  dense per-pop:"); for(int p=0;p<npop;p++)printf(" %.6f",dense_pp[p]); printf("\n");
    printf("  fold  per-pop:"); for(int p=0;p<npop;p++)printf(" %.6f",fold_pp_tot[p]); printf("\n");
    printf("  dense-vs-pythonref max rel err = %.3e\n",mdr);
    printf("  fold-vs-dense (C)  max rel err = %.3e  (port-defect check; exactness proven in py)\n",mfd);
    printf("  WALL-CLOCK (%d recipients): dense=%.2f ms  fold=%.2f ms (+ grouping %.2f ms built once)\n",
           nrecip, td*1e3, tf*1e3, tg*1e3);
    printf("    per-recipient (grouping amortized):  %.2fx   |   single-target (grouping incl): %.2fx\n",
           td/tf, td/(tf+tg));
    return 0;
}
