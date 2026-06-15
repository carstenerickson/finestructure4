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
        for(int i=0;i<K;i++) E[(size_t)l*K+i] = emis(r, donors[(size_t)i*N+l]);
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

/* ---------------- grouping (sort donors by (pop, block-substring)) ----------- */
static int gsb, geb;
static int cmp_donor(const void *pa, const void *pb){
    int ia=*(const int*)pa, ib=*(const int*)pb;
    if(pop_vec[ia]!=pop_vec[ib]) return pop_vec[ia]-pop_vec[ib];
    const uint8_t *ra=donors+(size_t)ia*N, *rb=donors+(size_t)ib*N;
    for(int l=gsb;l<geb;l++) if(ra[l]!=rb[l]) return (int)ra[l]-(int)rb[l];
    return 0;
}

/* ---------------- FOLDED engine (port of fold_ONU) -------------------------- */
typedef struct { int s,e,U; } Block;
typedef struct {           /* panel-fixed, target-independent: build ONCE */
    int B, nb, Umax;
    Block *blk;            /* [nb] */
    int *gidB;             /* [nb*K] gid per (block,donor) */
    int *sizeg,*repg,*popg;/* [nb*Umax] */
} Groups;

static Groups build_groups(int B){
    Groups G; G.B=B;
    int maxnb=N/2+2; G.blk=malloc(sizeof(Block)*maxnb); int nb=0;
    for(int s=0;s<N;s+=B){ int e=s+B; if(e>N)e=N; G.blk[nb].s=s; G.blk[nb].e=e; nb++; }
    if(nb>=2 && (G.blk[nb-1].e-G.blk[nb-1].s)<2){ G.blk[nb-2].e=G.blk[nb-1].e; nb--; }
    G.nb=nb;
    G.gidB=malloc(sizeof(int)*(size_t)nb*K);
    int *idx=malloc(sizeof(int)*K); int Umax=0;
    for(int b=0;b<nb;b++){
        gsb=G.blk[b].s; geb=G.blk[b].e;
        for(int i=0;i<K;i++) idx[i]=i;
        qsort(idx,K,sizeof(int),cmp_donor);
        int U=0; int *gb=G.gidB+(size_t)b*K;
        for(int t=0;t<K;t++){ if(t==0||cmp_donor(&idx[t],&idx[t-1])!=0) U++; gb[idx[t]]=U-1; }
        G.blk[b].U=U; if(U>Umax)Umax=U;
    }
    free(idx); G.Umax=Umax;
    G.sizeg=malloc(sizeof(int)*(size_t)nb*Umax); G.repg=malloc(sizeof(int)*(size_t)nb*Umax);
    G.popg=malloc(sizeof(int)*(size_t)nb*Umax);
    for(int b=0;b<nb;b++){ int U=G.blk[b].U; int *gb=G.gidB+(size_t)b*K;
        int *sz=G.sizeg+(size_t)b*Umax,*rp=G.repg+(size_t)b*Umax,*pg=G.popg+(size_t)b*Umax;
        for(int g=0;g<U;g++){sz[g]=0;rp[g]=-1;}
        for(int i=0;i<K;i++){int g=gb[i];sz[g]++; if(rp[g]<0){rp[g]=i;pg[g]=pop_vec[i];}}
    }
    return G;
}
static void free_groups(Groups *G){ free(G->blk);free(G->gidB);free(G->sizeg);free(G->repg);free(G->popg); }

/* lazy emission: the fold needs e only at group reps (O(N*U)) + per-donor at the
   O(K)-per-block boundaries - NOT the full O(N*K) matrix the dense engine fills. */
static double fold_cc(int rr, double *ccpop /*[npop], zeroed by caller*/, Groups *Gr){
    const uint8_t *rh = recip + (size_t)rr*N;
    #define EM(L,II) emis(rh[(L)], donors[(size_t)(II)*N+(L)])
    int nb=Gr->nb, Umax=Gr->Umax; Block *blk=Gr->blk; int *gidB=Gr->gidB;
    int *sizeg=Gr->sizeg, *repg=Gr->repg, *popg=Gr->popg;
    /* malloc not calloc: only [0,U_block) of each locus row is written+read, so
       zeroing all N*Umax pages (which grows with block size) is pure waste. */
    double *GF=malloc((size_t)N*Umax*sizeof(double)), *PF=malloc((size_t)N*Umax*sizeof(double));
    double *GB=malloc((size_t)N*Umax*sizeof(double)), *PB=malloc((size_t)N*Umax*sizeof(double));
    double *AENTRY=malloc(sizeof(double)*(size_t)nb*K), *WB=malloc(sizeof(double)*(size_t)nb*K);
    double *As=malloc(sizeof(double)*N), *Bs=malloc(sizeof(double)*N);
    double *G=malloc(sizeof(double)*Umax), *P=malloc(sizeof(double)*Umax), *Sx=malloc(sizeof(double)*Umax);

    /* FORWARD: store GF/PF, materialize AENTRY (block entry = a_{sblk-1}) */
    double *aprev=calloc(K,sizeof(double));
    for(int b=0;b<nb;b++){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K;
        int *sz=sizeg+(size_t)b*Umax, *rp=repg+(size_t)b*Umax;
        double *ent=AENTRY+(size_t)b*K;
        for(int i=0;i<K;i++) ent[i]= (b==0)?0.0:aprev[i];
        for(int g=0;g<U;g++) Sx[g]=0.0;
        if(b>0) for(int i=0;i<K;i++) Sx[gb[i]] += aprev[i];
        for(int l=sb;l<eb;l++){
            double rlm1 = (l>=2)? exp(As[l-2]-As[l-1]) : ((l==1)?exp(-As[0]):0.0);
            for(int g=0;g<U;g++){
                double eg=EM(l, rp[g]);
                if(l==sb && b==0){ G[g]=copyprob*eg; P[g]=0.0; }
                else if(l==sb){ G[g]=eg*copyprob; P[g]=eg*(1-T[l-1])*rlm1; }
                else { double fac=eg*(1-T[l-1])*rlm1; G[g]=eg*copyprob+fac*G[g]; P[g]=fac*P[g]; }
                GF[(size_t)l*Umax+g]=G[g]; PF[(size_t)l*Umax+g]=P[g];
            }
            double sumA=0.0; for(int g=0;g<U;g++) sumA += sz[g]*G[g] + P[g]*Sx[g];
            double s = (l<N-1)? sumA*T[l] : sumA;
            As[l] = (l>=1?As[l-1]:0.0) + log(s);
        }
        /* materialize a_{eb-1} -> aprev for next block */
        for(int i=0;i<K;i++){ int g=gb[i]; aprev[i]= GF[(size_t)(eb-1)*Umax+g] + PF[(size_t)(eb-1)*Umax+g]*ent[i]; }
    }

    /* BACKWARD: store GB/PB, materialize WB (= E[eb]*c_{eb}) per block */
    double *cexit=malloc(sizeof(double)*K); for(int i=0;i<K;i++) cexit[i]=1.0;
    { double sb0=0.0; for(int i=0;i<K;i++) sb0+=T[N-2]*copyprob*EM(N-1, i); Bs[N-1]=log(sb0); }
    for(int b=nb-1;b>=0;b--){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K;
        int *sz=sizeg+(size_t)b*Umax, *rp=repg+(size_t)b*Umax;
        int top = (eb-1 < N-2)? eb-1 : N-2;
        double *w=WB+(size_t)b*K;
        for(int i=0;i<K;i++) w[i]= EM(top+1, i)*cexit[i];
        for(int g=0;g<U;g++) Sx[g]=0.0;
        for(int i=0;i<K;i++) Sx[gb[i]] += w[i];
        for(int l=top;l>=sb;l--){
            double rb = (l+2<=N-1)? exp(Bs[l+2]-Bs[l+1]) : exp(-Bs[N-1]);
            for(int g=0;g<U;g++){
                if(l==top){ G[g]=1.0; P[g]=(1-T[l])*rb; }
                else { double fac=(1-T[l])*EM(l+1, rp[g])*rb; G[g]=1.0+fac*G[g]; P[g]=fac*P[g]; }
                GB[(size_t)l*Umax+g]=G[g]; PB[(size_t)l*Umax+g]=P[g];
            }
            if(l>0){ double sB=0.0; for(int g=0;g<U;g++){ double egl=EM(l, rp[g]); sB += egl*(sz[g]*G[g]+P[g]*Sx[g]); }
                Bs[l]=Bs[l+1]+log(T[l-1]*copyprob*sB); }
        }
        /* materialize c_{sb} -> cexit for next (leftward) block */
        for(int i=0;i<K;i++){ int g=gb[i]; cexit[i]= GB[(size_t)sb*Umax+g] + PB[(size_t)sb*Umax+g]*w[i]; }
    }

    /* CHUNKCOUNT: O(N*U) interior bilinear + O(K) boundary + start */
    double Asf=As[N-1];
    for(int p=0;p<npop;p++) ccpop[p]=0.0;
    for(int b=0;b<nb;b++){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K;
        int *sz=sizeg+(size_t)b*Umax, *rp=repg+(size_t)b*Umax, *pg=popg+(size_t)b*Umax;
        double *ent=AENTRY+(size_t)b*K, *w=WB+(size_t)b*K;
        double *Saent=calloc(U,sizeof(double)), *Sw=calloc(U,sizeof(double)), *Saw=calloc(U,sizeof(double));
        for(int i=0;i<K;i++){ int g=gb[i]; Saent[g]+=ent[i]; Sw[g]+=w[i]; Saw[g]+=ent[i]*w[i]; }
        for(int l=sb;l<eb;l++){
            if(l==N-1) continue;
            double BsR=(l+2<=N-1)?Bs[l+2]:0.0, Asm1=(l>=1)?As[l-1]:0.0;
            double KF1=exp(As[l]+BsR-Asf), KF=exp(Asm1+BsR-Asf), om=(1-T[l]);
            if(l < eb-1){ /* interior, O(U) */
                for(int g=0;g<U;g++){
                    double eg=EM(l+1, rp[g]);
                    double GFl=GF[(size_t)l*Umax+g], PFl=PF[(size_t)l*Umax+g];
                    double GFl1=GF[(size_t)(l+1)*Umax+g], PFl1=PF[(size_t)(l+1)*Umax+g];
                    double XG=GFl1*KF1 - GFl*KF*eg*om, XP=PFl1*KF1 - PFl*KF*eg*om;
                    double GBl1, PBl1;
                    if(l+1==N-1){ GBl1=1.0; PBl1=0.0; }
                    else { GBl1=GB[(size_t)(l+1)*Umax+g]; PBl1=PB[(size_t)(l+1)*Umax+g]; }
                    double contrib = GBl1*XG*sz[g] + GBl1*XP*Saent[g] + PBl1*XG*Sw[g] + PBl1*XP*Saw[g];
                    ccpop[pg[g]] += contrib;
                }
            } else { /* block-boundary locus, O(K) per-donor */
                if(b+1>=nb) continue; /* last block: eb-1==N-1 skipped above */
                int *gb2=gidB+(size_t)(b+1)*K; double *ent2=AENTRY+(size_t)(b+1)*K, *w2=WB+(size_t)(b+1)*K;
                for(int i=0;i<K;i++){
                    int g=gb[i], g2=gb2[i];
                    double a_l = GF[(size_t)l*Umax+g] + PF[(size_t)l*Umax+g]*ent[i];
                    double a_lp1 = GF[(size_t)(l+1)*Umax+g2] + PF[(size_t)(l+1)*Umax+g2]*ent2[i];
                    double c_lp1 = GB[(size_t)(l+1)*Umax+g2] + PB[(size_t)(l+1)*Umax+g2]*w2[i];
                    ccpop[pop_vec[i]] += a_lp1*c_lp1*KF1 - a_l*c_lp1*KF*EM(l+1, i)*om;
                }
            }
        }
        free(Saent); free(Sw); free(Saw);
    }
    /* start term, O(K) */
    { double k0=exp(Bs[1]-Asf); int *gb0=gidB; double *ent0=AENTRY, *w0=WB;
      for(int i=0;i<K;i++){ int g=gb0[i];
        double a0=GF[g]+PF[g]*ent0[i], c0=GB[g]+PB[g]*w0[i];
        ccpop[pop_vec[i]] += a0*c0*k0; } }

    free(GF);free(PF);free(GB);free(PB);free(AENTRY);free(WB);free(As);free(Bs);
    free(G);free(P);free(Sx);free(aprev);free(cexit);
    return 0;
}

int main(int argc, char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s cdata.bin blocksize\n",argv[0]); return 1; }
    int B = atoi(argv[2]);
    FILE *f=fopen(argv[1],"rb"); if(!f){perror("open");return 1;}
    int hdr[4]; fread(hdr,sizeof(int),4,f); K=hdr[0];N=hdr[1];npop=hdr[2];nrecip=hdr[3];
    double par[3]; fread(par,sizeof(double),3,f); rhobar=par[0];mut=par[1];copyprob=par[2];
    donors=malloc((size_t)K*N); fread(donors,1,(size_t)K*N,f);
    recip=malloc((size_t)nrecip*N); fread(recip,1,(size_t)nrecip*N,f);
    pos=malloc(sizeof(double)*N); fread(pos,sizeof(double),N,f);
    lam=malloc(sizeof(double)*N); fread(lam,sizeof(double),N,f);
    pop_vec=malloc(sizeof(int)*K); fread(pop_vec,sizeof(int),K,f);
    ref_cc=malloc(sizeof(double)*npop); fread(ref_cc,sizeof(double),npop,f);
    fclose(f);
    T=malloc(sizeof(double)*(N-1));
    for(int l=0;l<N-1;l++) T[l]=1.0-exp(-(pos[l+1]-pos[l])*rhobar*lam[l]);

    double *E=malloc(sizeof(double)*(size_t)N*K);
    double *a=malloc(sizeof(double)*(size_t)N*K), *c=malloc(sizeof(double)*(size_t)N*K);
    double *As=malloc(sizeof(double)*N), *Bs=malloc(sizeof(double)*N), *cc=malloc(sizeof(double)*K);
    double dense_pp[16]={0}, fold_pp_tot[16]={0};

    /* DENSE timed (per recipient: E fill + forward-backward-chunkcount) */
    double t0=now_s();
    for(int r=0;r<nrecip;r++){ fill_E(E,r); for(int i=0;i<K;i++)cc[i]=0.0;
        dense_cc(E,cc,a,c,As,Bs); for(int i=0;i<K;i++) dense_pp[pop_vec[i]]+=cc[i]; }
    double td=now_s()-t0;

    /* GROUPING built ONCE (panel-fixed, target-independent - amortizes over recipients) */
    double tg0=now_s(); Groups G=build_groups(B); double tg=now_s()-tg0;
    { long sumU=0, slots=0; for(int b=0;b<G.nb;b++){ sumU+=G.blk[b].U; slots+=(long)(G.blk[b].e-G.blk[b].s)*G.blk[b].U; }
      printf("  grouping: %d blocks, Umean=%.1f (vs K=%d -> fold ratio %.1fx), interior slots N*Umean=%ld vs N*K=%ld\n",
             G.nb, (double)sumU/G.nb, K, (double)K/((double)sumU/G.nb), slots, (long)N*K); }

    /* FOLD timed (per recipient: O(N*U) fold, emissions computed lazily per-group;
       NO O(N*K) emission fill - that's the dense engine's intrinsic cost, not the fold's) */
    double t1=now_s();
    double fold_pp[16];
    for(int r=0;r<nrecip;r++){
        for(int p=0;p<npop;p++) fold_pp[p]=0.0;
        fold_cc(r,fold_pp,&G); for(int p=0;p<npop;p++) fold_pp_tot[p]+=fold_pp[p]; }
    double tf=now_s()-t1;
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
