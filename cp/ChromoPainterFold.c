/* ChromoPainterFold.c - exact block-fold of ChromoPainter's chunk-count.
 *
 * A faithful port of the standalone cpfold engine (cpfold/cpfold.c, proven
 * fold==dense to ~3e-13) grafted onto cp's data structures. Produces the
 * per-population corrected_chunk_count IDENTICALLY to the dense forward-
 * backward-chunkcount, in O(N*Umean) instead of O(N*K).
 *
 * Scope: chunk counts only (the coancestry matrix). Requires the deterministic
 * fixed-parameter mode: uniform copy_prob, uniform (global) mutation rate, no
 * EM (-i 0). Reuses the caller's exact TransProb so the transition is identical.
 *
 * cpfold_perpop() is the entry point called from sampler() under -fold.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "ChromoPainterFold.h"

#define CF_SMALL_NUM 1e-20

/* engine globals (set per call by cpfold_perpop; single-threaded engine) */
static int K, N, npop;
static double copyprob, mut;
static uint8_t *donors;     /* [N*K] locus-major, built from existing_h */
static int *pop_vec;        /* [K] donor population */
static double *T;           /* [N-1] transition (the caller's TransProb) */

static double cf_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* emission e(recipient allele r, donor allele d) - matches cp exactly:
   r==9 -> 1.0 (recipient missing ignored); r==8 -> gap (SMALL_NUM); else (1-mut)/mut. */
static inline double emis(int r, int d){
    if(r==9) return 1.0;
    if(r==8) return (r==d)?(1-CF_SMALL_NUM):CF_SMALL_NUM;
    return (r==d)?(1-mut):mut;
}

typedef struct { int s,e,U; } Block;
typedef struct {
    int B, nb, Umax;
    Block *blk;
    int *gidB;
    int *sizeg,*repg,*popg;
    int *cellB, *celloff, *cellGb, *cellG2, maxcell;
} Groups;

/* panel-fixed (g_b,g_{b+1}) contingency for the L2 boundary join-fold */
static void build_contingency(Groups *G){
    int nb=G->nb, Umax=G->Umax; int *gidB=G->gidB;
    G->celloff=malloc(sizeof(int)*(size_t)(nb+1));
    G->cellB=malloc(sizeof(int)*(size_t)nb*K);
    int *cmap=malloc(sizeof(int)*(size_t)Umax*Umax);
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
static void free_groups(Groups *G){ free(G->blk);free(G->gidB);free(G->sizeg);free(G->repg);free(G->popg);
    free(G->cellB);free(G->celloff);free(G->cellGb);free(G->cellG2); }

/* PBWT adaptive variable-length blocks: grow each block by incrementally
   splitting the substring grouping column-by-column; cut when U reaches Ustar.
   16-way allele code (al&15) keeps all cp allele symbols 0-5,8,9 distinct. */
static Groups build_groups_adaptive(int Ustar){
    Groups G; G.B=Ustar; G.nb=0;
    int cap_blk=64; G.blk=malloc(sizeof(Block)*cap_blk); int *gidB=NULL;
    int *gid=malloc(sizeof(int)*K), *ng=malloc(sizeof(int)*K);
    int mapsz=(Ustar+16)*16; if(mapsz<512) mapsz=512;
    int *mp=malloc(sizeof(int)*mapsz), *mstamp=calloc(mapsz,sizeof(int)); int gen=0, Umax=0;
    int a=0; for(int i=0;i<K;i++) gid[i]=0; int U=1; int b=0;
    while(b<N){
        gen++; int newU=0; const uint8_t *col=donors+(size_t)b*K;
        for(int i=0;i<K;i++){ int code=col[i]&15; int ek=gid[i]*16+code;
            if(mstamp[ek]!=gen){ mstamp[ek]=gen; mp[ek]=newU++; } ng[i]=mp[ek]; }
        if(newU>Ustar && (b-a)>=2 && b<N-1){
            if(G.nb>=cap_blk){ cap_blk*=2; G.blk=realloc(G.blk,sizeof(Block)*cap_blk); }
            G.blk[G.nb].s=a; G.blk[G.nb].e=b; G.blk[G.nb].U=U;
            gidB=realloc(gidB,(size_t)(G.nb+1)*K*sizeof(int)); memcpy(gidB+(size_t)G.nb*K,gid,K*sizeof(int));
            if(U>Umax)Umax=U; G.nb++;
            a=b; for(int i=0;i<K;i++) gid[i]=0; U=1;
        } else { int *t=gid;gid=ng;ng=t; U=newU; b++; }
    }
    if(G.nb>=cap_blk){ cap_blk*=2; G.blk=realloc(G.blk,sizeof(Block)*cap_blk); }
    G.blk[G.nb].s=a; G.blk[G.nb].e=N; G.blk[G.nb].U=U;
    gidB=realloc(gidB,(size_t)(G.nb+1)*K*sizeof(int)); memcpy(gidB+(size_t)G.nb*K,gid,K*sizeof(int));
    if(U>Umax)Umax=U; G.nb++;
    free(gid);free(ng);free(mp);free(mstamp);
    G.gidB=gidB; G.Umax=Umax;
    G.sizeg=malloc(sizeof(int)*(size_t)G.nb*Umax); G.repg=malloc(sizeof(int)*(size_t)G.nb*Umax);
    G.popg=malloc(sizeof(int)*(size_t)G.nb*Umax);
    for(int bb=0;bb<G.nb;bb++){ int Ub=G.blk[bb].U; int *gb=gidB+(size_t)bb*K;
        int *sz=G.sizeg+(size_t)bb*Umax,*rp=G.repg+(size_t)bb*Umax;
        for(int g=0;g<Ub;g++){sz[g]=0;rp[g]=-1;}
        for(int i=0;i<K;i++){int g=gb[i];sz[g]++; if(rp[g]<0)rp[g]=i;}
    }
    build_contingency(&G);
    return G;
}

/* FOLDED forward+backward+chunkcount -> per-pop chunk counts (ccpop[npop]).
   Direct port of cpfold.c fold_cc; rh = the recipient allele row. */
static void fold_cc(const uint8_t *rh, double *ccpop, Groups *Gr){
    #define EM(L,II) emis(rh[(L)], donors[(size_t)(L)*K+(II)])
    int nb=Gr->nb, Umax=Gr->Umax; Block *blk=Gr->blk; int *gidB=Gr->gidB;
    int *sizeg=Gr->sizeg, *repg=Gr->repg;
    size_t *off=malloc(sizeof(size_t)*nb), tot=0;
    for(int b=0;b<nb;b++){ off[b]=tot; tot += (size_t)(blk[b].e-blk[b].s)*blk[b].U; }
    double *GF=malloc(tot*sizeof(double)), *PF=malloc(tot*sizeof(double)), *Eg=malloc(tot*sizeof(double));
    double *AENTRY=malloc(sizeof(double)*(size_t)nb*K), *WB=malloc(sizeof(double)*(size_t)nb*K);
    double *CEXIT=malloc(sizeof(double)*(size_t)nb*K);
    double *As=malloc(sizeof(double)*N), *Bs=malloc(sizeof(double)*N);
    double *SentS=malloc((size_t)nb*Umax*sizeof(double)), *SwS=malloc((size_t)nb*Umax*sizeof(double));
    double *G=malloc(Umax*sizeof(double)), *P=malloc(Umax*sizeof(double));
    double *GBp=malloc(Umax*sizeof(double)), *PBp=malloc(Umax*sizeof(double));
    double *GBc=malloc(Umax*sizeof(double)), *PBc=malloc(Umax*sizeof(double));
    double *szP=malloc((size_t)Umax*npop*sizeof(double)), *SaP=malloc((size_t)Umax*npop*sizeof(double));
    double *SwP=malloc((size_t)Umax*npop*sizeof(double)), *SawP=malloc((size_t)Umax*npop*sizeof(double));
    double *Mc=malloc((size_t)Gr->maxcell*npop*sizeof(double)), *Mce=malloc((size_t)Gr->maxcell*npop*sizeof(double));

    for(int b=0;b<nb;b++){ int sb=blk[b].s,eb=blk[b].e,U=blk[b].U; int *rp=repg+(size_t)b*Umax;
        double *base=Eg+off[b];
        for(int l=sb;l<eb;l++){ double *row=base+(size_t)(l-sb)*U; for(int g=0;g<U;g++) row[g]=EM(l, rp[g]); } }

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
        for(int g=0;g<U;g++) Sw[g]=0.0;
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
            if(l < eb-1){
                double *GFr1=GFb+(size_t)(l+1-sb)*U, *PFr1=PFb+(size_t)(l+1-sb)*U;
                for(int g=0;g<U;g++){
                    double eg=Er1[g];
                    double XG=GFr1[g]*KF1 - GFr[g]*KF*eg*om, XP=PFr1[g]*KF1 - PFr[g]*KF*eg*om;
                    double GBl1,PBl1; if(l+1==N-1){GBl1=1.0;PBl1=0.0;} else {GBl1=GBp[g];PBl1=PBp[g];}
                    double cA=GBl1*XG, cB=GBl1*XP, cC=PBl1*XG, cD=PBl1*XP; int base=g*npop;
                    for(int p=0;p<npop;p++){ double s_=szP[base+p]; if(s_==0.0) continue;
                        ccpop[p] += cA*s_ + cB*SaP[base+p] + cC*SwP[base+p] + cD*SawP[base+p]; }
                }
            } else if(b+1<nb){
                int co=Gr->celloff[b], nc=Gr->celloff[b+1]-co; int *cidB=Gr->cellB+(size_t)b*K;
                double *cx2=CEXIT+(size_t)(b+1)*K;
                double *GF2=GF+off[b+1], *PF2=PF+off[b+1], *Eg2=Eg+off[b+1];
                for(int x=0;x<nc*npop;x++){ Mc[x]=0.0; Mce[x]=0.0; }
                for(int i=0;i<K;i++){ int c=cidB[i], p=pop_vec[i]; double cv=cx2[i];
                    Mc[c*npop+p]+=cv; Mce[c*npop+p]+=cv*ent[i]; }
                int *cGb=Gr->cellGb+co, *cG2=Gr->cellG2+co;
                for(int c=0;c<nc;c++){ int g=cGb[c], g2=cG2[c];
                    double C0=KF1*GF2[g2], C1=KF1*PF2[g2]-KF*om*Eg2[g2];
                    double cf0=C0+C1*GFr[g], cf1=C1*PFr[g]; int base=c*npop;
                    for(int p=0;p<npop;p++) ccpop[p]+= cf0*Mc[base+p]+cf1*Mce[base+p]; }
            }
            { double *t; t=GBp;GBp=GBc;GBc=t; t=PBp;PBp=PBc;PBc=t; }
        }
        double *cxb=CEXIT+(size_t)b*K;
        for(int i=0;i<K;i++){ int g=gb[i]; double cv=GBp[g]+PBp[g]*w[i]; cexit[i]=cv; cxb[i]=cv; }
    }
    { double k0=exp(Bs[1]-Asf); int *gb0=gidB; double *ent0=AENTRY, *cx0=CEXIT; double *GF0=GF+off[0], *PF0=PF+off[0];
      for(int i=0;i<K;i++){ int g=gb0[i]; double a0=GF0[g]+PF0[g]*ent0[i]; ccpop[pop_vec[i]] += a0*cx0[i]*k0; } }

    free(off);free(GF);free(PF);free(Eg);free(AENTRY);free(WB);free(CEXIT);free(As);free(Bs);
    free(SentS);free(SwS);free(G);free(P);free(GBp);free(PBp);free(GBc);free(PBc);
    free(szP);free(SaP);free(SwP);free(SawP);free(aprev);free(cexit);free(Mc);free(Mce);
    #undef EM
}

/* Entry point. Builds locus-major donor buffer from existing_h (int** hap-major),
   the recipient row from newh, sets engine state, builds the (panel-fixed) grouping
   [timed -> *t_build] and runs the fold [timed -> *t_fold]. out_ccpop[ndonorpops]. */
void cpfold_perpop(int *newh, int **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *t_build, double *t_fold){
    K=nhaps; N=nloci; npop=ndonorpops;
    copyprob=copy_prob[0]; mut=MutProb_vec[0];
    pop_vec=pop_vec_in; T=TransProb;
    donors=malloc((size_t)N*K);
    for(int i=0;i<K;i++){ int *row=existing_h[i];
        for(int l=0;l<N;l++) donors[(size_t)l*K+i]=(uint8_t)row[l]; }
    uint8_t *rh=malloc(N); for(int l=0;l<N;l++) rh[l]=(uint8_t)newh[l];

    double tb0=cf_now();
    Groups Gr=build_groups_adaptive(Ustar);
    *t_build=cf_now()-tb0;

    double tf0=cf_now();
    fold_cc(rh, out_ccpop, &Gr);
    *t_fold=cf_now()-tf0;

    free_groups(&Gr); free(donors); free(rh);
}
