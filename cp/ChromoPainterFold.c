/* ChromoPainterFold.c - exact block-fold of ChromoPainter's chunk-count.
 *
 * A faithful port of the standalone cpfold engine (cpfold/cpfold.c, proven
 * fold==dense to ~3e-13) grafted onto cp's data structures. Produces the
 * per-population corrected_chunk_count IDENTICALLY to the dense forward-
 * backward-chunkcount, in O(N*Umean) instead of O(N*K).
 *
 * Produces per-population chunk counts, expected differences, expected chunk
 * lengths, the N_e (-in) and global-mutation (-iM) E-M quantities, and the forward
 * log-likelihood. Exact for uniform copy_prob + global mutation and for FIXED
 * per-population copy_prob (-p) / mutation (-m). It also returns the per-pop start
 * term separately, so the caller can drive the full per-pop E-M loop - copy
 * proportions (-ip) and per-pop mutation (-im, from the expected differences) -
 * exactly; it does not produce samples (-s>0). Reuses the caller's exact TransProb
 * so the transition is identical to the dense.
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

/* engine globals (set per call by cpfold_perpop; single-threaded engine) */
static int K, N, npop;
/* per-donor copy_prob / copy_probSTART / mutation rate. Uniform in the common
   case (one global value); per-population under -p (fixed per-pop copy_prob) and
   -m (fixed per-pop mutation), which the engine handles by grouping the state by
   (substring,pop) so copy_prob/mutation are constant within each group. */
static double *cf_cp, *cf_cps, *cf_mut;
static uint8_t *donors;     /* [N*K] locus-major, built from existing_h */
static int donors_built_N, donors_built_K;  /* dims of a retained `donors` (retain_panel) */
static int *pop_vec;        /* [K] donor population */
static double *T;           /* [N-1] transition (the caller's TransProb) */
static double *pos_g, *lam_g, delta_g, rho_g;  /* for the N_e (-in) EM update */

static double cf_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* emission is cp_emis() from ChromoPainterFold.h - shared with the dense FB. */

typedef struct { int s,e,U; } Block;
typedef struct {
    int B, nb, Umax;
    Block *blk;
    int *gidB;
    int *sizeg,*repg,*popg;
    int *cellB, *celloff, *cellGb, *cellG2, maxcell;
    int *gperm, *goff;   /* group->donor CSR per block (for -fold sampling): the donors
                            of block b's group g are gperm[b*K + goff[b*(Umax+1)+g] ..
                            goff[b*(Umax+1)+g+1]). Built once (panel-fixed). */
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
    free(G->cellB);free(G->celloff);free(G->cellGb);free(G->cellG2);
    free(G->gperm);free(G->goff); }

/* PBWT adaptive variable-length blocks: grow each block by incrementally
   splitting the substring grouping column-by-column; cut when U reaches Ustar.
   16-way allele code (al&15) keeps all cp allele symbols 0-5,8,9 distinct. */
static Groups build_groups_adaptive(int Ustar, int bypop){
    Groups G; G.B=Ustar; G.nb=0;
    int cap_blk=64; G.blk=malloc(sizeof(Block)*cap_blk); int *gidB=NULL;
    int *gid=malloc(sizeof(int)*K), *ng=malloc(sizeof(int)*K);
    /* map indexed by ek=gid[i]*16+code; gid[i] < K always (<= K groups), and the
       transient group count between block-start cuts can exceed Ustar on
       multi-allelic data, so size by K (not Ustar) to be overflow-proof. */
    int mapsz=K*16+16;
    int *mp=malloc(sizeof(int)*mapsz), *mstamp=calloc(mapsz,sizeof(int)); int gen=0, Umax=0;
    /* bypop seeds each block grouped by pop (gid=pop_vec), so every group is
       single-pop and copy_prob/mutation are group-constant; else substring-only. */
    int a=0; for(int i=0;i<K;i++) gid[i]=bypop?pop_vec[i]:0; int U=bypop?npop:1; int b=0;
    while(b<N){
        gen++; int newU=0; const uint8_t *col=donors+(size_t)b*K;
        for(int i=0;i<K;i++){ int code=col[i]&15; int ek=gid[i]*16+code;
            if(mstamp[ek]!=gen){ mstamp[ek]=gen; mp[ek]=newU++; } ng[i]=mp[ek]; }
        if(newU>Ustar && (b-a)>=2 && b<N-1){
            if(G.nb>=cap_blk){ cap_blk*=2; G.blk=realloc(G.blk,sizeof(Block)*cap_blk); }
            G.blk[G.nb].s=a; G.blk[G.nb].e=b; G.blk[G.nb].U=U;
            gidB=realloc(gidB,(size_t)(G.nb+1)*K*sizeof(int)); memcpy(gidB+(size_t)G.nb*K,gid,K*sizeof(int));
            if(U>Umax)Umax=U; G.nb++;
            a=b; for(int i=0;i<K;i++) gid[i]=bypop?pop_vec[i]:0; U=bypop?npop:1;
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
    /* group->donor CSR (gperm/goff), panel-fixed, for the -fold path sampler. */
    G.goff=malloc(sizeof(int)*(size_t)G.nb*(Umax+1)); G.gperm=malloc(sizeof(int)*(size_t)G.nb*K);
    { int *cur=malloc(sizeof(int)*(Umax+1));
      for(int bb=0;bb<G.nb;bb++){ int Ub=G.blk[bb].U; int *gb=gidB+(size_t)bb*K;
        int *sz=G.sizeg+(size_t)bb*Umax, *go=G.goff+(size_t)bb*(Umax+1), *gp=G.gperm+(size_t)bb*K;
        go[0]=0; for(int g=0;g<Ub;g++){ go[g+1]=go[g]+sz[g]; cur[g]=go[g]; }
        for(int i=0;i<K;i++){ int g=gb[i]; gp[cur[g]++]=i; } }
      free(cur); }
    build_contingency(&G);
    return G;
}

/* FOLDED forward+backward+chunkcount -> per-pop chunk counts (ccpop[npop]).
   Direct port of cpfold.c fold_cc; rh = the recipient allele row. */
/* etp_out[N-1] (per-locus transition prob, for -d) and ecp_out[N*npop] (per-locus
   per-pop copy posterior, for -b) are optional: filled only when non-NULL. */
/* Sample one donor ~ rescaled forward a[l][.] WITHOUT materializing the length-K
   vector: draw a group g prop to (size_g*GF + PF*Sentry_g), then a donor within g
   prop to (GF + PF*a_entry(i)) over that group's members (gperm/goff CSR). gw is a
   >=Umax scratch buffer. Uses the global rand() (seeded by -S). */
static int fold_sample_donor(int l, Groups *Gr, const double *GF, const double *PF,
                             const double *AENTRY, const double *SentS, const size_t *off,
                             const int *locblk, double *gw){
    int Umax=Gr->Umax, b=locblk[l], sb=Gr->blk[b].s, U=Gr->blk[b].U;
    const double *GFr=GF+off[b]+(size_t)(l-sb)*U, *PFr=PF+off[b]+(size_t)(l-sb)*U;
    const double *ent=AENTRY+(size_t)b*K, *Sx=SentS+(size_t)b*Umax;
    const int *sz=Gr->sizeg+(size_t)b*Umax, *go=Gr->goff+(size_t)b*(Umax+1), *gp=Gr->gperm+(size_t)b*K;
    double gtot=0.0;
    for(int g=0;g<U;g++){ double m=sz[g]*GFr[g]+PFr[g]*Sx[g]; if(m<0.0)m=0.0; gw[g]=m; gtot+=m; }
    double u=((double)rand()/RAND_MAX)*gtot, c=0.0; int g=0;
    for(g=0;g<U;g++){ c+=gw[g]; if(u<=c) break; } if(g>=U) g=U-1;
    int lo=go[g], hi=go[g+1]; double dtot=0.0;
    for(int t=lo;t<hi;t++){ double m=GFr[g]+PFr[g]*ent[gp[t]]; if(m>0.0)dtot+=m; }
    double u2=((double)rand()/RAND_MAX)*dtot, c2=0.0; int sel=gp[hi-1];
    for(int t=lo;t<hi;t++){ double m=GFr[g]+PFr[g]*ent[gp[t]]; if(m>0.0)c2+=m; if(u2<=c2){ sel=gp[t]; break; } }
    return sel;
}

/* samplesTOT>0 + sout!=NULL: also draw samplesTOT copying paths (hierarchical FFBS
   over the affine forward, no full Alphamat) into sout[s*N+l] = donor index. */
static void fold_cc(const uint8_t *rh, double *ccpop, double *startpop, double *ndiff, double *nlen, double *Ne_out, double *loglik_out, double *etp_out, double *ecp_out, int samplesTOT, int *sout, Groups *Gr){
    #define EM(L,II) cp_emis(rh[(L)], donors[(size_t)(L)*K+(II)], cf_mut[(II)])
    int nb=Gr->nb, Umax=Gr->Umax; Block *blk=Gr->blk; int *gidB=Gr->gidB;
    int *sizeg=Gr->sizeg, *repg=Gr->repg;
    /* per-(block,group) copy_prob / copy_probSTART (group-constant since each group
       is single-pop under -p; = the uniform scalar otherwise). */
    double *CPG=malloc((size_t)nb*Umax*sizeof(double)), *CPSG=malloc((size_t)nb*Umax*sizeof(double));
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
    double *sumA_arr = (samplesTOT>0 && sout) ? malloc(sizeof(double)*N) : NULL;  /* rescaled forward sum S~[l], for sampling */

    for(int b=0;b<nb;b++){ int sb=blk[b].s,eb=blk[b].e,U=blk[b].U; int *rp=repg+(size_t)b*Umax;
        double *base=Eg+off[b]; double *cpg=CPG+(size_t)b*Umax, *cpsg=CPSG+(size_t)b*Umax;
        for(int g=0;g<U;g++){ cpg[g]=cf_cp[rp[g]]; cpsg[g]=cf_cps[rp[g]]; }
        for(int l=sb;l<eb;l++){ double *row=base+(size_t)(l-sb)*U; for(int g=0;g<U;g++) row[g]=EM(l, rp[g]); } }

    double *aprev=calloc(K,sizeof(double));
    for(int b=0;b<nb;b++){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K; int *sz=sizeg+(size_t)b*Umax;
        double *ent=AENTRY+(size_t)b*K, *Sx=SentS+(size_t)b*Umax;
        double *GFb=GF+off[b], *PFb=PF+off[b], *Egb=Eg+off[b];
        double *cpg=CPG+(size_t)b*Umax, *cpsg=CPSG+(size_t)b*Umax;
        for(int g=0;g<U;g++) Sx[g]=0.0;
        for(int i=0;i<K;i++){ double a=(b==0)?0.0:aprev[i]; ent[i]=a; if(b>0) Sx[gb[i]]+=a; }
        for(int l=sb;l<eb;l++){
            double rlm1=(l>=2)?exp(As[l-2]-As[l-1]):((l==1)?exp(-As[0]):0.0);
            double *GFr=GFb+(size_t)(l-sb)*U, *PFr=PFb+(size_t)(l-sb)*U, *Er=Egb+(size_t)(l-sb)*U;
            double sumA=0.0;
            for(int g=0;g<U;g++){
                double eg=Er[g], gv,pv;
                if(l==sb && b==0){ gv=cpsg[g]*eg; pv=0.0; }     /* copy_probSTART at locus 0 */
                else if(l==sb){ gv=eg*cpg[g]; pv=eg*(1-T[l-1])*rlm1; }
                else { double fac=eg*(1-T[l-1])*rlm1; gv=eg*cpg[g]+fac*G[g]; pv=fac*P[g]; }
                G[g]=gv; P[g]=pv; GFr[g]=gv; PFr[g]=pv; sumA += sz[g]*gv + pv*Sx[g];
            }
            if(sumA_arr) sumA_arr[l]=sumA;
            double s=(l<N-1)?sumA*T[l]:sumA; As[l]=(l>=1?As[l-1]:0.0)+log(s);
        }
        double *GFl=GFb+(size_t)(eb-1-sb)*U, *PFl=PFb+(size_t)(eb-1-sb)*U;
        for(int i=0;i<K;i++){ int g=gb[i]; aprev[i]=GFl[g]+PFl[g]*ent[i]; }
    }

    /* ---- -fold path sampling: hierarchical FFBS over the affine forward ----
       For each path, sample the donor at N-1 ~ a[N-1][.], then walk backward: stay on
       the current donor j with prob (1-T)*a[l][j] / ((1-T)*a[l][j] + T*copy_prob[j]*S~[l])
       (no recombination), else resample i ~ a[l][.] (recombination). All quantities use
       the rescaled forward; the per-locus rescale cancels in both ratios. */
    if(samplesTOT>0 && sout){
        int *locblk=malloc(sizeof(int)*N);
        for(int b=0;b<nb;b++) for(int l=blk[b].s;l<blk[b].e;l++) locblk[l]=b;
        double *gw=malloc(sizeof(double)*Umax);
        for(int s=0;s<samplesTOT;s++){
            int j=fold_sample_donor(N-1, Gr, GF, PF, AENTRY, SentS, off, locblk, gw);
            sout[(size_t)s*N+(N-1)]=j;
            for(int l=N-2;l>=0;l--){
                int b=locblk[l], sb=blk[b].s, U=blk[b].U, gj=gidB[(size_t)b*K+j];
                double a_lj=GF[off[b]+(size_t)(l-sb)*U+gj] + PF[off[b]+(size_t)(l-sb)*U+gj]*AENTRY[(size_t)b*K+j];
                double w_norec=(1.0-T[l])*a_lj, w_rec=T[l]*cf_cp[j]*sumA_arr[l];
                double u=(double)rand()/RAND_MAX; int i;
                if(u*(w_norec+w_rec) < w_norec) i=j;   /* no recombination: copying continues */
                else i=fold_sample_donor(l, Gr, GF, PF, AENTRY, SentS, off, locblk, gw);
                sout[(size_t)s*N+l]=i; j=i;
            }
        }
        free(locblk); free(gw);
    }

    double Asf=As[N-1];
    for(int p=0;p<npop;p++) ccpop[p]=0.0;
    if(startpop) for(int p=0;p<npop;p++) startpop[p]=0.0;
    /* EM quantities (-in N_e, -iM global mutation), verified fold (wf wxk7vegbf):
       N_e from per-locus etp (the chunkcount total) rho-weighted by the dense gd_l;
       per-pop expected_differences from e_a_l_bc=a_l*c_l*KC folded over the moments. */
    double tot_prob_Ne=0.0, tot_gd=0.0;
    for(int p=0;p<npop;p++){ ndiff[p]=0.0; nlen[p]=0.0; }
    { double S_end=exp(As[N-2]-Asf);   /* last locus N-1: c=1, a[N-1]=aprev (post-forward) */
      for(int i=0;i<K;i++){ double am=aprev[i]*S_end;   /* = e_a_l_bc at l=N-1 (copy posterior) */
        if(rh[N-1]!=donors[(size_t)(N-1)*K+i]) ndiff[pop_vec[i]]+=am;
        if(ecp_out) ecp_out[(size_t)(N-1)*npop+pop_vec[i]]+=am; } }
    double *cexit=malloc(sizeof(double)*K); for(int i=0;i<K;i++) cexit[i]=1.0;
    { double sb0=0.0; for(int i=0;i<K;i++) sb0+=T[N-2]*cf_cp[i]*EM(N-1,i); Bs[N-1]=log(sb0); }
    for(int b=nb-1;b>=0;b--){
        int sb=blk[b].s, eb=blk[b].e, U=blk[b].U; int *gb=gidB+(size_t)b*K;
        int *sz=sizeg+(size_t)b*Umax;
        int top=(eb-1<N-2)?eb-1:N-2;
        double *w=WB+(size_t)b*K, *Sw=SwS+(size_t)b*Umax, *ent=AENTRY+(size_t)b*K;
        double *GFb=GF+off[b], *PFb=PF+off[b], *Egb=Eg+off[b]; double *cpg=CPG+(size_t)b*Umax;
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
                for(int g=0;g<U;g++) sB+=cpg[g]*Er[g]*(sz[g]*GBc[g]+PBc[g]*Sw[g]); Bs[l]=Bs[l+1]+log(T[l-1]*sB); }
            double BsR=(l+2<=N-1)?Bs[l+2]:0.0, Asm1=(l>=1)?As[l-1]:0.0;
            double KF1=exp(As[l]+BsR-Asf), KF=exp(Asm1+BsR-Asf), om=(1-T[l]);
            double *GFr=GFb+(size_t)(l-sb)*U, *PFr=PFb+(size_t)(l-sb)*U;
            double etpl=0.0;     /* per-locus chunkcount total = expected_transition_prob[l] (for N_e) */
            /* expected_chunk_length integrand = G_l*0.5*(e_a_lp1_bp + e_a_l_bc) (the
               tp_from_i_to_i terms cancel); G_l = 100*(pos[l+1]-pos[l])*delta*lambda[l] (cM). */
            double Glh = (lam_g[l]>=0) ? 0.5*100.0*(pos_g[l+1]-pos_g[l])*delta_g*lam_g[l] : 0.0;
            if(l < eb-1){
                double *GFr1=GFb+(size_t)(l+1-sb)*U, *PFr1=PFb+(size_t)(l+1-sb)*U;
                for(int g=0;g<U;g++){
                    double eg=Er1[g];
                    double XG=GFr1[g]*KF1 - GFr[g]*KF*eg*om, XP=PFr1[g]*KF1 - PFr[g]*KF*eg*om;
                    double GBl1,PBl1; if(l+1==N-1){GBl1=1.0;PBl1=0.0;} else {GBl1=GBp[g];PBl1=PBp[g];}
                    double cA=GBl1*XG, cB=GBl1*XP, cC=PBl1*XG, cD=PBl1*XP; int base=g*npop;
                    double aA=KF1*GFr1[g]*GBl1, aB=KF1*GFr1[g]*PBl1, aC=KF1*PFr1[g]*GBl1, aD=KF1*PFr1[g]*PBl1; /* e_a_lp1_bp */
                    for(int p=0;p<npop;p++){ double s_=szP[base+p]; if(s_==0.0) continue;
                        double inc=cA*s_ + cB*SaP[base+p] + cC*SwP[base+p] + cD*SawP[base+p];
                        ccpop[p]+=inc; etpl+=inc;
                        nlen[p]+=Glh*(aA*s_ + aB*SwP[base+p] + aC*SaP[base+p] + aD*SawP[base+p]); }
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
                    /* e_a_lp1_bp at boundary: a_{eb}=GF2+PF2*a_l, a_l=GFr+PFr*ent, so
                       a_{eb}*cx2 summed = (GF2+PF2*GFr)*Mc + PF2*PFr*Mce, times KF1. */
                    double bA=KF1*(GF2[g2]+PF2[g2]*GFr[g]), bB=KF1*PF2[g2]*PFr[g];
                    for(int p=0;p<npop;p++){ double inc=cf0*Mc[base+p]+cf1*Mce[base+p]; ccpop[p]+=inc; etpl+=inc;
                        nlen[p]+=Glh*(bA*Mc[base+p]+bB*Mce[base+p]); } }
            }
            /* N_e: dense gd_l rho-weight (pos/delta/lambda - NOT the T-form, which underflows).
               Skip zero-distance loci (gd==0, e.g. duplicate SNP position or lambda==0):
               their rho-weight is the 0/0 limit 1 but etpl->0 there, so the contribution is
               zero - including them would divide by zero. Matches the dense guard. */
            if(lam_g[l]>=0){ double gd=(pos_g[l+1]-pos_g[l])*delta_g*lam_g[l];
                if(gd>0.0){ tot_gd+=gd; tot_prob_Ne+=(rho_g*gd/(1.0-exp(-rho_g*gd)))*etpl; } }
            if(etp_out) etp_out[l]=etpl;   /* per-locus transition prob (for -d), l in 0..N-2 */
            /* per-pop expected_differences (e_a_l_bc=a_l*c_l*KC, mismatched donors) +
               the e_a_l_bc half of expected_chunk_length (all donors). CURRENT GBc/PBc.
               Summed over groups per pop, e_a_l_bc is the -b copy posterior at this locus. */
            { double KC=exp(Asm1+Bs[l+1]-Asf); int *rp=repg+(size_t)b*Umax;
              for(int g=0;g<U;g++){
                double gA=GFr[g],pA=PFr[g],gB=GBc[g],pB=PBc[g]; int base=g*npop;
                int mm = (rh[l]!=donors[(size_t)l*K+rp[g]]);
                for(int p=0;p<npop;p++){ double sz_=szP[base+p];
                    double albc=KC*(gA*gB*sz_ + gA*pB*SwP[base+p] + pA*gB*SaP[base+p] + pA*pB*SawP[base+p]);
                    nlen[p]+=Glh*albc; if(mm) ndiff[p]+=albc;
                    if(ecp_out) ecp_out[(size_t)l*npop+p]+=albc; } } }
            { double *t; t=GBp;GBp=GBc;GBc=t; t=PBp;PBp=PBc;PBc=t; }
        }
        double *cxb=CEXIT+(size_t)b*K;
        for(int i=0;i<K;i++){ int g=gb[i]; double cv=GBp[g]+PBp[g]*w[i]; cexit[i]=cv; cxb[i]=cv; }
    }
    { double k0=exp(Bs[1]-Asf); int *gb0=gidB; double *ent0=AENTRY, *cx0=CEXIT; double *GF0=GF+off[0], *PF0=PF+off[0];
      /* start term (= dense copy_prob_newSTART, the locus-0 posterior): folded into
         ccpop to give the full corrected chunk count, and reported separately in
         startpop so the caller can split off the no-start copy_prob_new (-ip). */
      for(int i=0;i<K;i++){ int g=gb0[i]; double a0=GF0[g]+PF0[g]*ent0[i]; double st=a0*cx0[i]*k0;
        ccpop[pop_vec[i]] += st; if(startpop) startpop[pop_vec[i]] += st; } }

    *Ne_out = (tot_gd>0.0) ? tot_prob_Ne/tot_gd : 0.0;   /* N_e EM estimate (-in); dense floor */
    if(*Ne_out < MIN_NE) *Ne_out = MIN_NE;
    if(loglik_out) *loglik_out = Asf;   /* forward log-likelihood (= dense Alphasum) */

    free(off);free(GF);free(PF);free(Eg);free(AENTRY);free(WB);free(CEXIT);free(As);free(Bs);
    free(SentS);free(SwS);free(G);free(P);free(GBp);free(PBp);free(GBc);free(PBc);
    free(szP);free(SaP);free(SwP);free(SawP);free(aprev);free(cexit);free(Mc);free(Mce);free(CPG);free(CPSG);
    free(sumA_arr);
    #undef EM
}

/* Entry point. Builds locus-major donor buffer from existing_h (int** hap-major),
   the recipient row from newh, sets engine state, builds the (panel-fixed) grouping
   [timed -> *t_build] and runs the fold [timed -> *t_fold]. out_ccpop[ndonorpops]. */
void cpfold_perpop(signed char *newh, signed char **existing_h, int nhaps, int nloci,
                   double *TransProb, double *MutProb_vec, double *copy_prob, double *copy_probSTART,
                   double *pos, double *lambda, double delta, double rhobar,
                   int *pop_vec_in, int ndonorpops, int Ustar,
                   double *out_ccpop, double *out_start, double *out_ndiff, double *out_nlen, double *out_Ne,
                   double *out_loglik, double *out_etp, double *out_ecp, int samplesTOT, int *out_samples,
                   double *t_build, double *t_fold, int retain_panel){
    K=nhaps; N=nloci; npop=ndonorpops;
    cf_cp=copy_prob; cf_cps=copy_probSTART; cf_mut=MutProb_vec;
    pop_vec=pop_vec_in; T=TransProb;
    pos_g=pos; lam_g=lambda; delta_g=delta; rho_g=rhobar;
    /* group state by (substring,pop) only when a per-pop parameter is actually
       non-uniform (-p or -m); substring-only otherwise (the fast common path). */
    int bypop=0;
    for(int i=1;i<K;i++){ if(copy_prob[i]!=copy_prob[0]||copy_probSTART[i]!=copy_probSTART[0]||MutProb_vec[i]!=MutProb_vec[0]){ bypop=1; break; } }
    /* Build the locus-major donor buffer from existing_h (hap-major). With retain_panel
       (single-recipient -fold) build only when the panel is not already resident for this
       (N,K): existing_h is then read just once. Otherwise (re)build every call - the panel
       changes between recipients. */
    if(!retain_panel || donors==NULL || donors_built_N!=N || donors_built_K!=K){
        if(retain_panel) free(donors);   /* drop a stale retained panel on a size change */
        donors=malloc((size_t)N*K);
        for(int i=0;i<K;i++){ signed char *row=existing_h[i];
            for(int l=0;l<N;l++) donors[(size_t)l*K+i]=(uint8_t)row[l];
            /* retain: this hap-major donor row is now fully copied into the locus-major
               buffer and is dead. Free it HERE - before build_groups/fold_cc allocate the
               fold working set - so the hap-major panel and the working set never coexist
               (that is what drops the peak; freeing after the fold returns does not, since
               the peak already happened). Freed via the existing_h alias into
               all_chromosomes; the caller nulls the canonical all_chromosomes entry next
               so DestroyData does not double-free. Leave-one-out keeps the recipient's own
               rows out of existing_h, so newh is never touched. */
            if(retain_panel){ free(row); existing_h[i]=NULL; }
        }
        donors_built_N=N; donors_built_K=K;
    }
    uint8_t *rh=malloc(N); for(int l=0;l<N;l++) rh[l]=(uint8_t)newh[l];

    double tb0=cf_now();
    Groups Gr=build_groups_adaptive(Ustar, bypop);
    *t_build=cf_now()-tb0;

    double tf0=cf_now();
    fold_cc(rh, out_ccpop, out_start, out_ndiff, out_nlen, out_Ne, out_loglik, out_etp, out_ecp, samplesTOT, out_samples, &Gr);
    *t_fold=cf_now()-tf0;

    /* Keep the donor buffer across calls when retaining (released by cpfold_cleanup);
       otherwise free it and null the static so cpfold_cleanup stays a safe no-op. */
    free_groups(&Gr); if(!retain_panel){ free(donors); donors=NULL; } free(rh);
}

/* Free a retained locus-major donor buffer. No-op when none is held; idempotent, so it
   is safe to call unconditionally at teardown (including the error path). */
void cpfold_cleanup(void){ free(donors); donors=NULL; donors_built_N=0; donors_built_K=0; }
