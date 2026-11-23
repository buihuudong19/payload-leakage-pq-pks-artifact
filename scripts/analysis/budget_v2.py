"""Bit budget by region A0/A1, bucket ceilings, signature bucket thresholds, single-bucket profile by alpha,
alpha0 matched by bucket, rejection probability of Kemeleon-R and number of receiver trials.

Design-stage calculation for the bucket ladder and the receiver bounds (paper, Sections 5.4-5.5).
Run: python scripts/analysis/budget_v2.py
"""
import math
n=512*512; tag=16; hdr=6
B=[256,512,1024,2048,4096,8192]
def bucket(L,Bs=B):
    for b in Bs:
        if L<=b: return b
    return None
for name,l0B in [("full",1536),("R",1252)]:
    l0=8*l0B; n0=math.ceil(l0/0.15); n1=n-n0
    print(f"[{name}] l0={l0} n0={n0} n1={n1}")
    for a in [0.05,0.1,0.2,0.4]:
        a1=(a*n-l0)/n1
        print(f"  alpha_global={a}: alpha1={a1:.4f}")
    for m,s,lab in [(256,0,'none'),(256,2420,'ML-DSA-44'),(256,3309,'ML-DSA-65'),(256,666,'FN-DSA-512~'),(256,1280,'FN-DSA-1024~'),(64,0,'none'),(1024,0,'none'),(4096,0,'none')]:
        b=bucket(14+m+s); tot=l0B+b+tag   # session-start image: header 14 B (with t0)
        print(f"  m={m} sig={lab}: 6+m+s={hdr+m+s} B={b} start={tot} B alpha_g={8*tot/n:.4f} alpha1={8*(b+tag)/n1:.4f}")
print("max start alpha full:",8*(1536+8192+16)/n," max next alpha:",8*(8192+16)/n)
print("achievable next alphas:",[round(8*(b+16)/n,4) for b in B+[16384]])
print("achievable start alphas (full):",[round(8*(1536+b+16)/n,4) for b in B+[16384]])
# ML-DSA bucket thresholds (session-start image, header 14 B; without t0 add 8 B)
for s,lab in [(2420,'44'),(3309,'65')]:
    for b in B:
        mmax=b-14-s
        if mmax>=0: print(f"ML-DSA-{lab}: bucket {b} holds m <= {mmax}")
# R-b per-alpha single-bucket profile
for a in [0.1,0.2,0.4]:
    tot=math.floor(a*n/8)
    print(f"alpha={a}: total bytes={tot}; start B_alpha={tot-1536-16}; next B_alpha={tot-16}; start(R) B={tot-1252-16}")
# matched alpha0 per bucket (full)
l0=12288
for b in B:
    ag=(l0+8*(b+16))/n; n0=math.ceil(l0/ag)
    print(f"matched: B={b} alpha_g={ag:.4f} n0={n0} ({n0/n:.1%}) alpha1={8*(b+16)/(n-n0):.4f}")
p=2**(8986-768*math.log2(3329))*(1-1/3329)**256   # per coefficient c2: Pr[0]=209/q, rejected with probability 1/209 -> 1/q
print("p_acc",p,"(1-p)^64=",(1-p)**64,"log2",64*math.log2(1-p))
print("reduction 1252/1536:",1-1252/1536)
# receiver trials
Smax,W,NB=16,8,6
# Number of receiver trials. Current code: each session tries the unseen j in the window plus EVERY seen j >= 2;
# the worst case (exhausting the state) is N + 6 with W = 8 when N >= W + 1 (= 2W - 1 + (N - W - 1)).
print("trials Prop2 (no replay search):",(1+Smax*(2*W-1))*NB)
for N in [8,16,32]:
    per=max(2*W-1, N+W-2) if N>W+1 else 2*W-1
    print(f"  ma hien tai, N={N}: <= {(1+Smax*per)*NB}")
# Integer STC width w = floor(n/m): m*w elements used, effective rate 1/w
def wid(nn,mbits):
    w=nn//mbits; return w, mbits*w/nn, 1/w if w else float('inf')
print("STC width (w, phan dung, toc do thuc):")
print("  A0 full:", wid(81920,12288))
for lab,nn,B_ in [("start A1 bucket 4096",180224,4096),("start A1 bucket 8192",180224,8192),("start A1 alpha=0.4 B=11555",180224,11555),("next bucket 256",262144,256),("next alpha=0.4 B=13091",262144,13091)]:
    print(f"  {lab}:", wid(nn,8*(B_+16)))
print("  REF-1S alpha=0.4:", wid(262144,int(0.4*262144)))
