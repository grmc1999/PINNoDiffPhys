import numpy as np


def group_U(dataframe,N_points=3):
    X_r_i=(
        lambda u,x,d: (
        np.concatenate((np.concatenate(*u),np.array(d["U"]))),
        np.concatenate((np.concatenate(*x),np.array(d["X"])))
        )
          )
    f=(lambda d:X_r_i(*tuple(zip(dataframe.sample(n=N_points,replace=True)[["U","X"]].values.T)),d))
    dataframe["U"],dataframe["X"]=zip(*(dataframe[["U","X"]].apply(f,axis=1)))
    return dataframe


import sympy

SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
K_rw0=sympy.symbols("k_{rw0}")
K_ro0=sympy.symbols("k_{ro0}")

Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)

K_rw=K_rw0*Sc**((2+3*lam)/(lam))
K_ro=K_ro0*((1-Sc)**2)*(1-Sc**((2+lam)/(lam)))

Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,0.05,0.45,0.5,1e5)
Se_f=lambda sw: sympy.lambdify((Sw,SOR,SWC),Sc)(sw,0.05,0.45)
K_rw_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_rw0),K_rw)(sw,0.05,0.45,0.5,1e5,0.3)
K_ro_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_ro0),K_ro)(sw,0.05,0.45,0.5,1e5,0.5)

def transform_sw_pt2pa(Sw,pt):
    pc=Pc_f(Sw)
    p_o=(pt+pc)/2
    p_w=(pt-pc)/2
    return p_o,p_w


def data_transform_sw_pt2pa(dataframe):
    dataframe["Po"],dataframe["Pw"]=transform_sw_pt2pa(dataframe["Ws"],dataframe["P"])
    return dataframe
