import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
plt.rcParams["figure.figsize"]=[6.0,15.0]


#%%

kk = np.logspace(-4,2,1000) # k in h/Mpc


Pk_lcdm = [] # P(k) in (Mpc/h)**3
Pk_exo_1 = [] # P(k) in (Mpc/h)**3
Pk_exo_2 = [] # P(k) in (Mpc/h)**3
Pk_exo_3 = []# P(k) in (Mpc/h)**3
Pk_exo_4 = []# P(k) in (Mpc/h)**3
Pk_exo_5 = []# P(k) in (Mpc/h)**3
Pk_exo_6 = []# P(k) in (Mpc/h)**3
Pk_exo_7 = []# P(k) in (Mpc/h)**3


common_settings={'output':'mPk',
                 'P_k_max_h/Mpc':200,
                 'z_max_pk':3,
                 'omega_b':0.02241,
                 'H0':67.56,
                 'tau_reio':0.05527,
                 'n_s':0.96144,
                 'ln10^{10}A_s':3.05,
                 'input_verbose':1,
                 'background_verbose':1,
                 'thermodynamics_verbose':1,
                 'perturbations_verbose':1,
                 'transfer_verbose':1,
                 'primordial_verbose':1,
                 'spectra_verbose':1,
                 'nonlinear_verbose':1,
                 'lensing_verbose':1,
                 'output_verbose':1}
# COMPUTE LCDM
M = Class()
M.set(common_settings)
M.set({'Omega_cdm':0.262,
       'N_ncdm':1,
       'N_ur':2.0328,
       'm_ncdm':0.06,
       'background_ncdm_psd':0})
M.compute()
h = M.h()

for k in kk:
    Pk_lcdm.append(M.pk(k*h,0)*h**3)
    
M.struct_cleanup()
M.empty()

fpk_lcdm=interp1d(kk,Pk_lcdm)

# COMPUTE EXO MODEL 1

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'0.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_1.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_1=interp1d(kk,Pk_exo_1)


# COMPUTE EXO MODEL 2

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'2.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_2.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_2=interp1d(kk,Pk_exo_2)

# COMPUTE EXO MODEL 3

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'4.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_3.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_3=interp1d(kk,Pk_exo_3)

# COMPUTE EXO MODEL 4

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'6.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_4.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_4=interp1d(kk,Pk_exo_4)


# COMPUTE EXO MODEL 5

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'10.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_5.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_5=interp1d(kk,Pk_exo_5)

# COMPUTE EXO MODEL 6

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'30.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_6.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_6=interp1d(kk,Pk_exo_6)

# COMPUTE EXO MODEL 7

M=Class()
M.set(common_settings)
M.set({'Omega_cdm':0.00001,
       'Omega_ncdm':'0,0.262',
       'N_ncdm':2,
       'N_ur':2.0328,
       'm_ncdm':'0.06,5000',
       'background_ncdm_psd':'0,1',
       'ncdm_psd_parameters':'50.0,1.0', #first is alpha, second is y_avg
#       'Quadrature strategy':'0,2',
#       'Number of momentum bins':'10,50'
       })
M.compute()
h = M.h()

for k in kk:
    Pk_exo_7.append(M.pk(k*h,0)*h**3)

M.struct_cleanup()
M.empty()

fpk_exo_7=interp1d(kk,Pk_exo_7)



#%% PLOT 1


plt.semilogx(kk,fpk_exo_1(kk)/fpk_lcdm(kk)-1.0, 'red',   label=r'$\alpha=0$')
plt.semilogx(kk,fpk_exo_2(kk)/fpk_lcdm(kk)-1.0, 'blue',  label=r'$\alpha=2$')
plt.semilogx(kk,fpk_exo_3(kk)/fpk_lcdm(kk)-1.0, 'green', label=r'$\alpha=4$')
plt.semilogx(kk,fpk_exo_4(kk)/fpk_lcdm(kk)-1.0, 'black', label=r'$\alpha=6$')
plt.semilogx(kk,fpk_exo_5(kk)/fpk_lcdm(kk)-1.0, 'orange', label=r'$\alpha=10$')
plt.semilogx(kk,fpk_exo_6(kk)/fpk_lcdm(kk)-1.0, 'purple', label=r'$\alpha=30$')
plt.semilogx(kk,fpk_exo_7(kk)/fpk_lcdm(kk)-1.0, 'olive',  label=r'$\alpha=50$')

#plt.semilogx(kk,fpk_exo_1(kk)/fpk_lcdm(kk)-1.0, 'red',   label=r'$\bar{y}=1$')
#plt.semilogx(kk,fpk_exo_2(kk)/fpk_lcdm(kk)-1.0, 'blue',  label=r'$\bar{y}=2$')
#plt.semilogx(kk,fpk_exo_3(kk)/fpk_lcdm(kk)-1.0, 'green', label=r'$\bar{y}=3$')
#plt.semilogx(kk,fpk_exo_4(kk)/fpk_lcdm(kk)-1.0, 'black', label=r'$\bar{y}=4$')


plt.title(r'$f_0(q) = \frac{1}{q^2} \left(\frac{q}{\bar{y}}\right)^{\alpha} e^{-(1+\alpha)q/\bar{y}} \, \, \, \, \, \, \, \, \, \, \, m_{\mathrm{NCDM}}= 5 \, \mathrm{keV}$', fontsize=15, pad=15)
plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$', fontsize=18)
plt.ylabel(r'$P_{\Lambda\mathrm{NCDM}}/P_{\Lambda\mathrm{CDM}}-1$', fontsize=18)
plt.xlim(1e-1,1e2)

plt.tick_params(axis="x",labelsize=18)
plt.tick_params(axis="y",labelsize=18)

plt.legend(loc='center left',fontsize=15, frameon=False,borderaxespad=0.)


plt.text(1, -0.6,r'$\bar{y}=1$',fontsize=15)
#plt.text(1, -0.6,r'$\alpha=2$',fontsize=15)

plt.show()
plt.clf()



  
