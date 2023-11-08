

from astropy import units as u, constants as const
import numpy as np

def get_tau_integ(sim, tau=2.0/3, plot=False, bin_size=10):

    index = sim.plasma.atomic_data.lines.nu.index
    taus = sim.plasma.tau_sobolevs.loc[index]
    freqs = sim.plasma.atomic_data.lines.nu.values
    order = np.argsort(freqs)
    freqs = freqs[order]
    taus = sim.plasma.tau_sobolevs.values[order]

    extra = bin_size-len(freqs)%bin_size
    extra_freqs = np.arange(extra+1)+1
    extra_taus = np.zeros((extra+1, taus.shape[1]))
    freqs = np.hstack((extra_freqs, freqs))
    taus = np.vstack((extra_taus, taus))

    bins_low = freqs[:-bin_size:bin_size]
    bins_high = freqs[bin_size::bin_size]
    delta_nu = bins_high - bins_low
    n_bins = len(delta_nu)

    taus = taus[1:n_bins*bin_size+1]
    freqs = freqs[1:n_bins*bin_size+1]

    ct = ((13*u.day*const.c).cgs.value)
    t_rad = sim.simulation_state.t_rad.cgs.value

    h = const.h.cgs.value
    c = const.c.cgs.value
    kb = const.k_B.cgs.value

    def B(nu, T):
        return 2*h*nu**3/c**2/(np.exp(h*nu/(kb*T))-1)
    
    def U(nu, T):
        return B(nu, T)**2 * (c/nu)**2 * (2*kb*T**2)**-1

    kappa_exp = (bins_low/delta_nu).reshape(-1, 1)/ct*(1-np.exp(-taus.reshape(n_bins, bin_size, -1))).sum(axis=1)
    kappa_thom = (sim.plasma.electron_densities.values*const.sigma_T.cgs.value)
    Bdnu = B(bins_low.reshape(-1, 1), t_rad.reshape(1, -1))*delta_nu.reshape(-1, 1)
    kappa_planck = (kappa_thom + (Bdnu*kappa_exp).sum(axis=0)/(Bdnu.sum(axis=0)))
    
    udnu = U(bins_low.reshape(-1, 1), t_rad.reshape(1, -1))*delta_nu.reshape(-1, 1)
    kappa_tot = kappa_thom + kappa_exp
    kappa_rossland = ((udnu*kappa_tot**-1).sum(axis=0)/(udnu.sum(axis=0)))**-1

    dr = (sim.simulation_state.r_outer-sim.simulation_state.r_inner).cgs.value
    dtau = kappa_planck*dr
    planck_integ_tau = np.cumsum(dtau[::-1])[::-1]
    rossland_integ_tau = np.cumsum((kappa_rossland*dr)[::-1])[::-1]
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(sim.simulation_state.v_inner, planck_integ_tau, label='Planck')
        plt.plot(sim.simulation_state.v_inner, rossland_integ_tau, label='Rossland')
        plt.plot(sim.simulation_state.v_inner, np.cumsum((dr*kappa_thom)[::-1])[::-1], label='Thomson')
        plt.plot(sim.simulation_state.v_inner, np.cumsum((dr*(Bdnu*kappa_exp).sum(axis=0)/(Bdnu.sum(axis=0)))[::-1])[::-1], label='Expansion')
        plt.legend()
        plt.yscale('log')
    kappa_planck
    return rossland_integ_tau, planck_integ_tau