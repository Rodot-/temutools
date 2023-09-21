

def run_config_find_v_inner(config, max_iter=2000, convergence_target=-0.40546510810816444):

    from tardis.simulation.base import Simulation
    import pandas as pd
    import numpy as np
    from astropy import units as u, constants as const
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from scipy.stats import linregress
    from mean_opacity import get_tau_integ

    MAX_ITER = max_iter
    CONVERGENCE_TARGET = convergence_target
    show_convergence_plots = False

    def convergence_target(sim):
        tau_rossland, tau_planck = get_tau_integ(sim, plot=False)
        return np.log(tau_rossland[0])
        #return sim.model.w[0]

    density_file = config.csvy_model
    model = pd.read_csv(density_file, sep=',', skiprows=35)
    #import pdb; pdb.set_trace()

    converged = False
    #target_density = 5e-2 # Will usually place it above the location where w > 0.5
    kind = 'nearest'

    # Guess the inner most possible edge from e-scattering
    dv = model.velocity.values[1:] - model.velocity.values[:-1]
    target_tau = 2/3
    t0 = 100*u.s
    t = config.supernova.time_explosion#13*u.day
    rhodvt2 = model.density.values[1:]*dv * u.g/u.cm**3*u.km/u.s/t**2
    M_p = u.u
    K = M_p*target_tau*2/t0**3/const.sigma_T
    K = K.cgs
    rhodvt2 = rhodvt2.cgs
    K, rhodvt2
    start_index = np.argmin((rhodvt2.value-K.value)**2)
    v_inner_guess = model.velocity.values[1:][start_index] 
    target_density = model.density.values[1:][start_index]



    # Initialize at the shell boundary closest to the target density, should help with a quicker first iteration
    print("Initial v_inner:", v_inner_guess)
    #import pdb; pdb.set_trace()
    config.model.v_inner_boundary = v_inner_guess * u.km/u.s
    #config.model.structure.v_outer_boundary = (model.velocity.values[-1] -1e-7)* u.km/u.s # The 1e-7 helps with numerical problems

    sim = Simulation.from_config(config, show_convergence_plots=show_convergence_plots)
    sim.run_convergence()

    i = 0

    convergence_scale = 10
    w_inner_tol = config.montecarlo.convergence_strategy.threshold * 0.5

    last_w_inner = 5
    last_v_inner = model.velocity.values[0] 
    w_inner = convergence_target(sim)
    v_inner = v_inner_guess
    best = [sim, w_inner]
    initial_w_inner = w_inner
    print("New w_inner:", w_inner)
    while not converged:

        i = i+1
        print("Iteration: ", i)
        damping_constant = config.montecarlo.convergence_strategy.damping_constant * 2
        #new_v_inner = v_inner + (np.log(0.5) - np.log(w_inner)) / (np.log(w_inner) - np.log(last_w_inner)) * (v_inner - last_v_inner) * damping_constant
        if i == 1:
            new_v_inner = (v_inner + model.velocity.values[-1])/2 # Do this to make sure we have two good baseline values
        else:
            new_v_inner = v_inner + (CONVERGENCE_TARGET  - w_inner) / (w_inner - last_w_inner) * (v_inner - last_v_inner) * damping_constant

        if new_v_inner <= model.velocity.values[1] or new_v_inner >= model.velocity.values[-2]: # This happens when things get too shakey, need 1 and -2 so we can extrapolate in the interpolators
            print("New v_inner is bad:", new_v_inner)
            print("    re-computing...")
            print('Previous v_inner, w_inner:', last_v_inner, last_w_inner)
            print('Current v_inner, w_inner:', v_inner, w_inner)
            new_v_inner = v_inner_guess # Go back to the original estimate and re-compute
            if new_v_inner <= model.velocity.values[1] or new_v_inner >= model.velocity.values[-2] or (new_v_inner == v_inner):
                new_v_inner = (v_inner + model.velocity.values[-1])/2
        last_v_inner = v_inner
        last_w_inner = w_inner
        v_inner = new_v_inner
        print("New v_inner:", new_v_inner, 'km/s')
        config.model.v_inner_boundary = (v_inner) * u.km/u.s
                
        ws = sim.model.w
        trads = sim.model.t_rad.value
        vel = sim.model.v_inner.value
        t_inner = sim.model.t_inner.value
        rho_e = sim.plasma.electron_densities

        interp_w = interp1d(vel, ws, fill_value='extrapolate', kind=kind)
        interp_t = interp1d(vel, trads, fill_value='extrapolate', kind=kind)
        interp_e = interp1d(vel, rho_e, fill_value='extrapolate', kind=kind)

        # For now just set the new t_inner to the old one since it will converge when the boundary is the same
        new_t_inner = interp_t(v_inner*1e5)*u.K / (trads[0]/t_inner)
        new_t_inner = sim.model.t_inner
        config.plasma.initial_t_inner = new_t_inner

        new_sim = Simulation.from_config(config, show_convergence_plots=show_convergence_plots)

        new_t_rad = interp_t(new_sim.model.v_inner) * u.K
        new_t_rad[new_t_rad.value < sim.model.t_rad.value.min()] = sim.model.t_rad[0]
        new_t_rad[new_t_rad.value > sim.model.t_rad.value.max()] = sim.model.t_rad[0]

        old_e = sim.plasma.electron_densities.values

        new_e = interp_e(new_sim.model.v_inner)
        new_e[new_e < old_e.min()] = old_e.min()
        new_e[new_e > old_e.max()] = old_e.max()

        new_sim.model.w = interp_w(new_sim.model.v_inner) 
        new_sim.model.w[new_sim.model.w < ws.min()] = ws.max()
        new_sim.model.w[new_sim.model.w > ws.max()] = ws.max()

        sim = new_sim

        sim.model.t_rad = new_t_rad
        sim.model.t_inner = new_t_inner
        sim.plasma.electron_densities.update(pd.Series(new_e))
        sim.plasma.update(t_rad=sim.model.t_rad, w=sim.model.w)
        sim.run_convergence()
        w_inner = convergence_target(sim)
        print("New w_inner:", w_inner)

        if sim.converged:
            if abs(w_inner - CONVERGENCE_TARGET) < (abs(best[1] - CONVERGENCE_TARGET)):
                best = [sim, w_inner]

        if i > MAX_ITER: # Set an iteration limit
            sim = best[0]
            w_inner = best[1]
            converged = True


        if abs(w_inner - CONVERGENCE_TARGET ) < w_inner_tol*convergence_scale:
            print("Got", abs(w_inner - CONVERGENCE_TARGET ), '<', w_inner_tol*convergence_scale)
            if not sim.converged:
                convergence_scale = max(convergence_scale * 0.5, 1.0)
                print("   Next Check Tolerance:", convergence_scale )
                config.montecarlo.iterations *= 2
                config.montecarlo.convergence_strategy.damping_constant *= 0.85
                print("   Simulation Not Converged, increasing iterations to", config.montecarlo.iterations)
                print("                         reducing damping constant to", config.montecarlo.convergence_strategy.damping_constant)
            elif abs(w_inner - CONVERGENCE_TARGET ) < w_inner_tol:
                converged = True
        else:
            print("Got", abs(w_inner - CONVERGENCE_TARGET ), '>', w_inner_tol*convergence_scale)
            print("   Continuing...")


    print("Converged!")
    print("  w_inner:", np.exp(w_inner))
    sim.run_final()
    return sim