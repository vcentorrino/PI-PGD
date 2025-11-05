### Author: Veronica Centorrino

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, lsq_linear
import matplotlib.pyplot as plt
from matplotlib import rc

# ============================================================================
#                               Functions
# ============================================================================
def soft(x, thr):
    """Soft thresholding operator.
    INPUT:
          x = state -- array_like
        thr = threshold -- float
    """
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0)

# --------------------------
#    Problem definitions
# --------------------------
def f_val(x):
    """Compute the value of the quadratic objective function."""
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]+1)**2
def f_grad(x):
    """Compute the gradient of the quadratic objective function."""
    return 2*np.array([x[0]-1, x[1]-2, x[2]+1])
def g_val(x):
    """Compute the L1 regularization term."""
    return np.sum(np.abs(x))
def h_fun(x):
    """Nonlinear equality constraint function."""
    return np.array([x[0]**2 + x[1] - 1,
                     np.sin(x[1]) + x[2] - 0.5])
def Dh_fun(x):
    """Jacobian matrix of nonlinear constraints h(x)."""
    return np.array([[2*x[0], 1, 0],
                     [0, np.cos(x[1]), 1]])
def objective(x, ls):
    """Compute full objective function."""
    return f_val(x) + ls*g_val(x)

# ==========================================================
# Compute consistent lambda* via KKT
# ==========================================================
def compute_multiplier_with_l1_subgrad(x_star, f_grad, h_fun, Dh_fun, ls):
    """
    Compute approximate Lagrange multipliers using KKT stationarity:
    0 ∈ grad f(x*) + ls * ∂||x*||_1 + Dh(x*)^T λ
    INPUT:
        x_star  = optimal primal variable -- array_like
        f_grad  = function handle for gradient of f(x)
        h_fun   = nonlinear constraint function
        Dh_fun  = Jacobian of constraint function
        ls      = sparsity coefficient -- float
    """
    grad_f = f_grad(x_star)
    Dh_x = Dh_fun(x_star)

    # Subgradient of L1 norm
    s = np.zeros_like(x_star)
    for i in range(len(x_star)):
        if x_star[i] > 1e-6:
            s[i] = 1.0
        elif x_star[i] < -1e-6:
            s[i] = -1.0
        else:
            s[i] = 0.0  # choose 0 inside [-1,1] for simplicity

    # Solve for λ:  Dh(x*)^T λ = - (grad_f + ls*s)
    rhs = -(grad_f + ls * s)
    lambd, *_ = np.linalg.lstsq(Dh_x.T, rhs, rcond=None)
    return lambd

# ==========================================================
# PI–PGD dynamics
# ==========================================================
def PI_PGD_l1(state, n, f_grad, h_fun, Dh_fun, ls, k_p, k_i, gamma):
    """Computes the PI-PGD dynamics for solving the optimization problem.
    INPUT:
        state  = Current state -- array_like
        n      = Dimension of primal variable x -- int
        f_grad = Gradient function of f(x)
        h_fun  = Nonlinear constraint function
        Dh_fun = Jacobian of h_fun
        ls     = Sparsity coefficient (L1 regularization weight) -- float
        k_p    = Proportional gain -- float
        k_i    = Integral gain -- float
        gamma  = Proximal parameter -- float
    """
    x, lambd = state[:n], state[n:]
    grad_f = f_grad(x)
    h_x = h_fun(x)
    Dh_x = Dh_fun(x)
    prox_arg = x - gamma * (grad_f + Dh_x.T @ lambd)
    prox_term = soft(prox_arg, gamma * ls)
    r = -x + prox_term
    dx_dt = r
    dlambda_dt = k_p * (Dh_x @ r) + k_i * h_x
    return np.concatenate([dx_dt, dlambda_dt])
# ==========================================================
# Simulation
# ==========================================================
#####
T = True
F = False
run_scipy = T      # set True to run CVXPy solve
save_scipy = F
run_dyn = T       # set True to run PI-PGD
save_dyn = F
run_log = F
save_log = F
save_fig = T
simulation_dyn_file = 'simulation_dyn_non_lin2.npz'
simulation_scipy_file = 'simulation_scipy_file_non_lin2.npz'
simulation_log_file = 'simulation_log_non_lin2.npz'

if __name__ == "__main__":
    #####
    # Parameters
    n, m = 3, 2
    rng = np.random.default_rng(2)
    state0 = 80*rng.random(n + m)
    k_p = 10#50
    k_i = 15#70
    ls = .5
    gamma = .5

    if run_scipy:
        constraints = [
            {'type': 'eq', 'fun': lambda x: x[0] ** 2 + x[1] - 1},
            {'type': 'eq', 'fun': lambda x: np.sin(x[1]) + x[2] - 0.5}
        ]
        x0 = np.array([0.5, 0.0, 0.0])
        # Pass ls using a lambda so that minimize() only sees x as input
        res = minimize(lambda x: objective(x, ls), x0, method='SLSQP', constraints=constraints)
        x_scipy = res.x
        print("Reference solution (SciPy):", x_scipy)
        print("Objective value:", res.fun)
        print("Success:", res.success)
        lamb_scipy = compute_multiplier_with_l1_subgrad(x_scipy, f_grad, h_fun, Dh_fun, ls)
        print("KKT multipliers:", lamb_scipy)
        if save_scipy:
            np.savez_compressed(simulation_scipy_file, x_scipy=x_scipy, lamb_scipy=lamb_scipy)
    else:
        try:
            data = np.load(simulation_scipy_file)
            x_scipy = data['x_scipy']
            lamb_scipy = data['lamb_scipy']
            print("Loaded scipy result from disk.")
        except Exception:
            x_scipy = None
            lamb_scipy = None
            print("No saved scipy data found. Set run_scipy=True to compute it.")

    t_start, t_end = 0, 10
    t_span = (t_start, t_end)
    t_eval = np.linspace(*t_span, 1000)
    if run_dyn:
        sol = solve_ivp(lambda t, y: PI_PGD_l1(y, n, f_grad, h_fun, Dh_fun, ls, k_p, k_i, gamma), t_span, state0,
                        t_eval=t_eval, method='RK45')
        if save_dyn:
            np.savez_compressed(simulation_dyn_file, solution_dyn=sol)
    else:
        try:
            load_dyn = np.load(simulation_dyn_file)
            sol = load_dyn['solution_dyn']
            print("PI-PGD data loaded successfully!")
        except Exception:
            raise FileNotFoundError(
                "Dynamics file not found. Set run_dyn=True to generate a dynamics solution "
                "or create a file named 'simulation_dyn.npz' with 'solution_dyn' saved."
            )
    x_sol = sol.y[:n, :]
    lambda_sol = sol.y[n:, :]
    h_sol = np.array([h_fun(x_sol[:, i]) for i in range(x_sol.shape[1])]).T

    x_final = x_sol[-1, :n]
    print(f"Error: ", abs(x_final - x_scipy))

    lambda_end = lambda_sol[:, -1]
    print("\nLambda at end of PI–PGD:", lambda_end)
    print("‖λ_end − λ*‖ =", np.linalg.norm(lambda_end - lamb_scipy))

    # ============================================================================
    #                                   PLOTS
    # ============================================================================
    # PLOT SETTINGS
    font = {'size': 16}
    plt.rc('font', **font)
    rc('text', usetex=True)
    rc('font', family='serif')

    colors_primal = plt.get_cmap('tab10').colors[:n]
    colors_dual = plt.get_cmap('tab10').colors[n:n + m]  # continue to avoid overlap
    colors_constraints = plt.get_cmap('tab10').colors[n + m: n + m + m]

    fig, ax = plt.subplots(3, 1, figsize=(7, 7))#(12, 8.5))
    for i in range(n):
        ax[0].plot(t_eval, x_sol[i, :], color=colors_primal[i], linewidth=1.5)
        ax[0].scatter(t_eval[-6], x_scipy[i], marker='x', color='r', label=rf'Optimal $x_{i+1}$', zorder=3)
    for i in range(m):
        ax[1].plot(t_eval, lambda_sol[i, :], color=colors_dual[i], linewidth=1.5)
        ax[1].scatter(t_eval[-6], lamb_scipy[i], marker='x', color='r', label=rf'Optimal $\lambda_{i+1}$', zorder=3)
    for i in range(m):
        ax[2].plot(t_eval, h_sol[i, :], color=colors_constraints[i], linestyle='--', label=rf'$h_{i+1}(t)$', linewidth=1.5)
    ax[0].set_xlim([t_start, t_end])
    ax[0].set_ylabel(r'$x(t)$', fontsize=16)
    ax[0].grid(True, alpha=1)
    ax[1].set_xlim([t_start, t_end])
    ax[1].set_ylabel(r'$\lambda(t)$', fontsize=16)
    ax[1].grid(True, alpha=1)
    ax[2].set_xlim([t_start, t_end])
    ax[2].set_ylabel(r'$h(x(t))$', fontsize=16)
    ax[2].grid(True, alpha=1)
    fig.text(0.53, 0, r'$t$', ha='center', fontsize=16)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'trajectories_nonlin2_arxiv_80.pdf', bbox_inches='tight')
    plt.show()

    ##########################
    if run_log:
        runs = 10
        log_norms = []
        # Weight matrix for weighted norm
        for i in range(runs):
            print(f"Run {i + 1}/{runs}")
            # Random initial states (scaled)
            x0_rand = 80 * np.random.rand(n)
            lambda0_rand = 80 * np.random.rand(m)
            state0 = np.concatenate([x0_rand, lambda0_rand])
            print('state0', state0)
            # Solve dynamics
            sol = solve_ivp(lambda t, y: PI_PGD_l1(y, n, f_grad, h_fun, Dh_fun, ls, k_p, k_i, gamma), t_span, state0,
                            t_eval=t_eval, method='RK45')
            # Compute log of weighted norms || z(t) - z* ||_P
            z_star = np.concatenate([x_scipy, lamb_scipy])
            diff = sol.y.T - z_star  # shape: (time_points, n+m)
            norm_diff = np.linalg.norm(diff.T, axis=0)  # weighted norm at each time
            log_norms.append(np.log(norm_diff))

        log_norms = np.array(log_norms)  # shape: (runs, time_points)
        if save_log:
            np.savez_compressed(simulation_log_file, log_norms=log_norms)
    else:
        # Load precomputed log norms
        data = np.load(simulation_log_file)
        log_norms = data['log_norms']

    # =============================
    # Cost function over time
    # =============================
    cost_sol = np.array([objective(x_sol[:, i], ls) for i in range(x_sol.shape[1])])
    plt.figure(figsize=(8, 3.5))
    plt.plot(t_eval, cost_sol, 'b', linewidth=1.5)
    plt.axhline(y=objective(x_scipy, ls), color='r', linestyle='--', label='Optimal cost')
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel('Cost', fontsize=16)
    plt.xlim(t_eval[0], t_eval[-1])
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    if save_fig:
        plt.savefig('cost_evolution_non_lin2_80.pdf', bbox_inches='tight')
    plt.show()

    # Different PI gain choices to test
    gain_pairs = [
        (4, 4),
        (10, 10),
        (15, 20),
        (20, 30),
        (30, 40),
        (40, 40)
    ]
    plt.figure(figsize=(8, 3.5))
    for k_p_val, k_i_val in gain_pairs:
        sol = solve_ivp(
            lambda t, y: PI_PGD_l1(y, n, f_grad, h_fun, Dh_fun, ls, k_p_val, k_i_val, gamma),
            t_span, state0, t_eval=t_eval, method='RK45'
        )
        z_sol = np.vstack([sol.y[:n, :], sol.y[n:, :]])
        z_star = np.concatenate([x_scipy, lamb_scipy])
        dist_to_opt = np.linalg.norm(z_sol - z_star[:, None], axis=0)
        dist_to_opt_log = np.log(dist_to_opt)
        plt.plot(t_eval, dist_to_opt_log, label=rf'$k_p={k_p_val}, k_i={k_i_val}$', linewidth=1.5)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r"$\log\left(\|z(t) - z^\star \|\right)$", fontsize=16)
    plt.xlim(t_eval[0], t_eval[-1])
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    if save_fig:
        plt.savefig('log_norm_diff_k_non_lin2_80.pdf', bbox_inches='tight')
    plt.show()