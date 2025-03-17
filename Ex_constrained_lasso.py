import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.linalg import eig

# ============================================================================
#                                  Functions
# ============================================================================
def generate_problem(n, m):
    """
    Generate a quadratic programming problem
    min (1/2) x^T W x 
    s.t. A x = b
    with normally distributed components.
    Ensures W is positive definite.
    """
    np.random.seed(0)
    W0 = np.random.randn(n, n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    W = 10 * np.eye(n) + W0 @ W0.T  # Positive definite matrix
    return W, A, b

# 1. Solve using CVXPY (Interior-Point Solver)
def solve_cvxpy(W, A, b, ls):
    """
    Solve the quadratic programming problem using CVXPY.
    min (1/2) x^T W x + ls ||x||_1
    s.t. A x = b
    Returns:
        Optimal solution (x) and dual variable of the equality constraint.
    """
    n = W.shape[0]
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, W) + ls*cp.norm1(x))
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    return x.value, constraints[0].dual_value


# 2. Solve using Dynamical System Method
def euler_step(dynamics, state, dt, *args):
    """
    Performs a single Euler step for ODE integration.
    
    INPUT:
        dynamics = function that defines the ODE
        state    = Current state -- array_like
        dt       = Time step size -- float
        args     = Any additional arguments to pass to the dynamics function
    
    """
    return state + dt * dynamics(state, *args)

def soft(x, thr):
    """Soft thresholding operator.
    INPUT:
          x = state -- array_like
        thr = threshold -- float
    """
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0)

def PI_PGD_l1(state, W, A, b, ls, k_p, k_i, gamma):
    n = W.shape[0]
    x = state[:n]
    lambd = state[n:]
    
    gradient_term = W @ x + A.T @ lambd
    prox_term = soft(x - gamma * gradient_term, gamma*ls)
    dx_dt = -x + prox_term
    dlambda_dt = (k_i - k_p) * (A @ x) + k_p * (A @ prox_term) - k_i * b
    return np.concatenate([dx_dt, dlambda_dt])

def PI_PGD_l1(state, W, A, b, ls, k_p, k_i, gamma):
    """Computes the PI-PGD dynamics for solving the optimization problem.
    
    INPUT:
        state = Current state -- array_like
        W     = positive definite matrix defing the quadratic cost -- array_like
        A     = matrix defining the constraint -- array_like
        b     = vector defining the constraint -- array_like
        ls    = sparsity coefficient -- float like
        k_p   = proportional parameter -- float like
        k_i   = integral parameter -- float like
        gamma = parameter prox -- float like
    
    """
    n = W.shape[0]
    x, lambd = state[:n], state[n:]
    gradient_term = W @ x + A.T @ lambd
    prox_term = soft(x - gamma * gradient_term, gamma * ls)
    dx_dt = -x + prox_term
    dlambda_dt = (k_i - k_p) * (A @ x) + k_p * (A @ prox_term) - k_i * b
    return np.concatenate([dx_dt, dlambda_dt])


def solve_dynamics(W, A, b, ls, k_p, k_i, gamma, dt, t_eval):
    """
    Solves the PI-PGD using Euler's method.
    
    INPUT:
        W      = positive definite matrix defing the quadratic cost -- array_like
        A      = matrix defining the constraint -- array_like
        b      = vector defining the constraint -- array_like
        ls     = sparsity coefficient -- float like
        k_p    = proportional parameter -- float like
        k_i    = integral parameter -- float like
        gamma  = parameter prox -- float like
        dt     = Time step size -- float like
        t_eval = Time interval simulation 
    
    """
    n, m = W.shape[0], A.shape[0]
    x0 = np.random.rand(n)
    lambda0 = np.random.rand(m)
    state = np.concatenate([x0, lambda0])
    solution = [state]
    start_time_dyn = time.perf_counter()
    for _ in range(len(t_eval)):
        state = euler_step(PI_PGD_l1, state, dt, W, A, b, ls, k_p, k_i, gamma)
        solution.append(state)
    time_sym_dyn = time.perf_counter() - start_time_dyn
    return np.array(solution), time_sym_dyn

def weighted_norm(vector, weight_matrix):
    """
    Compute the weighted norm of a vector.
    
    INPUT:
        vector        = The input vector -- array_like
        weight_matrix = The weighting matrix -- array_like

    """
    return np.linalg.norm(weight_matrix @ vector, axis=0, ord=2)

def check_condition(L, rho, k_p, p, gamma):
    """
    Checks whether the condition:
    4 * gamma^2 * p * K_p * rho - max{(gamma * p - K_p + gamma * K_p * rho)^2,
                                      (gamma * p - K_p + gamma * K_p * L)^2} > 0
    holds for given values of L, rho, K_p, and p and gamma.
    """
    term1 = 4 * (gamma**2) * p * k_p * rho
    penalty_max = max(
        (gamma * p - k_p + gamma * k_p * rho) ** 2,
        (gamma * p - k_p + gamma * k_p * L) ** 2
    )   
    result = term1 - penalty_max
    return result > 0, result
# ============================================================================
#                               Simulation
# ============================================================================
#####
T = True
F = False
run_cvxpy = F
save_cvxpy = T
run_dyn = F
save_dyn = F
run_log = F
save_log = F
save_fig = T
#####
# Parameters
n = 10
m = 5
W, A, b = generate_problem(n, m)
ls = 1

######   Find the optimal value using cvxpy
simulation_cvxpy = 'simulation_cvxpy.npz'
if run_cvxpy:
    start_time_cvxpy = time.perf_counter()
    x_cvxpy, lamb_cvxpy = solve_cvxpy(W, A, b, ls)
    time_sym_cvxpy = time.perf_counter() - start_time_cvxpy
    print(f"Time cvxpy: {time_sym_cvxpy:.6f} seconds")
    if save_cvxpy:
        np.savez_compressed(simulation_cvxpy, x_cvxpy=x_cvxpy, lamb_cvxpy=lamb_cvxpy)
else:
    load_cvxpy = np.load(simulation_cvxpy)
    x_cvxpy = load_cvxpy['x_cvxpy']
    lamb_cvxpy = load_cvxpy['lamb_cvxpy']
    print("cvxpy data loaded successfully!")

######   Find the optimal value using the PI-PGD
simulation_dyn = 'simulation_dyn.npz'

eigenvalues_W = np.linalg.eigvals(W)
eigenvalues_AAT = np.linalg.eigvals(A @ A.T)
L, rho = max(eigenvalues_W), min(eigenvalues_W)
amax, amin = max(eigenvalues_AAT), min(eigenvalues_AAT)
print("amin =", amin)
print("amax = ",amax)
print("L =", L)
print("rho = ",rho)
    
gamma = min(1/L, 4*rho/L**2 - 0.0001)
k_p = k_i = 20
p = k_p/gamma

is_condition_met, result_value = check_condition(L, rho, k_p, p, gamma)
print(f"Condition met: {is_condition_met}, Value: {result_value}")
#########
t_start, t_end, dt = 0, 30, 0.01
t_eval = np.arange(t_start, t_end, dt)

if run_dyn:
    solution_dyn, time_sym_dyn = solve_dynamics(W, A, b, ls, k_p, k_i, gamma, dt, t_eval)
    print(f"Time_dyn: {time_sym_dyn:.6f} seconds")
    x_dynamics = solution_dyn[-1, :n]
    print(f"Time_dyn: {time_sym_dyn:.6f} seconds")
    if save_dyn:
        np.savez_compressed(simulation_dyn, solution_dyn=solution_dyn)
else:
    load_dyn = np.load(simulation_dyn)
    solution_dyn = load_dyn['solution_dyn']
    print("PI-PGD data loaded successfully!")

#print(f"Error: ", abs(x_dynamics - x_cvxpy))
# ============================================================================
#                                   PLOTS
# ============================================================================
# PLOT SETTINGS
font = {'size': 16}
plt.rc('font', **font)
rc('text', usetex=True)
rc('font', family='serif')

fig, ax = plt.subplots(3, 1, figsize=(12, 8.5))
for i in range(n):
    ax[0].plot(t_eval, solution_dyn[1:, i], linewidth=1.5) #label=f'x1_{i+1}, p={p}', linewidth=1.5)
    ax[0].scatter(t_eval[-10], x_cvxpy[i], color='r', label=f'Optimal x_{i+1}', zorder=3)
for i in range(m):
    ax[1].plot(t_eval, solution_dyn[1:, n+i], linewidth=1.5) #label=f'x1_{i+1}, p={p}', linewidth=1.5)
    ax[1].scatter(t_eval[-10], lamb_cvxpy[i], color='r', label=f'Optimal x_{i+1}', zorder=3)
transformed_solution = (A @ solution_dyn[1:, :n].T).T
for i in range(m):
    ax[2].plot(t_eval, transformed_solution[:, i] - b[i], linestyle='--', linewidth=1.5)
ax[0].set_xlim([t_start, t_end])
ax[0].set_ylabel('Primal Variables')
ax[0].grid(True, alpha=1)
ax[1].set_xlim([t_start, t_end])
ax[1].set_ylabel('Dual Variables')
ax[1].grid(True, alpha=1)
ax[2].set_xlim([t_start, t_end])
ax[2].set_ylabel(r'$Ax(t) - b$')
ax[2].grid(True, alpha=1)
fig.text(0.53, 0, r'$t$', ha='center', fontsize=16)
plt.tight_layout()
if save_fig:
    plt.savefig(f'trajectories.pdf', bbox_inches='tight')
plt.show()


##########################
simulation_log = 'simulation_log.npz'
if run_log:
    log_norms = []
    P = np.block([[p * np.eye(n), np.zeros((n, m))], [np.zeros((m, n)), np.eye(m)]])
    for i in range(150):
        print(" i = ", i)
        x0 = 50*np.random.rand(n)
        lambda0 = 50*np.random.rand(m)
        state = []
        sol_dyn_rnd = []
        state = np.concatenate([x0, lambda0]) #+ np.random.normal(0, 1, n + m)
        sol_dyn_rnd = [state]
        for i in range(len(t_eval)):
            state = euler_step(PI_PGD_l1, state, dt, W, A, b, ls, k_p, k_i, gamma)
            sol_dyn_rnd.append(state)
        sol_dyn_rnd = np.array(sol_dyn_rnd)
        vector_rnd = sol_dyn_rnd - np.concatenate([x_cvxpy, lamb_cvxpy])
        norm_rnd = weighted_norm(vector_rnd.T, P)
        log_norm_rnd = np.log(norm_rnd)
        log_norms.append(log_norm_rnd)
    if save_log:
        np.savez_compressed(simulation_log, log_norms=log_norms)
else:
    load_log = np.load(simulation_log)
    log_norms = load_log['log_norms']
    
# Compute mean and confidence bounds
log_norms = np.array(log_norms)
mean_log_norm = np.mean(log_norms, axis=0)
std_dev_log_norm = np.std(log_norms, axis=0)
upper_bound = mean_log_norm + std_dev_log_norm
lower_bound = mean_log_norm - std_dev_log_norm

plt.figure(figsize=(8, 3.5))
#plt.rcParams['text.usetex'] = True
plt.plot(t_eval, mean_log_norm[1:], 'r', linewidth = 1.5, label='Mean')
plt.fill_between(t_eval, lower_bound[1:], upper_bound[1:], color='r', alpha=0.2, label='Std Deviation')
plt.xlabel(r'$t$', fontsize=19)
plt.ylabel(r"$\log\left(\|z(t) - z^\star \|_{P}\right)$", fontsize=19)
plt.xlim(left=0, right=t_end)
plt.grid(True, alpha=1)
plt.legend()
if save_fig:
    plt.savefig(f'log_norm_diff.pdf', bbox_inches='tight')
plt.show()
