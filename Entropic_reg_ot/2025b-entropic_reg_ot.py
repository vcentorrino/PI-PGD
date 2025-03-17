import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np
import ot
import math
from tqdm import tqdm
from tempfile import TemporaryDirectory
import imageio
import scipy.sparse as sparse
from sklearn.datasets import fetch_openml
from matplotlib import rc


# PLOT SETTINGS
font = {'size': 16}
plt.rc('font', **font)
rc('text', usetex=True)
rc('font', family='serif')
plt.rcParams['axes.facecolor'] = 'white'


# ============================================================================
#                                  Functions
# ============================================================================
#IMAGE
def convert_to_black_probabilities(im):
    flattened_array = im.reshape(-1)   # 0 is black, 255 is white
    blackness = 255 - flattened_array  # 0 is white, 255 is black
    blackness = blackness ** 3.25
    return blackness / blackness.sum()


def convert_pic_to_scatter_plot(black_probabilities, initial_shape, n_points=10000):
    points = np.random.choice(black_probabilities.shape[0], replace=True, size=(n_points,), p=black_probabilities)
    p1, p2 = np.unravel_index(points, initial_shape)
    rand = np.random.random((2, n_points))
    return p1 + rand[0], p2 + rand[1]


def display_scatter(source, show=True, size=None):
    x, y = source[..., 0], source[..., 1]
    if size is None:
        size = (y.max() - y.min(), x.max() - x.min())
    n_points = x.shape[0]
    plt.scatter(y, -x, s=4000. / n_points, c='k', marker=".", linewidths=2)
    plt.xlim(0, size[0] + 1)
    plt.ylim(-size[1], 0)
    plt.axis('off')
    if show:
        plt.show()


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def im2dots(original_img, n_points, size):
    pil_image = Image.new("RGBA", original_img.size, "BLACK")
    pil_image.paste(original_img, (0, 0), original_img)
    if size is not None:
        pil_image = pil_image.resize(size)
    pil_image = pil_image.convert(mode="L")
    pil_image = change_contrast(pil_image, 70)
    im = np.array(pil_image)
    probabilities = convert_to_black_probabilities(im)
    x, y = convert_pic_to_scatter_plot(probabilities, im.shape, n_points)
    return np.stack([x, y], axis=-1)


#####################
def get_points_at_t(xs, xt, G, t):
    # Compute the intermediate point positions given time t
    return (1 - t) * xs + t * G @ xt


def make_time_dimension(n_frames):
    if n_frames % 2 == 0:
        print("Adding a frame for symmetry purposes")
        n_frames += 1
    half = np.linspace(0, 1, math.ceil(n_frames / 2))
    return np.concatenate([half[:-1], np.flip(half)])

def make_gif(xs, xt, G, n_frames, gif_name):
    with TemporaryDirectory() as tmpdirname:
        time_dim = make_time_dimension(n_frames)
        n = len(time_dim)
        #fig = plt.figure(figsize=(6.4, 9.6))
        fig = plt.figure(figsize=(6.4, 8.5))

        def make_frame(index, xs, xt, G):
            t = time_dim[index]
            points = get_points_at_t(xs, xt, G, t)
            display_scatter(points, show=False, size=size)
            plt.savefig(f"{tmpdirname}/{index}.png", transparent=False, bbox_inches='tight')
            if index == 0:
                plt.savefig(f"start_{gif_name}.png", transparent=False, bbox_inches='tight')
            if index == (n - 1) / 2/2:
                plt.savefig(f"middle_{gif_name}.png", transparent=False, bbox_inches='tight')
            if index == (n - 1)/2:
                plt.savefig(f"final_{gif_name}.png", transparent=False, bbox_inches='tight')
            plt.clf()

        for index in tqdm(range(n), desc="Making pictures"):
            make_frame(index, xs, xt, G)

        plt.close()
        plt.cla()
        plt.clf()

        with imageio.get_writer(f"{gif_name}.gif", mode="I", fps=30) as writer:
            for i in tqdm(range(n), desc="Merging pictures"):
                filename = f"{tmpdirname}/{i}.png"
                image = imageio.v2.imread(filename)
                #image = imageio.imread(filename)
                writer.append_data(image)

            print("The merger is complete")
            writer.close()


################ PI-PGD
def construct_A(n, m):
    """Constructs the constraint matrix A for the vectorized optimal transport problem."""
    ones_m = np.ones((1, m))
    ones_n = np.ones((1, n))
    A = np.vstack([
        np.kron(ones_m, np.eye(n)),
        np.kron(np.eye(m), ones_n)])
    return A


def pi_pgd_dyn(state, grad_f, A, b_eq, K_p, K_i, gamma):
    """
    Computes the PI-proximal dynamics
    :param state: Current state vector (x, lambda)
    :param grad_f: Gradient of the cost function
    :param A: Constraint matrix
    :param b: Constraint vector
    :param K_p: Proportional gain
    :param K_i: Integral gain
    :param gamma: Step size parameter
    :return: Time derivative of state
    """
    mn = A.shape[1]
    p = state[:mn]
    lambd = state[mn:]
    prox_term = np.maximum(0, p - gamma * (grad_f + A.T @ lambd))
    dp_dt = -p + prox_term
    dlambda_dt = K_p * (A @ dp_dt) + K_i * (A @ p - b_eq)

    return np.concatenate((dp_dt, dlambda_dt))

def run_ot_pi_pgd_dyn(A, b_eq, C, epsilon, state_0, K_p, K_i, gamma, max_iter, dt=0.01, tol=1e-6):
    """
    Solves the entropic regularized optimal transport problem in vectorized form using projected gradient descent.
    :param a: Source distribution (n,)
    :param b: Target distribution (m,)
    :param C: Cost matrix (n, m)
    :param epsilon: Entropy regularization parameter
    :param max_iter: Number of iterations
    :param lr: Learning rate for gradient descent
    :return: Optimal transport plan in vectorized form (mn,)
    """
    mn = C.shape[0]

    # Euler step simulation
    state = state_0
    #solution = [state_0]
    p_solution_dyn = [state_0[:m * n]]
    for _ in range(max_iter):
        grad_f = C + epsilon * (1 + np.log(np.clip(state[:m * n], 1e-10, None)))
        state_new = state + dt * pi_pgd_dyn(state, grad_f, A, b_eq, K_p, K_i, gamma)

        if np.linalg.norm(state_new - state) < tol:
            print("The PI_PGD converged")
            break
        state = state_new
        p_solution_dyn.append(state[:m * n])

    p = state[:mn]
    lambd = state[mn:]
    print("p = ", p.shape)
    print("lambda = ", lambd.shape)
    return p, lambd, p_solution_dyn


def save_to_npz(filename, **kwargs):
    """ Append new data to an existing .npz file or create a new one. """
    try:
        existing_data = dict(np.load(filename, allow_pickle=True))  # Fix: allow_pickle=True
    except FileNotFoundError:
        existing_data = {}  # If file doesn't exist, start fresh
    
    existing_data.update(kwargs)  # Add new data
    np.savez_compressed(filename, **existing_data)  # Save everything back


# ============================================================================
#                               Simulation
# ============================================================================

run_mnist = False
run_image = False
run_sinkhorn = False
run_dyn = False
save_fig = True

simulation_minst = 'mnist_data_set.npz'
simulation_OT = 'simulation_OT.npz'

n_points = 100
size = (28, 28)
if run_mnist:
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    images = (255 - mnist.data.astype(np.float32)) / 255.0  # Normalize pixel values to [0, 1]
    labels = mnist.target.astype(np.int64)
    image1_idx = np.where(labels == 1)[0][0]
    image2_idx = np.where(labels == 2)[0][0]
    image3_idx = np.where(labels == 3)[0][0]
    image4_idx = np.where(labels == 4)[0][0]
    image5_idx = np.where(labels == 5)[0][0]
    image6_idx = np.where(labels == 6)[0][0]
    image7_idx = np.where(labels == 7)[0][0]
    image8_idx = np.where(labels == 8)[0][0]
    image9_idx = np.where(labels == 9)[0][0]

    # Reshape the flat image data to 28x28
    image1_array = images.iloc[image1_idx].values.reshape(28, 28)
    image2_array = images.iloc[image2_idx].values.reshape(28, 28)
    image3_array = images.iloc[image3_idx].values.reshape(28, 28)
    image4_array = images.iloc[image4_idx].values.reshape(28, 28)
    image5_array = images.iloc[image5_idx].values.reshape(28, 28)
    image6_array = images.iloc[image6_idx].values.reshape(28, 28)
    image7_array = images.iloc[image7_idx].values.reshape(28, 28)
    image8_array = images.iloc[image8_idx].values.reshape(28, 28)
    image9_array = images.iloc[image9_idx].values.reshape(28, 28)
    np.savez_compressed(simulation_minst,
                        image1_array = image1_array,
                        image2_array = image2_array,
                        image3_array = image3_array,
                        image4_array = image4_array,
                        image5_array = image5_array,
                        image6_array = image6_array,
                        image7_array = image7_array,
                        image8_array = image8_array,
                        image9_array = image9_array)


# PICK SOURCE AND TARGET IMAGES
# Convert NumPy arrays to PIL Images
if run_image:
    load_minst = np.load(simulation_minst)
    source = load_minst['image4_array']
    target = load_minst['image1_array']
    image_source = Image.fromarray((source * 255).astype(np.uint8)).convert("RGBA")
    image_target = Image.fromarray((target * 255).astype(np.uint8)).convert("RGBA")

    xs = im2dots(image_source, n_points, size)
    xt = im2dots(image_target, n_points, size)
    save_to_npz(simulation_OT, xs=xs, xt=xt)
    print("Image done!")
else:
    load_OT = np.load(simulation_OT, allow_pickle=True)
    if 'xs' in load_OT and 'xt' in load_OT:
        xs = load_OT['xs']
        xt = load_OT['xt']
        print("Image loaded successfully!")
    else:
        print("Warning: xs and xt not found in simulation_OT.npz!")


# Optimal transport parameters
epsilon = 0.001
C = ot.dist(xs, xt, metric='sqeuclidean')
n = m = n_points
a = np.ones(n_points)
a /= a.sum()
b = np.ones(n_points)
b /= b.sum()
K_p = 100
K_i = 100
gamma = 0.01#0.1
C_dyn = C.flatten(order='F')
A_dyn = construct_A(n, m)
b_dyn = np.concatenate((a, b))

max_it = 500000
if run_sinkhorn:
    start_time = time.perf_counter()
    P_sinkhorn = ot.sinkhorn(a, b, C, epsilon, numItermax=max_it) #, log=True)
    print("P_sinkhorn sum: ", P_sinkhorn.sum())
    end_time = time.perf_counter()
    sinkhorn_time = end_time - start_time

    save_to_npz(simulation_OT, P_sinkhorn=P_sinkhorn, sinkhorn_time=sinkhorn_time)
    print(f"Sinkhorn computation time: {sinkhorn_time:.6f} seconds")
else:
    load_OT = np.load(simulation_OT)
    if 'P_sinkhorn' in load_OT:
        P_sinkhorn = load_OT['P_sinkhorn']
        print("Sinkhorn data loaded successfully!")
    else:
        print("Warning: P_sinkhorn not found!")


if run_dyn:
    # Initial conditions
    np.random.seed(42)
    x0 = np.random.rand(n * m) + 0.01
    x0 /= x0.sum()

    A_dyn_reduced = A_dyn[:-1]
    b_dyn_reduced = np.concatenate((a, b[:-1]))  # Remove the last element (redundant constraint)
    # Initial conditions
    np.random.seed(331)
    lambda0_reduced = np.random.rand(A_dyn_reduced.shape[0])
    state0_reduced = np.concatenate((x0, lambda0_reduced))
    start_time_reduced = time.perf_counter()
    p_dyn_reduced, lambda_dyn_reduced, p_solution_dyn = run_ot_pi_pgd_dyn(A_dyn_reduced, b_dyn_reduced, C_dyn, epsilon, state0_reduced, K_p, K_i, gamma, max_it)
    end_time_reduced = time.perf_counter()
    dyn_time_reduced = end_time_reduced - start_time_reduced
    print(f"Dynamics reduced computation time: {dyn_time_reduced:.6f} seconds")

    P_dynamics_reduced = p_dyn_reduced.reshape(n, m, order='F')

    save_to_npz(simulation_OT, P_dynamics_reduced=P_dynamics_reduced, dyn_time_reduced=dyn_time_reduced, lambda_dyn_reduced=lambda_dyn_reduced, p_solution_dyn=p_solution_dyn)

else:
    load_OT = np.load(simulation_OT)

    if 'P_dynamics_reduced' in load_OT:
        P_dynamics_reduced = load_OT['P_dynamics_reduced']
        p_solution_dyn = load_OT['p_solution_dyn']
    else:
        print("Warning: P_dynamics_reduced not found!")

error_matrix = np.abs(P_dynamics_reduced - P_sinkhorn)
# ============================================================================
#                                   PLOTS
# ============================================================================
##########  GIF  ##########
G_dynamics_reduced = P_dynamics_reduced * n_points
make_gif(xs, xt, G_dynamics_reduced, 500, 'pi_pgd')


G_sinkhorn = P_sinkhorn * n_points
make_gif(xs, xt, G_sinkhorn, 500, 'sinkhorn')


##########################

fig1, axes1 = plt.subplots(1, 3, figsize=(12, 5))
axes1[0].imshow(P_dynamics_reduced, cmap='Blues')
axes1[0].set_title("PI-Prox")
axes1[0].set_xlabel("Target")
axes1[0].set_ylabel("Source")
axes1[1].imshow(P_sinkhorn, cmap='Blues')
axes1[1].set_title("Sinkhorn")
axes1[1].set_xlabel("Target")
axes1[1].set_ylabel("Source")
axes1[2].imshow(error_matrix, cmap='Reds', aspect='auto')
axes1[2].set_title("Error")
axes1[2].set_xlabel("Target")
axes1[2].set_ylabel("Source")
plt.colorbar(axes1[0].imshow(P_dynamics_reduced, cmap='Blues'), ax=axes1[0], shrink=0.6)
plt.colorbar(axes1[1].imshow(P_sinkhorn, cmap='Blues'), ax=axes1[1], shrink=0.6)
plt.colorbar(axes1[2].imshow(error_matrix, cmap='Reds'), ax=axes1[2], shrink=0.6)
plt.tight_layout()
if save_fig:
    plt.savefig(f"Error_pipgd_sinkhorn_eps%.3f.png" % epsilon, transparent=False, bbox_inches='tight')


plt.figure(figsize=(7, 7))
plt.imshow(P_dynamics_reduced, cmap='Blues')
plt.title("P", fontsize=20)
plt.xlabel("Target", fontsize=18)
plt.ylabel("Source", fontsize=18)
plt.colorbar(plt.imshow(P_dynamics_reduced, cmap='Blues'), shrink=0.8)
plt.tight_layout()
if save_fig:
    plt.savefig(f"OT_P_pipgd.png", transparent=False, bbox_inches='tight')

# Plot constraint error
norm_constraint_err = np.zeros(len(p_solution_dyn))
for i in range(len(p_solution_dyn)):
    norm_constraint_err[i] = np.linalg.norm(A_dyn@p_solution_dyn[i] - b_dyn)
plt.figure(figsize=(8, 5))
plt.plot(np.array(norm_constraint_err))
plt.ylabel(r"$\|Ap - d\|$", fontsize=20)
plt.xlim([0, len(p_solution_dyn)])
plt.xlabel("Iteration", fontsize=20)
plt.grid()
plt.tight_layout()
if save_fig:
    plt.savefig(f"OT_norm_error.png", transparent=False, bbox_inches='tight')

# Plot sum to 1 error
sum1_err = np.zeros(len(p_solution_dyn))
for i in range(len(p_solution_dyn)):
    sum1_err[i] = np.linalg.norm(np.abs(np.sum(p_solution_dyn[i])-1))
plt.figure(figsize=(8, 5))
plt.plot(np.array(sum1_err))
plt.ylabel(r"$|1_{nm}^T p - 1|$", fontsize=20)
plt.xlim([0, len(p_solution_dyn)])
plt.xlabel("Iteration", fontsize=20)
plt.grid()
plt.tight_layout()
if save_fig:
    plt.savefig(f"OT_sum1_error.png", transparent=False, bbox_inches='tight')


plt.figure(figsize=(7, 6))
err_con = np.abs(A_dyn@(P_dynamics_reduced.flatten(order='F')) - b_dyn)
plt.scatter(np.linspace(0, n+m-1, n+m), err_con)
plt.ylabel("$|Ap_i - d|$")
plt.xlabel("i")
plt.xlim(0, n+m)
plt.grid()
plt.tight_layout()
if save_fig:
    plt.savefig(f"Error_constraints_pipgd_eps%.3f.png" % epsilon, transparent=False, bbox_inches='tight')

plt.show()


