import argparse
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
"""
SVM with perturbed images of cats and dogs
lables: -1 -> cats, 1 -> dogs

"""
## Solver options for the optimization problem
solver_opts = {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 10000}

def svm_loss_func(x, y, w, b):
    return max(0.0, 1.0 - y * (np.dot(w, x) + b))


def load_images(dimension=64, load_in_gray=True, max_per_class=None):
    path = "train"
    images = []
    labels = []
    class_counts = {'cat': 0, 'dog': 0}
    
    for filename in tqdm(sorted(os.listdir(path)), desc="Loading images"):
        if filename.endswith('.jpg'):
            # Check if we've reached the limit for this class
            if max_per_class is not None:
                class_type = 'cat' if 'cat' in filename else 'dog'
                if class_counts[class_type] >= max_per_class:
                    continue
            
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = preprocess_image_loading(img_array, target_size=(dimension, dimension), to_gray=load_in_gray)
            images.append(img_array)
            label = -1 if 'cat' in filename else 1
            labels.append(label)
            
            # Update class count
            if max_per_class is not None:
                class_counts[class_type] += 1
    
    df = pd.DataFrame({'image': images, 'label': labels})
    return df

# Resize the images to 64 x 64 and flatten
def preprocess_image_loading(img, target_size=(64,64), to_gray=True):
    img = Image.fromarray(img)
    if to_gray:
        img = img.convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
    return img_array


def solve_svm_nominal(x, y, lambda_reg):
    n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    zeta = cp.Variable(n)
    ## Define the constraints
    constraints = []
    constraints += [zeta >= 0, cp.multiply(y, x @ w - b) >= 1 - zeta]
    
    ## Define the objective
    objective = cp.Minimize(lambda_reg * cp.sum_squares(w) + (1.0 / n) * cp.sum(zeta))
    
    ## Define the problem
    prob = cp.Problem(objective, constraints)
    
    prob.solve(solver=cp.OSQP, **solver_opts)
    
    w_val = w.value
    b_val = b.value
    zeta_val = zeta.value
    obj_val = prob.value
    print("Solver status (nominal):", prob.status)
    return w_val, b_val, zeta_val, obj_val

def solve_svm_cvar(x, y, lambda_reg, alpha, N):
    _ , n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    zeta = cp.Variable((n,N))
    u = cp.Variable(N)
    t = cp.Variable()
    
    ## Define the objective
    obj = t + (1.0 / ((1.0 - alpha) * float(N))) * cp.sum(u) + lambda_reg * cp.sum_squares(w)
    objective = cp.Minimize(obj)
    
    ## Define the constraints
    
    constraints = []
    
    ## Constraints for each pertubed scenario
    ## For scenario k: zeta[:,k] >= 1 - y * (Xk @ w - b); zeta[:,k] >= 0

    for k in range(N):
        x_k = x[k,:,:]
        constraints += [zeta[:, k] >= 0]
        constraints += [cp.multiply(y, x_k @ w - b) >= 1 - zeta[:, k]]
    
    ## Constraints for u
    for k in range(N):
        constraints += [u[k] >= 0]
        constraints += [u[k] >= ((1.0/n) * cp.sum(zeta[:, k]) - t)]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, **solver_opts)
    
    print("Solver status (CVaR):", problem.status)
    
    return w.value, b.value, problem.value, t.value

def calc_metrics(x, y, w, b):
    scores = x @ w + b
    y_pred = np.where(scores >= 0, 1, -1)
    accuracy = accuracy_score(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)
    return accuracy, confusion_mat

def evaluate_losses(x, y, w, b,  M):
    """Evaluate hinge losses for test samples."""
    losses = []
    for k in range(M):
        loss = np.array([svm_loss_func(x[k,i], y[i], w, b) for i in range(x.shape[1])])
        losses.append(loss.mean())
        
    return np.array(losses)

def cvar_empirical(values, alpha):
    """
    Empirical CVaR_alpha for a 1D array of loss values:
      CVaR = mean of the upper (1-alpha) tail (>= VaR_alpha).
    """
    values = np.asarray(values, dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if values.size == 0:
        return np.nan
    # Empirical VaR at level alpha
    var_alpha = np.quantile(values, alpha)
    tail = values[values >= var_alpha]
    return float(tail.mean())


def main():
    parser = argparse.ArgumentParser(description="SVM on Perturbed Images: nominal vs CVaR-optimized comparison")
    parser.add_argument('--alpha', type=float, default=0.95, help="CVaR level in (0,1).")
    parser.add_argument('--mode', choices=['compare','nominal','cvar'], default='compare',
                        help="Run nominal only, CVaR only, or both and compare (default).")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")    
    parser.add_argument('--lambda_reg', type=float, default=1e-3, help="Regularization parameter.")
    parser.add_argument('--dimension', type=int, default=64, help="Dimension of flattened images.")
    parser.add_argument('--grayscale', type=bool, default=True, help="Convert images to grayscale.")
    parser.add_argument('--n', type=int, default=300, help="Number of images to consider per class.")
    parser.add_argument('--N', type=int, default=10, help="Number of scenarios for training (optimization).")
    parser.add_argument('--M', type=int, default=100, help="Number of scenarios for evaluation (Test CVaR).")
    parser.add_argument('--sigma', type=float, default=0.1, help="Standard deviation of image noise.")
    
    args = parser.parse_args()
    ## Load the images and preprocess
    df = load_images(dimension=args.dimension, load_in_gray=args.grayscale, max_per_class=args.n)
    x = np.array(df['image'].tolist())  # Convert DataFrame column to 2D array
    y = df['label'].values
    np.random.seed(args.seed)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=args.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    ## Generate noisy pertubations of training and test data
    x_train_noisy = np.zeros((args.N, x_train.shape[0], x_train.shape[1]))
    ## Generate N different pertubations of training data
    for k in range(args.N):
        x_train_noisy[k,:,:] = x_train + np.random.normal(0, args.sigma, size=x_train.shape)
    
    x_test_noisy = np.zeros((args.M, x_test.shape[0], x_test.shape[1]))
    for k in range(args.M):
        x_test_noisy[k, :, :] = x_test + np.random.normal(0, args.sigma, size=x_test.shape)
    
    ## Solve nominal SVM
    if args.mode in ('compare','nominal'):
        w_nom, b_nom, zeta_nom, obj_nom = solve_svm_nominal(x_train, y_train, lambda_reg=args.lambda_reg)
        losses_nom = evaluate_losses(x_test_noisy, y_test, w_nom, b_nom, M=args.M)
        cvar_nom = cvar_empirical(losses_nom, args.alpha)
        mean_nom = float(losses_nom.mean())
        print("Nominal SVM Results:")
        print("  Value of objective at optimum = ", obj_nom)
        print(f"Test Set Results(test_N = {x_test.shape[0]}):")
        print(f"Mean hinge loss (test) = {mean_nom:.6f}")
        print(f"CVaR@alpha hinge loss (test) = {cvar_nom:.6f}")
        
        # train_acc_nom, train_conf_mat_nom = calc_metrics(x_train, y_train, w_nom, b_nom)
        test_acc_nom, test_conf_mat_nom = calc_metrics(x_test, y_test, w_nom, b_nom)
        # print(f"Training Accuracy: {train_acc_nom*100:.2f}%")
        # print("Training Confusion Matrix:")
        # print(train_conf_mat_nom)
        print(f"Test Accuracy: {test_acc_nom*100:.2f}%")
        print("Test Confusion Matrix:")
        print(test_conf_mat_nom)
    if args.mode in ('compare', 'cvar'):
        w_cvar, b_cvar, obj_cvar, var_cvar  = solve_svm_cvar(x_train_noisy, y_train, lambda_reg=args.lambda_reg, alpha=args.alpha, N=args.N)
        losses_cvar = evaluate_losses(x_test_noisy, y_test, w_cvar, b_cvar, M=args.M)
        cvar_cvar = cvar_empirical(losses_cvar, args.alpha)
        mean_cvar = float(losses_cvar.mean())
        print("CVaR-optimized SVM Results:")
        print("  Value of objective at optimum = ", obj_cvar)
        print(f"Test Set Results(test_N = {x_test.shape[0]}):")
        print(f"Mean hinge loss (test) = {mean_cvar:.6f}")
        print(f"CVaR@alpha hinge loss (test) = {cvar_cvar:.6f}")
        test_acc_cvar, test_conf_mat_cvar = calc_metrics(x_test, y_test, w_cvar, b_cvar)
        print(f"Test Accuracy: {test_acc_cvar*100:.2f}%")
        print("Test Confusion Matrix:")
        print(test_conf_mat_cvar)
    
    if args.mode == 'compare':
        print("Comparison (Test Set): CVaR@alpha=%.2f" % args.alpha)
        print("  CVaR(cvar)  vs  CVaR(nominal)  -->  %.6f  vs  %.6f" % (cvar_cvar, cvar_nom))
        print("  Mean(cvar)  vs  Mean(nominal)  -->  %.6f  vs  %.6f" % (mean_cvar, mean_nom))

if __name__ == "__main__":
    main()
