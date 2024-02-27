import numpy as np
from rng import rmvexp

def problem1b():
    cov_mat = np.genfromtxt("data/1b_matrix.csv", delimiter=',')
    mvexp_samples = rmvexp(200, cov_mat)
    print(f"Created samples: {mvexp_samples}")

if __name__ == "__main__":
    print("Solving problem 1b...")
    problem1b()