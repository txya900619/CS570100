# you must use python 3.10
# For linux, you must use download HomeworkFramework.cpython-310-x86_64-linux-gnu.so
# For Mac, you must use download HomeworkFramework.cpython-310-darwin.so
# If above can not work, you can use colab and download HomeworkFramework.cpython-310-x86_64-linux-gnu.so and don't forget to modify output's name.


# Environment:
# OS: Ubuntu 22.04.4 LTS
# CPU: Intel(R) Xeon(R) Gold 6426Y
# Python version: 3.10.16
# Numpy version: 2.2.5

import numpy as np
from HomeworkFramework import Function


class CMA_ES_optimizer(Function):  # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func_num = target_func  # Store func_num for evaluate

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        # CMA-ES specific parameters
        self.m = np.random.uniform(self.lower, self.upper, self.dim)  # Initial mean

        # Improved sigma initialization
        if isinstance(self.upper, (int, float)) and isinstance(
            self.lower, (int, float)
        ):
            self.sigma = (
                self.upper - self.lower
            ) * 0.2  # 0.2 for better initial exploration
        else:
            self.sigma = np.mean(self.upper - self.lower) * 0.2
            if self.sigma <= 0:
                self.sigma = 1.0

        # Optimized population size
        self.lambd = 4 + int(3 * np.log(self.dim))  # Population size
        self.mu = self.lambd // 2  # Number of parents/points for recombination

        # Pre-compute weights and mueff
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)  # More efficient calculation

        # Optimized learning rates
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff),
        )
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        # Initialize arrays with zeros
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)
        self.invsqrtC = np.eye(self.dim)

        # Pre-allocate arrays for better performance
        self.arz = np.zeros((self.lambd, self.dim))
        self.arx = np.zeros((self.lambd, self.dim))
        self.arfitness = np.zeros(self.lambd)

        self.eigeneval = 0
        self.chiN = self.dim**0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES):  # main part for your implementation
        count_eigen = 0
        min_sigma = 1e-12  # Minimum sigma value
        max_sigma = np.mean(self.upper - self.lower) * 0.5  # Maximum sigma value

        while self.eval_times < FES:
            print("=====================FE=====================")
            print(f"{self.eval_times}/{FES}")

            # --- Generate and evaluate lambda offspring ---
            for k in range(self.lambd):
                if self.eval_times >= FES:
                    break

                # Sample from multivariate normal distribution
                self.arz[k] = np.random.randn(self.dim)
                self.arx[k] = self.m + self.sigma * np.dot(
                    self.B, self.D * self.arz[k]
                )  # More efficient matrix multiplication
                self.arx[k] = np.clip(self.arx[k], self.lower, self.upper)

                value = self.f.evaluate(self.target_func_num, self.arx[k])
                self.eval_times += 1

                if isinstance(value, str) and value == "ReachFunctionLimit":
                    print("ReachFunctionLimit")
                    self.eval_times = FES
                    break

                self.arfitness[k] = float(value)

                if self.arfitness[k] < self.optimal_value:
                    self.optimal_solution[:] = self.arx[k]
                    self.optimal_value = self.arfitness[k]

            if self.eval_times >= FES:
                break

            # --- Sort and update ---
            arindex = np.argsort(self.arfitness)
            old_m = self.m.copy()
            best_arz = self.arz[arindex[: self.mu]]

            # Update mean with vectorized operations
            self.m = old_m + self.sigma * np.dot(
                self.B, self.D * np.dot(self.weights, best_arz)
            )

            # --- Step-size control ---
            # Update evolution paths with vectorized operations
            y = (self.m - old_m) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(
                self.cs * (2 - self.cs) * self.mueff
            ) * np.dot(self.invsqrtC, y)

            # Compute hsig more efficiently
            hsig = np.linalg.norm(self.ps) / np.sqrt(
                1 - (1 - self.cs) ** (2 * (self.eval_times / self.lambd))
            ) / self.chiN < 1.4 + 2 / (self.dim + 1)

            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
                self.cc * (2 - self.cc) * self.mueff
            ) * y

            # Adapt step-size sigma with bounds
            self.sigma *= np.exp(
                (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
            )
            self.sigma = np.clip(self.sigma, min_sigma, max_sigma)

            # --- Covariance matrix adaptation ---
            artmp = (1 / self.sigma) * (self.arx[arindex[: self.mu]] - old_m)

            # Rank-mu update with vectorized operations
            C_rank_mu_update = np.sum(
                [w * np.outer(y, y) for w, y in zip(self.weights, artmp)], axis=0
            )

            # Update covariance matrix
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1
                * (
                    np.outer(self.pc, self.pc)
                    + (1 - hsig) * self.cc * (2 - self.cc) * self.C
                )
                + self.cmu * C_rank_mu_update
            )

            # Enforce symmetry
            self.C = (self.C + self.C.T) / 2

            # Eigendecomposition of C with improved frequency
            if (self.eval_times - self.eigeneval) > self.lambd / (
                self.c1 + self.cmu
            ) / self.dim / 5:  # Increased update frequency
                self.eigeneval = self.eval_times
                count_eigen += 1
                try:
                    eigenvalues, self.B = np.linalg.eigh(self.C)
                    self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))
                    self.invsqrtC = np.dot(self.B, np.diag(1 / self.D) @ self.B.T)
                except np.linalg.LinAlgError:
                    print(
                        "Warning: LinAlgError in eigendecomposition. Resetting parameters."
                    )
                    self.C = np.eye(self.dim)
                    self.B = np.eye(self.dim)
                    self.D = np.ones(self.dim)
                    self.invsqrtC = np.eye(self.dim)

            # Check for numerical stability
            if (
                np.isnan(self.m).any()
                or np.isinf(self.m).any()
                or np.isnan(self.C).any()
                or np.isinf(self.C).any()
            ):
                print("Error: Numerical instability detected. Resetting parameters.")
                self.C = np.eye(self.dim)
                self.B = np.eye(self.dim)
                self.D = np.ones(self.dim)
                self.invsqrtC = np.eye(self.dim)
                self.sigma = max_sigma * 0.5
                self.pc = np.zeros(self.dim)
                self.ps = np.zeros(self.dim)

            print(
                f"Current best: {self.optimal_value:.4e}, Sigma: {self.sigma:.2e}, Evals: {self.eval_times}, EigenUpdates: {count_eigen}"
            )


if __name__ == "__main__":
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500

    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:  # func_num == 4
            fes = 2500

        print(f"\nOptimizing Function {func_num} with FES = {fes}")
        # you should implement your optimizer
        op = CMA_ES_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open(
            "{}_function{}.txt".format(__file__.split("_")[0], func_num), "w+"
        ) as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
