import numpy as np
import matplotlib.pyplot as plt

# Parameters for Futurity slot machine
pA = 0.4  # Winning probability for Arm A
pB = 0.6  # Winning probability for Arm B
J = 5     # Futurity threshold for refund
M = 10000  # Number of plays per simulation
repetitions = 1000  # Number of Monte Carlo simulations

# Define strategies
strategies = {
    "AB": [0, 1],
    "AABB": [0, 0, 1, 1],
    "AAABB": [0, 0, 0, 1, 1],
    "AAAABBBBAAAAAABBB": [0]*4 + [1]*4 + [0]*5 + [1]*3,
    "ABABABABABABABAB": [0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    "AAAAAABBBBBB": [0]*6 + [1]*6
}

# Function to simulate the Futurity slot machine
def simulate_futurity(strategy, pA, pB, J, M):
    sequence = np.tile(strategy, M // len(strategy) + 1)[:M]
    wins = 0
    failures = 0
    refunds = 0

    for choice in sequence:
        # Determine win/loss based on probabilities
        win = np.random.rand() < (pA if choice == 0 else pB)
        if win:
            wins += 1
            failures = 0  # Reset failures on a win
        else:
            failures += 1
            if failures == J:
                refunds += 1
                failures = 0  # Reset failures after refund

    # Calculate casino profit
    profit = (M - wins - J * refunds) / M
    return profit

# Monte Carlo simulation
results = {}
for name, strategy in strategies.items():
    profits = [simulate_futurity(strategy, pA, pB, J, M) for _ in range(repetitions)]
    results[name] = np.mean(profits)

# Visualization of results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), alpha=0.7, color='skyblue')
plt.xlabel('Strategy')
plt.ylabel('Casino Profit (Mean)')
plt.title('Casino Profit for Futurity Slot Machine under Different Strategies')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
