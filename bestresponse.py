# import modules
import numpy as np
import matplotlib.pyplot as plt

# define best response functions
def br1(sigma2, pop, dist, alpha, beta, delta):
    return np.exp(alpha*pop + beta*dist[0] + delta*sigma2)/(1 + np.exp(alpha*pop + beta*dist[0] + delta*sigma2))
def br2(sigma1, pop, dist, alpha, beta, delta):
    return np.exp(alpha*pop + beta*dist[1] + delta*sigma1)/(1 +np.exp(alpha*pop + beta*dist[1] + delta*sigma1))

# plot function
def BR_plot(pop, dist, alpha, beta, delta):
    grid_num = 200
    grid = np.linspace(0,1,grid_num)
    BRs = np.zeros((grid_num, 2))
    for i in range(grid_num):
        BRs[i, :] = [br1(grid[i], pop, dist, alpha, beta, delta), br2(grid[i], pop, dist, alpha, beta, delta)]
    plt.plot(grid, BRs[:, 1], label = "player 2's BR function")
    plt.plot(BRs[:, 0], grid, label = "player 1's BR function")
    plt.legend()
    plt.xlabel("player 1's entry probability")
    plt.ylabel("player 2's entry probability")
    
# find equilibrium
def findequi(pop, dist, alpha, beta, delta):
    grid_num = 1000
    grid = np.linspace(0, 1, grid_num)
    return [(br1(br2(i, pop, dist, alpha, beta, delta), pop, dist, alpha, beta, delta), br2(i, pop, dist, alpha, beta, delta)) for i in grid if (br1(br2(i, pop, dist, alpha, beta, delta), pop, dist, alpha, beta, delta) - i)**2 < 1.0e-7]