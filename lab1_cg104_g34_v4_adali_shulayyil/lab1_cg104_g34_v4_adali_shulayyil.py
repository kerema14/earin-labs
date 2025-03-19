#by Kerem Adalı K-8239 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from dataclasses import dataclass
from typing import List, Tuple


colors = ['purple', 'orange', 'cyan', 'magenta'] #bad code, very bad code

@dataclass
class History: #history class to store the results of the newton method
    solution: Tuple[float, float]
    iterations: int
    points_history: List[Tuple[float, float]]
    initial_guess: Tuple[float, float]
    learning_rate: float
        
def function(x, y):
    return 2*np.sin(x)+3*np.cos(y) #2sinx+3cosy=z

def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000)->History: #newton method function
    """
    Newton method
    
    Parameters:
    - initial_guess: initial 2D coordinate vector
    - alpha: step size parameter
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations

    """
    x1 = initial_guess[0]
    y1 = initial_guess[1]
    
    # Create history to store iterations for visualization
    history = [(x1, y1)]
    
    # Newton's method iterations
    for i in range(max_iter):
        # Compute gradient (first derivatives)
        grad_x = 2 * np.cos(x1)
        grad_y = -3 * np.sin(y1)
        gradient = np.array([grad_x, grad_y])
        
        # Compute Hessian (second derivatives)
        d2f_dx2 = -2 * np.sin(x1)
        d2f_dy2 = -3 * np.cos(y1)
        d2f_dxdy = 0  # Mixed derivative is zero for this function
        
        hessian = np.array([
            [d2f_dx2, d2f_dxdy],
            [d2f_dxdy, d2f_dy2]
        ])
        
        # Check if gradient is close to zero (we're at a critical point)
        if np.linalg.norm(gradient) < tol:
            return History((x1,y1),i,history,initial_guess,alpha)
        
        # Compute the Newton step: Δx = -H^(-1) * ∇f(x)
        try:
            delta = -alpha * np.dot(inv(hessian), gradient)
        except np.linalg.LinAlgError:
            # If the Hessian is singular, use gradient descent instead
            delta = -alpha * gradient
        
        # Update coordinates
        x1 += delta[0]
        y1 += delta[1]
        
        # Add new point to history
        history.append((x1, y1))
    
    # If we reach here, we've exceeded the maximum number of iterations
    return History((x1,y1),max_iter,history,initial_guess,alpha) #return history class here

def all(): #shows all points found by newton method between the given range
    """
    Visualization function: creates 3D plot of the function. 
    """
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax:Axes3D = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.9)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Additional settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('2*sin(x) + 3*cos(y)')
    
    # Plot critical points
    critical_points_markers = []
    # Find the minima and maxima (critical points) within the range
    for x_init in np.linspace(-5, 5, 7):
        for y_init in np.linspace(-5, 5, 7):
            min_point = newton_method([x_init, y_init], 1.0).solution
            
            # Check if point is within range and not already in the list
            if (abs(min_point[0]) <= 5 and abs(min_point[1]) <= 5 and #check if the point we've found is in [-5,5] for both x and y
                not any(np.allclose(min_point, cp) for cp in critical_points_markers)): #check if we are trying to add the same point twice
                critical_points_markers.append(min_point) 
    
    # Plot all found critical points
    for cp in critical_points_markers:
        ax.scatter(cp[0], cp[1], function(cp[0], cp[1]), color='green', s=300,depthshade=False, marker='*') #for each point in the list, draw it on the 3d plot with x,y,z,color,size and marker type as args
        ax.text(cp[0], cp[1], function(cp[0], cp[1])+0.7,"point" ,size=10, zorder=1) #str(""+str(cp[0]+0.4)[:5]+" "+ str(cp[1]+0.4)[:5]+" "+ str(function(cp[0], cp[1])+0.4)[:5])
    plt.show() #draw the plot
    
    return fig, ax #return both the figure and the 3d subplot

def visualize(histories:List[History]):
    # Visualization of paths for different initial guesses
    fig = plt.figure(figsize=(12, 10))
    ax:Axes3D = fig.add_subplot(111, projection='3d')

    # Create meshgrid for the surface
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.6)

    # Plot convergence paths
    """
    histories = [
        (history_1, initial_guess_1, 'r', 'Initial guess [2.0, 2.0], α=0.1'),
        (history_2, initial_guess_2, 'g', 'Initial guess [-3.0, 1.0], α=0.5'),
        (history_3, initial_guess_3, 'b', 'Initial guess [0.0, -4.0], α=1.0')
    ]
    """
    colorlist = list(mcolors.TABLEAU_COLORS.values())
    cnt = 0
    for history in histories:
        color = colorlist[cnt%len(colorlist)]
        cnt+=1
        # Extract x, y coordinates
        x_hist = [point[0] for point in history.points_history]
        y_hist = [point[1] for point in history.points_history]
        z_hist = [function(point[0], point[1]) for point in history.points_history]
        label = f"Initial guess {history.initial_guess}, α={history.learning_rate}"
        # Plot starting point
        ax.scatter(history.initial_guess[0], history.initial_guess[1], function(history.initial_guess[0], history.initial_guess[1]), color=color, s=100, marker='o')
        
        # Plot path
        ax.plot(x_hist, y_hist, z_hist, color=color, linestyle='-', marker='.', label=label)
        
        # Plot final point
        ax.scatter(x_hist[-1], y_hist[-1], z_hist[-1], color=color, s=100, marker='*')

    # Add colorbar and labels
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Newton Method Convergence Paths')
    ax.legend()
    plt.show()

def visualize_2d(histories:List[History], title="Newton Method Convergence Paths (2D View)", zoom_inset=False, zoom_point=None, zoom_radius=0.05):
    """
    Create 2D contour plot showing convergence paths
    
    Parameters:
    - histories: List of History objects to visualize
    - title: Plot title
    - zoom_inset: Whether to add a zoomed inset around a specific point
    - zoom_point: Center point for the zoom inset (tuple)
    - zoom_radius: Radius around zoom_point to show in the inset
    """
    plt.figure(figsize=(12, 10))
    
    # Create contour plot of the function
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    contour = plt.contourf(X, Y, Z, 50, cmap='coolwarm', alpha=0.7)
    plt.colorbar(contour)
    
    # Get colors for each history
    colorlist = list(mcolors.TABLEAU_COLORS.values())
    
    # Plot paths for different histories
    for i, history in enumerate(histories):
        color = colorlist[i % len(colorlist)]
        
        x_hist = [point[0] for point in history.points_history]
        y_hist = [point[1] for point in history.points_history]
        
        label = f"Initial guess {history.initial_guess}, α={history.learning_rate}, Iter={history.iterations}"
        plt.plot(x_hist, y_hist, color=color, linestyle='-', marker='.', label=label)
        plt.scatter(history.initial_guess[0], history.initial_guess[1], color=color, s=100, marker='o')
        plt.scatter(x_hist[-1], y_hist[-1], color=color, s=100, marker='*')
    
    # Add zoom inset if requested
    if zoom_inset and zoom_point is not None:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        axins = zoomed_inset_axes(plt.gca(), 6, loc=2)  # zoom factor: 6
        axins.contourf(X, Y, Z, 50, cmap='coolwarm', alpha=0.7)
        
        for i, history in enumerate(histories):
            color = colorlist[i % len(colorlist)]
            x_hist = [point[0] for point in history.points_history]
            y_hist = [point[1] for point in history.points_history]
            
            axins.plot(x_hist, y_hist, color=color, linestyle='-', marker='.')
            axins.scatter(x_hist[-1], y_hist[-1], color=color, s=100, marker='*')
        
        # Set the limits for the inset
        axins.set_xlim(zoom_point[0] - zoom_radius, zoom_point[0] + zoom_radius)
        axins.set_ylim(zoom_point[1] - zoom_radius, zoom_point[1] + zoom_radius)
        axins.grid(True)
        
        # Connect the inset with the main plot
        mark_inset(plt.gca(), axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.legend()
    plt.show()
# Main execution
fig, ax = all()

# Example 1
initial_guess_1 = [2.0, 2.0]
learning_rate_1 = 0.1 #step size
history_1 = newton_method(initial_guess_1, learning_rate_1)
print(f"Minimum approximation with initial guess {initial_guess_1}: {history_1.solution}, Iterations: {history_1.iterations}")

# Example 2
initial_guess_2 = [-3.0, 1.0]
learning_rate_2 = 0.5 #step size
history_2 = newton_method(initial_guess_2, learning_rate_2)
print(f"Minimum approximation with initial guess {initial_guess_2}: {history_2.solution}, Iterations: {history_2.iterations}")

# Example 3
initial_guess_3 = [0.0, -4.0]
learning_rate_3 = 1.0 #step size
history_3 = newton_method(initial_guess_3, learning_rate_3)
print(f"Minimum approximation with initial guess {initial_guess_3}: {history_3.solution}, Iterations: {history_3.iterations}")

visualize([history_1, history_2, history_3])

# Example 4 - Different step sizes with same initial point
initial_guess_4 = [1.0, 1.0]
step_sizes = [0.1, 0.5, 1.0, 1.5]
histories_step_size = []

# Collect histories for different step sizes
for alpha in step_sizes:
    history = newton_method(initial_guess_4, alpha)
    histories_step_size.append(history)
    print(f"Initial guess {initial_guess_4}, α={alpha}: {history.solution}, Iterations: {history.iterations}")

# Create a 2D visualization function


# Use the visualization functions for the step size comparison
visualize(histories_step_size)
visualize_2d(histories_step_size, title="Effect of Step Size on Newton Method Convergence (2D View)")

# Example 5 - Comparing different initial points converging to the same solution
initial_guesses = [[-2.0, 3.0], [0.0, 2.0], [3.0, 1.0], [1.0, -2.0]]
alpha_fixed = 0.5
histories_convergence = []

# Collect histories for different initial points
for init_guess in initial_guesses:
    history = newton_method(init_guess, alpha_fixed)
    histories_convergence.append(history)
    print(f"Initial guess {init_guess}, α={alpha_fixed}: {history.solution}, Iterations: {history.iterations}")

# Visualize convergence to same point from different starting positions
visualize(histories_convergence)
visualize_2d(histories_convergence, title="Convergence from Different Starting Points (2D View)")

# Example 6 - Testing behavior around a problematic area (near Hessian singularities)
# Try points where the function's second derivatives might be close to zero
singular_guesses = [[np.pi/2 - 0.1, 0], [np.pi/2 + 0.1, 0], [0, np.pi/2 - 0.1], [0, np.pi/2 + 0.1]]
alpha_7 = 0.3
histories_singular = []

for init_guess in singular_guesses:
    history = newton_method(init_guess, alpha_7)
    histories_singular.append(history)
    print(f"Singular test - Initial guess {init_guess}, α={alpha_7}: {history.solution}, Iterations: {history.iterations}")

# Visualize behavior around potential singularities
visualize(histories_singular)
visualize_2d(histories_singular, title="Newton Method Behavior Near Potential Singularities")