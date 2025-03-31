from typing import List, Dict, Tuple
class CSP:
    def __init__(self, variables, domains, constraints):
        """
        Initialization of the CSP class

        Parameters:
        - variables: list of tuples (i,j) representing cells in the 9x9 grid
        - domains: dictionary mapping variables to their possible values
        - constraints: dictionary mapping variables to variables they constrain
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None
        self.viz = []

    def print_sudoku(self, puzzle):
        """Prints the sudoku puzzle in a readable format"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - ")
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                print(puzzle[i][j], end=" ")
            print()

    def visualize(self):
        """Visualizes the steps of the solving algorithm"""
        print("\n" + "*" * 7 + " Visualization " + "*" * 7)
        for step, grid in enumerate(self.viz):
            print(f"\nStep {step+1}:")
            self.print_sudoku(grid)
            print("-" * 30)

    def solve(self):
        """
        Solves the sudoku puzzle using backtracking with forward checking
        
        Returns:
        - solution: dict mapping variables to values
        - viz: list of grid states for visualization
        """
        assignment = {}
        self.viz = []  # Reset visualization
        self.solution = self.backtrack(assignment)
        return self.solution, self.viz
    
    def forward_checking(self, var, value, assignment:dict):
        """
        Function that removes the value from the domains of free variables that are in the constraints of the var

        Parameters:
        - var: variable that was assigned the value
        - value: value that was assigned to the variable
        - assignment: dict with all the assignments to the variables
        """
        pruned = {}  # Track pruned values for backtracking
        
        # For each variable constrained by var
        for neighbor in self.constraints[var]:
            # Skip if already assigned
            if neighbor in assignment:
                continue
                
            # If the value is in the domain of the neighbor
            if value in self.domains[neighbor]:
                # Track pruning
                if neighbor not in pruned:
                    pruned[neighbor] = []
                pruned[neighbor].append(value)
                
                # Remove the value from domain
                self.domains[neighbor].remove(value)
                
                # If domain becomes empty, this assignment fails
                if not self.domains[neighbor]:
                    return pruned, False
        
        return pruned, True

    def backtrack(self, assignment:dict):
        """
        Backtracking algorithm

        Parameters:
        - assignment: dict with all the assignments to the variables

        Returns:
        - assignment: dict with all the assignments to the variables, or None if solution is not found. Return the first found solution
        """
        # If all variables are assigned, we're done
        if len(assignment) == len(self.variables):
            current_grid = self.create_grid_snapshot(assignment)
            self.viz.append(current_grid)
            return assignment
            
        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)
        
        # Create a grid snapshot for visualization
        current_grid = self.create_grid_snapshot(assignment)
        self.viz.append(current_grid)
        
        # Try assigning each value in var's domain
        for value in list(self.domains[var]):  # Use list to avoid modifying during iteration
            # Check if this assignment is consistent
            if self.is_consistent(var, value, assignment):
                # Assign the value
                assignment[var] = value
                
                # Apply forward checking
                pruned, is_consistent = self.forward_checking(var, value, assignment)
                
                # If assignment is consistent with forward checking
                if is_consistent:
                    # Recursive call
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                
                # If we get here, this assignment failed
                # Remove var from assignment
                del assignment[var]
                
                # Restore pruned values
                self.restore_pruned(pruned)
        
        # No solution found with current assignments
        return None
    
    def select_unassigned_variable(self, assignment:dict):
        """
        Selects an unassigned variable with minimum remaining values (MRV)
        """
        # Find variables that aren't assigned yet
        unassigned = [v for v in self.variables if v not in assignment]
        
        # Select the one with fewest remaining values in its domain
        return min(unassigned, key=lambda var: len(self.domains[var]))
    
    def is_consistent(self, var, value, assignment:dict):
        """
        Checks if assigning value to var is consistent with current assignment
        """
        # Check constraints with assigned variables
        for neighbor in self.constraints[var]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True
    
    def restore_pruned(self, pruned):
        """
        Restores pruned values to domains to take back the effect of forward checking
        """
        for var, values in pruned.items():
            for value in values:
                if value not in self.domains[var]:
                    self.domains[var].append(value)
    
    def create_grid_snapshot(self, assignment:dict):
        """
        Creates a grid snapshot for visualization
        """
        # Create empty grid
        grid = [[0 for _ in range(9)] for _ in range(9)]
        
        # Fill in values from the assignment
        for (i, j), value in assignment.items():
            grid[i][j] = value
            
        return grid


def create_sudoku_csp(puzzle):
    """
    Creates variables, domains, and constraints for the Sudoku CSP
    """
    # Create variables (all cells)
    variables = [(i, j) for i in range(9) for j in range(9)]
    
    # Create domains based on initial puzzle
    domains = {}
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == 0:  # Empty cell
                domains[(i, j)] = list(range(1, 10))
            else:  # Pre-filled cell
                domains[(i, j)] = [puzzle[i][j]]
    
    # Create constraints
    constraints = {}
    for i in range(9):
        for j in range(9):
            constraints[(i, j)] = []
            
            # Same row constraints
            for jj in range(9):
                if jj != j:
                    constraints[(i, j)].append((i, jj))
            
            # Same column constraints
            for ii in range(9):
                if ii != i:
                    constraints[(i, j)].append((ii, j))
            
            # Same 3x3 box constraints
            box_i, box_j = 3 * (i // 3), 3 * (j // 3)
            for ii in range(box_i, box_i + 3):
                for jj in range(box_j, box_j + 3):
                    if (ii, jj) != (i, j):
                        constraints[(i, j)].append((ii, jj))
    
    return variables, domains, constraints


puzzles = [
    # Easy Sudoku
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],

    # Medium Sudoku
    [
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 3],
        [0, 7, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 2],
        [0, 8, 0, 0, 4, 0, 0, 1, 0],
        [6, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 7, 8, 0],
        [5, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0]
    ],

    # Difficult Sudoku
    [
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 8, 0, 0, 0, 7, 0, 9, 0],
        [6, 0, 2, 0, 0, 0, 5, 0, 0],
        [0, 7, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 9, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 4, 0],
        [0, 0, 5, 0, 0, 0, 6, 0, 3],
        [0, 9, 0, 4, 0, 0, 0, 7, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0]
    ],

    # Impossible Sudoku (No valid solution)
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  
        [2, 2, 2, 2, 2, 2, 2, 2, 2],  
        [3, 3, 3, 3, 3, 3, 3, 3, 3],  
        [4, 4, 4, 4, 4, 4, 4, 4, 4],  
        [5, 5, 5, 5, 5, 5, 5, 5, 5],  
        [6, 6, 6, 6, 6, 6, 6, 6, 6],  
        [7, 7, 7, 7, 7, 7, 7, 7, 7],  
        [8, 8, 8, 8, 0, 8, 8, 8, 8],  
        [9, 9, 9, 9, 9, 9, 9, 9, 9]   
    ],

    # Corner Case 1: Already Solved Sudoku
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ],

    # Corner Case 2: Minimal Clues Sudoku (Hardest Solvable Sudoku)
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 1, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
        [9, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 5, 0, 0, 0, 0, 0]
    ]
    ,
    # Corner Case 3: Empty Grid (Completely Unfilled)
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],

    # Corner Case 4: Single Missing Cell (Almost Solved)
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 0]  # One missing value (should be 9)
    ],


    # Corner Case 5: Multiple Solutions Possible
    [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ]
]



# Test with a more difficult puzzle
def test_puzzle(puzzle:List[List[int]],visualize:bool=True):
    print("\n\n" + "=" * 30)
    print("TESTING DIFFICULT PUZZLE")
    print("=" * 30)
    
    variables, domains, constraints = create_sudoku_csp(puzzle)
    
    print('Original puzzle:')
    csp = CSP(variables, domains, constraints)
    csp.print_sudoku(puzzle)
    
    print('\nSolving...')
    sol, _ = csp.solve()
    
    solution = [[0 for i in range(9)] for i in range(9)]
    if sol is not None:
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0:
                    solution[i][j] = puzzle[i][j]
                elif (i, j) in sol:
                    solution[i][j] = sol[(i, j)]
        
        print('Solution:')
        csp.print_sudoku(solution)
        
        if visualize:
            csp.visualize()
    else:
        print("Solution does not exist")

for puzzle in puzzles:
    test_puzzle(puzzle)
    wait = input("Press Enter to continue...")

