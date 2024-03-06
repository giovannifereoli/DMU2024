using Pkg
pkg"add https://github.com/zsunberg/DMUStudent.jl"
pkg"add POMDPs"
pkg"add POMDPTools"
using DMUStudent.HW2
using POMDPs: states, actions, convert_s, stateindex
using POMDPTools: ordered_states
using Random
using LinearAlgebra
using SparseArrays
using MKL

############################
# Valute Iteration Functions
############################

function value_iteration_generic(m, gamma=0.95, tolerance=1e-8)
    # Exract MDP
    S = states(m)
    A = actions(m)
    T = transition_matrices(m)
    R = reward_vectors(m)

    # Initialize
    num_states = length(S)
    num_actions = length(A)
    V = rand(MersenneTwister(42), num_states)
    Vprime = rand(MersenneTwister(123), num_states)
    V_iter = zeros(num_states, num_actions)

    # Iterate until tol is met
    while maximum(abs.(V - Vprime)) > tolerance
        # Update 1
        copyto!(V, Vprime)

        # For a value of V
        for k in 1:num_actions
            V_iter[:, k] = R[A[k]] + gamma * T[A[k]] * V
        end
        
        # Update 2
        Vprime .= maximum(V_iter, dims=2)[:, 1]
    end

    return Vprime
end

function value_iteration_ACAS(m, gamma=0.99, epsilon=1e-8)
    # Exract MDP
    S = states(m)
    A = actions(m)
    T = transition_matrices(m, sparse=true)
    R = reward_vectors(m)

    # ACAS Reward and Transition Matrices
    R1_sparse = sparse(R[A[1]])
    T1_sparse = sparse(T[A[1]])
    R2_sparse = sparse(R[A[2]])
    T2_sparse = sparse(T[A[2]])
    R3_sparse = sparse(R[A[3]])
    T3_sparse = sparse(T[A[3]])

    # Initialize
    num_states = length(S)
    V = zeros(num_states)
    Vprime = rand!(MersenneTwister(43), zeros(num_states))
    
    # Iterate until tol is met
    while maximum(abs.(V - Vprime)) > epsilon
        # Update 1
        copyto!(V, Vprime)

        # Update 2, Belman Operator 
        Vprime = max.(R1_sparse + gamma * T1_sparse * V,
                        R2_sparse + gamma * T2_sparse * V, 
                        R3_sparse + gamma * T3_sparse * V)

    end
    return Vprime
end


############
# Question 3
############

# Solution
V = value_iteration_generic(grid_world)
display(render(grid_world, color=V))

############
# Question 4
############

# Solution and Evaluation
V = value_iteration_ACAS(UnresponsiveACASMDP(15))
HW2.evaluate(V, "giovanni.fereoli@colorado.edu")
