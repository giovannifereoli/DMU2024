using Pkg
# pkg"add https://github.com/zsunberg/DMUStudent.jl"
# pkg"add POMDPs"
# pkg"add POMDPTools"
# pkg"add D3Trees"
# pkg"add StaticArrays"
# pkg"add BenchmarkTools"
using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using LinearAlgebra: norm

############
# Question 2
############

## Roll-out generation function 
function rollout(m, policy_rollout, s0, depth=100)
    # Initialization
    r_total = 0.0
    t = 0
    s = s0

    # Generate a trajetory 
    while !isterminal(m, s) && t < depth
        a = policy_rollout(m, s)
        s, r = @gen(:sp, :r)(m, s, a)
        r_total += discount(m)^t * r
        t += 1
    end

    return r_total
end

## Policy functions
# Random policy
function random_policy(m, s)
    return rand(actions(m))
end

# Heuristic policy following Euclidean distance for 60x60 grid
function heuristic_policy60(m, s)
    #Initialization
    max_index = 40
    reward_locations = [(i, j) for i in 20:20:max_index for j in 20:20:max_index]

    # Calculate the distance to each reward location
    distances = [norm((s[1] - loc[1], s[2] - loc[2])) for loc in reward_locations]

    # Choose the nearest reward location
    nearest_reward_idx = argmin(distances)
    nearest_reward_location = reward_locations[nearest_reward_idx]

    # Determine the direction to move towards the nearest reward location
    x_distance = nearest_reward_location[1] - s[1]
    y_distance = nearest_reward_location[2] - s[2]

    # Choose the action that moves towards the nearest reward location
    if abs(x_distance) > abs(y_distance)
        # Move along x-direction
        if sign(x_distance) == 1
            return actions(m)[4]  # Move right
        else
            return actions(m)[3]  # Move left
        end
    else
        # Move along y-direction
        if sign(y_distance) == 1
            return actions(m)[1]  # Move up
        else
            return actions(m)[2]  # Move down
        end
    end
end

## Run Statistics
function compute_mean_and_sem(rewards)
    mean_reward = mean(rewards)
    sem = std(rewards) / sqrt(length(rewards))
    return mean_reward, sem
end

## Call MDP
m = HW3.DenseGridWorld(seed=3)

## Run results and compute Statistics (MC Simulations)
results_random = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:10000]
results_heuristic = [rollout(m, heuristic_policy60, rand(initialstate(m))) for _ in 1:10000]
@show compute_mean_and_sem(results_random)
@show compute_mean_and_sem(results_heuristic)

############
# Question 3
############

## Functions
# MCTS solver
function mcts_solver(m, s0, policy, iterations, depth, exp_c)
    # Initialization Q, N, t dictionaries and actions space
    Q = Dict{Tuple{statetype(m),actiontype(m)},Float64}()
    N = Dict{Tuple{statetype(m),actiontype(m)},Int}()
    T = Dict{Tuple{statetype(m),actiontype(m),statetype(m)},Int}()
    Action_space = actions(m)

    # MCTS iterations
    for _ in 1:iterations
        simulate!(m, s0, Q, N, T, Action_space, policy, depth, exp_c)
    end

    return Q, N, T
end

# MCTS simulations
function simulate!(m, s, Q, N, T, Action_space, policy, depth, exp_c)
    # Expand until max depth
    if depth <= 0
        return rollout(m, policy, s)
    end

    # Initialization new nodes
    if !haskey(N, (s, rand(Action_space)))
        for a in Action_space
            N[(s, a)] = 0
            Q[(s, a)] = 0.0
        end
        return rollout(m, policy, s)
    end

    # Selection and expansion
    a = explore(s, Action_space, Q, N, exp_c)
    sp, r = @gen(:sp, :r)(m, s, a)
    q = r + discount(m) * simulate!(m, sp, Q, N, T, Action_space, policy, depth - 1, exp_c)

    # Backpropagation
    if !haskey(T, (s, a, sp))
        T[(s, a, sp)] = 0
    end
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    T[(s, a, sp)] += 1

    return q
end

# MCTS exploration through UCB (Upper Confidence Bound)
# Function for bonus term
function bonus(Nsa, Ns)
    if Nsa == 0
        return Inf
    else
        return sqrt(log(Ns) / Nsa)
    end
end

# Function to choose MCTS action
function explore(s, A, Q, N, exp_c)
    # Initialization
    Ns = sum(N[(s, a)] for a in A)

    # Return chosen action
    return argmax(a -> Q[(s, a)] + exp_c * bonus(N[(s, a)], Ns), A)
end

## Tree solution
# MDP initialization
m = DenseGridWorld(seed=4)

# Tree visualization for GridWorld 
# Inputs: iterations = 7, depth = 50, seed = 4, s0 = SA[19, 19], exp_c=150
Q, N, T = mcts_solver(m, SA[19, 19], heuristic_policy60, 7, 50, 150)
inchrome(visualize_tree(Q, N, T, SA[19, 19]))

############
# Question 4
############

## Functions
# MCTS open-loop planner for 60x60 grid
function mcts_planner60(m, s)
    # Initialization dictionaries and Action Space
    Q = Dict{Tuple{statetype(m),actiontype(m)},Float64}()
    N = Dict{Tuple{statetype(m),actiontype(m)},Int}()
    T = Dict{Tuple{statetype(m),actiontype(m),statetype(m)},Int}()
    Action_space = actions(m)

    for _ in 1:1000
        # MCTS simulation
        simulate!(m, s, Q, N, T, Action_space, heuristic_policy60, 50, 150)
    end

    # Select best action looking MCTS results
    return argmax(a -> Q[(s, a)], Action_space)
end

# Solution planning
results_mcts = [rollout(m, mcts_planner60, rand(initialstate(m))) for _ in 1:100]
@show compute_mean_and_sem(results_mcts)

############
# Question 5
############

## Functions
# Heuristic policy following Euclidean distance for 100x100 grid
function heuristic_policy100(m, s)
    #Initialization
    max_index = 80
    reward_locations = [(i, j) for i in 20:20:max_index for j in 20:20:max_index]

    # Calculate the distance to each reward location
    distances = [norm((s[1] - loc[1], s[2] - loc[2])) for loc in reward_locations]

    # Choose the nearest reward location
    nearest_reward_idx = argmin(distances)
    nearest_reward_location = reward_locations[nearest_reward_idx]

    # Determine the direction to move towards the nearest reward location
    x_distance = nearest_reward_location[1] - s[1]
    y_distance = nearest_reward_location[2] - s[2]

    # Choose the action that moves towards the nearest reward location
    if abs(x_distance) > abs(y_distance)
        # Move along x-direction
        if sign(x_distance) == 1
            return actions(m)[4]  # Move right
        else
            return actions(m)[3]  # Move left
        end
    else
        # Move along y-direction
        if sign(y_distance) == 1
            return actions(m)[1]  # Move up
        else
            return actions(m)[2]  # Move down
        end
    end
end

# MCTS open-loop planner for 100x100 grid
function mcts_planner100(m, s)
    # Initialization dictionaries and Action Space
    Q = Dict{Tuple{statetype(m),actiontype(m)},Float64}()
    N = Dict{Tuple{statetype(m),actiontype(m)},Int}()
    T = Dict{Tuple{statetype(m),actiontype(m),statetype(m)},Int}()
    Action_space = actions(m)
    start = time_ns()

    for _ in 1:1000
        # MCTS simulation
        simulate!(m, s, Q, N, T, Action_space, heuristic_policy100, 50, 150)
    end

    # Select best action looking MCTS results
    return argmax(a -> Q[(s, a)], Action_space)
end

## Evaluation
HW3.evaluate(mcts_planner100, "giovanni.fereoli@colorado.edu", time=true)
