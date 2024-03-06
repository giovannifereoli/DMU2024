############
# Libraries
############

using Pkg
# pkg"add https://github.com/zsunberg/DMUStudent.jl"
# pkg"add POMDPs"
# pkg"add POMDPTools"
# pkg"add StaticArrays"
# pkg"add POMDPModels"
# pkg"add POMDPTools"
# pkg"add SparseArrays"
# pkg"add CommonRLInterface"
# pkg"add Interact"
using DMUStudent.HW4: HW4, gw, render
using POMDPModels: SimpleGridWorld
using LinearAlgebra: I
using CommonRLInterface: actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
import POMDPTools
using SparseArrays
using Statistics: mean
using Plots
using Interact

############
# Functions
############

# SARSA episode generation
function sarsa_episode!(Q, env; eps=0.10, gamma=0.99, alpha=0.2)
    start = time()

    function policy(s)
        if rand() < eps
            return rand(actions(env))
        else
            return argmax(a -> Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]

    while !terminated(env)
        ap = policy(sp)

        Q[(s, a)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a)])

        s = sp
        a = ap
        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    Q[(s, a)] += alpha * (r - Q[(s, a)])

    return (hist=hist, Q=copy(Q), time=time() - start)
end


# Q-LEARNING episode generation
function q_learning_episode!(Q, env; eps=0.10, gamma=0.99, alpha=0.2)
    start = time()

    function policy(s)
        if rand() < eps
            return rand(actions(env))
        else
            return argmax(a -> Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    hist = [s]

    while !terminated(env)
        a = policy(s)
        r = act!(env, a)
        sp = observe(env)

        Q[(s, a)] += alpha * (r + gamma * maximum(ap -> Q[sp, ap], actions(env)) - Q[(s, a)])

        s = sp
        push!(hist, sp)
    end

    Q[(s, a)] += alpha * (r - Q[(s, a)])

    return (hist=hist, Q=copy(Q), time=time() - start)
end

# SARSA training
function sarsa!(env; n_episodes=100, eps_min=0.10, gamma=0.99, alpha=0.2)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []

    for i in 1:n_episodes
        reset!(env)
        push!(episodes, sarsa_episode!(Q, env;
            eps=max(eps_min, 1 - i / n_episodes), gamma, alpha))
    end

    return episodes
end

# Q-LEARNING training
function q_learning!(env; n_episodes=100, eps_min=0.10, gamma=0.99, alpha=0.2)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []

    for i in 1:n_episodes
        reset!(env)
        push!(episodes, q_learning_episode!(Q, env;
            eps=max(eps_min, 1 - i / n_episodes), gamma, alpha))
    end

    return episodes
end

# Evaluate Policy 
function evaluate(env, policy, n_episodes=1000, max_steps=1000, gamma=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = observe(env)
        while !terminated(env)
            a = policy(s)
            r += gamma^t * act!(env, a)
            s = observe(env)
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

############
# Results
############

## Training
# Initialization fro HW4
#m = HW4.GridWorldEnv()
#env = convert(AbstractEnv, m)
#n_episodes = 20000
#eps_min = 0.05
#gamma = 0.99
#alpha = 0.01

# Initialization for SimpleGridWorld POMPDj
m = SimpleGridWorld()
env = convert(AbstractEnv, m)
n_episodes = 10000
eps_min = 0.1
gamma = 0.99
alpha = 0.2

# Training SARSA and SARSA-lambda
sarsa_episodes = sarsa!(env; n_episodes, eps_min, gamma, alpha);
q_learning_episodes = q_learning!(env; n_episodes, eps_min, gamma, alpha);

## Plots
# Initialization
episodes = Dict("SARSA" => sarsa_episodes, "Q-Learning" => q_learning_episodes)
n = 10
stop = 1000

# Environment Step - Average Return
p1 = plot(xlabel="Environment Step [-]", ylabel="Average Return [-]")
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    xs = [0]
    ys = [mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newsteps)
        Q = eps[i].Q
        push!(ys, mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env)))))
    end
    plot!(p1, xs, ys, label=name)
end
plot!(p1, legend=true)


# Clock Wall Time - Average Return 
#p2 = plot(xlabel="Wall Clock Time [s]", ylabel="Average Return [-]")
#for (name,eps) in episodes
#    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
#    xs = [0.0]
#    ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
#    for i in n:n:min(stop, length(eps))
#        newtime = sum(ep.time for ep in eps[i-n+1:i])
#        push!(xs, last(xs) + newtime)
#        Q = eps[i].Q
#        push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))
#    end    
#    plot!(p2, xs, ys, label=name)
#end
#plot!(p2, legend=true)


