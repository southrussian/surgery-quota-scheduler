using Random

mutable struct Agent
    complexity::Float64
    completeness::Float64
    urgency::Float64
    k::Float64
    position::Int
end

Agent(complexity, completeness, urgency) = Agent(complexity, completeness, urgency, (complexity + (1 - completeness)) * urgency, 0)

mutable struct Environment
    N::Int
    C::Int
    b::Float64
    P::Float64
    S::Int
    Urgency_pool::Vector{Int}
    Completeness_pool::Vector{Int}
    Complexity_values::Vector{Int}
    agents::Vector{Agent}
    schedule::Vector{Int}
    step::Int
end

function Environment(N=12, C=4, b=0.2, P=0.05)
    S = div(N^2, C)
    Urgency_pool = [1, 2, 3]
    Completeness_pool = [0, 1]
    Complexity_values = [0, 1]
    agents = [Agent(ug, cm, cx) for ug in Urgency_pool for cm in Completeness_pool for cx in Complexity_values]
    schedule = zeros(Int, S)
    step = 0
    Environment(N, C, b, P, S, Urgency_pool, Completeness_pool, Complexity_values, agents, schedule, step)
end

function move!(env::Environment, agent::Agent, action::String)
    if action == "forward"
        agent.position = min(agent.position + 1, env.S - 1)
    elseif action == "backward"
        agent.position = max(agent.position - 1, 0)
    end
end

function hold(env::Environment, agent::Agent)
    n = env.schedule[agent.position]
    if n <= env.C
        return (env.C - n) * env.b
    else
        return -(n - env.C) * env.b
    end
end

function observe(env::Environment, agent::Agent)
    observation = env.schedule[max(0, agent.position - 3) : min(env.S, agent.position + 4)]
    append!(observation, [round(Int, i * rand(0.5:0.01:1.5)) for i in env.schedule[max(0, agent.position - 10) : agent.position - 3]])
    append!(observation, [round(Int, i * rand(0.5:0.01:1.5)) for i in env.schedule[agent.position + 4 : min(env.S, agent.position + 11)]])
    return observation
end

function step!(env::Environment, actions::Vector{String})
    for (agent, action) in zip(env.agents, actions)
        if action in ["forward", "backward"]
            move!(env, agent, action)
        elseif action == "hold"
            reward = hold(env, agent)
            if rand() < env.P
                agent.position = rand(0:env.S - 1)
            end
            env.schedule[agent.position] += 1
        end
    end
    env.step += 1
    if env.step >= env.S
        return true, reward
    else
        return false, reward
    end
end