####################################
# Explicit Euler
#
# numeric integration file for the
# mathematical pendulum
#
# - explicit euler
#
####################################


mutable struct Integrator
    delta_t::Float64
    timesteps::Int64
    Integrator(delta_t, timesteps) = new(delta_t, timesteps)
end


## run one integration time step
function run_step(int::Integrator, type, pendulum)
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        run_central_diff_step(int, pendulum)
    else
        println("... integration type not understood ...")
    end
end


## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum)
    a = -(pendulum.g / pendulum.l) * sin(pendulum.phi) - pendulum.c * pendulum.phi_dot
    pendulum.phi = pendulum.phi + int.delta_t * pendulum.phi_dot
    pendulum.phi_dot = pendulum.phi_dot + a * int.delta_t
end


## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum)
    fg = -pendulum.g / pendulum.l * pendulum.phi
    a = (pendulum.m / int.delta_t^2) + (pendulum.c / (2 * int.delta_t))
    b = (pendulum.k) - (2 * pendulum.m / int.delta_t^2)
    c = (pendulum.m / int.delta_t^2) - (pendulum.c / (2 * int.delta_t))

    phi_temp = pendulum.phi
    pendulum.phi = (fg - b * pendulum.phi - c * pendulum.phi_prev) / a
    pendulum.phi_prev = phi_temp
end
