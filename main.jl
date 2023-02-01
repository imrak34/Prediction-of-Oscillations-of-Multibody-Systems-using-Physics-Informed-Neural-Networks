using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux
using Statistics
using LinearAlgebra


# Store iteration data
mutable struct Data
	y::Vector
	x::Vector
	Data() = new()
end

# Struct of hyperparameters
mutable struct Lambda
    mse::Float64	# MSE loss weugth
	res::Float64	# RES loss weight
	decay::Float64	# Weight decay loss weight
end

show_plot = false
save_plot = false

## File Loaded and Case Selection
println("-- Pendulum Solver --")
case = "central_diff"
#case = "euler"


## Load Pendulum Object
include("Dynsys.jl")
initial_phi = 1.0
initial_vel = 0.0
phi = initial_phi
phi_dot = initial_vel


g = 10.0                      #gravity
m = 1.0                       #mass
c = 5.0                        #damping coefficient
k = 300.0                     #stiffness coefficient
l = m*g/k                     # length of the pendulum
pendulum = Dynsys.Math_pendulum(l, g, m, c, k, initial_phi, initial_vel)


## Load Integrator Object
t_end = 1.0     # End time
ts = 2000		# Number of timesteps
dt = t_end / ts	# Time step size
Integ = Dynsys.Integrator(dt, ts)
t=zeros(Integ.timesteps)
data = Data()
data.y = zeros(ts)
data.x = zeros(ts)


## Setup Time Intetration
# initial setting
if show_plot == true
	fig = Dynsys.create_fig(pendulum)
	Dynsys.plot_state(pendulum)
	display(fig)
end

# compute phi_-1
if case == "central_diff"
	acceleration = (-pendulum.g / pendulum.l * pendulum.phi) - (k/m * pendulum.phi) - (c/m * pendulum.phi_dot)
	pendulum.phi_prev = pendulum.phi - (Integ.delta_t * pendulum.phi_dot) + (0.5 * Integ.delta_t * Integ.delta_t * acceleration)
else
	pendulum.phi_prev = 0
end



## ------------------------- ##
##      Data Deneration      ##
## ------------------------- ##
println("Using '" * case * "' Solver!\nGenerating Data")
for i in 1:Integ.timesteps
	# Integration step
	Dynsys.run_step(Integ, case, pendulum)

	# Plot the state
	if show_plot == true
		fig = Dynsys.create_fig(pendulum)
		Dynsys.plot_state(pendulum)
		display(fig)
	end

	# Save the step
	data.y[i] = pendulum.phi
	data.x[i] = dt * i
end

println("Data Generation Complete")
if save_plot == true
    println("Saving Data Plot")
    plot(data.x,data.y)
    savefig( "Numerical_solution.png")
    println("Save Complete")
else
	plot(data.x,data.y)
end



## ------------------------- ##
##       Neural Network      ##
## ------------------------- ##
## Network Creation
activation_function = "tanh"
neural_network = Chain(
    Dense(1, 30, tanh),
    Dense(30,30, tanh),
    Dense(30, 1)
)

optimizer = Adam(0.01)						# Selected optimizer
parameters = Flux.params(neural_network)	# Get NN parameters


## ----- Normalisation of Input Data ----- ##
normalization = true
if normalization == true
	if activation_function == "relu" || activation_function == "sigmoid" #|| activation_function == "tanh"
		min_val = findmin(data.y)[1]     # Minimum value from dataset
		max_val = findmax(data.y)[1]     # Max value from dataset
	    difference = max_val - min_val   # Range of data
	else
		mean1 = mean(data.y)
		std1 = std(data.y)
	end
end

function normalize_data(data_in)
	if !normalization
		return data_in
	end

	if activation_function == "tanh"
		return (data_in .-mean1)./std1
	else
		return  ((data_in .- min_val) / difference)
	end
end

function denormalize_data(data_in)
	if !normalization
		return data_in
	end

	if activation_function == "tanh"
		return (data_in .* std1).+ mean1
	else
		return data_in .* difference .+ min_val
	end
end


## ----- Data Splitting and Loading ----- ##
training_ts = Int(floor(0.35 * ts))
training_data_t = data.x[1:2:training_ts]'		# Time data (training data set)
training_data_y = data.y[1:2:training_ts]'		# Training data (training data set)
validation_data_t = data.x[2:2:training_ts]'	# Time data (validation data set)
validation_data_y = data.y[2:2:training_ts]'	# Training data (validation data set)

training_data_y = normalize_data(training_data_y)
validation_data_y = normalize_data(validation_data_y)
input_data = Flux.Data.DataLoader((training_data_t, training_data_y), batchsize=25, shuffle=true)	# Load data


## ----- Loss Functions ----- ##
# Mean Squared Error Loss
function mse_loss(x, y)
	return mean((y .- neural_network(x)).^2)	# Compute mse loss
end

# Weight Decay (using L1 and L2 Regularization)
function weight_loss()
	weights = hcat(reshape(parameters[1],1,30),reshape(parameters[3],1,900),reshape(parameters[5],1,30))
	return  1e-4*sum(abs2.(weights)) #+ 1e-4 * sum(abs.(weights))
end

# Residual Loss
res_dt = 0.001
res_n_points = 50
res_data_center = collect(range(res_dt, t_end, res_n_points))'	# Training data set for residaul computation (center)
res_data_right = res_data_center .+ res_dt								# Training data set for residaul computation (right)
res_data_left = res_data_center .- res_dt								# Training data set for residaul computation (left)

function res_loss()
	vals_center = denormalize_data(neural_network(res_data_center))		# Central values for CFD
	vals_right = denormalize_data(neural_network(res_data_right))		# Right values for CFD
	vals_left = denormalize_data(neural_network(res_data_left))			# Left values for CFD

    v = (vals_right .- vals_left) / (2*res_dt)									# Compute phi_dot using central difference
    a = (vals_right .- 2*vals_center .+ vals_left) / (res_dt^2)					# Compute phi_ddot using central difference
    res = (a) .+ (c/m * v) .+ (k/m * vals_center) .+ (g / l * vals_center)		# Compute the physics residual (acceleration + velocity + displacement - external)

	return mean(abs2.(res))
end

# Total Loss Function
lambda = Lambda(1, 1e-4, 0)
function loss_function(x, y)
	mse = mse_loss(x, y)
	res = res_loss()
	weight = weight_loss()

    return lambda.mse * mse + lambda.res * res + lambda.decay * weight  # Compute the total loss function (mse + res + weight decay)
end


## ----- Main Training Loop ----- ##
tolerance = 1e-5	# Early termination tolerance
max_itrs = 10000	# Maximum number of training iterations
n_itrs = max_itrs
itr_per_save = 1000

convergence_res = Data()
convergence_res.x = zeros(max_itrs)
convergence_res.y = zeros(max_itrs)
convergence_val = Data()
convergence_val.x = zeros(max_itrs)
convergence_val.y = zeros(max_itrs)
z = zeros(max_itrs)
plot_data = zeros((Int(floor(max_itrs/itr_per_save)),ts))

println("Training Network")
for e in 1:max_itrs
	Flux.train!(loss_function, parameters, input_data, optimizer)   # Training call
    convergence_res.y[e] = loss_function(training_data_t, training_data_y)
    convergence_val.y[e] = mse_loss(validation_data_t, validation_data_y)

    convergence_res.x[e] = e
    convergence_val.x[e] = e
    z[e] = optimizer.eta
    # Print info every 100 itrs
	if e % 100 == 0
		loss = mse_loss(validation_data_t, validation_data_y) + lambda.res * res_loss_validation()
		@printf("Iterations: %d, Loss: %.4e, Learning Rate: %.2e\n", e, loss, optimizer.eta)
	end

    # Save convergence plots every n itrs
	if e % itr_per_save == 0
		plot_data[Int(e/itr_per_save),:] = neural_network(data.x')
	end

	# Check loss function on validation dataset
	if mse_loss(validation_data_t, validation_data_y) + lambda.res * res_loss_validation() < tolerance
		@printf("Training converged in %d iterations\n", e)
		global n_itrs = e
		break
	end

	#Update training rate
	if e == 1500
		optimizer.eta = 0.001
	elseif e == 3000
		optimizer.eta = 0.0005
	elseif e == 6000
		optimizer.eta = 0.0001
	elseif e == 10000
		optimizer.eta = 0.00005
	elseif e == 15000
		optimizer.eta = 0.00001
	end
end
println("Training Complete")



## ----- Plotting Results ----- ##
## Plot Trained Network Output and Compare
trained_y = denormalize_data(neural_network(data.x'))
plot(data.x, [trained_y', data.y], title = "Predicting Oscillation of a Pendulum with PINN" , label = ["PINN" "Numerical Solution"] , xlabel = "Time [s]" , ylabel = "ϕ [rad]")
println("Saving Fitted Plot")
savefig( "FittedData.png")
println("Save Complete")


## Plot Evolution
plot_evolution = true
if plot_evolution == true
	plot(data.x, denormalize_data(plot_data[1,:]), title="Prediction Evolution", label=("Iteration: " * string(itr_per_save)), xlabel = "Time [s]" , ylabel = "ϕ [rad]")
	for i in 2:Int(floor(n_itrs/itr_per_save))
		plot!(data.x, denormalize_data(plot_data[i,:]),label=("Iteration: " * string(i*itr_per_save)))
	end
	println("Saving Evolution Plot")
	savefig("TrainingEvolution.png")
	println("Save Complete")
end


## Plot convergence
plot_convergence = true
if plot_convergence == true
	plot(convergence_res.x[1:n_itrs], [convergence_res.y[1:n_itrs], convergence_val.y[1:n_itrs]],
        yaxis=:log, title = "Loss Function Convergence" , xlabel = "Iterations" , ylabel = "Loss",
        label=["Training Dataset Convergence" "Validation Dataset Convergence"])
	println("Saving Convergence Plot")
	savefig("Convergence.png")
	println("Save Complete")
end

## Plot Learning Rate Updation

plot(convergence_res.x[1:n_itrs], z[1:n_itrs] , yscale = :log10, legend = false, title = "Learning Rate Updation" , xlabel = "Iterations" , ylabel = "Learning Rate")
savefig( "Learning_Rate_Updation.png")
