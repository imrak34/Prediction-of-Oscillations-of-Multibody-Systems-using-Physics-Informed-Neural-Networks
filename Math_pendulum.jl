####################################
# Mathematical pendulum
#
# store variables
# plot state
#
####################################

using Plots

## create object pendulum
mutable struct Math_pendulum
    l::Float64
	g::Float64
    m::Float64
	c::Float64
	k::Float64
	phi::Float64
	phi_dot::Float64
	phi_prev::Float64
    Math_pendulum(l, g, m, c, k, phi, phi_dot) = new(l, g, m, c, k, phi, phi_dot)
end


## create empty figure
function create_fig(mp::Math_pendulum, hfig=500, bfig=500, size_factor = 1.1)
    p_mat = zeros(2,5)
	p_mat[:,1] = zeros(2)   + [-mp.l,-mp.l]*size_factor
	p_mat[:,2] = p_mat[:,1] + [2*mp.l,0]*size_factor
	p_mat[:,3] = p_mat[:,2] + [0,2*mp.l]*size_factor
	p_mat[:,4] = p_mat[:,3] + [-2*mp.l,0]*size_factor
	p_mat[:,5] = p_mat[:,1]
	box_border_plot = plot(p_mat[1,:],p_mat[2,:],border=:none,aspect_ratio=1,
	            legend=false,color="gray",lw=1,fmt=:pdf)
	plot!(size=(bfig,hfig))
	return box_border_plot
end

## plot the state of the pendulum
function plot_state(mp::Math_pendulum)
	plot_rod(mp)
	plot_mass(mp)
	plot_bearing(mp)
	plot_hinge(mp)
end

## plot bearing
function plot_bearing(mp::Math_pendulum,size=0.1,linew=1.0)
	bearing = zeros(2,4)
	bearing[:,1] = [0,0]
	bearing[:,2] = bearing[:,1] + [size*0.75,size]
	bearing[:,3] = bearing[:,2] + [-2*size*0.75,0]
	bearing[:,4] = bearing[:,3] + [size*0.75,-size]
	return plot!(bearing[1,:],bearing[2,:],color="black",lw=linew)
end

## plot status of the pendulum
function plot_rod(mp::Math_pendulum)
	pos_x = +mp.l*sin(mp.phi)
	pos_y = -mp.l*cos(mp.phi)
	rod = zeros(2, 2)
	rod[:,1] = [pos_x, pos_y]
	return plot!(rod[1,:],rod[2,:],color="black", lw=2)
end

## plot mass of the mathematical pendulum
function plot_mass(mp::Math_pendulum, size=15.0)
	pos_x = +mp.l*sin(mp.phi)
	pos_y = -mp.l*cos(mp.phi)
	return scatter!([pos_x],[pos_y],markersize=size,
	                color="black", markerstrokecolor="black")
end

## plot the hinge at the bearing
function plot_hinge(mp::Math_pendulum, size=6.0)
	return scatter!([0.0],[0.0],markersize=size,color="white",
	                markerstrokecolor="black",markerstrokewidths=2.0)
end
