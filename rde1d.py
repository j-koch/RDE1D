#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from clawpack import riemann 
import numpy as np
from clawpack import pyclaw
from scipy.integrate import solve_ivp


def getPrimitive(q,state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']

    rho = q[0,:]
    u = q[1,:]/q[0,:]
    z = q[3,:]/q[0,:]
    P = (gamma-1.)*(q[2,:] - 0.5*(q[1,:]**2/q[0,:]))
    T = P/rho
    
    return rho,u,P,T,z

def step_Euler(solver,state,dt):
    # Get parameters:
    gamma = state.problem_data['gamma']
    Da = state.problem_data['Da']
    Ea = state.problem_data['Ea']
    Tign = state.problem_data['Tign']
    Pref = state.problem_data['Pref']
    rhoref = state.problem_data['rhoref']
    AR = state.problem_data['AR']
    hv = state.problem_data['hv']
    s = state.problem_data['s']

    # ODE solved explicitly with 2-stage, 2nd-order Runge-Kutta method.
    dt2 = dt/2.
    q = state.q
    
    # Get state prior to integrating:
    rho,u,P,T,z = getPrimitive(q,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-10.0,0.5) - np.heaviside(state.t-12.0,0.5))*np.heaviside(1.0-xc,0.0) + np.heaviside(state.t-10.0,0.5)*K
    Ic = np.sqrt(gamma)*(2.0/(gamma+1.0))**((gamma + 1.0)/(2.0*(gamma - 1.0)))
    r = (1.0+(gamma-1.0)/2.0)**(-gamma/(gamma-1.0))
    uref = np.sqrt(Pref/rhoref)
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    s = s*np.heaviside(state.t - 10.0,0.0)
    alpha = Ic*np.sqrt(Pref*rhoref)/(rhoref*uref)
    omega = K*rho*(1.0-z)*Da

    qstar = np.empty(q.shape)
    qstar[0,:] = q[0,:] + dt2*alpha*(H*AR - np.sqrt(P*rho))
    qstar[1,:] = q[1,:]
    qstar[2,:] = q[2,:] + dt2*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    qstar[3,:] = q[3,:] + dt2*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

    # Update primitive variables:
    rho,u,P,T,z = getPrimitive(qstar,state)
    K = np.exp(-Ea*((1.0/T) - (1.0/Tign)))*np.heaviside(T - 1.01,0.0)
    K = 5.0*(np.heaviside(state.t-10.0,0.5) - np.heaviside(state.t-12.0,0.5))*np.heaviside(1.0-xc,0.0) + np.heaviside(state.t-10.0,0.5)*K
    H = (1.0 - np.heaviside(P-r,0.0)*(P-r)/(1.0-r))*np.heaviside(1.0-P,0.0)
    omega = K*rho*(1.0-z)*Da
    
    q[0,:] = q[0,:] + dt*alpha*(H*AR - np.sqrt(P*rho))
    q[1,:] = q[1,:]
    q[2,:] = q[2,:] + dt*((alpha/(gamma-1.0)*(H*AR - T*np.sqrt(P*rho))) + omega*hv)
    q[3,:] = q[3,:] + dt*(omega - (rho*s*H*z) + alpha*(H*AR - np.sqrt(P*rho))*z)

def init(state):
    # Get area and problem data:
    gamma = state.problem_data['gamma']
    L = state.problem_data['L']
    xc = state.grid.x.centers
    
    P = 1.0 + 0.0*xc
    T = 1.0 + 0.0*xc
    z = 0.5*np.sin((2.0*np.pi/L)*xc)+0.5
    rho = P/T

    state.q[0,:] = rho
    state.q[1,:] = 0.0 
    state.q[2,:] = P/(gamma-1.)
    state.q[3,:] = rho*z

    
# Specify Riemann Solver and instantiate solver object:
rs_HLL = riemann.euler_1D_py.euler_rq1D
rs_HLLC = riemann.euler_1D_py.euler_hllc_rq1D_counterProp
solver = pyclaw.ClawSolver1D(rs_HLLC)
solver.kernel_language = 'Python'

# Set Boundary Conditions:
solver.step_source = step_Euler
solver.bc_lower[0]=pyclaw.BC.periodic # custom inlet
solver.bc_upper[0]=pyclaw.BC.periodic # custom outlet
solver.aux_bc_lower[0]=pyclaw.BC.extrap
solver.aux_bc_upper[0]=pyclaw.BC.extrap
solver.max_steps = 100000
solver.cfl_desired = 0.1

# Working Fluid:
pref = 1.0
rhoref = 1.0
gamma = 1.29
R = 1.0
Tref = 1.0

# Injector/mixing:
s = 0.07

# Kinetics:
Da = 208.3
Tign = 6.2
hv = 24.577
Ea = 11.0

# Geometry:
L = 24.0
mx = 4800

# 1-D domain specification:
x = pyclaw.Dimension(0,L,mx,name='x')
domain = pyclaw.Domain([x])
state = pyclaw.State(domain,num_eqn=4,num_aux=4)

# Working fluid:
state.problem_data['gamma'] = gamma
state.problem_data['gamma1'] = gamma-1.0
state.problem_data['Pref'] = pref
state.problem_data['rhoref'] = rhoref
state.problem_data['R'] = R
state.problem_data['Tref'] = Tref

# Injector/mixing:
state.problem_data['s'] = s
state.problem_data['AR'] = AR
state.problem_data['L'] = L

# Kinetics and heat release:
state.problem_data['Da'] = Da
state.problem_data['Tign'] = Tign
state.problem_data['hv'] = hv
state.problem_data['Ea'] = Ea

# Initialize domains, including the aux cells:
init(state)
#initDetTube(state)

# Get coordinates of domain:
xc = state.grid.x.centers
# Initialize arrays. These will be added together to form complete area profile.
state.aux[0,:] = np.ones(xc.shape) #area - needs to be ones always
state.aux[1,:] = np.zeros(xc.shape) #tracker for heat release through time 

# Set up PyClaw Controller:
claw = pyclaw.Controller()
claw.tfinal = 100.0
claw.solution = pyclaw.Solution(state,domain)
claw.solver = solver
claw.num_output_times = 1000
claw.outdir = './_output'
claw.keep_copy = True
claw.write_aux_always = False

#q^T = [rho*A, rho*u*A, E*A, rho*Z]
#2 waves for HLL, 3 waves for HLLC
claw.solver.num_eqn = 4
claw.solver.num_waves = 3 

# Run the simulation:
claw.run()