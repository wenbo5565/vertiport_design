import pyomo.environ as pyo
from pyomo.core.kernel.conic import primal_exponential
import numpy as np

# Create a model named 'MixedLinearExponential'

#################################################
# Section 1 - Initiate models
#################################################
svd = pyo.ConcreteModel('R-SVD')

#################################################
# Section 2 - Define parameters and sets
#################################################
num_region = 10 # num of demand regions (e.g. 10 administrative region within Shenzhen City)
num_loc = 5 # num of candidate vertiports
num_c_loc = 2 # num of competitor vertiports
h_v = 2 # maximal num of open vertiports
h_t = 2 # maximal num of air taxis
M = 1 # number of eVOTL allowed
M_set = [M]
theta = 0.33 # 0.33

avg_travel_time = 0.4 # average ground travel time in hours from i to a vertiport
avg_daily_demand = 400 # 400 single flight daily citywise
avg_hour_demand = avg_daily_demand / 10 / num_region # service starts from 8am and end at 6pm 
avg_waiting_time_compt = 0.8 # average waiting time at competitior port

avg_ev_trip = 0.5 # first order moment of single-trip flight time
avg_ev_load = 0.1 # first order moment of loading and unloading
avg_ev_charging = 0.5 # first order moment of re-charging for next trip

avg_ev_trip_sm = 4 / 3 *  avg_ev_trip ** 2 # second order moment expoential uniform
avg_ev_load_sm = 4 / 3 * avg_ev_load ** 2 # second order moment expoential uniform
avg_ev_charging_sm = 4 / 3 * avg_ev_charging ** 2 # second order moment 

avg_service_time = 2 * avg_ev_trip  + 2 * avg_ev_load + avg_ev_charging
avg_service_time_sm = 4 * avg_ev_trip_sm + 4 * avg_ev_load_sm + avg_ev_charging_sm + \
                      8 * avg_ev_trip * avg_ev_load + 4 * avg_ev_trip * avg_ev_charging + \
                      4 * avg_ev_load * avg_ev_charging

I = list(np.arange(1, num_region + 1))
J = np.arange(1, num_loc + 1)
J_c = np.arange(num_loc + 1, num_loc + num_c_loc + 1) # J_c and J do not share index
J_star = np.concatenate((J, J_c))
size_J_star = len(J_star)

np.random.seed(2025)
lam = np.random.poisson(lam = avg_hour_demand, size = num_region)
lam = lam / sum(lam)

q = np.random.uniform(low = avg_travel_time - 0.2, high = avg_travel_time + 0.2  , size = (num_region, size_J_star))
w_hat = np.random.uniform(low = avg_waiting_time_compt - 0.2, high = avg_waiting_time_compt + 0.2, size = num_c_loc)


###### adding parameters to the model object (index starts from 1)
svd.lam = pyo.Param(I, initialize = {i + 1: lam[i] for i in range(len(lam))})
svd.t = pyo.Param(I, J_star, initialize = {(i, j): q[i - 1, j - 1] for i in I for j in J_star})
svd.w_hat = pyo.Param(J_c, initialize = {j + num_loc + 1: w_hat[j] for j in range(len(w_hat))})

svd.mu_s = pyo.Param(J, initialize = avg_service_time) # first order moment: service time from j to destination (another city)
svd.mu_s_sm = pyo.Param(J, initialize = avg_service_time_sm) # second order moment

svd.theta = pyo.Param(initialize = theta)

svd.J_c = pyo.Set(J_c)
svd.J = pyo.Set(J)

def param_rule_c(svd, i, j):
    return svd.theta * svd.t[i, j] + svd.w_hat[j]
svd.c = pyo.Param(I, J_c, rule = param_rule_c)
# =============================================================================
# def param_rule_mu_s2(svd, j):
#     """
#         assign value to mu s squared
#     """
#     return 1
# svd.mu_s2 = pyo.Param(J, rule = param_rule_mu_s2)
# =============================================================================

### upper and lower bound for continous variable
def param_rule_U_up(svd, j, m):
    return 2
svd.U_up = pyo.Param(J, M_set, rule = param_rule_U_up)

def param_rule_U_low(svd, j, m):
    return 0
svd.U_low = pyo.Param(J, M_set, rule = param_rule_U_low)

def param_rule_eta_up(svd, j):
    return 1
svd.eta_up = pyo.Param(J, rule = param_rule_eta_up)

def param_rule_eta_low(svd, j):
    return 0
svd.eta_low = pyo.Param(J, rule = param_rule_eta_low)

def param_rule_v_up(svd, i, j):
    return 1
svd.v_up = pyo.Param(I, J, rule = param_rule_v_up)

def param_rule_v_low(svd, i, j):
    return 0
svd.v_low = pyo.Param(I, J, rule = param_rule_v_low)

def param_rule_alpha_up(svd, i, l):
    return 1
svd.alpha_up = pyo.Param(I, J, rule = param_rule_alpha_up)

def param_rule_alpha_low(svd, i, l):
    return 0
svd.alpha_low = pyo.Param(I, J, rule = param_rule_alpha_low)

#################################################
# Section 3 - Create decision variables
#################################################
svd.y = pyo.Var(J, domain = pyo.Binary)  # A nonnegative variable for the linear part
# svd.y[1].fix(1)
# svd.y[2].fix(1)


svd.p = pyo.Var(I, J_star, domain = pyo.NonNegativeReals, bounds = (0, 1))
# svd.p[1, 1].fix(0.9)

svd.gamma = pyo.Var(J, M_set, domain = pyo.Binary)
svd.eta = pyo.Var(J, domain = pyo.NonNegativeReals)
svd.beta = pyo.Var(I, J, J, domain = pyo.NonNegativeReals)
svd.alpha = pyo.Var(I, J, domain = pyo.NonNegativeReals)
svd.v = pyo.Var(I, J, domain = pyo.NonNegativeReals)
svd.z = pyo.Var(I, J, domain = pyo.NonNegativeReals)
svd.w = pyo.Var(J, domain = pyo.NonNegativeReals)
svd.U = pyo.Var(J, M_set, domain = pyo.NonNegativeReals)
svd.omega = pyo.Var(J, M_set, domain = pyo.NonNegativeReals)
svd.pi = pyo.Var(J, domain = pyo.NonNegativeReals)

#################################################
# Section 4 - Create Constraint
#################################################

########## reformulated constraint for deterministic constraint

svd.constr_hv = pyo.Constraint(expr = pyo.summation(svd.y) <= h_v)
svd.constr_ht = pyo.Constraint(expr = sum(m * svd.gamma[j, m] for j in J for m in M_set) <= h_t)

def constr_rule_My(svd, j):
    return sum(m * svd.gamma[j, m] for m in M_set) <= M * svd.y[j]
svd.constr_My = pyo.Constraint(J, rule = constr_rule_My)

def constr_rule_gamma(svd, j):
    return sum(svd.gamma[j, m] for m in M_set) <= 1
svd.constr_gamma = pyo.Constraint(J, rule = constr_rule_gamma)

def constr_rule_eta(svd, j):
    return svd.eta[j] == sum(svd.lam[i] * svd.p[i, j] for i in I)
svd.constr_eta = pyo.Constraint(J, rule = constr_rule_eta)

def constr_rule_plessy(svd, i, j):
    return svd.p[i, j] <= svd.y[j]
svd.constr_plessy = pyo.Constraint(I, J, rule = constr_rule_plessy)

########### reformulated constraints for equilibrium probability

##### linear constraint
def constr_linear_p(svd, i, j):
    return sum(svd.beta[i, j, l] for l in J) + svd.p[i, j] * sum(np.exp(-svd.c[i, l]) for l in J_c) <= svd.alpha[i, j]
svd.constr_linear_p = pyo.Constraint(I, J, rule = constr_linear_p)



# svd.constr_exp_v_gr_e = pyo.Constraint(I, J, rule = constr_exp_v_gr_e)

##### exponential cone constraint
# =============================================================================
# def constr_exp_v_gr_e(svd, i, j):
#     return primal_exponential(r = svd.v[i, j], x1 = 1, x2 = svd.z[i, j])
# svd.constr_exp_v_gr_e = pyo.Constraint(I, J, rule = constr_exp_v_gr_e)
# 
# =============================================================================

##### linear approximation to e^{-z} = 1 - z when -z is around 0
def constr_rule_v_approx(svd, i, j):
    return svd.v[i, j] == 1 - svd.z[i, j]
svd.constr_v_approx = pyo.Constraint(I, J, rule = constr_rule_v_approx)

##### McCormick set for alpha
def constr_rule_mc_alpha1(svd, i, j):
    return svd.alpha[i, j] >= svd.v_low[i, j] * svd.y[j]
svd.constr_mc_alpha1 = pyo.Constraint(I, J, rule = constr_rule_mc_alpha1)

def constr_rule_mc_alpha2(svd, i, j):
    return svd.alpha[i, j] >= svd.v_up[i, j] * (svd.y[j] - 1) + svd.v[i, j]
svd.constr_mc_alpha2 = pyo.Constraint(I, J, rule = constr_rule_mc_alpha2)

def constr_rule_mc_alpha3(svd, i, j):
    return svd.alpha[i, j] <= svd.v_up[i, j] * svd.y[j]
svd.constr_mc_alpha3 = pyo.Constraint(I, J, rule = constr_rule_mc_alpha3)

def constr_rule_mc_alpha4(svd, i, j):
    return svd.alpha[i, j] <= svd.v_low[i, j] * (svd.y[j] - 1) + svd.v[i, j]
svd.constr_mc_alpha4 = pyo.Constraint(I, J, rule = constr_rule_mc_alpha4)


##### McCormick set for beta
def constr_rule_mc_beta1(svd, i, j, l):
    return svd.beta[i, j, l] >= svd.p[i, j] * svd.alpha_low[i, l]
svd.constr_mc_beta1 = pyo.Constraint(I, J, J, rule = constr_rule_mc_beta1)

def constr_rule_mc_beta2(svd, i, j, l):
    return svd.beta[i, j, l] >= svd.alpha[i, l] + svd.p[i, j] * svd.alpha_up[i, l] - svd.alpha_up[i, l]
svd.constr_mc_beta2 = pyo.Constraint(I, J, J, rule = constr_rule_mc_beta2)

def constr_rule_mc_beta3(svd, i, j, l):
    return svd.beta[i, j, l] <= svd.alpha[i, l] + svd.p[i, j] * svd.alpha_low[i, l] - svd.alpha_low[i, l]
svd.constr_mc_beta3 = pyo.Constraint(I, J, J, rule = constr_rule_mc_beta3)

def constr_rule_mc_beta4(svd, i, j, l):
    return svd.beta[i, j, l] <= svd.p[i, j] * svd.alpha_up[i, l]
svd.constr_mc_beta4 = pyo.Constraint(I, J, J, rule = constr_rule_mc_beta4)

##### Auxiliary for z
def constr_rule_aux_z(svd, i, j):
    return svd.z[i, j] == svd.theta * (svd.t[i, j] + svd.w[j])
svd.constr_aux_z = pyo.Constraint(I, J, rule = constr_rule_aux_z)

# =============================================================================
# def constr_exp_v_le_e(svd, i, j):
#     return primal_exponential(r = -svd.v[i, j], x1 = -1, x2 = -svd.z[i, j])
# svd_constr_exp_v_le_e = pyo.Constraint(I, J, rule = constr_exp_v_le_e)
# =============================================================================

        



########### reformulated constraints for queueing delay

### McCormick sets for omega
def constr_rule_mc_omega1(svd, j, m):
    return svd.omega[j, m] >= svd.U_low[j, m] * svd.gamma[j, m]
svd.constr_mc_omega1 = pyo.Constraint(J, M_set, rule = constr_rule_mc_omega1)

def constr_rule_mc_omega2(svd, j, m):
    return svd.omega[j, m] >= svd.U_up[j, m] * (svd.gamma[j, m] - 1) + svd.U[j, m]
svd.constr_mc_omega2 = pyo.Constraint(J, M_set, rule = constr_rule_mc_omega2)

def constr_rule_mc_omega3(svd, j, m):
    return svd.omega[j, m] <= svd.U_up[j, m] * svd.gamma[j, m]
svd.constr_mc_omega3 = pyo.Constraint(J, M_set, rule = constr_rule_mc_omega3)

def constr_rule_mc_omega4(svd, j, m):
    return svd.omega[j, m] <= svd.U_low[j, m] * (svd.gamma[j, m] - 1) + svd.U[j, m]
svd.constr_mc_omega4 = pyo.Constraint(J, M_set, rule = constr_rule_mc_omega4)

### McCormick sets for pi
def constr_rule_mc_pi1(svd, j):
    return svd.pi[j] >= svd.eta_low[j] * svd.U[j, 1] + svd.eta[j] * svd.U_low[j, 1] - svd.eta_low[j] * svd.U_low[j, 1]
svd.constr_mc_pi1 = pyo.Constraint(J, rule = constr_rule_mc_pi1)

def constr_rule_mc_pi2(svd, j):
    return svd.pi[j] >= svd.eta_up[j] * svd.U[j, 1] + svd.eta[j] * svd.U_up[j, 1] - svd.eta_up[j] * svd.U_up[j, 1]
svd.constr_mc_pi2 = pyo.Constraint(J, rule = constr_rule_mc_pi2)

def constr_rule_mc_pi3(svd, j):
    return svd.pi[j] <= svd.eta_up[j] * svd.U[j, 1] + svd.eta[j] * svd.U_low[j, 1] - svd.eta_up[j] * svd.U_low[j, 1]
svd.constr_mc_pi3 = pyo.Constraint(J, rule = constr_rule_mc_pi3)

def constr_rule_mc_pi4(svd, j):
    return svd.pi[j] <= svd.eta[j] * svd.U_up[j, 1] + svd.eta_low[j] * svd.U[j, 1] - svd.eta_low[j] * svd.U_up[j, 1]
svd.constr_mc_pi4 = pyo.Constraint(J, rule = constr_rule_mc_pi4)

### non-McCormick constraints related to the queueing delay
def constr_rule_w_omega(svd, j, m):
    return svd.w[j] == 1/2 * svd.omega[j, m]
svd.constr_w_omega = pyo.Constraint(J, M_set, rule = constr_rule_w_omega)

def constr_rule_U1(svd, j):
    return svd.U[j, 1] == svd.pi[j] * svd.mu_s[j] + svd.eta[j] * svd.mu_s_sm[j]
svd.constr_U1 = pyo.Constraint(J, rule = constr_rule_U1)

def constr_rule_gamma_tight(svd, j, m):
    return svd.y[j] <= svd.gamma[j, m]
svd.constr_gamma_tight = pyo.Constraint(J, M_set, rule = constr_rule_gamma_tight)

################## probability constraint
def constr_rule_p_sum(svd, i):
    return sum(svd.p[i, j] for j in J_star) == 1
svd.constr_p = pyo.Constraint(I, rule = constr_rule_p_sum)

#################################################
# Section 5 - Create Objective function
#################################################
svd.objective = pyo.Objective(rule = sum(svd.lam[i] * svd.p[i, j] for i in I for j in J), sense = pyo.maximize)

##################################################
# Section 6 - Solving the model and report result
##################################################


solver = pyo.SolverFactory('mosek')
solver_options = {
    'iparam.num_threads': 4,
    'dparam.intpnt_co_tol_rel_gap': 1e-6,
    'iparam.mio_max_num_branches': 100000,
}
results = solver.solve(svd, tee = True)

########### report result
svd.objective()


svd.pprint()
svd.y.display()
svd.p.display()

svd.v.display()
svd.z.display()
svd.t.display()
svd.U.display()
svd.w.display()
svd.gamma.display()

svd.w_hat.display()

# =============================================================================
# for i in I:
#     for j in J:
#         print('======================================')
#         print(f'v {i} {j} is {svd.v[i, j].value}')
#         print(f'exp is {np.exp(-svd.z[i, j].value)}')
# =============================================================================
