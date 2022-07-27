#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:23:32 2022

@author: simonl
"""
import numpy as np
# import aa
import matplotlib.pyplot as plt
from time import perf_counter

def cons_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('itj,itj->tj'
                          ,taxed_price**(1-p.sigma[None,:,None]) 
                          ,b.share_cons_o_np 
                          )
    
    Q = np.einsum('tj,itj -> itj' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) ,  
                  taxed_price ** (-p.sigma[None,:,None]))
    return Q   

def iot_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('itj,itjs->tjs'
                          ,taxed_price**(1-p.eta[None,:,None]) 
                          ,b.share_cs_o_np 
                          )
    
    M = np.einsum('tjs,itj -> itjs' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) , 
                  taxed_price ** (-p.eta[None,:,None]))
    return M 

def compute_I_hat(p_hat, E_hat, params, baseline):
    p = params
    b = baseline
    
    iot_hat_unit = iot_eq_unit(p_hat, params, baseline) 
    cons_hat_unit = cons_eq_unit(p_hat, params, baseline)
    
    A = b.va_np + np.einsum('it,itj,itjs,itjs->js' , 
                          p_hat,
                          p.carb_cost_np*b.co2_intensity_np[:,:,None],
                          iot_hat_unit,
                          b.iot_np)  

    K = b.cons_tot_np - np.einsum( 'it,itj,itj,itj -> j', 
                                              p_hat, 
                                              p.carb_cost_np*b.co2_intensity_np[:,:,None], 
                                              cons_hat_unit , 
                                              b.cons_np )       
    I_hat = (np.einsum('js,js->j' , E_hat , A) + b.deficit_np) / K
    return I_hat

# def solve_p(E, params, baseline, price_init = None):
#     p = params
#     b = baseline
#     C = b.country_number
#     S = b.sector_number
#
#     tol_p = 1e-8
#
#     price_new = None
#     if price_init is None:
#         price_old = np.ones(C*S).reshape(C,S)
#     else:
#         price_old = price_init
#
#     condition = True
#     count = 0
#
#     dim = C*S
#     mem = 10
#     type1 = True
#     regularization = 1e-10
#     relaxation=1
#     safeguard_factor=1
#     max_weight_norm=1e6
#     aa_wrk = aa.AndersonAccelerator(dim, mem, type1,
#                                     regularization=regularization,
#                                     relaxation=relaxation,
#                                     safeguard_factor=safeguard_factor,
#                                     max_weight_norm=max_weight_norm)
#
#     while condition:
#         if count!=0:
#             price_new = price_new.ravel()
#             price_old = price_old.ravel()
#             aa_wrk.apply(price_new, price_old)
#             price_old = price_new.reshape(C,S)
#         taxed_price = np.einsum('it,itj->itj',
#                                 price_old,
#                                 (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
#         price_agg_no_pow = np.einsum('itj,itjs->tjs'
#                                   ,taxed_price**(1-p.eta[None,:,None])
#                                   ,b.share_cs_o_np
#                                   )
#         price_agg = np.divide(1,
#                         price_agg_no_pow ,
#                         out = np.ones_like(price_agg_no_pow),
#                         where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
#         prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
#         wage_hat = np.einsum('js,js->j', E , b.va_share_np )
#         price_new = wage_hat[:,None]**b.gamma_labor_np * prod
#
#         condition = np.linalg.norm(price_new - price_old)/np.linalg.norm(price_new) > tol_p
#         count+=1
#
#     return price_new

def E_func(E_old,params,baseline,price_init = None):
    p = params
    b = baseline    

    price = solve_p(E_old, params, baseline, price_init)
    
    iot_hat_unit = iot_eq_unit(price, params, baseline) 
    cons_hat_unit = cons_eq_unit(price, params, baseline)    
   
    A = b.va_np + np.einsum('it,itj,it,itjs,itjs->js' , 
                          price,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          iot_hat_unit,
                          b.iot_np)  
    B = np.einsum('itj,itj->itj',
                  cons_hat_unit,
                  b.cons_np)    

    K = b.cons_tot_np - np.einsum( 'it,itj,it,itj,itj -> j', 
                                              price, 
                                              p.carb_cost_np,
                                              b.co2_intensity_np, 
                                              cons_hat_unit , 
                                              b.cons_np )
    Z = np.einsum('itjs,itjs->itjs',
                  iot_hat_unit,
                  b.iot_np)        

    one_over_K = np.divide(1, K) 
    F = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
    T = np.einsum('js,itjs -> it', E_old , F )
    
    E_new = price * (T + np.einsum('itj,j,j->it',B,b.deficit_np,one_over_K)) / b.output_np
    
    E_new = E_new / E_new.mean()
    
    return E_new, price

# def solve_E(params, baseline, E_init = None):
#     C = baseline.country_number
#     S = baseline.sector_number
#
#     tol_E = 1e-8
#     convergence_window = 2
#     smooth_large_jumps = True
#     plot_history = False
#     plot_convergence = False
#
#     count = 0
#     condition = True
#
#     E_new = None
#     price_init = None
#     if E_init is None:
#         E_old = np.ones(C*S).reshape(C,S)
#     else:
#         E_old = E_init
#
#     dim = C*S
#     mem = 5
#     type1 = False
#     regularization = 1e-10
#     relaxation=1/4
#     safeguard_factor=1
#     max_weight_norm=1e6
#     aa_wrk = aa.AndersonAccelerator(dim, mem, type1,
#                                     regularization=regularization,
#                                     relaxation=relaxation,
#                                     safeguard_factor=safeguard_factor,
#                                     max_weight_norm=max_weight_norm)
#     if plot_history or plot_convergence:
#         E_history = []
#         t1 = perf_counter()
#
#     while condition:
#         if count!=0:
#             E_new = E_new.ravel()
#             E_old = E_old.ravel()
#             aa_wrk.apply(E_new, E_old)
#
#             if smooth_large_jumps:
#                 high_jumps_too_big = E_new > 1000*E_old
#                 if np.any(high_jumps_too_big):
#                     E_new[high_jumps_too_big] = E_old[high_jumps_too_big]*1/2+E_new[high_jumps_too_big]*1/2
#                 low_jumps_too_big = E_new < 1000*E_old
#                 if np.any(low_jumps_too_big):
#                     E_new[low_jumps_too_big] = E_old[low_jumps_too_big]*1/2+E_new[low_jumps_too_big]*1/2
#
#             E_new[E_new<0]=0
#             E_old = E_new.reshape(C,S)
#
#         E_new, price_init = E_func(E_old,params,baseline, price_init)
#
#         assert not np.any(np.isnan(E_new)), "nan in E solver"
#
#         if count == 0:
#             convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
#         else:
#             convergence = np.append(convergence ,
#                                     np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
#
#         condition = np.any(convergence[-convergence_window:] > tol_E)
#         count += 1
#
#         if plot_history or plot_convergence:
#             E_history.append(E_old)
#
#
#
#     if plot_history or plot_convergence:
#         t2=perf_counter()
#         distance_history = [np.linalg.norm(E_new - E) for E in E_history]
#     if plot_history:
#         plt.plot(np.array([E_new.ravel() - E.ravel() for E in E_history]).T, lw=1)
#         plt.title(str(t2-t1))
#         plt.show()
#     if plot_convergence:
#         plt.semilogy(distance_history)
#         plt.title(str(t2-t1))
#         plt.show()
#         plt.plot([np.linalg.norm(E) for E in E_history], lw = 2)
#         plt.plot([np.linalg.norm(E_new) for E in E_history], lw = 2)
#         plt.show()
#     return E_new

def solve_E_p(params , baseline, E_init = None):
    
    num_index = baseline.country_list.index(baseline.num_country)
    E = solve_E(params, baseline, E_init)
    
    if baseline.num_type == 'output':
        norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.output_np[num_index,:])
    if baseline.num_type == 'wage':
        norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.va_share_np[num_index,:] )
        
    E = E / norm_factor
    
    price = solve_p(E, params, baseline)
    
    results = {'E_hat': E,'p_hat':price}
    return results

# def solve_fair_tax(params, baseline):
#
#     p = params
#     b = baseline
#     C = b.country_number
#     baseline_deficit_np = b.deficit_np
#
#     T_tol = 1e-6
#     T_old = np.zeros(C)
#     T_new = None
#     E_init = None
#
#     count = 0
#     condition =True
#
#     dim = C
#     mem = 5
#     type1 = False
#     regularization = 1e-10
#     relaxation=1
#     safeguard_factor=1
#     max_weight_norm=1e6
#     aa_wrk = aa.AndersonAccelerator(dim, mem, type1,
#                                     regularization=regularization,
#                                     relaxation=relaxation,
#                                     safeguard_factor=safeguard_factor,
#                                     max_weight_norm=max_weight_norm)
#
#     while condition:
#         print('iteration :',count)
#         if count !=0:
#             aa_wrk.apply(T_new, T_old)
#             T_old = T_new
#
#         b.deficit_np = baseline_deficit_np + T_old
#
#         results = solve_E_p(p, b, E_init)
#         E_hat_sol = results['E_hat']
#         p_hat_sol = results['p_hat']
#
#         iot_hat_unit = iot_eq_unit(p_hat_sol, p, b)
#         cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)
#         beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
#         taxed_price = np.einsum('it,itj->itj',
#                                 p_hat_sol,
#                                 (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
#         price_agg_no_pow = np.einsum('itj,itj->tj'
#                                   ,taxed_price**(1-p.sigma[None,:,None])
#                                   ,b.share_cons_o_np
#                                   )
#         price_agg = price_agg_no_pow ** (1/(1 - p.sigma[:,None]))
#         H = b.cons_tot_np*(price_agg**(beta)).prod(axis=0)
#
#         I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, p, b)
#         iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
#         cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
#         va_new = E_hat_sol * b.va_np
#         G = np.einsum('js->j',va_new) \
#             + np.einsum('itj,it,itjs->j',p.carb_cost_np,b.co2_intensity_np,iot_new) \
#             + np.einsum('itj,it,itj->j',p.carb_cost_np,b.co2_intensity_np,cons_new) \
#             + baseline_deficit_np
#
#         T_new = (H*G.sum()/H.sum()-G)
#
#         condition = min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()) > T_tol
#         print('condition', min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()))
#
#         count += 1
#
#         E_init = E_hat_sol
#
#     results = {'E_hat': E_hat_sol,'p_hat':p_hat_sol,'contrib':T_new}
#     return results

def compute_emissions_utility(results, params, baseline):
    p = params
    b = baseline
    E_hat_sol = results['E_hat']
    p_hat_sol = results['p_hat']
    
    q_hat_sol = E_hat_sol /p_hat_sol       
    emissions_sol = np.einsum('js,js->', q_hat_sol, b.co2_prod_np)

    I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, params, baseline)
    
    cons_hat_sol = np.einsum('j,itj->itj',  I_hat_sol , cons_eq_unit(p_hat_sol, params, baseline)) 
    
    utility_cs_hat_sol = np.einsum('itj,itj->tj', 
                                    cons_hat_sol**((p.sigma[None,:,None]-1)/p.sigma[None,:,None]) , 
                                    b.share_cons_o_np ) ** (p.sigma[:,None] / (p.sigma[:,None]-1))
    beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
    
    utility_c_hat_sol = (utility_cs_hat_sol**beta).prod(axis=0)
    
    utility_hat_sol = np.einsum('j,j->' , utility_c_hat_sol , b.cons_tot_np/(b.cons_tot_np.sum()))
    
    return emissions_sol, utility_hat_sol, utility_c_hat_sol