#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:23:32 2022

@author: simonl
"""
import numpy as np

def cons_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('itj,itj->tj'
                          ,taxed_price**(1-p.sigma) 
                          ,b.share_cons_o_np 
                          )
    
    Q = np.einsum('tj,itj -> itj' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) ,  
                  taxed_price ** (-p.sigma))
    return Q   

def iot_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('itj,itjs->tjs'
                          ,taxed_price**(1-p.eta) 
                          ,b.share_cs_o_np 
                          )
    
    M = np.einsum('tjs,itj -> itjs' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) , 
                  taxed_price ** (-p.eta))
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

def solve_p(E, params, baseline):
    p = params
    b = baseline
    C = b.country_number
    S = b.sector_number
    
    tol_p = 1e-8
    p_step = 1
       
    price_new = np.ones(C*S).reshape(C,S)
    price_old = np.zeros((C,S))    
    
    while np.linalg.norm(price_new - price_old)/np.linalg.norm(price_new) > tol_p:        
        price_old = (p_step * price_new + (1-p_step) * price_old)       
        taxed_price = np.einsum('it,itj->itj',
                                price_old,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))        
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta) 
                                  ,b.share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta - 1))            
        prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)       
        wage_hat = np.einsum('js,js->j', E , b.va_share_np )       
        price_new = wage_hat[:,None]**b.gamma_labor_np * prod
    
    return price_new

def solve_E(params, baseline):
    p = params
    b = baseline
    C = b.country_number
    S = b.sector_number
        
    E_step = 2/3
    tol_E = 1e-8
    
    count = 0
    condition = True
    window = 10

    E_new = np.ones(C*S).reshape(C,S)
    E_old = np.zeros((C,S))
    convergence = np.ones(1)
    
    while condition:
    
        E_old = (E_step * E_new + (1-E_step) * E_old)
        
        price = solve_p(E_old, params, baseline)
        
  
        iot_hat_unit = iot_eq_unit(price, params, baseline) 
        
        cons_hat_unit = cons_eq_unit(price, params, baseline)    
           
        A = b.va_np + np.einsum('it,itj,itjs,itjs->js' , 
                              price,
                              p.carb_cost_np*b.co2_intensity_np[:,:,None],
                              iot_hat_unit,
                              b.iot_np)  
        B = np.einsum('itj,itj->itj',
                      cons_hat_unit,
                      b.cons_np)    

        K = b.cons_tot_np - np.einsum( 'it,itj,itj,itj -> j', 
                                                  price, 
                                                  p.carb_cost_np*b.co2_intensity_np[:,:,None], 
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
        
        if count == 0:
            convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
        else:    
            convergence = np.append( convergence , 
                                    np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
        if count > window:
            condition = np.any(convergence[-window:] > tol_E) 
        
        count += 1
        
    return E_new

def solve_E_p(params , baseline):
    
    num_index = baseline.country_list.index(baseline.num_country)
    E = solve_E(params, baseline)
    
    if baseline.num_type == 'output':
        norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.output_np[num_index,:])
    if baseline.num_type == 'wage':
        norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.va_share_np[num_index,:] )
        
    E = E / norm_factor
    
    price = solve_p(E, params, baseline)
    
    results = {'E_hat': E,'p_hat':price}
    return results

def solve_fair_tax(params, baseline):
    
    p = params
    b = baseline
    C = b.country_number
    baseline_deficit_np = b.deficit_np
    
    step = 1/3
    T_tol = 1e-6
    T_old = np.zeros(C)
    T_new = np.ones(C)
    count = 0

    while min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()) > T_tol:
        print('iteration :',count)
        if count !=0:
            T_old=T_new*step+T_old*(1-step)
        
        b.deficit_np = baseline_deficit_np + T_old
        
        results = solve_E_p(p, b)    
        E_hat_sol = results['E_hat']
        p_hat_sol = results['p_hat']

        iot_hat_unit = iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
        taxed_price = np.einsum('it,itj->itj',
                                p_hat_sol,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        price_agg_no_pow = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma) 
                                  ,b.share_cons_o_np 
                                  )       
        price_agg = price_agg_no_pow ** (1/(1 - p.sigma))    
        H = b.cons_tot_np*(price_agg**(beta)).prod(axis=0)
        
        I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va_new = E_hat_sol * b.va_np
        G = np.einsum('js->j',va_new) \
            + np.einsum('itj,itjs->j',p.carb_cost_np*b.co2_intensity_np[:,:,None] ,iot_new) \
            + np.einsum('itj,itj->j',p.carb_cost_np*b.co2_intensity_np[:,:,None] ,cons_new) \
            + b.deficit_np

        T_new = (H*G.sum()/H.sum()-G)

        print('condition', min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()))
        
        count += 1
    
    results = {'E_hat': E_hat_sol,'p_hat':p_hat_sol,'contrib':T_new}
    return results

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
                                    cons_hat_sol**((p.sigma-1)/p.sigma) , 
                                    b.share_cons_o_np ) ** (p.sigma / (p.sigma-1))
    beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
    
    utility_c_hat_sol = (utility_cs_hat_sol**beta).prod(axis=0)
    
    utility_hat_sol = np.einsum('j,j->' , utility_c_hat_sol , b.cons_tot_np/(b.cons_tot_np.sum()))
    
    return emissions_sol, utility_hat_sol, utility_c_hat_sol
    