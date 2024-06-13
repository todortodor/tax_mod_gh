#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:23:32 2022

@author: simonl
"""
import numpy as np
try:
    import aa
except:
    pass
import matplotlib.pyplot as plt
# from time import perf_counter

def cons_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    # taxed_price = np.einsum('it,itj,itj->itj',
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            # p.tau_hat,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    if p.tax_scheme == 'eu_style':
        # taxed_price = np.einsum('it,itj,itj->itj',
        taxed_price = np.einsum('it,itj->itj',
                                price,
                                # p.tau_hat,
                                (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
    
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
                            # p.tau_hat,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    if p.tax_scheme == 'eu_style':
        taxed_price = np.einsum('it,itj->itj',
                                price,
                                # p.tau_hat,
                                (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
    
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

# def compute_I_hat(p_hat, E_hat, params, baseline):
#     p = params
#     b = baseline
    
#     iot_hat_unit = iot_eq_unit(p_hat, params, baseline) 
#     cons_hat_unit = cons_eq_unit(p_hat, params, baseline)
    
#     A = b.va_np + np.einsum('it,itj,itjs,itjs->js' , 
#                           p_hat,
#                           p.carb_cost_np*b.co2_intensity_np[:,:,None],
#                           iot_hat_unit,
#                           b.iot_np)  

#     K = b.cons_tot_np - np.einsum( 'it,itj,itj,itj -> j', 
#                                               p_hat, 
#                                               p.carb_cost_np*b.co2_intensity_np[:,:,None], 
#                                               cons_hat_unit , 
#                                               b.cons_np )       
#     I_hat = (np.einsum('js,js->j' , E_hat , A) + b.deficit_np) / K
#     return I_hat

def solve_p(E, params, baseline, price_init = None):
    p = params
    b = baseline
    C = b.country_number
    S = b.sector_number
    
    tol_p = 1e-8
       
    price_new = None
    if price_init is None:
        price_old = np.ones(C*S).reshape(C,S)  
    else:
        price_old = price_init
        
    condition = True
    count = 0
    
    dim = C*S
    mem = 10
    type1 = True
    regularization = 1e-10
    relaxation=1
    safeguard_factor=1
    max_weight_norm=1e6
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)

    while condition:
        if count>0:
            price_new = price_new.ravel()
            price_old = price_old.ravel()
            aa_wrk.apply(price_new, price_old)
            price_old = price_new.reshape(C,S)        
        taxed_price = np.einsum('it,itj->itj',
                                price_old,
                                # p.tau_hat,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None])) 
        if p.tax_scheme == 'eu_style':
            taxed_price = np.einsum('it,itj->itj',
                                    price_old,
                                    # p.tau_hat,
                                    (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta[None,:,None]) 
                                  ,b.share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
        # price_agg = np.divide(1, 
        #                 price_agg_no_pow , 
        #                 out = np.full_like(price_agg_no_pow,np.inf), 
        #                 where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
        # plt.plot(price_agg_no_pow.ravel())    
        # plt.show()        
        prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
        wage_hat = np.einsum('js,js->j', E , b.va_share_np )    
        price_new = wage_hat[:,None]**b.gamma_labor_np * prod
        
        condition = np.linalg.norm(price_new - price_old)/np.linalg.norm(price_new) > tol_p
        count+=1
        # plt.plot(price_new.ravel())
        # plt.show()
    
    return price_new

# old solver funcs, needs to be updated for tau_hat

# def E_func(E_old,params,baseline,price_init = None):
#     p = params
#     b = baseline    
    
#     price = solve_p(E_old, params, baseline, price_init)
    
#     iot_hat_unit = iot_eq_unit(price, params, baseline) 
#     cons_hat_unit = cons_eq_unit(price, params, baseline)    
   
#     A = b.va_np + np.einsum('it,itj,it,itjs,itjs->js' , 
#                           price,
#                           p.carb_cost_np,
#                           b.co2_intensity_np,
#                           iot_hat_unit,
#                           b.iot_np)  
#     B = np.einsum('itj,itj->itj',
#                   cons_hat_unit,
#                   b.cons_np)    

#     K = b.cons_tot_np - np.einsum( 'it,itj,it,itj,itj -> j', 
#                                               price, 
#                                               p.carb_cost_np,
#                                               b.co2_intensity_np, 
#                                               cons_hat_unit , 
#                                               b.cons_np )
#     Z = np.einsum('itjs,itjs->itjs',
#                   iot_hat_unit,
#                   b.iot_np)        

#     one_over_K = np.divide(1, K) 
#     F = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
#     T = np.einsum('js,itjs -> it', E_old , F )
    
#     E_new = price * (T + np.einsum('itj,j,j->it',B,b.deficit_np,one_over_K)) / b.output_np
    
#     # plt.plot(T.ravel(),label = 'T')
#     # plt.plot(np.einsum('itj,j,j->it',B,b.deficit_np,one_over_K).ravel(), label = 'B')
#     # plt.legend()
#     # plt.show()
    
#     E_new = E_new / E_new.mean()
    
#     return E_new, price

# def solve_E(params, baseline, E_init = None):
#     C = baseline.country_number
#     S = baseline.sector_number
        
#     tol_E = 1e-8
#     convergence_window = 2
#     smooth_large_jumps = True
#     plot_history = False
#     plot_convergence = False
    
#     count = 0
#     condition = True

#     E_new = None
#     price_init = None
#     if E_init is None:
#         E_old = np.ones(C*S).reshape(C,S)
#     else:
#         E_old = E_init

#     dim = C*S
#     mem = 5 #memory of Anderson Acceleration, empirically 5 is good, is optimization between
#             #number of steps and computational time of 1 step
#     type1 = False #type of Anderson Acceleration, empirically should stay 2, so False
#     regularization = 1e-10 #lower value is faster, higher value is more stable, stable between 1e-8 and 1e-12
#     relaxation=1/4 #doesnt matter
#     safeguard_factor=1 #1 is max and it's fine
#     max_weight_norm=1e6 #prevents diverging
#     aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
#                                     regularization=regularization,
#                                     relaxation=relaxation, 
#                                     safeguard_factor=safeguard_factor, 
#                                     max_weight_norm=max_weight_norm)
#     if plot_history or plot_convergence:
#         E_history = []
#         t1 = perf_counter()
        
#     while condition:
#         # print(count)
#         if count>0:
#             E_new = E_new.ravel()
#             E_old = E_old.ravel()
#             aa_wrk.apply(E_new, E_old)
            
#             if smooth_large_jumps:
#                 high_jumps_too_big = E_new > 1000*E_old
#                 if np.any(high_jumps_too_big):
#                     E_new[high_jumps_too_big] = E_old[high_jumps_too_big]*1/2+E_new[high_jumps_too_big]*1/2
#                 low_jumps_too_big = E_new < 1000*E_old
#                 if np.any(low_jumps_too_big):
#                     E_new[low_jumps_too_big] = E_old[low_jumps_too_big]*1/2+E_new[low_jumps_too_big]*1/2
                    
#             E_new[E_new<0]=0
#             E_old = E_new.reshape(C,S)
#             # E_old = (E_new.reshape(C,S)+E_old.reshape(C,S))/2
        
#         E_new, price_init = E_func(E_old,params,baseline, price_init)
        
#         assert not np.any(np.isnan(E_new)), "nan in E solver"
        
#         if count == 0:
#             convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
#         else:    
#             convergence = np.append(convergence , 
#                                     np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
#         condition = np.any(convergence[-convergence_window:] > tol_E)
#         count += 1
        
#         # plt.plot(E_new.ravel()-E_old.ravel())
#         # plt.show()
        
#         # plt.semilogy(convergence)
#         # plt.show()
        
#         if plot_history or plot_convergence:
#             E_history.append(E_old)
            
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

# def solve_E_p(params , baseline, E_init = None):
    
#     try:
#         num_index = baseline.country_list.index(baseline.num_country)
#     except:
#         pass
#     E = solve_E(params, baseline, E_init)
    
#     if baseline.num_type == 'output':
#         if baseline.num_country == 'WLD':
#             norm_factor = np.einsum('js,js->', E , baseline.output_np) / baseline.output_np.sum()
#         else:
#             norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.output_np[num_index,:])
#     if baseline.num_type == 'wage':
#         if baseline.num_country == 'WLD':
#             norm_factor = np.einsum('js,js,j->', E , baseline.va_share_np, baseline.va_np.sum(axis=1)) / baseline.va_np.sum()
#         else:
#             norm_factor = np.einsum('s,s->', E[num_index,:] , baseline.va_share_np[num_index,:] )
        
#     E = E / norm_factor
    
#     price = solve_p(E, params, baseline)
    
#     results = {'E_hat': E,'p_hat':price}
#     return results

def solve_one_loop(params, baseline, vec_init = None, tol=1e-9, damping=5):
    C = baseline.country_number
    S = baseline.sector_number
    p = params
    b = baseline  

    tol = tol
    smooth_large_jumps = True
    
    count = 0
    condition = True

    vec_new = None
    p_new = None
    E_new = None
    I_new = None
    if vec_init is None:
        E_old = np.ones(C*S).reshape(C,S)
        p_old = np.ones(C*S).reshape(C,S)
        I_old = np.ones(C).reshape(C)
    else:
        E_old = vec_init[:C*S].reshape(C,S)
        p_old = vec_init[C*S:2*C*S].reshape(C,S)
        I_old = vec_init[2*C*S:].reshape(C)
        # I_old = vec_init[C*S:].reshape(C)

    dim = 2*C*S+C
    # dim = C*S+C
    mem = 20 #memory of Anderson Acceleration, empirically 5 is good, is optimization between
            #number of steps and computational time of 1 step
    type1 = False #type of Anderson Acceleration, empirically should stay 2, so False
    regularization = 1e-10 #lower value is faster, higher value is more stable, stable between 1e-8 and 1e-12
    relaxation=1/4 #doesnt matter
    safeguard_factor=1 #1 is max and it's fine
    max_weight_norm=1e6 #prevents diverging
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)
    # damping = 5
    # damping = 2
    safe_convergence = 5e-3
    convergence = None
    # print(params.carb_cost_df)
    while condition:
        if count>0:
            vec_new = np.concatenate([E_new.ravel(),p_new.ravel(),I_new.ravel()])
            vec_old = np.concatenate([E_old.ravel(),p_old.ravel(),I_old.ravel()])
            # vec_new = np.concatenate([E_new.ravel(),I_new.ravel()])
            # vec_old = np.concatenate([E_old.ravel(),I_old.ravel()])
            # aa_wrk.apply(vec_new, vec_old)
            
            if convergence[-1] < safe_convergence:
                aa_wrk.apply(vec_new, vec_old)
            # if convergence[-1] < safe_convergence:
            #     damping = 1
                # damping = 2
            
            # if smooth_large_jumps:
            #     high_jumps_too_big = vec_new > 1000*vec_old
            #     if np.any(high_jumps_too_big):
            #         vec_new[high_jumps_too_big] = vec_old[high_jumps_too_big]*1/2+vec_new[high_jumps_too_big]*1/2
            #     low_jumps_too_big = vec_new < 1000*vec_old
            #     if np.any(low_jumps_too_big):
            #         vec_new[low_jumps_too_big] = vec_old[low_jumps_too_big]*1/2+vec_new[low_jumps_too_big]*1/2
            
            vec_new[vec_new<0]=0
            vec_old = (vec_new + (damping-1)*vec_old)/damping
            # vec_old = vec_new
            E_old = vec_old[:C*S].reshape(C,S)
            p_old = vec_old[C*S:2*C*S].reshape(C,S)
            # I_old = vec_old[C*S:].reshape(C)
            I_old = vec_old[2*C*S:].reshape(C)
        
        # price = solve_p(E_old, params, baseline, price_init)
        
        # p_old = solve_p(E_old, params, baseline)
        
        iot_hat_unit = iot_eq_unit(p_old, params, baseline) 
        cons_hat_unit = cons_eq_unit(p_old, params, baseline)    
       
        # A = b.va_np + np.einsum('it,itj,it,itjs,itjs->js' , 
        #                       p_old,
        #                       p.carb_cost_np,
        #                       b.co2_intensity_np,
        #                       iot_hat_unit,
        #                       b.iot_np)  
        # B = np.einsum('itj,itj->itj',
        #               cons_hat_unit,
        #               b.cons_np)    

        # K = b.cons_tot_np - np.einsum( 'it,itj,it,itj,itj -> j', 
        #                                           p_old, 
        #                                           p.carb_cost_np,
        #                                           b.co2_intensity_np, 
        #                                           cons_hat_unit , 
        #                                           b.cons_np )
        # Z = np.einsum('itjs,itjs->itjs',
        #               iot_hat_unit,
        #               b.iot_np)        

        # one_over_K = np.divide(1, K) 
        # F = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
        # T = np.einsum('js,itjs -> it', E_old , F )
        
        # E_new = p_old * (T + np.einsum('itj,j,j->it',B,b.deficit_np,one_over_K)) / b.output_np
        
        A = np.einsum('j,it,itj,itj->it',
                      I_old,
                      p_old,
                      # p.tau_hat,
                      cons_hat_unit,
                      b.cons_np)
        
        B = np.einsum('js,it,itjs,itjs->it',
                      E_old,
                      p_old,
                      # p.tau_hat,
                      iot_hat_unit,
                      b.iot_np)
        
        E_new = (A+B)/b.output_np
        
        # print(p.tax_scheme)
        
        if p.tax_scheme == 'consumer':
        
            A = np.einsum('js,js->j',
                          E_new,
                          b.va_np)
            
            B = np.einsum('j,it,itj,it,itj,itj->j',
                          I_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          cons_hat_unit,
                          b.cons_np)
            
            K = np.einsum('js,it,itj,it,itjs,itjs->j',
                          E_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          iot_hat_unit,
                          b.iot_np)
        
        if p.tax_scheme == 'producer':
        
            A = np.einsum('js,js->j',
                          E_new,
                          b.va_np)
            
            B = np.einsum('i,jt,jti,jt,jti,jti->j',
                          I_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          cons_hat_unit,
                          b.cons_np)
            
            K = np.einsum('is,jt,jti,jt,jtis,jtis->j',
                          E_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          iot_hat_unit,
                          b.iot_np)
            
        if p.tax_scheme == 'eu_style':
        
            A = np.einsum('js,js->j',
                          E_new,
                          b.va_np)
            
            B_1 = np.einsum('i,jt,jti,jt,jti,jti->j',
                          I_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          cons_hat_unit,
                          b.cons_np)
            
            B_2 = np.einsum('j,it,itj,it,itj,itj->j',
                          I_old,
                          p_old,
                          # p.tau_hat,
                          np.maximum((p.carb_cost_np-np.einsum('itj->jti',p.carb_cost_np)),0),
                          b.co2_intensity_np,
                          cons_hat_unit,
                          b.cons_np)
            
            B = B_1 + B_2
            
            K_1 = np.einsum('is,jt,jti,jt,jtis,jtis->j',
                          E_old,
                          p_old,
                          # p.tau_hat,
                          p.carb_cost_np,
                          b.co2_intensity_np,
                          iot_hat_unit,
                          b.iot_np)
            
            K_2 = np.einsum('js,it,itj,it,itjs,itjs->j',
                          E_old,
                          p_old,
                          # p.tau_hat,
                          np.maximum((p.carb_cost_np-np.einsum('itj->jti',p.carb_cost_np)),0),
                          b.co2_intensity_np,
                          iot_hat_unit,
                          b.iot_np)
            
            K = K_1 + K_2
        
        # I_new = (A+B+K+b.deficit_np) / np.einsum('itj->j',b.cons_np)
        # print('a',A.shape,'b',B.shape,'k',K.shape)
        I_new = (A+B+K+b.deficit_np) / b.cons_tot_np
        
        taxed_price = np.einsum('it,itj->itj',
                                p_old,
                                # p.tau_hat,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        
        if p.tax_scheme == 'eu_style':
            taxed_price = np.einsum('it,itj->itj',
                                    p_old,
                                    # p.tau_hat,
                                    (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
        
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta[None,:,None]) 
                                  ,b.share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
  
        prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
        wage_hat = np.einsum('js,js->j', E_old , b.va_share_np )    
        
        p_new = wage_hat[:,None]**b.gamma_labor_np * prod
        
        # p_new = p_old
        
        # E_new = E_new / E_new.mean()
        # I_new = I_new / E_new.mean()
        # p_new = p_new / E_new.mean()
        
        # E_new, price_init = E_func(E_old,params,baseline, price_init)
        if count == 0:
            convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
        else:    
            vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
            vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
            convergence = np.append(convergence , 
                                    np.linalg.norm(vec_new - vec_old)/np.linalg.norm(vec_old) )
            # vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
            # vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
            # convergence = np.append(convergence , 
            #                         np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
        condition = convergence[-1] > tol
        count += 1
        # print(E_new,p_new)
        if count % 100 == 0 and count>0:
            plt.semilogy(convergence)
            # plt.plot(convergence)
            plt.show()
            print(count,convergence[-1])
            # damping = damping*2
        # if count > 50:
        #    #  plt.plot(vec_new-vec_old)
        #    #  # # plt.plot(convergence)
        #    #  plt.show()
        #    # # print((vec_new/vec_old).max(),(vec_new/vec_old).min())
        #    print(np.linalg.norm(vec_new))
        
    # E_new = E_new / E_new.mean()
    # I_new = I_new / E_new.mean()
    # p_new = p_new / E_new.mean()
            
    # plt.semilogy(convergence)
    # # plt.plot(convergence)
    # plt.show()
    
    try:
        num_index = baseline.country_list.index(baseline.num_country)
    except:
        pass
    
    if baseline.num_type == 'output':
        if baseline.num_country == 'WLD':
            norm_factor = np.einsum('js,js->', E_new , baseline.output_np) / baseline.output_np.sum()
        else:
            norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.output_np[num_index,:])
    if baseline.num_type == 'wage':
        if baseline.num_country == 'WLD':
            norm_factor = np.einsum('js,js,j->', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1)) / baseline.va_np.sum()
        else:
            norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.va_share_np[num_index,:] )
    if baseline.num_type == 'no_scaling':
        norm_factor = 1
    
    E_new = E_new / norm_factor
    p_new = p_new / norm_factor
    I_new = I_new / norm_factor
    
    results = {'E_hat': E_new,'p_hat':p_new,'I_hat':I_new}
    return results

# def solve_autarky(params, baseline, vec_init = None, tol=1e-10):
#     C = baseline.country_number
#     S = baseline.sector_number
#     p = params
#     b = baseline  

#     tol = tol
#     smooth_large_jumps = True
    
#     count = 0
#     condition = True

#     vec_new = None
#     p_new = None
#     E_new = None
#     I_new = None
#     if vec_init is None:
#         E_old = np.ones(C*S).reshape(C,S)
#         p_old = np.ones(C*S).reshape(C,S)
#         I_old = np.ones(C).reshape(C)
#     else:
#         E_old = vec_init[:C*S].reshape(C,S)
#         p_old = vec_init[C*S:2*C*S].reshape(C,S)
#         I_old = vec_init[2*C*S:].reshape(C)
#         # I_old = vec_init[C*S:].reshape(C)

#     dim = 2*C*S+C
#     # dim = C*S+C
#     mem = 20 #memory of Anderson Acceleration, empirically 5 is good, is optimization between
#             #number of steps and computational time of 1 step
#     type1 = False #type of Anderson Acceleration, empirically should stay 2, so False
#     regularization = 1e-10 #lower value is faster, higher value is more stable, stable between 1e-8 and 1e-12
#     relaxation=1/4 #doesnt matter
#     safeguard_factor=1 #1 is max and it's fine
#     max_weight_norm=1e6 #prevents diverging
#     aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
#                                     regularization=regularization,
#                                     relaxation=relaxation, 
#                                     safeguard_factor=safeguard_factor, 
#                                     max_weight_norm=max_weight_norm)
#     damping = 5
#     safe_convergence = 1e-3
#     convergence = None
    
#     while condition:
#         if count>0:
#             vec_new = np.concatenate([E_new.ravel(),p_new.ravel(),I_new.ravel()])
#             vec_old = np.concatenate([E_old.ravel(),p_old.ravel(),I_old.ravel()])
#             # vec_new = np.concatenate([E_new.ravel(),I_new.ravel()])
#             # vec_old = np.concatenate([E_old.ravel(),I_old.ravel()])
#             # aa_wrk.apply(vec_new, vec_old)
            
#             if smooth_large_jumps:
#                 high_jumps_too_big = vec_new > 1000*vec_old
#                 if np.any(high_jumps_too_big):
#                     vec_new[high_jumps_too_big] = vec_old[high_jumps_too_big]*1/2+vec_new[high_jumps_too_big]*1/2
#                 low_jumps_too_big = vec_new < 1000*vec_old
#                 if np.any(low_jumps_too_big):
#                     vec_new[low_jumps_too_big] = vec_old[low_jumps_too_big]*1/2+vec_new[low_jumps_too_big]*1/2
#             if convergence[-1] < safe_convergence:
#                 damping = 2
                    
#             vec_new[vec_new<0]=0
#             vec_old = (vec_new + (damping-1)*vec_old)/damping
#             # vec_old = (vec_new/vec_new.mean() + (damping-1)*vec_old)/damping
#             # vec_old = vec_new
#             E_old = vec_old[:C*S].reshape(C,S)
#             p_old = vec_old[C*S:2*C*S].reshape(C,S)
#             # I_old = vec_old[C*S:].reshape(C)
#             I_old = vec_old[2*C*S:].reshape(C)
        
#         # price = solve_p(E_old, params, baseline, price_init)
        
#         # p_old = solve_p(E_old, params, baseline)
        
#         # iot_hat_unit = iot_eq_unit(p_old, params, baseline) 
#         # cons_hat_unit = cons_eq_unit(p_old, params, baseline)    
       
#         # A = b.va_np + np.einsum('it,itj,it,itjs,itjs->js' , 
#         #                       p_old,
#         #                       p.carb_cost_np,
#         #                       b.co2_intensity_np,
#         #                       iot_hat_unit,
#         #                       b.iot_np)  
#         # B = np.einsum('itj,itj->itj',
#         #               cons_hat_unit,
#         #               b.cons_np)    

#         # K = b.cons_tot_np - np.einsum( 'it,itj,it,itj,itj -> j', 
#         #                                           p_old, 
#         #                                           p.carb_cost_np,
#         #                                           b.co2_intensity_np, 
#         #                                           cons_hat_unit , 
#         #                                           b.cons_np )
#         # Z = np.einsum('itjs,itjs->itjs',
#         #               iot_hat_unit,
#         #               b.iot_np)        

#         # one_over_K = np.divide(1, K) 
#         # F = np.einsum('itj,js,j -> itjs' , B , A , one_over_K ) + Z    
#         # T = np.einsum('js,itjs -> it', E_old , F )
        
#         # E_new = p_old * (T + np.einsum('itj,j,j->it',B,b.deficit_np,one_over_K)) / b.output_np
        
#         A = np.einsum('i,it,ti->it',
#                       I_old,
#                       1/(1+np.einsum('iti,it->it',p.carb_cost_np,b.co2_intensity_np)),
#                       b.cons_np.sum(axis=0))
        
#         B = np.einsum('is,it,tis->it',
#                       E_old,
#                       1/(1+np.einsum('iti,it->it',p.carb_cost_np,b.co2_intensity_np)),
#                       b.iot_np.sum(axis=0))
        
#         E_new = (A+B)/b.output_np

#         A = np.einsum('js,js->j',
#                       E_new,
#                       b.va_np)
        
#         B = np.einsum('j,jt,jt,tj->j',
#                       I_old,
#                       np.einsum('iti,it->it',p.carb_cost_np,b.co2_intensity_np),
#                       1/(1+np.einsum('iti,it->it',p.carb_cost_np,b.co2_intensity_np)),
#                       b.cons_np.sum(axis=0))
        
#         K = np.einsum('j,jt,jt,tjs->j',
#                       I_old,
#                       np.einsum('jtj,jt->jt',p.carb_cost_np,b.co2_intensity_np),
#                       1/(1+np.einsum('jtj,jt->jt',p.carb_cost_np,b.co2_intensity_np)),
#                       b.iot_np.sum(axis=0))
        
#         # I_new = (A+B+K+b.deficit_np) / np.einsum('itj->j',b.cons_np)
#         # print('a',A.shape,'b',B.shape,'k',K.shape)
#         # I_new = (A+B+K+b.deficit_np) / b.cons_tot_np
#         I_new = (A+B+K) / b.cons_tot_np
        
#         # taxed_price = np.einsum('jt,jt->jt',
#         #                         p_old,
#         #                         (1+np.einsum('jtj,jt->jt',p.carb_cost_np,b.co2_intensity_np)))
        
#         taxed_price = np.einsum('it,itj->itj',
#                                 p_old,
#                                 (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
                
#         price_agg_no_pow = np.einsum('jtj,jtjs->tjs'
#                                   ,taxed_price**(1-p.eta[None,:,None]) 
#                                   ,b.share_cs_o_np 
#                                   )       
#         price_agg = np.divide(1, 
#                         price_agg_no_pow , 
#                         out = np.ones_like(price_agg_no_pow), 
#                         where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))     
  
#         prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
#         wage_hat = np.einsum('js,js->j', E_old , b.va_share_np )    
        
#         p_new = wage_hat[:,None]**b.gamma_labor_np * prod
        
#         # p_new = p_old
        
#         # vec_new = np.concatenate()
        
        
        
#         # E_new, price_init = E_func(E_old,params,baseline, price_init)
#         vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
#         vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
#         if count == 0:
#             convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
#         else:    
#             convergence = np.append(convergence , 
#                                     np.linalg.norm(vec_new - vec_old)/np.linalg.norm(vec_old) )
#             # vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
#             # vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
#             # convergence = np.append(convergence , 
#             #                         np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
#         condition = convergence[-1] > tol
#         count += 1
#         # print(E_new,p_new)
#         if count % 50 == 0:
#             plt.semilogy(convergence)
#             # plt.plot(convergence)
#             plt.show()
#             print(count,convergence[-1])
#         # if count > 50:
#         #     plt.plot(vec_new-vec_old)
#         #     # # plt.plot(convergence)
#         #     plt.show()
#         #     print((vec_new/vec_old).max(),(vec_new/vec_old).min())
        
#         # E_new = E_new / vec_new.mean()
#         # I_new = I_new / vec_new.mean()
#         # p_new = p_new / vec_new.mean()
        
#     # plt.semilogy(convergence)
#     # # plt.plot(convergence)
#     # plt.show()
    
#     try:
#         num_index = baseline.country_list.index(baseline.num_country)
#     except:
#         pass
    
#     if baseline.num_type == 'output':
#         if baseline.num_country == 'WLD':
#             norm_factor = np.einsum('js,js->', E_new , baseline.output_np) / baseline.output_np.sum()
#         else:
#             norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.output_np[num_index,:])
#     if baseline.num_type == 'wage':
#         if baseline.num_country == 'WLD':
#             norm_factor = np.einsum('js,js,j->', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1)) / baseline.va_np.sum()
#         else:
#             norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.va_share_np[num_index,:] )
        
#     E_new = E_new / norm_factor
#     p_new = p_new / norm_factor
#     I_new = I_new / norm_factor
    
#     results = {'E_hat': E_new,'p_hat':p_new,'I_hat':I_new}
#     return results

def autarky_cons_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('jt,jtj->jt',
                            price,
                            # p.tau_hat,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('jt,jtj->jt'
                          ,taxed_price**(1-p.sigma[None,:]) 
                          ,b.share_cons_o_np 
                          )
    
    Q = np.einsum('jt,jt -> tj' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) ,  
                  taxed_price ** (-p.sigma[None,:]))
    return Q   

def autarky_iot_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('jt,jtj->jt',
                            price,
                            # p.tau_hat,
                            (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
    
    price_agg_no_pow = np.einsum('jt,jtjs->tjs'
                          ,taxed_price**(1-p.eta[None,:]) 
                          ,b.share_cs_o_np 
                          )
    
    M = np.einsum('tjs,jt -> tjs' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) , 
                  taxed_price ** (-p.eta[None,:]))
    return M 

def solve_autarky(params, baseline, vec_init = None, tol=1e-10):
    C = baseline.country_number
    S = baseline.sector_number
    p = params
    b = baseline  

    tol = tol
    smooth_large_jumps = True
    
    count = 0
    condition = True

    vec_new = None
    p_new = None
    E_new = None
    I_new = None
    if vec_init is None:
        E_old = np.ones(C*S).reshape(C,S)
        p_old = np.ones(C*S).reshape(C,S)
        I_old = np.ones(C).reshape(C)
    else:
        E_old = vec_init[:C*S].reshape(C,S)
        p_old = vec_init[C*S:2*C*S].reshape(C,S)
        I_old = vec_init[2*C*S:].reshape(C)
        # I_old = vec_init[C*S:].reshape(C)

    dim = 2*C*S+C
    # dim = C*S+C
    mem = 20 #memory of Anderson Acceleration, empirically 5 is good, is optimization between
            #number of steps and computational time of 1 step
    type1 = False #type of Anderson Acceleration, empirically should stay 2, so False
    regularization = 1e-10 #lower value is faster, higher value is more stable, stable between 1e-8 and 1e-12
    relaxation=1/4 #doesnt matter
    safeguard_factor=1 #1 is max and it's fine
    max_weight_norm=1e6 #prevents diverging
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)
    damping = 5
    safe_convergence = 1e-3
    convergence = None
    
    while condition:
        if count>0:
            vec_new = np.concatenate([E_new.ravel(),p_new.ravel(),I_new.ravel()])
            vec_old = np.concatenate([E_old.ravel(),p_old.ravel(),I_old.ravel()])
            # vec_new = np.concatenate([E_new.ravel(),I_new.ravel()])
            # vec_old = np.concatenate([E_old.ravel(),I_old.ravel()])
            aa_wrk.apply(vec_new, vec_old)
            # if count>0:
            #     E_new = E_new / E_new.mean(axis=1)[:,None]
            #     I_new = I_new / E_new.mean(axis=1)[:,None]
            #     p_new = p_new / E_new.mean(axis=1)[:,None]
            # if count>0:
            #     E_new = E_new / vec_new.mean()
            #     I_new = I_new / vec_new.mean()
            #     p_new = p_new / vec_new.mean()
            
            if smooth_large_jumps:
                high_jumps_too_big = vec_new > 1000*vec_old
                if np.any(high_jumps_too_big):
                    vec_new[high_jumps_too_big] = vec_old[high_jumps_too_big]*1/2+vec_new[high_jumps_too_big]*1/2
                low_jumps_too_big = vec_new < 1000*vec_old
                if np.any(low_jumps_too_big):
                    vec_new[low_jumps_too_big] = vec_old[low_jumps_too_big]*1/2+vec_new[low_jumps_too_big]*1/2
            if convergence[-1] < safe_convergence:
                damping = 2
                    
            vec_new[vec_new<0]=0
            vec_old = (vec_new + (damping-1)*vec_old)/damping
            # vec_old = vec_old/vec_old.mean()
            # vec_old = vec_new
            E_old = vec_old[:C*S].reshape(C,S)
            p_old = vec_old[C*S:2*C*S].reshape(C,S)
            # I_old = vec_old[C*S:].reshape(C)
            I_old = vec_old[2*C*S:].reshape(C)
            # if count>0:
            #     E_old = E_old / E_old.mean(axis=1)[:,None]
            #     I_old = I_old / E_old.mean(axis=1)
            #     p_old = p_old / E_old.mean(axis=1)[:,None]
            
        
        iot_hat_unit = autarky_iot_eq_unit(p_old, params, baseline) 
        cons_hat_unit = autarky_cons_eq_unit(p_old, params, baseline)    
        
        A = np.einsum('j,jt,tj,jtj->jt',
                      I_old,
                      p_old,
                      # p.tau_hat,
                      cons_hat_unit,
                      b.cons_np)
        
        B = np.einsum('js,jt,tjs,jtjs->jt',
                      E_old,
                      p_old,
                      # p.tau_hat,
                      iot_hat_unit,
                      b.iot_np)
        
        E_new = (A+B)/b.output_np
        
        A = np.einsum('js,js->j',
                      E_new,
                      b.va_np)
        
        B = np.einsum('j,jt,jtj,jt,tj,jtj->j',
                      I_old,
                      p_old,
                      # p.tau_hat,
                      p.carb_cost_np,
                      b.co2_intensity_np,
                      cons_hat_unit,
                      b.cons_np)
        
        K = np.einsum('js,jt,jtj,jt,tjs,jtjs->j',
                      E_old,
                      p_old,
                      # p.tau_hat,
                      p.carb_cost_np,
                      b.co2_intensity_np,
                      iot_hat_unit,
                      b.iot_np)
        
        # I_new = (A+B+K+b.deficit_np)/b.cons_tot_np
        I_new = (A+B+K)/b.cons_tot_np
        
        taxed_price = np.einsum('jt,jtj->jt',
                                p_old,
                                # p.tau_hat,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        
        price_agg_no_pow = np.einsum('jt,jtjs->tjs'
                                  ,taxed_price**(1-p.eta[None,:]) 
                                  ,b.share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
  
        prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
        wage_hat = np.einsum('js,js->j', E_old , b.va_share_np )    
        
        p_new = wage_hat[:,None]**b.gamma_labor_np * prod
        
        # p_new = p_old
        
        
        # E_new, price_init = E_func(E_old,params,baseline, price_init)
        if count == 0:
            convergence = np.array([np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old)])
        else:    
            vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
            vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
            convergence = np.append(convergence , 
                                    np.linalg.norm(vec_new - vec_old)/np.linalg.norm(vec_old) )
            # vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
            # vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
            # convergence = np.append(convergence , 
            #                         np.linalg.norm(E_new - E_old)/np.linalg.norm(E_old) )
        
        condition = convergence[-1] > tol
        count += 1
        # print(E_new,p_new)
        if count % 500 == 0:
            plt.semilogy(convergence)
            # plt.plot(convergence)
            plt.show()
            print(count,convergence[-1])
        # if count > 500:
        #     plt.plot(vec_new-vec_old)
        #     # # plt.plot(convergence)
        #     plt.show()
        #     print((vec_new/vec_old).max(),(vec_new/vec_old).min(),np.linalg.norm(vec_old))
            
    # plt.semilogy(convergence)
    # # plt.plot(convergence)
    # plt.show()
    
    # try:
    #     num_index = baseline.country_list.index(baseline.num_country)
    # except:
    #     pass
    
    # if baseline.num_type == 'output':
    #     if baseline.num_country == 'WLD':
    #         norm_factor = np.einsum('js,js->', E_new , baseline.output_np) / baseline.output_np.sum()
    #     else:
    #         norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.output_np[num_index,:])
    # if baseline.num_type == 'wage':
    #     if baseline.num_country == 'WLD':
    #         norm_factor = np.einsum('js,js,j->', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1)) / baseline.va_np.sum()
    #     else:
    #         norm_factor = np.einsum('s,s->', E_new[num_index,:] , baseline.va_share_np[num_index,:] )
        
    # E_new = E_new / np.einsum('js,js,j->j', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1))[:,None]
    # p_new = p_new / np.einsum('js,js,j->j', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1))[:,None]
    # I_new = I_new / np.einsum('js,js,j->j', E_new , baseline.va_share_np, baseline.va_np.sum(axis=1))
    
    results = {'E_hat': E_new,'p_hat':p_new,'I_hat':I_new}
    return results

def solve_fair_tax(params, baseline):
    
    p = params
    b = baseline.make_np_arrays().compute_shares_and_gammas()
    C = b.country_number
    baseline_deficit_np = b.deficit_np
    
    assert np.all(p.tau_hat == 1), 'Fair tax solver mot updated for counterfactual trade costs'
    
    T_tol = 1e-9
    T_old = np.zeros(C)
    T_new = None
    # E_init = None
    vec_init = None
    
    count = 0
    condition =True
    
    dim = C
    mem = 5
    type1 = False
    regularization = 1e-10
    relaxation=1
    safeguard_factor=1
    max_weight_norm=1e6 
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)
    
    while condition:
        print('iteration :',count)
        if count !=0:
            aa_wrk.apply(T_new, T_old)
            T_old = T_new
        
        b.deficit_np = baseline_deficit_np + T_old
        
        # results = solve_E_p(p, b, E_init)    
        results = solve_one_loop(p, b, vec_init)    
        E_hat_sol = results['E_hat']
        p_hat_sol = results['p_hat']
        I_hat_sol = results['I_hat']

        iot_hat_unit = iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
        taxed_price = np.einsum('it,itj->itj',
                                p_hat_sol,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        if p.tax_scheme == 'eu_style':
            taxed_price = np.einsum('it,itj->itj',
                                    p_hat_sol,
                                    (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
        price_agg_no_pow = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma[None,:,None]) 
                                  ,b.share_cons_o_np 
                                  )       
        price_agg = price_agg_no_pow ** (1/(1 - p.sigma[:,None]))    
        H = b.cons_tot_np*(price_agg**(beta)).prod(axis=0)
        
        # I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va_new = E_hat_sol * b.va_np
        if p.tax_scheme == 'consumer':
            G = np.einsum('js->j',va_new) \
                + np.einsum('itj,it,itjs->j',p.carb_cost_np,b.co2_intensity_np,iot_new) \
                + np.einsum('itj,it,itj->j',p.carb_cost_np,b.co2_intensity_np,cons_new) \
                + baseline_deficit_np
        if p.tax_scheme == 'producer':
            G = np.einsum('js->j',va_new) \
                + np.einsum('jti,jt,jtis->j',p.carb_cost_np,b.co2_intensity_np,iot_new) \
                + np.einsum('jti,jt,jti->j',p.carb_cost_np,b.co2_intensity_np,cons_new) \
                + baseline_deficit_np
        if p.tax_scheme == 'eu_style':
            G = np.einsum('js->j',va_new) \
                + np.einsum('jti,jt,jtis->j',p.carb_cost_np,b.co2_intensity_np,iot_new) \
                + np.einsum('itj,it,itjs->j',np.maximum((p.carb_cost_np-np.einsum('itj->jti',p.carb_cost_np)),0),b.co2_intensity_np,iot_new) \
                + np.einsum('jti,jt,jti->j',p.carb_cost_np,b.co2_intensity_np,cons_new) \
                + np.einsum('itj,it,itj->j',np.maximum((p.carb_cost_np-np.einsum('itj->jti',p.carb_cost_np)),0),b.co2_intensity_np,cons_new) \
                + baseline_deficit_np

        T_new = (H*G.sum()/H.sum()-G)
        
        condition = min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()) > T_tol
        print('condition', min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()))
        
        count += 1
        
        U_new = (G + T_old) / H
        
        # if count > 1:
        #     print(U_new-1)
        #     plt.plot((U_new-U_old)/np.abs(U_old-1))
        #     plt.show()
        # if count == 1:
        #     print(U_new-1)
        #     plt.plot(U_new-1)
        #     plt.show()
        
        U_old = U_new.copy()
        
        # E_init = E_hat_sol
        vec_init = np.concatenate([results['E_hat'].ravel(),
                                    results['p_hat'].ravel(),
                                    results['I_hat'].ravel()] )
    
    results = {'E_hat': E_hat_sol,'p_hat':p_hat_sol,'I_hat':I_hat_sol,'contrib':T_new}
    return results

def solve_pol_pay_tax(params, baseline):
    
    p = params
    b = baseline.make_np_arrays().compute_shares_and_gammas()
    C = b.country_number
    baseline_deficit_np = b.deficit_np
    
    assert np.all(p.tau_hat == 1), 'Polluter pays solver mot updated for counterfactual trade costs'
    
    T_tol = 1e-9
    T_old = np.zeros(C)
    T_new = None
    # E_init = None
    vec_init = None
    
    count = 0
    condition =True
    
    dim = C
    mem = 5
    type1 = False
    regularization = 1e-10
    relaxation=1
    safeguard_factor=1
    max_weight_norm=1e6 
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)
    
    while condition:
        print('iteration :',count)
        if count !=0:
            aa_wrk.apply(T_new, T_old)
            T_old = T_new
        
        b.deficit_np = baseline_deficit_np + T_old
        
        # results = solve_E_p(p, b, E_init)    
        results = solve_one_loop(p, b, vec_init)    
        E_hat_sol = results['E_hat']
        p_hat_sol = results['p_hat']
        I_hat_sol = results['I_hat']

        iot_hat_unit = iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
        taxed_price = np.einsum('it,itj->itj',
                                p_hat_sol,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        if p.tax_scheme == 'eu_style':
            taxed_price = np.einsum('it,itj->itj',
                                    p_hat_sol,
                                    (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
        price_agg_no_pow = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma[None,:,None]) 
                                  ,b.share_cons_o_np 
                                  )       
        price_agg = price_agg_no_pow ** (1/(1 - p.sigma[:,None]))    
        H = b.cons_tot_np*(price_agg**(beta)).prod(axis=0)
        
        # I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va_new = E_hat_sol * b.va_np
        G = np.einsum('js->j',va_new) \
            + np.einsum('itj,it,itjs->j',p.carb_cost_np,b.co2_intensity_np,iot_new) \
            + np.einsum('itj,it,itj->j',p.carb_cost_np,b.co2_intensity_np,cons_new) \
            + baseline_deficit_np
        
        omega = b.cumul_emissions_share_np/b.cons_tot_np
        
        T_new = H-G+omega*H*(G-H).sum()/(omega*H).sum()
        
        condition = min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()) > T_tol
        print('condition', min((np.abs(T_old-T_new)).max(),(np.abs(T_old-T_new)/T_new).max()))
        
        count += 1
        
        vec_init = np.concatenate([results['E_hat'].ravel(),
                                    results['p_hat'].ravel(),
                                    results['I_hat'].ravel()] )
    
    results = {'E_hat': E_hat_sol,'p_hat':p_hat_sol,'I_hat':I_hat_sol,'contrib':T_new}
    return results

def solve_fair_carb_price(params, baseline, temp):
    
    p = params
    b = baseline.make_np_arrays().compute_shares_and_gammas()
    C = b.country_number
    
    assert np.all(p.tau_hat == 1), 'Fair carb price solver not updated for counterfactual trade costs'
    
    carb_price_tol = 1e-8
    # carb_price_old = np.ones(C)*4.5e-5 / b.num
    carb_price_old = np.ones(C)*45 / b.num
    carb_price_new = None
    # E_init = None
    vec_init = None
    
    count = 0
    condition =True
    
    dim = C
    mem = 5
    type1 = False
    regularization = 1e-10
    relaxation=1
    safeguard_factor=1
    max_weight_norm=1e6 
    aa_wrk = aa.AndersonAccelerator(dim, mem, type1, 
                                    regularization=regularization,
                                    relaxation=relaxation, 
                                    safeguard_factor=safeguard_factor, 
                                    max_weight_norm=max_weight_norm)
    
    convergence = []
    damping = 5
    
    # temporary for further extension
    delta = 1
    lamb = 1
    uniform_eq_utility = np.ones(C)
    emissions_eq = 40483.75805
    
    while condition:
        print('iteration :',count)
        if count !=0:
            aa_wrk.apply(carb_price_new, carb_price_old)
            carb_price_old = (carb_price_new+(damping-1)*carb_price_old)/damping
            # carb_price_old = carb_price_new
            carb_price_old[carb_price_old<1e-2/b.num] = 1e-2/b.num
            # if np.min(carb_price_old) < 1e-6:
            #     carb_price_old = carb_price_old + (1e-6-np.min(carb_price_old))
            
        params.update_carb_cost(carb_price_old)
        # if count == 0:
        #     params.num_scale_carb_cost(baseline.num, inplace = True)
        p=params
        # print((params.carb_cost_df*1e6*b.num).round(1))
        print((params.carb_cost_df*b.num).round(1))
        # print((params.carb_cost_np == 1e-6).sum()/64/42)
        # results = solve_E_p(p, b, E_init)    
        results = solve_one_loop(p, b, vec_init)    
        E_hat_sol = results['E_hat']
        p_hat_sol = results['p_hat']
        I_hat_sol = results['I_hat']
        
        # print((E_hat_sol*b.co2_prod_np).sum())
        

        iot_hat_unit = iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
        taxed_price = np.einsum('it,itj->itj',
                                p_hat_sol,
                                (1+p.carb_cost_np*b.co2_intensity_np[:,:,None]))
        price_agg_no_pow = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma[None,:,None]) 
                                  ,b.share_cons_o_np 
                                  )       
        price_agg = price_agg_no_pow ** (1/(1 - p.sigma[:,None]))    
        H = b.cons_tot_np*(price_agg**(beta)).prod(axis=0)
        
        iot_new = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons_new = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va_new = E_hat_sol * b.va_np
        
        omega = b.cumul_emissions_share_np/b.cons_tot_np
        
        A = va_new.sum(axis=1)+b.deficit_np
        B = np.einsum('itj,it->j',
                        cons_new,
                        baseline.co2_intensity_np) +\
            np.einsum('itjs,it->j',
                        iot_new,
                        baseline.co2_intensity_np)
        
        # print(B.sum()/1e6)
        print(B.sum())
        
        # carb_price_new = (np.einsum('j,j,j,,->j',
        #                   H,
        #                   (omega+1)**(delta*(1-lamb)),
        #                   uniform_eq_utility**(1-delta),
        #                   emissions_eq + (A/carb_price_old).sum(),
        #                   1/np.sum(H*(omega+1)**(delta*(1-lamb))/carb_price_old)
        #                   ) - A)/B
        # carb_price_new = (np.einsum('j,,->j',
        #                   H,
        #                   emissions_eq + (A/carb_price_old).sum(),
        #                   1/np.sum(H/carb_price_old)
        #                   ) - A)/B
        
        emissions_sol, utility, utility_countries = compute_emissions_utility(results, 
                                                                              params, 
                                                                              baseline, 
                                                                              autarky=False)
        plt.plot(b.country_list,utility_countries-1)
        plt.show()
        
        carb_price_new = H * (
            (emissions_eq + np.sum(A/carb_price_old))/np.sum(H/carb_price_old)
            ) / (B + A/carb_price_old)
        
        # carb_price_new = H * utility_countries.mean() / (B + A/carb_price_old)
        
            # (np.einsum('j,,->j',
            #               H,
            #               emissions_eq + (A/carb_price_old).sum(),
            #               1/np.sum(H/carb_price_old)
            #               ) - A)/B
        
        condition = max((np.abs(carb_price_old-carb_price_new)).max(),(np.abs(carb_price_old-carb_price_new)/carb_price_old).max()
                        ) > carb_price_tol and count < 5000
        print(count,'condition', max((np.abs(carb_price_old-carb_price_new)).max(),(np.abs(carb_price_old-carb_price_new)/carb_price_old).max()
                        ))
        # print((carb_price_new*1e6).round(1))
        
        count += 1
        
        vec_init = np.concatenate([results['E_hat'].ravel(),
                                    results['p_hat'].ravel(),
                                    results['I_hat'].ravel()] )
        
        temp['E_hat'] = E_hat_sol
        temp['p_hat'] = p_hat_sol
        temp['I_hat'] = I_hat_sol
        temp['carb_price'] = carb_price_new
        
        convergence.append((np.abs(carb_price_old-carb_price_new)/carb_price_old).max())
        plt.semilogy(convergence)
        plt.show()
    
    results = {'E_hat': E_hat_sol,'p_hat':p_hat_sol,'I_hat':I_hat_sol,'carb_price':carb_price_new}
    return results

class helper_for_min:
    def __init__(self):
        self.guess = None
        self.count = 0
        self.history = []
    
    def update_initial_guess(self,guess):
        self.guess = guess
        
    def update_count(self):
        self.count = self.count+1
        
    def show_progress(self,emissions_sol,utility_countries):
        if self.count > 0 and self.count % 50 == 0:
            print('count:',self.count,',norm:',self.norm,',emissions:',emissions_sol)
            plt.plot(utility_countries)
            plt.show()

def to_minimize_under_constraint_from_carb_price(carb_price, params, baseline, helper):
    params.update_carb_cost(carb_price)
    p = params
    b = baseline
    C = b.country_number
    helper.update_count()
    results = solve_one_loop(p, b, helper.guess)    
    emissions_sol, utility, utility_countries = compute_emissions_utility(results, 
                                                                          params, 
                                                                          baseline, 
                                                                          autarky=False)
    helper.results = results
    helper.emissions = emissions_sol
    helper.utility_countries = utility_countries
    helper.carb_price = carb_price
    
    # res = np.concatenate([(100*(utility_countries-utility_countries.mean()))**8,
    #                       (100*np.array(emissions_sol-emissions_target)/emissions_target)[None]])
    
    res = (100*(utility_countries-utility_countries.mean()))**2
    
    helper.res = res
    
    helper.update_initial_guess(np.concatenate([results['E_hat'].ravel(),
                                results['p_hat'].ravel(),
                                results['I_hat'].ravel()] ))
    helper.show_progress(emissions_sol,utility_countries)
    helper.norm = np.linalg.norm(res)
    
    helper.history.append(carb_price)
    
    return np.linalg.norm(res)

def constraint_from_carb_price(carb_price, params, baseline, helper):
    params.update_carb_cost(carb_price)
    p = params
    b = baseline
    C = b.country_number
    helper.update_count()
    results = solve_one_loop(p, b, helper.guess)    
    emissions_sol, utility, utility_countries = compute_emissions_utility(results, 
                                                                          params, 
                                                                          baseline, 
                                                                          autarky=False)
    helper.results = results
    helper.emissions = emissions_sol
    helper.utility_countries = utility_countries
    helper.carb_price = carb_price
    
    # res = np.concatenate([(100*(utility_countries-utility_countries.mean()))**8,
    #                       (100*np.array(emissions_sol-emissions_target)/emissions_target)[None]])
    
    res = (100*(utility_countries-utility_countries.mean()))**2
    
    helper.res = res
    
    helper.update_initial_guess(np.concatenate([results['E_hat'].ravel(),
                                results['p_hat'].ravel(),
                                results['I_hat'].ravel()] ))
    helper.show_progress(emissions_sol,utility_countries)
    helper.norm = np.linalg.norm(res)
    
    helper.history.append(carb_price)
    
    return (emissions_sol - 39399.01705150698)/39399.01705150698

def to_minimize_from_carb_price(carb_price, params, baseline, helper,
                                emissions_target=39387.3):
    params.update_carb_cost(carb_price)
    p = params
    b = baseline
    C = b.country_number
    helper.update_count()
    results = solve_one_loop(p, b, helper.guess)    
    emissions_sol, utility, utility_countries = compute_emissions_utility(results, 
                                                                          params, 
                                                                          baseline, 
                                                                          autarky=False)
    helper.results = results
    helper.emissions = emissions_sol
    helper.utility_countries = utility_countries
    helper.carb_price = carb_price
    
    # res = np.concatenate([(100*(utility_countries-utility_countries.mean()))**8,
    #                       (100*np.array(emissions_sol-emissions_target)/emissions_target)[None]])
    
    res = (100*(utility_countries-utility_countries.mean()))
    
    helper.res = res
    
    helper.update_initial_guess(np.concatenate([results['E_hat'].ravel(),
                                results['p_hat'].ravel(),
                                results['I_hat'].ravel()] ))
    helper.show_progress(emissions_sol,utility_countries)
    helper.norm = np.linalg.norm(res)
    
    # helper.history.append(carb_price)
    
    return res

def scalar_to_minimize_from_carb_price(carb_price, params, baseline, helper,
                                emissions_target=39387.3):
    params.update_carb_cost(carb_price)
    p = params
    b = baseline
    C = b.country_number
    helper.update_count()
    results = solve_one_loop(p, b, helper.guess)    
    emissions_sol, utility, utility_countries = compute_emissions_utility(results, 
                                                                          params, 
                                                                          baseline, 
                                                                          autarky=False)
    helper.results = results
    helper.emissions = emissions_sol
    helper.utility_countries = utility_countries
    helper.carb_price = carb_price
    
    # res = np.concatenate([(100*(utility_countries-utility_countries.mean()))**2,
    #                      (100*np.array(emissions_sol-emissions_target)/emissions_target)[None]])
    
    res = (100*(utility_countries-utility_countries.mean()))**8
    
    helper.update_initial_guess(np.concatenate([results['E_hat'].ravel(),
                                results['p_hat'].ravel(),
                                results['I_hat'].ravel()] ))
    helper.show_progress(emissions_sol,utility_countries)
    helper.norm = np.linalg.norm(res)
    
    return np.linalg.norm(res)

def compute_emissions_utility(results, params, baseline, autarky = False):
    p = params
    # b = baseline
    b = baseline.make_np_arrays().compute_shares_and_gammas()
    E_hat_sol = results['E_hat']
    p_hat_sol = results['p_hat']
    I_hat_sol = results['I_hat']
    
    q_hat_sol = E_hat_sol /p_hat_sol       
    emissions_sol = np.einsum('js,js->', q_hat_sol, b.co2_prod_np)
    # emissions_sol = np.einsum('js,js->', E_hat_sol, b.co2_prod_np)
    
    if 'contrib' in results.keys():
        b.deficit_np = b.deficit_np + results['contrib']
    
    # I_hat_sol = compute_I_hat(p_hat_sol, E_hat_sol, params, b)
    
    cons_hat_sol = np.einsum('j,itj->itj',  I_hat_sol , cons_eq_unit(p_hat_sol, params, baseline)) 
    
    utility_cs_hat_sol = np.einsum('itj,itj->tj', 
                                    cons_hat_sol**((p.sigma[None,:,None]-1)/p.sigma[None,:,None]) , 
                                    b.share_cons_o_np ) ** (p.sigma[:,None] / (p.sigma[:,None]-1))
    beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
    
    utility_c_hat_sol = (utility_cs_hat_sol**beta).prod(axis=0)
    
    utility_hat_sol = np.einsum('j,j->' , utility_c_hat_sol , b.cons_tot_np/(b.cons_tot_np.sum()))
    
    if autarky:
        cons_hat_sol = np.einsum('j,tj->tj',  I_hat_sol , autarky_cons_eq_unit(p_hat_sol, params, baseline)) 
        
        utility_cs_hat_sol = np.einsum('tj,jtj->tj', 
                                        cons_hat_sol**((p.sigma[:,None]-1)/p.sigma[:,None]) , 
                                        b.share_cons_o_np ) ** (p.sigma[:,None] / (p.sigma[:,None]-1))
        beta = np.einsum('jtj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)
        
        utility_c_hat_sol = (utility_cs_hat_sol**beta).prod(axis=0)
        
        utility_hat_sol = np.einsum('j,j->' , utility_c_hat_sol , b.cons_tot_np/(b.cons_tot_np.sum()))
        
    
    return emissions_sol, utility_hat_sol, utility_c_hat_sol


