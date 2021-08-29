'''
Created on Dec 14, 2017

@author: dduque
'''

'''
Regression model
r_i = b0 + b1*m_i + b2*i + b3* m_i^2

m_i = number of constraints at iteration i

SUMMARY OUTPUT                                
                                
Regression Statistics                                
Multiple R    0.483331535                            
R Square    0.233609373                            
Adjusted R Square    0.233606008                            
Standard Error    0.001940854                            
Observations    683340                            
                                
ANOVA                                
    df    SS    MS    F    Significance F            
Regression    3    0.784621593    0.261540531    69430.95271    0            
Residual    683336    2.57406896    3.76692E-06                    
Total    683339    3.358690553                        
                                
    Coefficients    Standard Error    t Stat    P-value    Lower 95%    Upper 95%    Lower 95.0%    Upper 95.0%
Intercept    -0.000236431    1.10051E-05    -21.48366715    2.3934E-102    -0.000258001    -0.000214861    -0.000258001    -0.000214861
num_ctr    2.19498E-06    3.66744E-08    59.85051378    0    2.1231E-06    2.26686E-06    2.1231E-06    2.26686E-06
pass    -2.02414E-07    1.66117E-08    -12.18503248    3.76569E-34    -2.34973E-07    -1.69856E-07    -2.34973E-07    -1.69856E-07
num_ctr2    1.08989E-09    3.06675E-11    35.53879132    2.2194E-276    1.02978E-09    1.14999E-09    1.02978E-09    1.14999E-09
'''

b0 = -0.00023641
b1 = -2.02414E-07
b2 = 0
b3 = 2.19498E-06
b4 = 1.08989E-09
def predict(m,i):
    return b0 + b1*(i) + b2*(i**2) + b3*(m+i) + b4*(m+i)**2
hline = '----------------------------------------------------------------------------------------------------------------------------------------'    
    
print(hline)
print( '%15s %15s %15s %15s %15s %15s %15s' %('Num. reservoirs' , 'T', '# cuts','Prob.size', 'Pred. RT', 'Pred. RT','Pred. RT'))
print( '%15s %15s %15s %15s %15s %15s %15s' %('' , '', '',' ratio CS/ESS',   'ESS', 'CS', 'ratio CS/ESS'))
print(hline)
num_reservoirs = [50,100,200,500,1000]
iters_test = [i for i  in [10,100,1000,10000]]
horizons = [12,26,52]
sigma_card = 30
for (k,r) in enumerate(num_reservoirs):
    for T in horizons:
        for i in iters_test:
            cs_m = r*3+1
            ess_m = r*4+1
            cs_lp_pred = predict(r*3+1, i)
            ess_lp_pred = predict(r*4+1, i)
            ratio = cs_lp_pred/ess_lp_pred
            
            cs_rt_pred = (T-1)*sum(predict(cs_m, j) for j in range(0,i))  + sum(sigma_card*predict(cs_m, 0) + sigma_card*(T-2)*predict(cs_m, j) for j in range(0,i))
            ess_rt_pred = (T-1)*sum(predict(ess_m, j) for j in range(0,i))  + sum(sigma_card*predict(ess_m, 0) + sigma_card*(T-2)*predict(ess_m, j) for j in range(0,i))
            print( '%15i %15i %15i %15.4f %15.1f %15.1f %15.3f' %(r , T,   i, (r*3+1+i)/(r*4+1+ i) , ess_rt_pred, cs_rt_pred, cs_rt_pred/ess_rt_pred))
    print(hline)        
