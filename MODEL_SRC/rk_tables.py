# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020


def rk_coef(expl_order, p_solves='only_final'):
    """
    Lists the Runge-Kutta table coefficients for the 
    explicit integration and implicit pressure correction.
    Alternative schemes are commented out.

    expl_order... order of the time integration
    p_solves... only_final: use pressure solver at final stage only
                default: use pressure solver at every stage

    coeff_sub... a-coefficients to form the next sub-state
    corr_coef... b-coefficients for the implicit pressure correction
    pres_solves... array that indicates at which substeps to use pressure solver instead of guesses
    """

    
    if expl_order == 1:
        #Euler forward scheme
        coeff_sub = [[1.0]]
        corr_coef = [1.0]
        pres_solves = [1]

    elif expl_order == 2:
        #Mid-point method
        coeff_sub = [[0.5, 0.0], [0.0, 1.0]]
        corr_coef = [0.5, 1.0]

        #Heun's second order method
        #coeff_sub = [[0.5, 0.0], [0.5, 0.5]]
        #corr_coef = [0.5, 1.0]  
        if p_solves == 'only_final':
            pres_solves = [0, 1]
        else:
            pres_solves = [1, 1]

    elif expl_order == 3:
        #Strong stability preserving method
        coeff_sub = [[1.0 , 0.0, 0.0], [1.0 / 4.0, 1.0 / 4.0, 0.0], [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]]
        corr_coef = [1.0, 1.0 / 2.0, 1.0]

        #Kutta's method
        #coeff_sub = [[1.0 / 2.0 , 0.0, 0.0], [-1.0, 2.0, 0.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]]
        #corr_coef = [1.0 / 2.0, 1.0, 1.0]
        #Heun's third order method
        #coeff_sub = [[1.0 / 3.0 , 0.0, 0.0], [0.0, 2.0 / 3.0, 0.0], [1.0 / 4.0, 0.0, 3.0 / 4.0]]
        #corr_coef = [1.0 / 3.0, 2.0 / 3.0, 1.0]
        #Ralston's method
        #coeff_sub = [[1.0 / 2.0, 0.0, 0.0], [0.0, 3.0 / 4.0, 0.0], [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0]]
        #corr_coef = [1.0 / 2.0, 3.0 / 4.0, 1.0]

        if p_solves == 'only_final':
            pres_solves = [0, 0, 1]
        else:
            pres_solves = [1, 1, 1]

    elif expl_order == 4:
        #Classic Runge Kutta
        coeff_sub = [[1.0 / 2.0 , 0.0, 0.0, 0.0], [0.0, 1.0 / 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]]
        corr_coef = [1.0 / 2.0, 1.0 / 2.0, 1.0, 1.0]

        #3/8-rule 
        #coeff_sub = [[1.0 / 3.0 , 0.0, 0.0, 0.0], [-1.0 / 3.0, 1.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0], [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0]]
        #corr_coef = [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0]
        if p_solves == 'only_final':
            # only final pressure solve is not sufficient to maintain 4th order in time
            pres_solves = [0, 0, 1, 1]
        else:
            pres_solves = [1, 1, 1, 1]

    else:
        if rank == 0:
            print "Error: There is no time scheme implemented with order greater than 4"
        raise ValueError

   
    return coeff_sub, corr_coef, pres_solves

