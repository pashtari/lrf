*SENSE:Minimize
NAME          test123
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
 G  None_elastic_SubProblem_Constraint
COLUMNS
    None_elastic_SubProblem_free_bound  None_elastic_SubProblem_Constraint   1.000000000000e+00
    None_elastic_SubProblem_neg_penalty_var  None_elastic_SubProblem_Constraint   1.000000000000e+00
    None_elastic_SubProblem_neg_penalty_var  obj       -9.000000000000e-01
    None_elastic_SubProblem_pos_penalty_var  None_elastic_SubProblem_Constraint   1.000000000000e+00
    None_elastic_SubProblem_pos_penalty_var  obj        9.000000000000e-01
    w         None_elastic_SubProblem_Constraint   1.000000000000e+00
    w         obj        1.000000000000e+00
    x         c1         1.000000000000e+00
    x         c2         1.000000000000e+00
    x         obj        1.000000000000e+00
    y         c1         1.000000000000e+00
    y         c3        -1.000000000000e+00
    y         obj        4.000000000000e+00
    z         c2         1.000000000000e+00
    z         c3         1.000000000000e+00
    z         obj        9.000000000000e+00
RHS
    RHS       c1         5.000000000000e+00
    RHS       c2         1.000000000000e+01
    RHS       c3         7.000000000000e+00
    RHS       None_elastic_SubProblem_Constraint  -1.000000000000e+00
BOUNDS
 FX BND       None_elastic_SubProblem_free_bound   0.000000000000e+00
 MI BND       None_elastic_SubProblem_neg_penalty_var
 UP BND       None_elastic_SubProblem_neg_penalty_var   0.000000000000e+00
 FR BND       w       
 UP BND       x          4.000000000000e+00
 LO BND       y         -1.000000000000e+00
 UP BND       y          1.000000000000e+00
ENDATA
