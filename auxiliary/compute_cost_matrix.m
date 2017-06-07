% Define the problem train_cost function. The input X is a structure with
% fields U1, U2, U3, G representing a rank (r1,r2,r3) tensor.
% f(X) = 1/2 * || P.*(X - A) ||^2
function f = compute_cost_matrix(X, P, A)
    Diff = P.*X - P.*A;
    f = .5*norm(Diff , 'fro')^2;        
end

