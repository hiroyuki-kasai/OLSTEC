% Define the problem train_cost function. The input X is a structure with
% fields U1, U2, U3, G representing a rank (r1,r2,r3) tensor.
% f(X) = 1/2 * || P.*(X - A) ||^2
function f = compute_cost_tensor(X, P, PA, tensor_dims)
    n1 = tensor_dims(1);
    n2 = tensor_dims(2);
    n3 = tensor_dims(3);
    
    Diff = P.*X - PA;
    Diff_flat = reshape(Diff, n1*n2, n3);
    
    f = .5*norm(Diff_flat , 'fro')^2;          
end

