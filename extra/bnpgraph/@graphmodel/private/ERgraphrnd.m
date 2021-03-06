function G = ERgraphrnd(n, p)

% ERgraphrnd samples a simple Erdos-Renyi graph  with n nodes and probability of
% connection p and returns the adjacency matrix G
% G = ERgraphrnd(n, p)

G = rand(n)<p;
% Symmetrize and remove self-nodes
G = tril(G) + tril(G)' -2*diag(diag(G));
% Remove nodes with no connections
deg = sum(G);
ind = deg>0;
G = G(ind, :);
G = G(:, ind);
G = sparse(logical(G));