function [q N Z rank_Z h q_N] = getNetParams(net, train)
    
    % liczba parametrow sieci
    w = net.IW{1}';             
    bin = net.b{1};               
    v = net.LW{2,1}';           
    bout = net.b{2}; 
    q = numel(w)+numel(bin)+numel(v)+numel(bout);
    
    % jacobian 
    Z = calc_jacobian(net, train);
    rank_Z = rank(Z);
    
    % obliczanie dzwigni h
    [U W V] = svd(Z);
    W = diag(diag(W));
    ZTZ = V*W*W*(V');
    H = Z*(inv(ZTZ))*Z';
    h = diag(H);

    % obliczanie q/N
    N = size(train, 1);
    q_N = q/N;