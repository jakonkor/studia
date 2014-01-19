function [y] = randSet(x)
    permutation = randperm(length(x));
    tmp = x;
    for i = 1:1:length(x)
        tmp(i,:) = x(permutation(i),:);
    end
    y = tmp;	
    clear tmp;
