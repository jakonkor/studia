%Normalizacja i centrowanie
function [y] = normSet(x)
    result = x;
    tmp = studentize(x);
    normX = norm(tmp(:,1), inf);
    normY = norm(tmp(:,2), inf);
    result(:,1) = tmp(:,1) ./ normX;
    result(:,2) = tmp(:,2) ./ normY;
    y = result;