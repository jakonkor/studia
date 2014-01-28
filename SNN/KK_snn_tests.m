% Konrad Koryciński 
% Projekt z przedmiotu SNN - aproksymacja

clc;
clear;

disp('Start:');

params = [ -0.164888 2.526858 -8.026429 3.296518 -1.037379 0.790668 7.362406 8.882589 -8.775881 -3.256581 -0.791625 -14.125703 -5.251103 2.043099 -2.259130 -10.511632 ]; 
mHiddenNeuronOpt = 5;

%-0.164888 ;2.526858 ;-8.026429 ;3.296518 ;-1.037379 ;0.790668 ;7.362406 ;8.882589 ;-8.775881 ;-3.256581 ;-0.791625 ;-14.125703 ;-5.251103 ;2.043099 ;-2.259130 ;-10.511632 ;


mDane = load('snn_b.txt');
mDane_test = load('snn_test.txt');

mDane = normSet(mDane);
mDane_test = normSet(mDane_test);
% podzial zbioru na zbior uczacy i testowy
[mTrain, mTest] = subset(mDane',1,1,1/2);
mTrain = mTrain';
mTest = mTest';

mTrain = randSet(mTrain);
N = size(mDane_test, 1);

net = newff([min(mTrain(:,1)) max(mTrain(:,1))],[mHiddenNeuronOpt 1],{'tansig', 'purelin'}, "trainlm", "learngdm", "mse");
%inicjalizacja wag sieci 
net.b{1}=params(1:5)';
net.IW{1,1} = params(6:10)';
net.LW{2,1}=params(12:16);
net.b{2}=params(11);
    
net.trainFcn = "trainlm"; 
net.trainParam.goal = 0.01; %warunek stopu - poziom błędu 
net.trainParam.epochs =400; %maksymalna liczba epok
x = mTrain(:,1)'
y = mTrain(:,2)'
net = train(net,x,y); %uczenie sieci

[q N Z rank_Z h q_N] = getNetParams(net, mTrain);

figure(1);
hist(h, 30);
title ('Histogram hkk:');
print -djpg "histogram_hkk.jpg";

y = sim(net,mDane_test(:,1)');
error = (sum((y-mDane_test(:,2)').^2)./2)./N;
error

t_stud = 1.96;  % alfa=0.05, N=inf

s = sqrt(sum((mDane_test(:,2)'-y).^2)/(N-q));
d_E = t_stud*s*sqrt(h);
E_max = y+d_E;
E_min = y-d_E;

figure(2);
hold on;
plot(mDane_test(:,1), mDane_test(:,2), '.g');
plot(mDane_test(:,1), y, 'b');
xlabel ("x");
ylabel ("y");
title ('Predykcja danych testowych');
legend('zbior testowy', 'wyjscie sieci');
print -djpg "wykres_funkcji.jpg";

figure(3);
hold on;
plot(mDane_test(:,1), mDane_test(:,2), '.g');
plot(mDane_test(:,1), y, 'b');
plot(mDane_test(:,1),E_max, 'r');
plot(mDane_test(:,1),E_min, 'r');
xlabel ("x");
ylabel ("y");
title ('Predykcja danych testowych z zaznaczeniem graniczych bledow');
legend('zbior testowy', 'wyjscie sieci', 'blad maxymalny', 'blad minimalny');
print -djpg "wykres_funkcji_errors.jpg";

% regresja wielomianowa 
poly_logs = fopen('poly.txt', "w+");
L = 25;
error_poly = zeros(1,L);
tmp = linspace(1,L,L);
for i=1:1:L
    p = polyfit(mDane_test(:,1)',mDane_test(:,2)',i);
    yn = polyval(p,mDane_test(:,1)');
    error_poly(i) = (sum((yn-mDane_test(:,2)').^2)./2)./N;
    fprintf(poly_logs, '%i; %f;', i, error_poly(i));
    fprintf(poly_logs, '\n');
end
fclose(poly_logs);

figure(4)
plot(mDane_test(:,1)',mDane_test(:,2)','rx',mDane_test(:,1)',yn,'-b');
xlabel ("x");
ylabel ("y");
title ('Regresja wielomianowa');
legend('zbior testowy', 'wynik polyval');
print -djpg "regresja_wielomianowa.jpg";

figure(5)
plot(mDane_test(:,1)',mDane_test(:,2)','rx',mDane_test(:,1)',y,'-b');
xlabel ("x");
ylabel ("y");
title ('Dzialanie sieci neuronowej na tle zbioru testowego');
legend('zbior testowy', 'wyjscie sieci');
print -djpg "test_vs_snn.jpg";

figure(6)
plot(tmp,error_poly,'x-r');
xlabel ("stopien wielomianu");
ylabel ("blad");
title ('Blad regresji w zaleznosci od stopnia wielomianu');
print -djpg "regresja_blad.jpg";
error_poly