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
