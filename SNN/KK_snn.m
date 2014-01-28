% Konrad Koryciński 
% Projekt z przedmiotu SNN - aproksymacja

clc;
clear;

disp('Start:');
% Import danych z pliku tekstowego
% Opis danych:
% kolumna 1 - x
% kolumna 2 - y=f(x)

mDane = load('snn_b.txt');
mDane_test = load('snn_test.txt');
% mniejszy zbior do szybkiego testowania
%mDane = load('test_dane.txt');

figure(1);
plot(mDane(:,1),mDane(:,2), "ob", mDane_test(:,1),mDane_test(:,2), "*g");
xlabel ("x");
ylabel ("y");
title ('Dane z plikow snn b.txt i snn test.txt');
legend('snn b.txt', 'snn test.txt');
print -djpg "dane_all.jpg";


% podzial zbioru na zbior uczacy i testowy
[mTrain, mTest] = subset(mDane',1,1,1/2);
mTrain = mTrain';
mTest = mTest';

mDane = normSet(mDane);
figure(2);
plot(mDane(:,1),mDane(:,2));
print -djpg "dane_norm.jpg";

figure(3);
plot(mTrain(:,1),mTrain(:,2), "ob",mTest(:,1),mTest(:,2), "*g" );
xlabel ("x");
ylabel ("y");
title ('Zbior uczacy i testowy przed normowaniem');
legend('zbior trenujacy', 'zbior testowy');
print -djpg "dane_train_test.jpg";

mTrain = normSet(mTrain);
mTest = normSet(mTest);
figure(4);
plot(mTrain(:,1),mTrain(:,2), "ob",mTest(:,1),mTest(:,2), "*g" );
xlabel ("x");
ylabel ("y");
title ('Zbior uczacy i testowy po normowaniu');
legend('zbior trenujacy', 'zbior testowy');
print -djpg "dane_train_test_norm.jpg";

disp('Wizualizacja danych: zrobiona');

logs = fopen('logs.txt', "w+");
%-----------------------------------------
% dobór liczby neuronów metodą porównania błędu śrenio-kwadratowego
% oraz metoda wirtualnej skrajnej ocenykrzyzowej

mse_logs = fopen('mse.txt', "w+");
loo_logs = fopen('loo.txt', "w+");
mHiddenNeuronMax = 15;
mTestNumber = 15;

mErrorTrain = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTest = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTrainAvr = zeros(mHiddenNeuronMax, 1);
mErrorTestAvr = zeros(mHiddenNeuronMax, 1);
mErrorTrainMin = zeros(mHiddenNeuronMax, 1);
mErrorTestMin = zeros(mHiddenNeuronMax, 1);

var_hkk = zeros(mHiddenNeuronMax, mTestNumber);

for neuronNum = 1:1:mHiddenNeuronMax

    fprintf(loo_logs, '%i; ', neuronNum);
    for testNum = 1:1:mTestNumber
       
        mTrain = randSet(mTrain);
        N = size(mTrain, 1);
        net = createNet(mTrain, neuronNum);
        [q N Z rank_Z h q_N] = getNetParams(net, mTrain);
        
        %symulacja sieci
        y = sim(net,mTrain(:,1)');
        mErrorTrain(neuronNum, testNum) = (sum((y-mTrain(:,2)').^2)./2)./length(mTrain(:,1));
    
        y = sim(net,mTest(:,1)');
        mErrorTest(neuronNum, testNum) = (sum((y-mTest(:,2)').^2)./2)./length(mTest(:,1));

        rk = y - mTrain(:,2);
        rk_k = rk./(ones(size(h))-h);
        
        w_hkk = sqrt(sum((q_N*ones(size(h))-h).^2)/N);
        var_hkk(neuronNum, testNum) = w_hkk;
        fprintf(loo_logs, 'P; %i; rank; %i;', testNum, rank_Z);
        fprintf(loo_logs, 'P; %i; hkk; %f;', testNum, w_hkk);
      
    end
    
    mErrorTrainAvr(neuronNum)=mean(mErrorTrain(neuronNum,:)');
    mErrorTestAvr(neuronNum)=mean(mErrorTest(neuronNum,:)');
    mErrorTrainMin(neuronNum)=min(mErrorTrain(neuronNum,:)');
    mErrorTestMin(neuronNum)=min(mErrorTest(neuronNum,:)');
     
    fprintf(mse_logs, '%i; %f; %f; %f; %f', neuronNum, mErrorTrainAvr(neuronNum), mErrorTestAvr(neuronNum), mErrorTrainMin(neuronNum), mErrorTestMin(neuronNum));
    fprintf(mse_logs, '\n');
    fprintf(loo_logs, 'mean; %f;', mean(var_hkk(neuronNum,:)'));
    fprintf(loo_logs, '\n');
end

figure(5);
x=linspace(1,mHiddenNeuronMax,mHiddenNeuronMax)
plot(x,mErrorTrainAvr','o-r',x,mErrorTestAvr','o-g');
xlabel ("liczba neuronow ukrytych");
ylabel ("blad");
title ("Blad sredni");
legend('zbior trenujacy', 'zbior testowy');
print -djpg "sredni_MSE.jpg";
plot(x,mErrorTrainMin','o-r',x,mErrorTestMin','o-g');
xlabel ("liczba neuronow ukrytych");
ylabel ("blad");
title ("Blad minimalny");
legend('zbior trenujacy', 'zbior testowy');
print -djpg "min_MSE.jpg";

fclose(mse_logs);
fclose(loo_logs);


% symulacja 50 sieci z optymalna liczna neuronow
params_logs = fopen('params.txt', "w+");
epu_logs = fopen('epu.txt', "w+");
mHiddenNeuronOpt = 5;
mTestNumber = 50;
Ep=zeros(1,mTestNumber);
u=zeros(1,mTestNumber);
index = 1;
for testNum = 1:1:mTestNumber
            
    mTrain = randSet(mTrain);
    N = size(mTrain, 1);
    net = createNet(mTrain, mHiddenNeuronOpt);
    y = sim(net,mTrain(:,1)');
    [q N Z rank_Z h q_N] = getNetParams(net, mTrain);
    if rank_Z == q 
        rk = y' - mTrain(:,2);
        rk_k = rk./(ones(size(h))-h);
        Ep_tmp = sqrt(1/N*sum((rk_k).^2));
        u_tmp = 1/N*sum(sqrt((N/q)*h));
        Ep(index) = Ep_tmp;
        u(index) = u_tmp;
        fprintf(epu_logs, '%i; %f; %f;', testNum, Ep_tmp, u_tmp);
        fprintf(epu_logs, '\n');
        
        params = [net.b{1}' net.IW{1,1}' net.b{2} net.LW{2,1}];
        fprintf(params_logs, '%f ;' , params);
        fprintf(params_logs,'\n');
        index = index + 1;
    end 
end

Ep = Ep(1:index-1)
u = u(1:index-1)
plot(Ep,u,'.g');
ylabel('u');
xlabel('Ep');
print -djpg "EpU.jpg";
fclose(params_logs);
fclose(epu_logs);

disp('all done');