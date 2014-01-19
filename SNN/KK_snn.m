% Konrad Koryciński 
% Projekt z przedmiotu SNN - aproksymacja

clc;
clear;
% Import danych z pliku tekstowego
mDane=load('snn_b.txt');
%mDane = load('test_dane.txt');

% Opis danych:
% kolumna 1 - x
% kolumna 2 - y=f(x)
plot(mDane(:,1),mDane(:,2));
print -djpg "dane_oryginal.jpg";

%Normalizacja i centrowanie
% to nizej to to samo co studentize :) 
%mPrestd = prestd(mDane(:,2))
mTmp = studentize(mDane);
mNormX = norm(mTmp(:,1), inf);
mNormY = norm(mTmp(:,2), inf);
% odchylanie standardowe
%mStd = std(mDane)
mDane(:,1) = mTmp(:,1) ./ mNormX;
mDane(:,2) = mTmp(:,2) ./ mNormY;

%prestd(dane)
%mDane(:,1) = mDane(:,1) - mean(mDane(:,1))
%mDane

% podzial zbioru na zbior uczacy i testowy
[mTrain, mTest] = subset(mDane',1,1,1/2);
mTrain = mTrain';
mTest = mTest';
%poststd(prestd(dane))
%trastd(dane)
%mapstd(dane)
%dane
%dane(:)
%dane(:,1)'
%dane(:,2)'
plot(mDane(:,1),mDane(:,2));
print -djpg "dane_norm.jpg";
%print -deps foo.eps


logs = fopen('log.txt', "w+");

%-----------------------------------------
% pkt 2 - dobór liczby neuronów metodą porównania błędu śrenio-kwadratowego

mHiddenNeuronMax = 6;
mTestNumber = 2;

mErrorTrain = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTest = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTrainAvr = zeros(mHiddenNeuronMax, 1);
mErrorTestAvr = zeros(mHiddenNeuronMax, 1);
mErrorTrainMin = zeros(mHiddenNeuronMax, 1);
mErrorTestMin = zeros(mHiddenNeuronMax, 1);

for neuronNum = 1:1:mHiddenNeuronMax
    for testNum = 1:1:mTestNumber
       
       

        
        mTrain = randSet(mTrain);
        N = size(mTrain, 1);
        
        
        net = newff([min(mTrain(:,1)) max(mTrain(:,1))],[neuronNum 1],{"tansig" , "purelin"} , "trainlm", "learngdm", "mse");
        
        rand('state',sum(100*clock)); %inicjalizacja generatora liczb pseudolosowych 
        
        %inicjalizacja wag sieci 
        net.IW{1} = (rand(neuronNum, 1)-0.5)/(0.5/0.15); 
        net.LW{2} = (rand(1, neuronNum) - 0.5) / (0.5 / 0.15); 
        net.b{1} = (rand(neuronNum, 1) - 0.5) / (0.5 / 0.15); 
        net.b{2} = (rand() - 0.5) / (0.5 / 0.15); 
        net.trainParam.goal = 0.01; %warunek stopu - poziom błędu 
        net.trainParam.epochs = 100; %maksymalna liczba epok 
        net=train(net,mTrain(:,1)',mTrain(:,2)'); %uczenie sieci
        
        w = net.IW{1}';             
        bin=net.b{1};               
        v = net.LW{2,1}';           
        bout = net.b{2}; 
        paramsCount = numel(w)+numel(bin)+numel(v)+numel(bout)
        
        %symulacja sieci
        y = sim(net,mTrain(:,1)');
        mErrorTrain(neuronNum, testNum) = (sum((y-mTrain(:,2)').^2)./2)./length(mTrain(:,1));
    
        y = sim(net,mTest(:,1)');
        mErrorTest(neuronNum, testNum) = (sum((y-mTest(:,2)').^2)./2)./length(mTest(:,1));

        %dobór liczby neuronów metodą loo
        fputs(logs, ['Neurony ukryte: ' num2str(neuronNum)]);
        fputs(logs, ['Próba: ' num2str(testNum)]);
        %disp(['JAKOBIAN: ']);
        Z = calc_jacobian(net, mTrain);
        mZrank=rank(Z);
        fputs(logs, ['Rząd jacobaianu Z: ' num2str(mZrank) ]);
        fputs(logs, ['Liczba parametrów sieci: ' num2str(paramsCount)]);
        [U W V] = svd(Z);
        W = diag(diag(W));
        ZTZ = V*W*W*(V');
        H = Z*(inv(ZTZ))*Z';
        h = diag(H);
        
        rk = y - mTrain(:,2);
        tmp = ones(size(h))
        rk_k = rk./(tmp-h);
        % hkk wariancja
        mHkkVar = sum((mTrain(:,2)-y).^2)/size(y,1)
        
        q_N = paramsCount/N;
        
        w_hkk=sqrt(sum((q_N*ones(size(h))-h).^2)/N);
        fputs(logs,  ['Wariancja hkk: ' num2str(w_hkk) ]);
%disp(['Wariancja wartosci hkk=' num2str(w_hkk)]);
        
          w = net.IW{1}';              %weights inputs->hidden neurons
    bin=net.b{1};               %input bias
    v = net.LW{2,1}';           %weights hidden neurons->output
    bout = net.b{2}; 
        paramsCount = numel(w)+numel(bin)+numel(v)+numel(bout)
        
        


    end
    
    mErrorTrainAvr(neuronNum)=mean(mErrorTrain(neuronNum,:)');
    mErrorTestAvr(neuronNum)=mean(mErrorTest(neuronNum,:)');
    mErrorTrainMin(neuronNum)=min(mErrorTrain(neuronNum,:)');
    mErrorTestMin(neuronNum)=min(mErrorTest(neuronNum,:)');
    
    %csvwrite(logs, mErrorTrainAvr);
    %csvwrite(logs, mErrorTestAvr);
    %fprintf(logs, '%f ', mErrorTrainMin);
    %fprintf(logs, '%f ', mErrorTestMin);
    
    
end

plot(mDane(:,1),mDane(:,2));
print -djpg "dane_norm.jpg";

x=linspace(1,mHiddenNeuronMax,mHiddenNeuronMax)
plot(x,mErrorTrainAvr','o-r',x,mErrorTestAvr','o-g');
print -djpg "sredni_MSE.jpg";
plot(x,mErrorTrainMin','o-r',x,mErrorTestMin','o-g');
print -djpg "min_MSE.jpg";


%dobór liczby neuronów metodą loo


