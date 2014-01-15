% Konrad Koryciński 
% Projekt z przedmiotu SNN - aproksymacja

clc;
clear;

% Import danych z pliku tekstowego
mDane=load('A_test_14.txt');
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

%-----------------------------------------
% pkt 2 - dobór liczby neuronów metodą porównania błędu śrenio-kwadratowego

mHiddenNeuronMax = 30;
mTestNumber = 30;

mErrorTrain = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTest = zeros(mHiddenNeuronMax, mTestNumber);
mErrorTrainAvr = zeros(mHiddenNeuronMax, 1);
mErrorTestAvr = zeros(mHiddenNeuronMax, 1);
mErrorTrainMin = zeros(mHiddenNeuronMax, 1);
mErrorTestMin = zeros(mHiddenNeuronMax, 1);

for neuronNum = 1:1:mHiddenNeuronMax
    for testNum = 1:1:mTestNumber
       
        #randomizacja danych
        permutation = randperm(length(mTrain));
        mTrainRand = mTrain;
        for i = 1:1:length(mTrain)
            mTrainRand(i,:) = mTrain(permutation(i),:);
        end
        % czyszenie pamięci
        mTrain = mTrainRand;	
        clear mTrainRand;
        
        
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
        
        %symulacja sieci
        y = sim(net,mTrain(:,1)');
        mErrorTrain(neuronNum, testNum) = (sum((y-mTrain(:,2)').^2)./2)./length(mTrain(:,1));
    
        y = sim(net,mTest(:,1)');
        mErrorTest(neuronNum, testNum) = (sum((y-mTest(:,2)').^2)./2)./length(mTest(:,1));

    end
    
    mErrorTrainAvr(neuronNum)=mean(mErrorTrain(neuronNum,:)');
    mErrorTestAvr(neuronNum)=mean(mErrorTest(neuronNum,:)');
    mErrorTrainMin(neuronNum)=min(mErrorTrain(neuronNum,:)');
    mErrorTestMin(neuronNum)=min(mErrorTest(neuronNum,:)');
    
end

plot(mDane(:,1),mDane(:,2));
print -djpg "dane_norm.jpg";

x=linspace(1,mHiddenNeuronMax,mHiddenNeuronMax)
plot(x,mErrorTrainAvr','o-r',x,mErrorTestAvr','o-g');
print -djpg "sredni_MSE.jpg";
plot(x,mErrorTrainMin','o-r',x,mErrorTestMin','o-g');
print -djpg "min_MSE.jpg";