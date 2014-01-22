% tworzy i uczy siec neuronowa

% dane wejsciowe 
% mTrain - zbior uczacy 
% n_hidden - liczba nauronow ukrytych

% dane wyjsciowe 
% net - wytrenowana siec wg podanych parametrow
% N - liczba parametrow sieci

function [net]  = createNet(mTrain, n_hidden) 

    net = newff([min(mTrain(:,1)) max(mTrain(:,1))],[n_hidden 1],{'tansig', 'purelin'}, "trainlm", "learngdm", "mse");
    rand('state',sum(100*clock)); %inicjalizacja generatora liczb pseudolosowych 
    
    %inicjalizacja wag sieci 
    net.IW{1} = (rand(n_hidden, 1)-0.5)/(0.5/0.15); 
    net.LW{2} = (rand(1, n_hidden) - 0.5) / (0.5 / 0.15); 
    net.b{1} = (rand(n_hidden, 1) - 0.5) / (0.5 / 0.15); 
    net.b{2} = (rand() - 0.5) / (0.5 / 0.15); 
    
    % w octave nie zaimplementowane    
    %net.trainFcn = "traingd"; 
    %net.trainParam.goal = 0.01; %warunek stopu - poziom błędu 
    %net.trainParam.epochs = 400; %maksymalna liczba epok
    %x = mTrain(:,1)'
    %y = mTrain(:,2)'
    %net = train(net,x,y); %uczenie sieci
    
    net.trainFcn = "trainlm"; 
    net.trainParam.goal = 0.01; %warunek stopu - poziom błędu 
    net.trainParam.epochs =400; %maksymalna liczba epok
    x = mTrain(:,1)'
    y = mTrain(:,2)'
    net = train(net,x,y); %uczenie sieci
