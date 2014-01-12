function [net]= train_net(train_set,labels,hidden_neurons_count)
    %Opis: funkcja tworz飡 i ucz飡 sie栮euronow銠   %Parametry:
    %   train_set: zbi㲠ucz飹 - kolejne punkty w kolejnych wierszach
    %   labels:    etykiety punkt㷠- {-1,1}
    %   hidden_neurons_count: liczba neuron㷠w warstwie ukrytej
    %Warto즠zwracana:
    %   net - obiekt reprezentuj飹 sie栮euronow銊    %inicjalizacja obiektu reprezentuj飥go sie栮euronow銠   %funkcja aktywacji: neuron㷠z warstwy ukrytej - tangens hiperboliczny,
    %                   neuronu wyj죩owego - liniowa
    %funkcja ucz飡: Levenberg-Marquard

    input_count=size(train_set,2);
    pr=min_max(train_set');                 %okre쬥nie minimalnych i 
                                            %maksymalnych warto죩 dla
                                            %ka拉go wej죩a
    net=newff(pr, [hidden_neurons_count 1],{'tansig', 'purelin'}, 'trainlm');

    rand('state',sum(100*clock));           %inicjalizacja generatora liczb 
                                            %pseudolosowych
    
    %inicjalizacja wag sieci
    net.IW{1} = (rand(hidden_neurons_count, input_count) - 0.5) / (0.5 / 0.15);
    net.LW{2} = (rand(1, hidden_neurons_count) - 0.5) / (0.5 / 0.15);
    net.b{1}  = (rand(hidden_neurons_count, 1) - 0.5) / (0.5 / 0.15);
    net.b{2}  = (rand() - 0.5) / (0.5 / 0.15);

    net.trainParam.goal = 0.01;             %warunek stopu - poziom b㪤u
    net.trainParam.epochs = 500;            %maksymalna liczba epok
    net=train(net,train_set',labels');      %uczenie sieci
