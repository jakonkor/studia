% plik:  main.m 
% opis:  przyk㡤owy skrypt pokazuj飹 u﹣ie sieci neuronowych w 
%        programie Octave
% autor: Zbigniew Szymaᳫi <z.szymanski@ii.pw.edu.pl>
% data:  2013-12-16

clc;        %wyczyszczenie okna komend
clear;      %usuni꣩e wszystkich zmiennych

% Import danych z pliku tekstowego
dane=load('dane.txt');
% Opis tablicy 'dane':
% kolumny 1,2 - wsp㳲z꤮e punkt㷠do klasyfikacji
% kolumna 3   - etykieta punktu {-1,1}

% Uczenie sieci neuronowej 
liczba_neuronow_ukrytych=8;
[net]=train_net(dane(:,1:2),dane(:,3),liczba_neuronow_ukrytych);
%klasyfikacja danych ze zbioru ucz飥go
wyniki=sign(sim(net,dane(:,1:2)')');
%analiza wynik㷠klasyfikacji
TP=size(find(dane(idx_poz,3)==1),1)     %liczba True Positives
TN=size(find(dane(idx_neg,3)==-1),1)    %liczba True Negatives
FP=size(find(dane(idx_poz,3)==-1),1)    %liczba False Positives
FN=size(find(dane(idx_neg,3)==1),1)     %liczba False Negatives
                                        %TP+TN+FP+FN == rozmiar zbioru

%Wizualizacja wynik㷠klasyfikacji
idx_poz=find(wyniki(:)==1);             %indeksy przykladow 
                                        %zaklasyfikowanych jako poz
idx_neg=find(wyniki(:)==-1);            %indeksy przykladow 
                                        %zaklasyfikowanych jako neg
idx_blad=find(wyniki(:)~=dane(:,3));    %indeksy b㪤nie zaklasyfikowanych
                                        %przyk㡤㷍

figure(100);
plot(dane(idx_blad,1),dane(idx_blad,2),'ob'); %wykre쬥nie b㪤nie
                                              %zaklasyfikowanych pr㢥k
hold on;
%wyniki klasyfikacji - klasa pozytywna
plot(dane(idx_poz,1),dane(idx_poz,2),'.r');   
%wyniki klasyfikacji - klasa negatywna
plot(dane(idx_neg,1),dane(idx_neg,2),'.k');
hold off;
