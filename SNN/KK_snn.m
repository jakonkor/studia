% Konrad Koryciński 
% Projekt z przedmiotu SNN - aproksymacja

clc;
clear;

% Import danych z pliku tekstowego
%mDane=load('A_test_14.txt');
mDane = load('test_dane.txt');
% Opis danych:
% kolumna 1 - x
% kolumna 2 - y=f(x)
plot(mDane(:,1),mDane(:,2));
print -djpg "dane_oryginal.jpg";

%Normalizacja i centrowanie
mDane
% to nizej to to samo co studentize :) 
%mPrestd = prestd(mDane(:,2))
mTmp = studentize(mDane)
mNormX = norm(mTmp(:,1), inf)
mNormY = norm(mTmp(:,2), inf)
% odchylanie standardowe
%mStd = std(mDane)
mDane(:,1) = mTmp(:,1) ./ mNormX;
mDane(:,2) = mTmp(:,2) ./ mNormY;
mDane

%prestd(dane)
%mDane(:,1) = mDane(:,1) - mean(mDane(:,1))
%mDane
% podzial zbioru na zbior uczacy i testowy
[mTrain, mTest] = subset(mDane',1,1,1/2)
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