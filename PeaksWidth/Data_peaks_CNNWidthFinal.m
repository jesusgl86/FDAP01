clear all
clc
%% Making training data
N=10000;
Toutput=zeros(N,1);
Nm=28;
sigma=0;
XTrain=zeros(Nm,Nm,1,N);
clear ToutputPks ToutputHgt ToutputWdt
Np=1;
for i=1:N
    i
    x=linspace(0,50,1000);
    n=round(Np*rand()+1);
    %Pos= 100*randi([2 8],1,n);
    Pos=round(50*rand(n,1)'+1);
    R=randi(100);

    Hgt= 1*randi(R,1,n)+1*round(mod(i,1000),-2);
    if n==2
        Hgt/2;
    end
    Wdt=round(50*rand(n,1)'+1);
    for k=1:n
        Gauss(k,:)=Hgt(k)*exp(-((x-Pos(k))/Wdt(k)).^2);
    end
    PeakSig=sum(Gauss);
%     imshow(I)
    %I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
    [Pks,locs,width]=findpeaks(PeakSig);
    
    if isempty(Pks)
        ToutputPks(i)=0;
        ToutputHgt(i)=0;
        ToutputWdt(i)=0;
    else
        ToutputPks(i)=numel(Pks);
        [maxHgt maxi]=max(Pks);
        ToutputHgt(i)=maxHgt;
        [maxWidth maxi]=max(width);
        ToutputWdt(i,1)=maxWidth;
    end
    
    PeakSig=(PeakSig+sigma*randn(size(x)))/max(PeakSig);
    %Peaksig=PeakSig-mean(PeakSig);
    M=diffmat(PeakSig);
    % plot
    I = mat2gray(M);
    I=imresize(I,[Nm Nm]);
    XTrain(:,:,1,i)=I;
    Sigtrain(i,:)=PeakSig;
end

ToutputPks=categorical(ToutputPks);
categories(ToutputPks)
I=XTrain(:,:,1,1);
figure('visible','on')
imshow(I);
figure
histogram(ToutputPks)
figure
histogram(ToutputHgt)

%% Making validation data
N=1000;
Toutvalid=zeros(N,1);
Xvalid=zeros(Nm,Nm,1,N);
clear ToutputPksvalid
for i=1:N
    i
    x=linspace(0,50,1000);
    n=round(Np*rand()+1);
    Pos=round(50*rand(n,1)'+1);
    R=randi(100);

    Hgt= 1*randi(R,1,n)+1*round(mod(i,1000),-2);
    if n==2
        Hgt/2;
    end
    Wdt=round(50*rand(n,1)'+1);
    for k=1:n
        Gauss(k,:)=Hgt(k)*exp(-((x-Pos(k))/Wdt(k)).^2);
    end
    PeakSig=sum(Gauss);
%     imshow(I)
    %I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
    [Pks,locs,width]=findpeaks(PeakSig);
    if isempty(Pks)
        ToutputPksvalid(i)=0;
        ToutputHgtvalid(i)=0;
        ToutputWdtvalid(i)=0;
    else
    ToutputPksvalid(i)=numel(Pks);
    [maxHgt maxi]=max(Pks);
    ToutputHgtvalid(i)=maxHgt;
    [maxWidth maxi]=max(width);
    ToutputWdtvalid(i,1)=maxWidth;
    end
    
    PeakSig=(PeakSig+sigma*randn(size(x)))/max(PeakSig);
    %Peaksig=PeakSig-mean(PeakSig);
    M=diffmat(PeakSig);
    % plot
    I = mat2gray(M);
    I=imresize(I,[Nm Nm]);
    Xvalid(:,:,1,i)=I;
    Sigvalid(i,:)=PeakSig;
end
I=Xvalid(:,:,1,1);
figure('visible','on')
imshow(I);
ToutputPksvalid=categorical(ToutputPksvalid);
categories(ToutputPksvalid)
figure
histogram(ToutputPksvalid)
figure
histogram(ToutputHgtvalid)

%% Making testing data
N=1000;
Touttest=zeros(N,1);
Xtest=zeros(Nm,Nm,1,N);
clear ToutputPkstest

for i=1:N
    i
    x=linspace(0,50,1000);
    n=round(Np*rand()+1);
    Pos=round(50*rand(n,1)'+1);
    R=randi(100);

    Hgt= 1*randi(R,1,n)+1*round(mod(i,1000),-2);
    if n==2
        Hgt/2;
    end
    Wdt=round(50*rand(n,1)'+1);
    for k=1:n
        Gauss(k,:)=Hgt(k)*exp(-((x-Pos(k))/Wdt(k)).^2);
    end
    PeakSig=sum(Gauss);
%     imshow(I)
    %I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
    [Pks,locs,width]=findpeaks(PeakSig);
    if isempty(Pks)
        ToutputPkstest(i)=0;
        ToutputHgttest(i)=0;
        ToutputWdttest(i)=0;    
    else
        ToutputPkstest(i)=numel(Pks);
        [maxHgt maxi]=max(Pks);
        ToutputHgttest(i)=maxHgt;
        [maxWidth maxi]=max(width);
        ToutputWdttest(i,1)=maxWidth;
    end
    
    PeakSig=(PeakSig+sigma*randn(size(x)))/max(PeakSig);
    %Peaksig=PeakSig-mean(PeakSig);
    M=diffmat(PeakSig);
    % plot
    I = mat2gray(M);
    I=imresize(I,[Nm Nm]);
    Xtest(:,:,1,i)=I;
    Sigtest(i,:)=PeakSig;
end
ToutputPkstest=categorical(ToutputPkstest);
categories(ToutputPkstest)
figure
histogram(ToutputPkstest)
figure
histogram(ToutputHgttest)

%%
maxval=max([max(ToutputHgt) max(ToutputHgtvalid) max(ToutputHgttest)]);
maxwith=max([max(ToutputWdt) max(ToutputWdtvalid) max(ToutputWdttest)]);
ToutputHgt=ToutputHgt'/maxval;
ToutputHgtvalid=ToutputHgtvalid'/maxval;
ToutputHgttest=ToutputHgttest'/maxval;
Toutput=ToutputWdt/maxwith;
Toutvalid=ToutputWdtvalid/maxwith;
Touttest=ToutputWdttest/maxwith;
save('Peaks_train_datawidth.mat','ToutputPks','ToutputHgt','ToutputWdt','XTrain')
save('Peaks_valid_datawidth.mat','ToutputPksvalid','ToutputHgtvalid','ToutputWdtvalid','Xvalid')
save('Peaks_test_datawidth.mat','ToutputPkstest','ToutputHgttest','ToutputWdttest','Xtest')