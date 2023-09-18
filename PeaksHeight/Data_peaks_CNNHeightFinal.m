clear all
clc
%% Making training data
N1=10000;
Toutput=zeros(N1,1);
Nm=28;
sigma=1;
XTrain=zeros(Nm,Nm,1,N1);
clear ToutputPks ToutputHgt ToutputWdt
Np=1;
for i=1:N1
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
        ToutputWdt(i)=width(maxi);
    end
    Sigtrain0(i,:)=PeakSig+sigma*randn(size(x));
    %PeakSig=PeakSig/max(PeakSig);
    %M=diffmat(PeakSig);
    % plot
    %I = mat2gray(M);
    %I=imresize(I,[Nm Nm]);
    %XTrain(:,:,1,i)=I;

end

%% Making validation data
N2=1000;
Toutvalid=zeros(N2,1);
Xvalid=zeros(Nm,Nm,1,N2);
clear ToutputPksvalid
for i=1:N2
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
    ToutputWdtvalid(i)=maxWidth;
    end
    Sigvalid0(i,:)=PeakSig+sigma*randn(size(x));
%     PeakSig=PeakSig/max(PeakSig);
%     M=diffmat(PeakSig);
%     % plot
%     I = mat2gray(M);
%     I=imresize(I,[Nm Nm]);
%     Xvalid(:,:,1,i)=I;
end

%% Making testing data
N3=1000;
Touttest=zeros(N3,1);
Xtest=zeros(Nm,Nm,1,N3);
clear ToutputPkstest

for i=1:N3
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
        ToutputWdttest(i)=maxWidth;
    end
    Sigtest0(i,:)=PeakSig+sigma*randn(size(x));
%     PeakSig=PeakSig/max(PeakSig);
%     M=diffmat(PeakSig);
%     % plot
%     I = mat2gray(M);
%     I=imresize(I,[Nm Nm]);
%     Xtest(:,:,1,i)=I;
end

%%
maxval=max([max(ToutputHgt) max(ToutputHgtvalid) max(ToutputHgttest)]);
Toutput=ToutputHgt'/maxval;
Toutvalid=ToutputHgtvalid'/maxval;
Touttest=ToutputHgttest'/maxval;
Sigtrain=(Sigtrain0+sigma*randn(size(x)))/maxval;
Sigvalid=(Sigvalid0+sigma*randn(size(x)))/maxval;
Sigtest=(Sigtest0+sigma*randn(size(x)))/maxval;
%%
for i=1:N1
    M=diffmat(Sigtrain(i,:));
    % plot
    I = mat2gray(M,[0 1]);
    I=imresize(I,[Nm Nm]);
    XTrain(:,:,1,i)=I;
    i
end
I=XTrain(:,:,1,1);
figure('visible','on');
imshow(I);
ToutputPks=categorical(ToutputPks);
categories(ToutputPks)
figure
histogram(ToutputPks)
figure
histogram(Toutput)
%%
for i=1:N2
    M=diffmat(Sigvalid(i,:));
    % plot
    I = mat2gray(M, [0 1]);
    I=imresize(I,[Nm Nm]);
    Xvalid(:,:,1,i)=I;  
    i
end
I=Xvalid(:,:,1,900);
figure('visible','on')
imshow(I);
ToutputPksvalid=categorical(ToutputPksvalid);
categories(ToutputPksvalid)
figure
histogram(ToutputPksvalid)
figure
histogram(Toutvalid)
%%
for i=1:N3
    M=diffmat(Sigtest(i,:));
    % plot
    I = mat2gray(M,[0 1]);
    I=imresize(I,[Nm Nm]);
    Xtest(:,:,1,i)=I;
    i
end
ToutputPkstest=categorical(ToutputPkstest);
categories(ToutputPkstest)
figure
histogram(ToutputPkstest)
figure
histogram(Touttest)
%%

save('Peaks_train_data.mat','ToutputPks','ToutputHgt','ToutputWdt','XTrain')
save('Peaks_valid_data.mat','ToutputPksvalid','ToutputHgtvalid','ToutputWdtvalid','Xvalid')
save('Peaks_test_data.mat','ToutputPkstest','ToutputHgttest','ToutputWdttest','Xtest')