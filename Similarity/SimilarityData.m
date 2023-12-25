clear all
close all
clc
f1=zeros(0,1);
f2=zeros(0,1);
N=500;
XTrain=zeros(28,28,1,4*N);
for i=1:N
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
%I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
plot(t,u)
hold on
c=-.03+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
plot(t,v)
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);

Toutput(i)=0; 

XTrain1(:,:,1,i)=Iu;
XTrain2(:,:,1,i)=Iv;

f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end
hold off

figure

histogram(f1, 'FaceColor', 'blue', 'DisplayName', 'f_1', 'Normalization', 'probability');
hold on; 
histogram(f2, 'FaceColor', 'red', 'DisplayName', 'f_2', 'Normalization', 'probability');
hold off; 

% Create a legend
legend('Location', 'best'); % 'best' places the legend in the best available position
xlabel('Measure'); % Replace 'X-Axis Label' with your actual label
ylabel('Relative Frequency'); % Replace 'Y-Axis Label' with your actual label

figure
f1=zeros(0,1);
f2=zeros(0,1);
for i=1:N
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
plot(t,u)
hold on
c=-.01+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);
plot(t,v)

Toutput(i+N)=1; 

XTrain1(:,:,1,i+N)=Iu;
XTrain2(:,:,1,i+N)=Iv;


f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end
hold off

figure
histogram(f1, 'FaceColor', 'blue', 'DisplayName', 'f_1', 'Normalization', 'probability');
hold on;
histogram(f2, 'FaceColor', 'red', 'DisplayName', 'f_2', 'Normalization', 'probability');
hold off; 

% Create a legend
legend('Location', 'best'); % 'best' places the legend in the best available position
xlabel('Measure'); % Replace 'X-Axis Label' with your actual label
ylabel('Relative Frequency'); % Replace 'Y-Axis Label' with your actual label
%%

Nv=50;
XTrain=zeros(28,28,1,2*N);
for i=1:Nv
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
%I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
%plot(t,u)
%hold on
c=-.03+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
% plot(t,v)
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);

TValid(i)=0; 

XValid1(:,:,1,i)=Iu;
XValid2(:,:,1,i)=Iv;

f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end

f1=zeros(0,1);
f2=zeros(0,1);
for i=1:Nv
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
%plot(t,u)
%hold on
c=-.01+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);
% plot(t,v)

TValid(i+Nv)=1; 

XValid1(:,:,1,i+Nv)=Iu;
XValid2(:,:,1,i+Nv)=Iv;


f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end
hold off

%%

Nt=50;
XTrain=zeros(28,28,1,2*N);
for i=1:Nt
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
%I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
%plot(t,u)
%hold on
c=-.03+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
% plot(t,v)
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);

TTest(i)=0; 

XTest1(:,:,1,i)=Iu;
XTest2(:,:,1,i)=Iv;

f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end

figure
histogram(f1, 'FaceColor', 'blue', 'DisplayName', 'f_1', 'Normalization', 'probability');
hold on;
histogram(f2, 'FaceColor', 'red', 'DisplayName', 'f_2', 'Normalization', 'probability');
hold off; 
% Create a legend
legend('Location', 'northeast'); % 'best' places the legend in the best available position
xlabel('Measure'); % Replace 'X-Axis Label' with your actual label
ylabel('Relative Frequency'); % Replace 'Y-Axis Label' with your actual label
f1=zeros(0,1);
f2=zeros(0,1);
for i=1:Nt
t = 5:120;
c=-.01+.001*randn;
u = 100*(1./(1+exp(c.*(t-6))));
Mu=diffmat(u);
Iu = mat2gray(Mu);
Iu=imresize(Iu,[28 28]);
%plot(t,u)
%hold on
c=-.01+.001*randn;
v = 100*(1./(1+exp(c.*(t-6))));
Mv=diffmat(v);
Iv = mat2gray(Mv);
Iv=imresize(Iv,[28 28]);
% plot(t,v)

TTest(i+Nt)=1; 

XTest1(:,:,1,i+Nt)=Iu;
XTest2(:,:,1,i+Nt)=Iv;


f1=[f1,100*sum(abs(u-v))/sum(u)];
f2=[f2,50*log(100*(1+mean((u-v).^2))^(-.5))/log(10)];
end
hold off

figure
histogram(f1, 'FaceColor', 'blue', 'DisplayName', 'f_1', 'Normalization', 'probability');
hold on;
histogram(f2, 'FaceColor', 'red', 'DisplayName', 'f_2', 'Normalization', 'probability');
hold off; 
% Create a legend
legend('Location', 'northeast'); % 'best' places the legend in the best available position
xlabel('Measure'); % Replace 'X-Axis Label' with your actual label
ylabel('Relative Frequency'); % Replace 'Y-Axis Label' with your actual label
save('Sim_train_data.mat','XTrain1','XTrain2','Toutput')
save('Sim_valid_data.mat','XValid1','XValid2','TValid')
save('Sim_test_data.mat','XTest1','XTest2','TTest','Nt')