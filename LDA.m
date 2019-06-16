function [Output_y,W_lda] = LDA(InputSignal, Target, nclass)

%**************************************************************************
% LDA (also known as Fishers LDA)
%**************************************************************************

% InputSignal - the input data of size n x p
% n - # of trials
% p - number of features per trial
% Target - is a vector (n x 1) containing the target class of the trials

clear idxlda
c = nclass;
Sw = zeros(size(InputSignal,2),size(InputSignal,2));
for i = 1:c
    idxlda{i} = find(Target== i);
    mu{i} = mean(InputSignal(idxlda{i},:));
    term = bsxfun(@minus,InputSignal(idxlda{i},:),mu{i})'; % term - (x-uc)
    Sw = Sw + term*term'; % Within class scatter matrix
end
    
Sb = (mu{1}'-mu{2}')*(mu{1}'-mu{2}')'; % Between class scatter matrix - for 2 class

[W,D] = eig(pinv(Sw)*Sb);
[D,idxs1] = sort(diag(D),'descend');
W = W(:,idxs1);

clear y
clear W_lda
W_lda = W(:,1:c-1);
Output_y = (W_lda'*InputSignal')';


% plot(y(idxs{1}'),ones(1,size(idxs{1},1)),'ro')
% hold on
% plot(y(idxs{2}'),ones(1,size(idxs{2},1)),'g*')


%**************************************************************************