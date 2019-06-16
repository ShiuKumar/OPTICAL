function [CSP_Transformed_Data, Wcsp, Ccsp] = CSP(InputSignal,Target,m)

% This CSP is for 2 class problem only. Classes to be labelled 1 & 2
%InputSignal is your Trials with each Trial NxTxM (N-dimension(# of channels), T is # of
% samples per dimension and M is # of Trails)

% Target is a row vector containing the traget class (Mx1)
% m is the number of first and last rows used for calculating the features

% output contains feature vector along the rows with last column
% indicating class of signal

clear Ccsp
nclass = length(unique(Target));
% C_bar = zeros(size(InputSignal,1),size(InputSignal,1),nclass);
% C_cap = C_bar;
% Z1 = zeros(m,size(InputSignal,2));


for y = 1:nclass
    n{y} = find(Target==y);
    
%     clear Ccsp
    for j = 1:1:size(n{y})
        % InputSignal(:,:,n{i}(j)) = InputSignal(:,:,n{i}(j)) - repmat(mean(InputSignal(:,:,n{i}(j))'),size(InputSignal(:,:,n{i}(j)),2),1)';
        Ccsp{y}(:,:,j) = InputSignal(:,:,n{y}(j))*InputSignal(:,:,n{y}(j))';
        Ccsp{y}(:,:,j) = Ccsp{y}(:,:,j)./trace(Ccsp{y}(:,:,j));
    end
    Ccsp{y} = Ccsp{y}(:,:,(sum(sum(isnan(Ccsp{y})))==0));
    C_bar(:,:,y) = mean(Ccsp{y},3); % average covariance matrix for class i
end

Cc = sum(C_bar,3); % Composite covariance matrix

[u1,s1] = eig(Cc);
[s1,idx1] = sort(diag(s1),'descend');
 u1 = u1(:,idx1);
 
 P = sqrt(inv(diag(s1 + 0.001* eye(size(s1)))))*u1'; % whitening matrix
 %clear C_cap
 for y = 1:nclass
     C_cap(:,:,y) = P*C_bar(:,:,y)*P';
 end
 
[u2,s2] = eig(C_cap(:,:,1)); % either 1 or 2 will work
[s2,idx2] = sort(diag(s2),'descend');
 u2 = u2(:,idx2);
    
W_csp = u2'*P; % CSP spatial filters
%   size(W_csp)
% W_csp
Wcsp = W_csp([1:m end-m+1:end],:);

% for i=1:size(Wcsp,1)
%     Wcsp(i,:)=Wcsp(i,:)./norm(Wcsp(i,:)); 
% end

%Sort columns, take first and last columns first, etc
% Wcsp=zeros(m*2,length(W_csp));
% W_csp1 = W_csp([1:m end-m+1:end],:);
% i=0;
% for d=1:m*2
%     if (mod(d,2)==0)
%         Wcsp(d,:)=W_csp1(m*2-i,:);
%         i=i+1;
%     else
%         Wcsp(d,:)=W_csp1(1+i,:);
%     end
% end
 
 for y = 1:1:size(Target,1)
    Z1(:,:,y) = Wcsp*InputSignal(:,:,y); % Z - csp transformed data
%     variance = var(Z1(:,:,i)');
%     f = log(variance./sum(variance));
%     F(i,1:size(variance,2)) = f;
 end
 
%  features = [F Target];
 CSP_Transformed_Data = Z1;

        
        
        
        
        
        
        
        