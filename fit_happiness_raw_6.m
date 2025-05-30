function [fitoutput] = fit_happiness_raw_6(inx,lb,ub, data,fitoutput_choice)


options = optimset('Display','off','MaxIter',10000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',1000,'LargeScale','off');

warning off;
rating_ind =[2 4 6 8 10 12 15 18 21 23 25 28 30 33 35 38 41 44 46 48 51 54 56 58 60 63 66 69 71 74 76 78 81 84 87 90];

fitoutput.rawhappy = data.rawhappy;
fitoutput.certain = data.certain;
fitoutput.gamble_1 = data.gamble_1;
fitoutput.gamble_2 = data.gamble_2;
fitoutput.type = data.type;
fitoutput.choice = data.choice;
fitoutput.outcome = data.outcome;

%fitoutput.win = zeros(length(fitoutput.outcome),1);
%fitoutput.loss = zeros(length(fitoutput.outcome),1);

for i=1:length(fitoutput.outcome)
    if fitoutput.outcome(i,1)>=0
        fitoutput.win(i,1)=fitoutput.outcome(i,1);
        fitoutput.loss(i,1)=0;
    elseif fitoutput.outcome(i,1)<2
        fitoutput.win(i,1)=0;
        fitoutput.loss(i,1)=fitoutput.outcome(i,1);
    end
end

for m=1:length(rating_ind)
    win_mtx(m,1:length(1:rating_ind(m))) = fliplr(transpose(fitoutput.win(1:rating_ind(m))));
    loss_mtx(m,1:length(1:rating_ind(m))) = fliplr(transpose(fitoutput.loss(1:rating_ind(m))));
end
       
fitoutput.inx = inx;
fitoutput.lb  = lb;
fitoutput.ub  = ub;
fitoutput.options = options;

rating = fitoutput.rawhappy;

[b, ~, ~, ~, ~, ~, H] = fmincon(@model, inx, [],[],[],[],lb,ub,[], options, win_mtx, loss_mtx, rating,rating_ind);
    fitoutput.b  = b;
    fitoutput.se = transpose(sqrt(diag(inv(H)))); %does not always work
    [sse, happypred, happyr2] = model(b, win_mtx, loss_mtx, rating,rating_ind);
    fitoutput.happypred = happypred;
    fitoutput.r2 = happyr2;
    fitoutput.sse = sse;
    fitoutput.BIC       = length(happypred)*log(sse/length(happypred)) + length(b)*log(length(happypred));
    fitoutput.AIC       = length(happypred)*log(sse/length(happypred)) + 2*length(b);
end


function [sse, happypred, happyr2] = model(x, win_mtx, loss_mtx, rating,rating_ind)
weight_win = x(1);
weight_loss = x(2);
gamma = x(3);
const=x(4);
weight_t=x(5);

decayvec  = gamma.^[0:(size(win_mtx,2)-1)]; 
decayvec  = decayvec(:);
dec       = decayvec;

happypred = weight_win*win_mtx*dec + weight_loss*loss_mtx*dec + weight_t*rating_ind' + const;
% for i =1:length(CR)
%     happypred(i,1) =  weight_CR*CR(1:i)'*fliplr(dec(1:i))  + weight_EV*EV(1:i)'*fliplr(dec(1:i))  +weight_RPE*RPE(1:i)'*fliplr(dec(1:i))  + const;
% end
sse         = sum((rating-happypred).^2); %sum least-squares error
re          = sum((rating-mean(rating)).^2); 
happyr2     = 1-sse/re;

end
    
   