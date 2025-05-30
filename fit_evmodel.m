function [fitoutput] = fit_evmodel(inx,lb,ub, data)


options = optimset('Display','off','MaxIter',10000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',1000,'LargeScale','off');
warning off;

fitoutput.certain = data.certain;
fitoutput.gamble_1 = data.gamble_1;
fitoutput.gamble_2 = data.gamble_2;
fitoutput.type = data.type;
fitoutput.choice = data.choice;

fitoutput.inx = inx;
fitoutput.lb  = lb;
fitoutput.ub  = ub;
fitoutput.options = options;


    inx0 = inx;
    [b, loglike, exitflag, output, lambda, grad, H] = fmincon(@model, inx0, [],[],[],[],lb,ub,[], options, fitoutput.certain, fitoutput.gamble_1, fitoutput.gamble_2, fitoutput.choice);
    se = transpose(sqrt(diag(inv(H))));
    fitoutput.b       = b;
    fitoutput.modelLL = -loglike;
    fitoutput.nullmodelLL = log(0.5)*size(fitoutput.choice,1);         %LL of random-choice model
    fitoutput.pseudoR2    = 1 + loglike / (fitoutput.nullmodelLL); %pseudo-R2 statistic
    fitoutput.exitflag    = exitflag;
    fitoutput.output      = output;
    fitoutput.H           = H; %Hessian
    fitoutput.dof = length(b);
    fitoutput.BIC = log(length(fitoutput.choice))*fitoutput.dof -2*fitoutput.modelLL;
    fitoutput.AIC = 2*fitoutput.dof -2*fitoutput.modelLL;
end   


function [loglike] = model(x, certain, gamble_1, gamble_2, choice)
tau = x(1);

    ev_certain = certain;
    ev_gamble = 0.5.*(gamble_1+gamble_2);
    ev_diff = ev_gamble - ev_certain;
    probchoice         = 1 ./ (1 + exp(-tau * (ev_diff)));        %convert ev to probability; for gamble
    probchoice(find(probchoice == 0)) = eps;      %to prevent fmin crashing from a log zero
    probchoice(find(probchoice == 1)) = 1 - eps;

    choice = double(choice == 2);
    
    loglike = - (transpose(choice(:)) * log(probchoice(:)) + transpose(1-choice(:)) * log(1-probchoice(:)));
end
    
   