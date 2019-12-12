%% Piping Optimisation Code
tic;
%% Problem Formulation
% Heat Transfer in the pipe
kp = 385; % thermal conductivity of copper
kins = 0.021; % thermal conductivity of insulation
hint = 100; % heat transfer coefficient (internal)
hext = 8; % heat transfer coefficient (external)
tinf = 15; % temperature of air surrounding insulation
tbar = 55; % average temperature of water in the pipes 
Lp = 1; % length of piping
fl = 0.1; % L/s volumetric flow rate in heating system
r1_b = [0.0001, 0.2667]; % Bounds for r1, interal pipe radius
r2_b = [0.0015,0.267]; % Bounds for r2, external pipe radius
r3_b = [0.0015,0.5]; % Bounds for r3, insulation radius


Q = @(r) ((2*pi*r(3)*Lp*(tbar-tinf))/((r(3)/(hint*r(1)))+((r(3)*log(r(2)/r(1)))/kp)+((r(3)*log(r(3)/r(2)))/kins)+(1/hext)));

% Pressure Loss
mu = 0.0005039; % Pa*s viscosity of water at 55 degrees
rho = 985.7; % kg m^-3 density of water at 55 degrees
e = 0.0000015; % m absolute roughness of copper pipe

Pf = @(r) (frictionFactor(r(1), rho, fl, mu, e)*(Lp/(2*r(1)))*(0.5*rho*flowSpeed(r(1), fl)^2))/rho;
%% Assumptions
%{
Pipe external diameter -> 6mm -> 567 mm
Pipe thickness -> 0.3mm -> 6mm
Insulation thickness -> 0 -> 100mm
Pipe has no bends
Heat transfer coefficients are constant
%}

%% Problem Visualisation
%% Critical Radius of Insulation (Figure 5.2.1)
r3 = linspace(0.001,0.05,100);
r1 = 0.0005*ones(1,100);
r2 = 0.0008*ones(1,100);

r = num2cell([r1;r2,;r3],1);
q = cellfun(Q, r);
figure;
plot(r3,q)
xlabel('Insulation radius, r3 (m)')
ylabel('Heat Loss, Q (W)')
title('Figure 5.2.1')

%% Monotonicity
%% Sweeping through r1 (Figure 5.3.1)
r1 = linspace(r1_b(1),r1_b(2), 100);
r2 = ((r2_b(1)+r2_b(2))/2)*ones(1,100);
r3 = ((r3_b(1)+r3_b(2))/2)*ones(1,100);

r = num2cell([r1;r2,;r3],1);

out = cellfun(Q, r);
figure;
plot(r1, out)
xlabel('Internal pipe radius, R1 (m)')
title('Varying R1 with constant R2 and R3')

%% Multiobjective Solving

%% Pareto Plot gamultiobj (Figure 5.4.1)
fcn = @(r) [Q(r),Pf(r)];

rng default % For reproducibility
A = [1,-1,0;0,1,-1;0,-1,0];
b = [-0.0003;0;-0.003];
Aeq = [];
beq = [];
lb = [r1_b(1),r2_b(1)];
ub = [r1_b(2),r2_b(2)]; 

[x,fval] = gamultiobj(fcn,3,A,b,Aeq,beq,lb,ub);

figure;
plot(fval(:,1),fval(:,2),'ko')
axis([0.3 2 -1e14 8e14])
xlabel('Heat loss, Q')
ylabel('Pressure Loss, E')
title('Pareto Points in Function Space')

%% Utility Function
%{
Maximise the amount of hot water delivered to the midpoint of the system
(takes into account both pressure loss and heat loss)
%}

u = @(r) Q(r)+Pf(r)*flowSpeed(r(1),fl); % utility function - overall energy loss

r0 = [(r1_b(1)+r1_b(2))/2, (r2_b(1)+r2_b(2))/2, (r3_b(1)+r3_b(2))/2]; % initial point

disp(r0)

%% Trust Region - unconstrained attempt - unfeasible solution
options = optimoptions('fminunc','Algorithm','quasi-newton', 'MaxFunEvals',1000);
[r, fval] = fminunc(u,r0,options);

disp(['Initial Objective: ' num2str(Q(r0))])
disp(table(r(1),r(2),r(3), 'VariableNames',{'r1', 'r2', 'r3'}))
disp(['Final Objective - Trust Region: ' num2str(Q(r))])
disp('r2 < r1 so objective is unfeasible')

%% Interior point
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunEvals',1000);
[r, fval, exitflag, output, lambda, grad, hessian] = fmincon(u,r0,A,b,Aeq,beq,lb,ub,@cons,options);

disp(['Initial Objective: ' num2str(u(r0))])
disp(table(r(1),r(2),r(3), 'VariableNames',{'r1', 'r2', 'r3'}))
disp(['Final Objective - Interior Point: ' num2str(u(r))])



%% SQP to determine more accurate solution

r0 = [r(1),r(2),r(3)];
options = optimoptions('fmincon','Display','iter','Algorithm','sqp', 'MaxFunEvals',2000,'StepTolerance',1e10);
[r, fval, exitflag, output, lambda, grad, hessian] = fmincon(u,r0,A,b,Aeq,beq,lb,ub,@cons,options);

disp(['Initial Objective: ' num2str(u(r0))])
disp(table(r(1),r(2),r(3), 'VariableNames',{'r1', 'r2', 'r3'}))
disp(['Final Objective - SQP: ' num2str(u(r))])
disp('Lagrangian Multipliers:')

disp(table(lambda.ineqlin,[lambda.ineqnonlin;NaN],lambda.lower,lambda.upper, 'VariableNames',{'ineqlin','ineqnonlin','lower','upper'}))



%% Sensistivity Analysis
disp(table(grad(1),grad(2),grad(3), 'VariableNames',{'Gradient r1', 'Gradient r2', 'Gradient r3'}))

%% sweeping through r1 - Figure 5.4.2
r1 = linspace(r0(1)-r0(1)/2,r0(1)+r0(1)/2, 100);
r2 = r0(2)*ones(1,100);
r3 = r0(3)*ones(1,100);

r = num2cell([r1;r2,;r3],1);
out = cellfun(u, r);
figure;
plot(r1, out)
hold on
plot(r0(1), out(50), 'ko')
xlabel('Internal pipe radius, r1 (m)')
ylabel('Utility, U (W)')

%% sweeping through r2 - Figure 5.4.3
r1 = r0(1)*ones(1,100);
r2 = linspace(r0(2)-r0(2)/2,r0(2)+r0(2)/2, 100);
r3 = r0(2)*ones(1,100);

r = num2cell([r1;r2,;r3],1);
out = cellfun(u, r);
figure;
plot(r2, out)
hold on
plot(r0(2), out(50), 'ko')
xlabel('External pipe radius, r2 (m)')
ylabel('Utility, U (W)')

%% sweeping through r3 - Figure 5.4.4
r1 = r0(1)*ones(1,100);
r2 = r0(2)*ones(1,100);
r3 = linspace(r0(3)-r0(3)/2,r0(3)+r0(3)/2, 100);

r = num2cell([r1;r2,;r3],1);
out = cellfun(u, r);
figure;
plot(r3, out)
hold on
plot(r0(3), out(50), 'ko')
xlabel('Insulation pipe radius, r3 (m)')
ylabel('Utility, U (W)')

%% Paramteric Study

%% Varying Hint - Figure 5.4.5
hint_range = linspace(10,1000,100);
h = zeros(1,100);
for i = 1:100
    hint = hint_range(i);
    Q = @(r) ((2*pi*r(3)*Lp*(tbar-tinf))/((r(3)/(hint*r(1)))+((r(3)*log(r(2)/r(1)))/kp)+((r(3)*log(r(3)/r(2)))/kins)+(1/hext)));
    Pf = @(r) ((frictionFactor(r(1), rho, fl, mu, e)*(Lp/(2*r(1)))*(0.5*rho*flowSpeed(r(1), fl)^2))/rho)*flowSpeed(r(1),fl);
    u = @(r) Q(r)+Pf(r);
    h(i) = u(r0);
end

figure;
plot(hint_range, h)
xlabel('Internal Heat Transfer Coefficient')
ylabel('Overall power loss (W)')

%% Varying Hext - Figure 5.4.6
hext_range = linspace(0.08,80,100);
h = zeros(1,100);
for i = 1:100
    hext = hext_range(i);
    Q = @(r) ((2*pi*r(3)*Lp*(tbar-tinf))/((r(3)/(hint*r(1)))+((r(3)*log(r(2)/r(1)))/kp)+((r(3)*log(r(3)/r(2)))/kins)+(1/hext)));
    Pf = @(r) ((frictionFactor(r(1), rho, fl, mu, e)*(Lp/(2*r(1)))*(0.5*rho*flowSpeed(r(1), fl)^2))/rho)*flowSpeed(r(1),fl);
    u = @(r) Q(r)+Pf(r);
    h(i) = u(r0);
end

figure;
plot(hext_range, h)
xlabel('External Heat Transfer Coefficient')
ylabel('Overall power loss (W)')

%% Varying fl 
fl_range = linspace(0.001,1,100);
u_fl = zeros(1,100);
for i = 1:100
    fl = fl_range(i);
    Q = @(r) ((2*pi*r(3)*Lp*(tbar-tinf))/((r(3)/(hint*r(1)))+((r(3)*log(r(2)/r(1)))/kp)+((r(3)*log(r(3)/r(2)))/kins)+(1/hext)));
    Pf = @(r) ((frictionFactor(r(1), rho, fl, mu, e)*(Lp/(2*r(1)))*(0.5*rho*flowSpeed(r(1), fl)^2))/rho)*flowSpeed(r(1),fl);
    u = @(r) Q(r)+Pf(r);
    u_fl(i) = u(r0);
end

figure;
loglog(fl_range, u_fl)
xlabel('Flow Rate')
ylabel('Overall power loss (W)')

toc
%% Non Linear Constraints

function [c,ceq] = cons(r)
ceq= [];
c4 = -r(2)+0.003;
c6 = (pi*(r(3)^2-r(2)^2)/2) - (pi*(r(2)^2-r(1)^2));
c = [c4,c6];
end

%% Functions
function f = frictionFactor(r1, rho, fl, mu, e)
    Re = (rho*flowSpeed(r1, fl)*2*r1)/mu; % Reynolds number
    ed = e/(2*r1);
    
    colebrook_fed = @(f, ed) 1/sqrt(f)+2*log10((ed/3.7)+(2.51)/(Re*sqrt(f)));
    
    if Re > 4000 %turbulent
        f = fzero(@(f) colebrook_fed(f, ed), [0.008, 0.1]);
    elseif Re < 2000 %laminar
        f = 64/Re;
    else %transitional
        f = (((Re-2000)/(4000-2000))*(0.1-0.008))+0.008;
    end
end

function u = flowSpeed(r1, fl)
    u = fl/(pi*r1^2); % m s^-1 average speed of water 
end