function Lorenz_data_assim
T     = 5;                                   % Time Horizon
dt    = 1e-3;                                 % Integrator interpolation time step
T_min = -50;                                  % Integrate Lorenz from Tmin to T. Discard transient section (Tmin to 0).
P         = 20;                               % Number of segments
tp        = (0:T/P:T)';
tspan     = (T_min:dt:T);                     % Time interval for the Lorenz
init_cond = 0.1*ones(3,1);                        % Lorenz Initial Condition Vector
sigma = 10;
rho   = 28;                                   % RHO Value
beta  = 8/3;
opts  = odeset('RelTol',1e-5,'AbsTol',1e-5);  % Integrator tolerances
tol   = 1e-6;                                 % GMRES tol
maxit = 3*P;                                  % Maximum number of GMRES Iterations
gamma = 0;                                    % Tikhonov regularisation parameter

for p=1:2:length(tp)-2
    tspan_con_fd = tp(p):dt:tp(p+1);
    tspan_con_bw = tp(p+2):-dt:tp(p+1);
    t_span_con(:,p:p+1) = [tspan_con_fd' tspan_con_bw'];
end

% Compute a trajectory which we want to track
[t,u] = ode45(@(t,u) lorenz_solve(t,u,sigma,rho,beta), tspan, init_cond, opts);  % Solve the Lorenz System

% Discard transient section
t0_loc  = ceil((length(t))*(-T_min/(T-T_min)));
t       = t(t0_loc:end);
x       = u(t0_loc:end,1); y  = u(t0_loc:end,2); z  = u(t0_loc:end,3);           % Lorenz Solutions

figure(1)
plot(t,x,'k')
hold on
plot(t,y,'k')
plot(t,z,'k')
hold on

% Reference Data
yr =y; zr = z; 

% Assume that we cannot meausure a variable
x = zeros(1+T/dt,1);
Q = [0 0 0; 0 1 0;0 0 1];

% Compute primal Residuals
f1      = sigma*(y-x);     f2 = x.*(rho-z)-y;    f3 = x.*y - beta*z;             % f(u,t)

dx = (x(3:end)-x(1:end-2))/(2*dt); dx0 = (x(2)-x(1))/(dt); dxT = (x(end)-x(end-1))/(dt); Dx = [dx0;dx;dxT];
dy = (y(3:end)-y(1:end-2))/(2*dt); dy0 = (y(2)-y(1))/(dt); dyT = (y(end)-y(end-1))/(dt); Dy = [dy0;dy;dyT];
dz = (z(3:end)-z(1:end-2))/(2*dt); dz0 = (z(2)-z(1))/(dt); dzT = (z(end)-z(end-1))/(dt); Dz = [dz0;dz;dzT];

Rx = Dx-f1; Ry = Dy-f2; Rz = Dz-f3;
Rxbar = (1/length(Rx))*sum(norm(Rx)); Rybar = (1/length(Rx))*sum(norm(Ry)); Rzbar = (1/length(Rx))*sum(norm(Rz));

% Assume initial adjoint distributions

w1 = zeros(1+T/dt,1); 
% w1 = sin((2*pi*t)/T);
w2 = w1; w3 = w1;

% Compute adjoint residuals
g1 = sigma*w1  - (rho-z).*w2 - y.*w3 - Q(1,1)*(0);  g2 = -sigma*w1 + w2 - x.*w3 - Q(2,2)*(y-yr); g3 = x.*w2 +  beta*w3 - Q(3,3)*(z-zr);

dw1 = (w1(3:end)-w1(1:end-2))/(2*dt); dw10 = (w1(2)-w1(1))/(dt); dw1T = (w1(end)-w1(end-1))/(dt); Dw1 = [dw10;dw1;dw1T];
dw2 = (w2(3:end)-w2(1:end-2))/(2*dt); dw20 = (w2(2)-w2(1))/(dt); dw2T = (w2(end)-w2(end-1))/(dt); Dw2 = [dw20;dw2;dw2T];
dw3 = (w3(3:end)-w3(1:end-2))/(2*dt); dw30 = (w3(2)-w3(1))/(dt); dw3T = (w3(end)-w3(end-1))/(dt); Dw3 = [dw30;dw3;dw3T];

% Compute Adjoint Residuals
Rw1 = Dw1-g1; Rw2 = Dw2-g2; Rw3 = Dw3-g3;
Rw1bar = (1/length(Rx))*sum(norm(Rw1)); Rw2bar = (1/length(Rx))*sum(norm(Rw2)); Rw3bar = (1/length(Rx))*sum(norm(Rw3));

% Norm vector
Res = [Rxbar;Rybar;Rzbar;Rw1bar;Rw2bar;Rw3bar];

figure(2)
semilogy(0,Res,'-o')
hold on

b = cell(P/2,1);
j = 0;
%% Start loop
while max(Res) > 1e-3
    
j = j+1;    
% Compute RHS terms
for p = 1:2:P-1
    i =ceil(p/2);
    
[tc,h_b_fw] = ode45(@(tc,h_b_fw) newton_raphson_solve(tc,h_b_fw,t,x,y,z,rho,sigma,beta,Rx,Ry,Rz,Rw1,Rw2,Rw3,Q), t_span_con(:,p), zeros(6,1), opts);
[tc,h_b_bw] = ode45(@(tc,h_b_bw) newton_raphson_solve(tc,h_b_bw,t,x,y,z,rho,sigma,beta,Rx,Ry,Rz,Rw1,Rw2,Rw3,Q), t_span_con(:,p+1), zeros(6,1), opts); % Constraint Equations

b{i,:}   = (h_b_fw(end,:)' - h_b_bw(end,:)');

end

B = cell2mat(b);
rhs = -B;

[hp,flag,relres,iter,resvec] = gmres(@Ah_vector,rhs,[],tol,maxit);

HP = [hp(1:3);zeros(3,1);hp(4:end);zeros(3,1)];
k = 0;
for p = 1:2:P-1
    k = k+1;
    i =ceil(p/2);    
    [tc_fw,h_fw] = ode45(@(tc,h_fw) newton_raphson_solve(tc,h_fw,t,x,y,z,rho,sigma,beta,Rx,Ry,Rz,Rw1,Rw2,Rw3,Q), t_span_con(:,p), HP(1+6*(i-1):6*i), opts);
    [tc_bw,h_bw] = ode45(@(tc,h_bw) newton_raphson_solve(tc,h_bw,t,x,y,z,rho,sigma,beta,Rx,Ry,Rz,Rw1,Rw2,Rw3,Q), t_span_con(:,p+1), HP(1+(6*i):6*(i+1)), opts);
    
    tc_bw = flipud(tc_bw); h_bw = flipud(h_bw); % flip adjoints and time
    
    figure(20)
    plot(tc_fw,h_fw)
    hold on
    plot(tc_bw,h_bw)
    
    if k == 1
    du{k,:} = [h_fw(1:end,1:3);h_bw(2:end,1:3)];
    dw{k,:} = [h_fw(1:end,4:6);h_bw(2:end,4:6)];
    end

    if k>1
    du{k,:} = [h_fw(2:end,1:3);h_bw(2:end,1:3)];
    dw{k,:} = [h_fw(2:end,4:6);h_bw(2:end,4:6)];
    end
    
end

DU = cell2mat(du);
DW = cell2mat(dw);  

% Compute new trajectory and adjoints

x  = x + DU(:,1); y   = y + DU(:,2); z   = z + DU(:,3);
w1 = w1 + DW(:,1); w2 = w2 + DW(:,2); w3 = w3 + DW(:,3);

% Compute Residuals
% Compute primal Residuals
f1      = sigma*(y-x);     f2 = x.*(rho-z)-y;    f3 = x.*y - beta*z;             % f(u,t)

dx = (x(3:end)-x(1:end-2))/(2*dt); dx0 = (x(2)-x(1))/(dt); dxT = (x(end)-x(end-1))/(dt); Dx = [dx0;dx;dxT];
dy = (y(3:end)-y(1:end-2))/(2*dt); dy0 = (y(2)-y(1))/(dt); dyT = (y(end)-y(end-1))/(dt); Dy = [dy0;dy;dyT];
dz = (z(3:end)-z(1:end-2))/(2*dt); dz0 = (z(2)-z(1))/(dt); dzT = (z(end)-z(end-1))/(dt); Dz = [dz0;dz;dzT];

Rx = Dx-f1; Ry = Dy-f2; Rz = Dz-f3;
Rxbar = (1/length(Rx))*sum(norm(Rx)); Rybar = (1/length(Rx))*sum(norm(Ry)); Rzbar = (1/length(Rx))*sum(norm(Rz));

% Compute adjoint residuals
g1 = sigma*w1  - (rho-z).*w2 - y.*w3 - Q(1,1)*(0);  g2 = -sigma*w1 + w2 - x.*w3 - Q(2,2)*(y-yr); g3 = x.*w2 +  beta*w3 - Q(3,3)*(z-zr);

dw1 = (w1(3:end)-w1(1:end-2))/(2*dt); dw10 = (w1(2)-w1(1))/(dt); dw1T = (w1(end)-w1(end-1))/(dt); Dw1 = [dw10;dw1;dw1T];
dw2 = (w2(3:end)-w2(1:end-2))/(2*dt); dw20 = (w2(2)-w2(1))/(dt); dw2T = (w2(end)-w2(end-1))/(dt); Dw2 = [dw20;dw2;dw2T];
dw3 = (w3(3:end)-w3(1:end-2))/(2*dt); dw30 = (w3(2)-w3(1))/(dt); dw3T = (w3(end)-w3(end-1))/(dt); Dw3 = [dw30;dw3;dw3T];

Rw1 = Dw1-g1; Rw2 = Dw2-g2; Rw3 = Dw3-g3;
Rw1bar = (1/length(Rx))*sum(norm(Rw1)); Rw2bar = (1/length(Rx))*sum(norm(Rw2)); Rw3bar = (1/length(Rx))*sum(norm(Rw3));

% Norm vector
Res = [Rxbar;Rybar;Rzbar;Rw1bar;Rw2bar;Rw3bar];

figure(2)
semilogy(j,Res,'-o')
hold on

end

figure(1)
plot(t,x,'b')
hold on
plot(t,y,'b')
plot(t,z,'b')
hold on

%%
% Lorenz solver. typical values: rho = 28; sigma = 10; beta = 8/3;
function dudt = lorenz_solve(t,u,sigma,rho,beta)
dudt = zeros(3,1);
dudt(1) = sigma*(u(2) - u(1));
dudt(2) = u(1)*(rho - u(3)) - u(2);
dudt(3) = u(1)*u(2) - beta*u(3);
end

function dhdt = newton_raphson_solve(tc,h,t,x,y,z,rho,sigma,beta,Rx,Ry,Rz,Rw1,Rw2,Rw3,Q)
          x  = interp1(t,x,tc);    y  = interp1(t,y,tc);   z  = interp1(t,z,tc);
          Rx = interp1(t,Rx,tc);   Ry = interp1(t,Ry,tc);  Rz = interp1(t,Rz,tc);
          Rw1 = interp1(t,Rw1,tc);   Rw2 = interp1(t,Rw2,tc);  Rw3 = interp1(t,Rw3,tc);

          dhdt    = zeros(6,1);
          dhdt(1) =  -sigma*h(1)  + sigma*h(2) - Rx;
          dhdt(2) = (rho-z)*h(1)  +      -h(2)            -x*h(3) - Ry;
          dhdt(3) =       y*h(1)  +     x*h(2)         -beta*h(3) - Rz;
          dhdt(4) = -Q(1,1)*h(1)  + sigma*h(4)     - (rho-z)*h(5)     - y*h(6) - Rw1;
          dhdt(5) = -Q(2,2)*h(2)   -sigma*h(4)             + h(5)     - x*h(6) - Rw2;
          dhdt(6) = -Q(3,3)*h(3)                         + x*h(5) +  beta*h(6) - Rw3;
    
end

function dhdt = newton_raphson_hom_solve(tc,h,t,x,y,z,rho,sigma,beta,Q)
          x  = interp1(t,x,tc);    y  = interp1(t,y,tc);   z  = interp1(t,z,tc);

          dhdt    = zeros(6,1);
          dhdt(1) =  -sigma*h(1)  + sigma*h(2) ;
          dhdt(2) = (rho-z)*h(1)  +      -h(2)            -x*h(3) ;
          dhdt(3) =       y*h(1)  +     x*h(2)         -beta*h(3) ;
          dhdt(4) = -Q(1,1)*h(1)  + sigma*h(4)     - (rho-z)*h(5)     - y*h(6) ;
          dhdt(5) = -Q(2,2)*h(2)   -sigma*h(4)             + h(5)     - x*h(6) ;
          dhdt(6) = -Q(3,3)*h(3)                         + x*h(5) +  beta*h(6) ;
    
end

function Ah = Ah_vector(h)
  H = [h(1:3);zeros(3,1);h(4:end);zeros(3,1)];

 for p = 1:2:P-1
     i =ceil(p/2);
     [ta,h_fw] = ode45(@(tc,h_fw) newton_raphson_hom_solve(tc,h_fw,t,x,y,z,rho,sigma,beta,Q), t_span_con(:,p), H(1+6*(i-1):6*i), opts);
     [ta,h_bw] = ode45(@(tc,h_bw) newton_raphson_hom_solve(tc,h_bw,t,x,y,z,rho,sigma,beta,Q), t_span_con(:,p+1), H(1+(6*i):6*(i+1)), opts);
     Ah{i,:}   = (h_fw(end,:)'-h_bw(end,:)');
 end
    Ah = cell2mat(Ah);
end

end