clear; close all; clc;

%% Settings
a = 1;
b = 2;
c = 3;

%% 1-D: RANSAC, Linear Regression, Hough Transform ...


%% Start
x = 1:100;
y = a * x.^2 + b * x + c;

%% Curve Fitting
a_est = 0.9;
b_est = 1.9;
c_est = 2.9;
y_est = a_est * x.^2 + b_est * x + c_est;

%% Plotting
figure(1);
plot(x, y, 'b-+', 'MarkerSize', 10, 'LineWidth', 5)
hold on;
plot(x, y_est, 'r*', 'MarkerSize', 10, 'LineWidth', 5)