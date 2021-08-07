clear; close all;

%% simulation parameters
frame_size = 100;
dx = 1;
dy = 1;
dt = 1/100;
sigma_n = 300;
sigma_v = 25;
u_estimation_range = -50:49;
w_estimation_range = -50:49;

%% configuration parameters
u_options = [30, 0];
w_options = [0, 30];
frame_symmetry_options = [ "blank wall", "vertical", "horizontal" ,  "diagonal", "random"];

%% Main loop - loop over configurations and plot them
figure;

prior = calculate_prior(u_estimation_range, w_estimation_range, sigma_v);
for frame_symmetry_index = 1:numel(frame_symmetry_options)
    
    frame_symmetry = frame_symmetry_options(frame_symmetry_index);
    
    frames_t = create_frame(frame_size, frame_symmetry);
    
    subplot(numel(u_options) + 1, numel(frame_symmetry_options), frame_symmetry_index);
    imshow(frames_t, [0,1]);
    title(frame_symmetry)
    
    for velocity_index = 1:numel(u_options)
        
        u = u_options(velocity_index);
        w = w_options(velocity_index);
        
        [frames_t_plus_dt] = propagate_frame_in_time(frames_t, u, w, dt);
        
        [Ix, Iy, It] = calculate_gradients(frames_t, frames_t_plus_dt, dx, dy, dt);
        
        p_I_given_u_w = calculate_P_I_given_u_w(...
            u_estimation_range, w_estimation_range,  Ix, Iy, It, sigma_n);
        p_u_w_given_I = p_I_given_u_w .* prior;
        
        [estimated_u, estimated_w] =...
            estimate_u_w(p_u_w_given_I, u_estimation_range, w_estimation_range);
        
    subplot_index =  velocity_index * numel(frame_symmetry_options) + frame_symmetry_index;
    subplot(numel(u_options) + 1, numel(frame_symmetry_options), subplot_index);
    imshow(p_u_w_given_I, []);
    end
end

%% functions

function [prior] = calculate_prior(u_estimation_range, w_estimation_range, sigma_v)
    [U, W] = meshgrid(u_estimation_range, w_estimation_range);
    prior = exp(-1*(U.^2 + W.^2)/(2*sigma_v^2));
end

function [frames_t] = create_frame(frame_size, frame_symmetry)
    frames_t = zeros(frame_size,frame_size);
    switch frame_symmetry
        case "blank wall"
            
        case "horizontal"
            frames_t(1:10:end,:) = 1;
        case "vertical"
            frames_t(:,1:10:end) = 1;
        case 'diagonal'
            frames_t(:,1:10:end) = 1;
            for i = 1:frame_size
                frames_t(i,:) = circshift(frames_t(i,:), -1*i, 2);
            end
        case "random"
            frames_t = floor((randi(frame_size/10, ...
                [frame_size,frame_size]) - 1)*(1/(frame_size/10-1)));
                         
    frames_t = frames_t + randn(size(frames_t)) * 10^-5;
    end
end

function [Ix, Iy, It] = calculate_gradients(frames_t, frames_t_plus_dt, dx, dy, dt)
    Ix = (frames_t(2:end,2:end) - frames_t(2:end,1:end-1))/dx;
    Iy = (frames_t(2:end,2:end) - frames_t(1:end-1,2:end))/dy;
    It = (frames_t_plus_dt(2:end,2:end) - frames_t(2:end,2:end))/dt;
end

function [frame_propagated_in_time] = propagate_frame_in_time(frames_t, u, w, dt)
    convxy = conv2([1-w*dt,w*dt],[1-u*dt,u*dt], frames_t);
    frame_propagated_in_time = convxy(1:end-1,1:end-1);
end

function [p_I_given_u_w] = calculate_P_I_given_u_w(...
    u_estimation_range, w_estimation_range,  Ix, Iy, It, sigma_n)
    p_I_given_u_w = zeros(numel(u_estimation_range), numel(w_estimation_range));
    for u_ind = 1 : numel(u_estimation_range)
        for w_ind = 1 : numel(w_estimation_range)
            current_u = u_estimation_range(u_ind);
            current_w = w_estimation_range(w_ind);
            differential_constancy_assumption = ...
                -1*sum((Ix(:)*current_u + Iy(:)*current_w + It(:)).^2)/(2*sigma_n^2);
            p_I_given_u_w(w_ind, u_ind) = exp(differential_constancy_assumption);
        end
    end
end

function [estimated_u, estimated_w] = estimate_u_w(p_u_w_given_I, u_estimation_range, w_estimation_range)
    [~, estimated_index] = max(p_u_w_given_I(:));
    estimated_u = u_estimation_range(floor((estimated_index-1) / numel(w_estimation_range)) + 1);
    estimated_w = w_estimation_range(mod(estimated_index-1, numel(w_estimation_range)) + 1);
end