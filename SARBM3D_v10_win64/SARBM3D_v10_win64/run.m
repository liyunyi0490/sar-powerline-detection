% =========================================================================
% SAR-BM3D 去噪脚本 - 并行版本（修复版）
% =========================================================================
clear; clc;

% --- 1. 设置路径 ---
dataset_root = 'C:\Users\32956\Desktop\sar(L=20)\train'; 
noisy_dir = fullfile(dataset_root, '');
result_dir = fullfile(dataset_root, 'BM3D');

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% --- 2. 获取文件列表 ---
fprintf('正在读取文件列表...\n');
files = dir(fullfile(noisy_dir, '*.jpg'));
num_files = length(files);
fprintf('共找到 %d 张图片，准备开始并行处理...\n', num_files);

% --- 2.1 预处理：将文件信息转换为单元格数组 ---
% 这样每个工作进程都能独立访问文件信息
file_info = cell(num_files, 3);
for i = 1:num_files
    file_info{i, 1} = files(i).name;          % 文件名
    file_info{i, 2} = files(i).folder;        % 文件路径
    file_info{i, 3} = fullfile(files(i).folder, files(i).name);  % 完整路径
end

% --- 3. 初始化并行池 ---
try
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('正在启动并行池...\n');
        parpool('local', 16);  % 设置为 几 个进程
    else
        fprintf('使用现有的并行池（%d 个工作进程）...\n', pool.NumWorkers);
    end
catch ME
    fprintf('并行池创建失败，将使用串行模式: %s\n', ME.message);
    % 继续执行，后续会自动使用串行模式
end

% --- 4. 开始并行处理 ---
fprintf('开始并行处理图像...\n');

parfor i = 1:num_files
    % 从单元格数组中获取文件信息
    filename = file_info{i, 1};
    full_path = file_info{i, 3};
    
    % --- 提取视数 L ---
    [~, name_no_ext, ~] = fileparts(filename);
    tokens = regexp(name_no_ext, 'L(\d+(?:\.\d+)?)', 'tokens');
    
    if isempty(tokens)
        L_val = 20;  % 默认视数
        fprintf('使用默认视数 L=%d 处理文件: %s\n', L_val, filename);
    else
        L_str = tokens{1}{1};
        L_val = str2double(L_str);
        fprintf('提取视数 L=%d 处理文件: %s\n', L_val, filename);
    end
    
    % --- 读取图像 ---
    img_in = imread(full_path);
    img_in = double(img_in);
    
    % --- 转换为振幅图像 ---
    img_amplitude = sqrt(img_in);
    
    % --- 核心调用 SARBM3D_v10 ---
    try
        y_hat_amplitude = SARBM3D_v10(img_amplitude, L_val);
        y_hat_intensity = y_hat_amplitude .^ 2;
        img_out = uint8(y_hat_intensity);
        save_path = fullfile(result_dir, filename);
        imwrite(img_out, save_path);
        fprintf('处理完成: %s\n', filename);
        
    catch ME
        fprintf('处理 %s 时出错: %s\n', filename, ME.message);
    end
end

fprintf('所有图像处理完成！\n');