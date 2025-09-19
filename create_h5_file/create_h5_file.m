clear;
close all;

fprintf('starting script...\n');

savepath   = 'D:\Documents\MATLAB\MEF-GAN-H5-builder/training/train.h5';
size_input = 144;
stride     = 200;
Format     = '*.jpg';
img_ch     = 3;   % 1 for gray, 3 for RGB
ratio      = 1;   % >1 to enlarge before patching

% ---------------------------------------
% Allocate for 3 stacks: OE, UE, GT (3*img_ch channels)
% Use uint8 for compact storage (normalize later in Python).
% ---------------------------------------
data = zeros(size_input, size_input, 3*img_ch, 1, 'uint8');

folder1   = 'D:\Documents\MATLAB\MEF-GAN-H5-builder/training/over';   % OE
filepath1 = dir(fullfile(folder1, Format));

folder2   = 'D:\Documents\MATLAB\MEF-GAN-H5-builder/training/under';  % UE
filepath2 = dir(fullfile(folder2, Format));

folder3   = 'D:\Documents\MATLAB\MEF-GAN-H5-builder/training/gt';     % GT
filepath3 = dir(fullfile(folder3, Format));

count = 0;
L1 = length(filepath1);
L2 = length(filepath2);
L3 = length(filepath3);
L  = min([L1, L2, L3]);

tic
for i = 1:L
    % --- read images ---
    img1 = imread(fullfile(folder1, filepath1(i).name));  % OE
    img2 = imread(fullfile(folder2, filepath2(i).name));  % UE
    img3 = imread(fullfile(folder3, filepath3(i).name));  % GT

    % --- ensure 3 channels (some images may be grayscale) ---
    if size(img1,3) == 1, img1 = repmat(img1, [1 1 3]); end
    if size(img2,3) == 1, img2 = repmat(img2, [1 1 3]); end
    if size(img3,3) == 1, img3 = repmat(img3, [1 1 3]); end

    % --- convert to uint8 consistently (avoid mixing double/int for storage) ---
    % If you prefer saving normalized floats, use: imgXf = im2single(imgX);
    img1u = im2uint8(img1);
    img2u = im2uint8(img2);
    img3u = im2uint8(img3);

    % --- pick a common target size so all three match BEFORE patching ---
    % Use img1's size as reference (or min across all three to avoid upscaling)
    H1 = size(img1u,1); W1 = size(img1u,2);
    targetH = max(1, fix(H1 * ratio));
    targetW = max(1, fix(W1 * ratio));

    % Resize all three to the SAME target
    i1 = imresize(img1u, [targetH, targetW]);
    i2 = imresize(img2u, [targetH, targetW]);
    i3 = imresize(img3u, [targetH, targetW]);

    fprintf('Processing image %d/%d: %s\n', i, L, filepath1(i).name);

    [height, width, ~] = size(i1);

    % --- extract aligned patches ---
    for x = 1:stride:(height - size_input + 1)
        for y = 1:stride:(width  - size_input + 1)
            sub_img1 = i1(x:x+size_input-1, y:y+size_input-1, :); % OE
            sub_img2 = i2(x:x+size_input-1, y:y+size_input-1, :); % UE
            sub_img3 = i3(x:x+size_input-1, y:y+size_input-1, :); % GT

            count = count + 1;

            % Channels: [1:img_ch]=OE, [img_ch+1:2*img_ch]=UE, [2*img_ch+1:3*img_ch]=GT
            data(:, :,              1:img_ch,              count) = sub_img1;
            data(:, :,  img_ch+1 :  2*img_ch,              count) = sub_img2;
            data(:, :, 2*img_ch+1 : 3*img_ch,              count) = sub_img3;

            % ------- Optional augmentations (vertical/horizontal flips) -------
            %{
            % Vertical flip
            count = count + 1;
            data(:, :,              1:img_ch,              count) = flipud(sub_img1);
            data(:, :,  img_ch+1 :  2*img_ch,              count) = flipud(sub_img2);
            data(:, :, 2*img_ch+1 : 3*img_ch,              count) = flipud(sub_img3);

            % Horizontal flip
            count = count + 1;
            data(:, :,              1:img_ch,              count) = fliplr(sub_img1);
            data(:, :,  img_ch+1 :  2*img_ch,              count) = fliplr(sub_img2);
            data(:, :, 2*img_ch+1 : 3*img_ch,              count) = fliplr(sub_img3);
            %}
        end
    end

    if mod(i,10) == 0
        i
        toc
    end

    % prevent size carryover into next iteration
    clear i1 i2 i3 img1u img2u img3u img1 img2 img3
end

% Randomize sample order once at the end
order = randperm(count);
data  = data(:, :, :, order);

% ---- write to HDF5 (store2hdf5 must be on your MATLAB path) ----
chunksz     = 12;
totalct     = 0;
create_flag = true;

for batch_num = 1:floor(count / chunksz)
    lastread   = (batch_num - 1) * chunksz;
    batch_data = data(:, :, :, lastread+1 : lastread+chunksz);
    startloc   = struct('dat', [1, 1, 1, totalct + 1]);

    if batch_num ~= 1
        create_flag = false;
    end

    curr_dat_sz = store2hdf5(savepath, batch_data, create_flag, startloc, chunksz);
    totalct     = curr_dat_sz(end);
end

fprintf('Done. Saved %d patches.\n', count);
fprintf('Final dataset shape (H, W, C, N): [%d %d %d %d]\n', size_input, size_input, 3*img_ch, totalct);
