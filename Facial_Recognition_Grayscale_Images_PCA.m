clc
clearvars
close all

path_img = 'Path to image';
imgFiles = dir(fullfile(path_img, '*.jpg')); 
nr_img = numel(imgFiles);

% matricea de date
date = zeros(numel(imread(fullfile(path_img, imgFiles(1).name))), nr_img);

for i = 1:nr_img
    img = imread(fullfile(path_img, imgFiles(i).name));
    date(:, i) = img(:); 
end

%% aplicare PCA
%  standardizarea datelor
medie = mean(date, 2);
date = date - medie;

% matricea de covarianta
mat_cov = date*date';

% vectori si valori proprii
[vectori_proprii, valori_proprii] = eig(mat_cov);

% sortarea vectorilor proprii considerand ordinea descrescatoare a valorilor proprii
[~, idx] = sort(diag(valori_proprii), 'descend');
vectori_proprii = vectori_proprii(:, idx);
	
	%creare vector caracteristic din vectorii proprii
nr_comp_principale = 50; 
vectori_proprii_selectati = vectori_proprii(:, 1:nr_comp_principale);

% proiectarea datelor in noul subspatiu 
proiectii_date = vectori_proprii_selectati' * date;

%% testare imagini noi
test_set = imageDatastore('Location of data set');
for i=1:numel(test_set.Files)
    img_noua = readimage(test_set,i);

    % standardizare imagine
    img_noua_vector = double(img_noua(:)) - medie; 

    % proiectarea imaginii in subspatiul definit de componentele principale
    img_noua_proiectie = vectori_proprii_selectati' * img_noua_vector;

    % calculul scorului de similaritate prin norma euclidiana
    scor_similaritate = zeros(1, nr_img);

    for i = 1:nr_img
        scor_similaritate(i) = norm(img_noua_proiectie - proiectii_date(:, i));
    end

    % identificarea celei mai potrivite imagini (cea cu cea mai mica diferenta)
    [scor_minim, idx_potrivire] = min(scor_similaritate);

    %afisare rezultat final
    figure;
    subplot(1, 2, 1); imshow(img_noua); title('Imaginea Noua');
    subplot(1, 2, 2); imshow(uint8(reshape(date(:, idx_potrivire) + medie, size(img_noua)))); title('Imagine Recunoscuta');
    disp(['Scor de similaritate: ', num2str(scor_minim)]);
end
