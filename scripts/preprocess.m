function preprocess(x)
img = imread(x);
img = transpose(img);
img = reshape(img, 1, []);
fileName = strcat(x, '.txt');
file = fopen(fileName, 'w');
fprintf(file, '%d\n', img);
