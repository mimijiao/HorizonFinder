function showHorizon(x)
houghfile = fopen('houghLines.txt', 'r');
houghs = fscanf(houghfile, '%f %f\n', [2 Inf]);
houghs = houghs';

img = imread(x);
figure, imshow(img), hold on
h_line = houghs(1,:);
n = size(img, 2);
x1 = polyval(h_line, 0);
y1 = polyval(h_line, n);
line([1,n],[x1,y1], 'Color', 'r', 'LineWidth', 1);
hold off
