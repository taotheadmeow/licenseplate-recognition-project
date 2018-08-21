I = imread('real3.JPG');
%I = I>128;
[ocr, t] = evaluateOCRTraining(I);
disp(t.Text)
figure(1); imshow(insertOCRAnnotation(I, t));