
% Auto-generated by ocrTrainer app on 05-May-2018
%-------------------------------------------------------
%
% evaluateOCRTraining Evaluate OCR training
% results = evaluateOCRTraining(I) runs OCR using the trained language model
% and returns an image, ocrI, that has been annotated with the recognition
% results.
%
% ocrI = evaluateOCRTraining(..., roi) optionally limit OCR to a
% rectangular region of interest, roi, within I. Use this if I contains a
% lot of non-text background, which can hinder OCR performance.
%
% [..., results] = evaluateOCRTraining(I) optionally returns the detailed
% recogition results in an ocrText object.
%
% Notes
% -----
%
% 1) You may need to pre-process your test images to remove noise and
%    improve text segmentation.
%
% 2) You may need to modify the OCR 'TextLayout' parameter used below to
%    something that is more suitable for your input images.
%
% 3) You may need to re-train your language using more training samples.
%
% 4) You may need to modify this function and pass a region of interest
%    (ROI) to the OCR function. This is required if your image has a lot
%    of non-text background. Alternatively, use IMCROP to manually crop the
%    image before evaluating OCR.
%
% Example 1 - Draw an ROI and evaluate OCR training
% ---------------------------------------------------
%  I = imread('D:\Google Drive\Z480 Sync\Projects\license-plate-recognition\58cda470-9c74-4706-afd9-b6d64ffde00b.jpg');
%  figure
%  imshow(I)
%
%  % Draw a region of interest
%  h = imrect
%
%  % Evaluate OCR within ROI
%  roi = h.getPosition;
%  ocrI = evaluateOCRTraining(I, roi);
%
%  % Show results
%  figure
%  imshow(ocrI)
%
%  Example 2 - Get all OCR results
%  -------------------------------
%  I = imread('D:\Google Drive\Z480 Sync\Projects\license-plate-recognition\58cda470-9c74-4706-afd9-b6d64ffde00b.jpg');
%  [ocrI, results] = evaluateOCRTraining(I);
%  results.Text
%  results.CharacterConfidences
%
%  Example 3 - Batch OCR evaluation
%  --------------------------------
%  % Save this generated function and use it in the imageBatchProcessor
%  imageBatchProcessor
%
% See also ocr, ocrTrainer, ocrText
function [ocrI, results] = evaluateOCRTraining(I, roi)

% Location of trained OCR language data
trainedLanguage = 'D:\Projects\license-plate-recognition\ThaiLang\tessdata\ThaiLang.traineddata';

% Run OCR using trained language. You may need to modify OCR parameters or
% pre-process your test images for optimal results. Also, consider
% specifying an ROI input to OCR in case your images have a lot of non-text
% background.
layout = 'Block';
if nargin == 2
    results = ocr(I, roi, ...
        'Language', trainedLanguage, ...
        'TextLayout', layout);
else
    results = ocr(I, ...
        'Language', trainedLanguage, ...
        'TextLayout', layout);
end

ocrI = insertOCRAnnotation(I, results);

%--------------------------------------------------------------------------
% Annotate I with OCR results.
%--------------------------------------------------------------------------
