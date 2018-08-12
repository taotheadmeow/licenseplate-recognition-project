function J = insertOCRAnnotation(I, results)
text = results.Text;

I = im2uint8(I);
if isempty(deblank(text))
    % Text not recognized.
    text = 'Unable to recognize any text.';
    [M,N,~] = size(I);
    J = insertText(I, [N/2 M/2], text, ...
        'AnchorPoint', 'Center', 'FontSize', 24, 'Font', "Sarun's ThangLuang");
    
else
    location = results.CharacterBoundingBoxes;
    
    % Remove new lines from results.
    newlines = text == char(10);
    text(newlines) = [];
    location(newlines, :) = [];
    
    % Remove spaces from results
    spaces = isspace(text);
    text(spaces) = [];
    location(spaces, :) = [];
    
    % Convert text array into cell array of strings.
    text = num2cell(text);
    
    % Pad the image to help annotate results close to the image border.
    I = padarray(I, [50 50], uint8(255));
    location(:,1:2) = location(:,1:2) + 50;
    
    % Insert text annotations.
    J  = insertObjectAnnotation(I, 'rectangle', location, text);
end