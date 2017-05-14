
data = load("train_32x32.mat");
fieldnames(data)

X = data.X;
y = data.y;

whos

%
%   Attr Name        Size                     Bytes  Class
%   ==== ====        ====                     =====  ===== 
%        X          32x32x3x73257         225045504  uint8
%        ans     73257x1                     586056  double
%        data        1x1                  225631560  struct
%        y       73257x1                     586056  double
%

for i = 1:rows(y)
	i
	img = X(:,:,:,i);
	label = y(i)
	string_label = num2str(label)
	filename = strcat("./", string_label,"/","svhn_", string_label,"_",num2str(i), ".jpg")
	imwrite(img, filename, 'jpg');

endfor
