# cp symbolic.py /opt/conda/lib/python3.6/site-packages/torch/onnx/symbolic.py
echo ""
echo "This export script requires a fairly recent version of PyTorch. Please make sure to update PyTorch if you run into errors."
echo ""
echo "If you want to export an ensemble model, please edit `models.py` and change line 65"
echo ""
echo "        input_incept = F.interpolate(x, (self.input_sizes[0], self.input_sizes[0]), mode='bilinear')"
echo "        input_resnet = F.interpolate(x, (self.input_sizes[1], self.input_sizes[1]), mode='bilinear')" 
echo ""
echo "to"
echo ""
echo "        input_incept = x # F.interpolate(x, (self.input_sizes[0], self.input_sizes[0]), mode='bilinear')"
echo "        input_resnet = x # F.interpolate(x, (self.input_sizes[1], self.input_sizes[1]), mode='bilinear')" 
echo ""
echo "This disables an additional resizing of the input and will use the same resolution for both networks."
echo ""
echo "Press enter to continue, otherwise CTRL+c to cancel"
read
echo "Running export..."
python onnx_export.py --model_type resnext101 --image_size 224 --weights_path /path/to/model_best.pth.tar \
	--num_classes 1000 \
	--output_prefix myprefix
echo ""
echo "Finished. If you saw a stack trace, then you probably didn't patch symbolic.py as adviced above. Add the custom lines and try to run the script again."
