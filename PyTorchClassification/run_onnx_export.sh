# cp symbolic.py /opt/conda/lib/python3.6/site-packages/torch/onnx/symbolic.py
echo ""
echo "This export script assumes that you patched the pytorch file in /path/to/conda/env/lib/python3.6/site-packages/torch/onnx/symbolic.py by adding "
echo ""
echo "def adaptive_avg_pool2d(g, input, output_size):"
echo "    assert output_size == [1, 1], \"Only output_size=[1, 1] is supported\""
echo "    return g.op(\"GlobalAveragePool\", input)"
echo "def adaptive_max_pool2d(g, input, output_size):"
echo "    assert output_size == [1, 1], \"Only output_size=[1, 1] is supported\""
echo "    return g.op("GlobalMaxPool", input), None"
echo ""
echo "to it. "
echo "Press enter to continue, otherwise CTRL+c to cancel"
read
echo "Running export..."
python onnx_export.py --model_type resnext101 --image_size 224 --weights_path /path/to/model_best.pth.tar \
	--num_classes 1000 \
	--output_prefix myprefix
echo ""
echo "Finished. If you saw a stack trace, then you probably didn't patch symbolic.py as adviced above. Add the custom lines and try to run the script again."
