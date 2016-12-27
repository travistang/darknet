#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

const int NUM_CLASSES = 80;
char *my_coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
typedef struct
{
	box* boxes;
	int* classes;
	float* max_prob;
	int num_boxes;
} result;

result* get_result(box* boxes, float** probs, int num_boxes)
{
	result* res = malloc(sizeof(result));
	// memory allocation
	res -> num_boxes = num_boxes;
	res -> boxes = calloc(num_boxes,sizeof(box));
	res -> max_prob = calloc(num_boxes,sizeof(float));
	res -> classes = calloc(num_boxes,sizeof(int));

	// content assignment
	memcpy(res -> boxes, boxes, num_boxes * sizeof(box));

	for(int i = 0; i < num_boxes; i++)
	{
		res -> classes[i] = max_index(probs[i],NUM_CLASSES);
		res -> max_prob[i] = probs[i][res -> classes[i]];
	}
	return res;
};
void free_result(result* r)
{
	free(r -> boxes);
	free(r -> classes);
	free(r -> max_prob);
	free(r);
};
result* inference(network* net, image* im)
{
	// network config
	detection_layer l = net -> layers[net -> n - 1];
	set_batch_network(net,1);

	// image config
	image size = resize_image(*im,net -> w, net -> h);
	srand(2222222);
	float thresh = .2;
	float nms = .4;
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    	for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	float* X = size.data;
	network_predict(*net,X);
	get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	int num_boxes = l.side*l.side*l.n;
	return get_result(boxes,probs,num_boxes);
}
int main(int argc, char** argv)
{
	// network configuration
	network net = parse_network_cfg("cfg/yolov1/tiny-coco.cfg");
	load_weights(&net,"tiny-yolo.weights");
	image im = load_image_color("image.jpg",0,0);
	result* res = inference(&net,&im);
	for (int i = 0; i < res -> num_boxes; i++)
	{
		box* b = &(res -> boxes[i]);
		char* predicted_class = my_coco_classes[res -> classes[i]];
		printf("%f %f %f %f class: %s prob: %f,\n",b -> x, b -> y, b -> w, b -> h,predicted_class,res -> max_prob[i]);
	}
	free_result(res);
	free_network(net);
	free_image(im);
	

	return 0;
}
