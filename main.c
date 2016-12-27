#include <parser.h>

int main(int argc, char** argv)
{
	network net = parse_network_cfg("cfg/yolov1/tiny-coco.cfg");
	return 0;
}
