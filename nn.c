
// gcc -std=c99 -O0 nn.c

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_TENSOR_SIZE (256 * 256 * 4)

typedef float Storage[MAX_TENSOR_SIZE];

typedef struct {
	float* data;
	int32_t size0;
	int32_t size1;
	int32_t size2;
	int32_t size3;
	int32_t stride0;
	int32_t stride1;
	int32_t stride2;
	int32_t stride3;
} Tensor;

void assert(bool condition, int32_t line) {
	if(!condition) {
		printf("Assert %i\n", line);
		exit(999);
	}
}

#define ASSERT(condition) assert(condition, __LINE__)

void error() {
	printf("Error\r");
	exit(666);
}

void tensor_make_4d(Tensor* this, float* data, int32_t size0, int32_t size1, int32_t size2, int32_t size3) {
	this->data = data;
	ASSERT((size0 * size1 * size2 * size3) <= MAX_TENSOR_SIZE);
	this->size0 = size0;
	this->size1 = size1;
	this->size2 = size2;
	this->size3 = size3;
	this->stride3 = 1;
	this->stride2 = this->stride3 * size3;
	this->stride1 = this->stride2 * size2;
	this->stride0 = this->stride1 * size1;
}

int32_t tensor_size(Tensor* this) {
	return this->size0 * this->size1 * this->size2 * this->size3;
}

static int n = 0;

int32_t tensor_index_4d(Tensor* this, int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
	ASSERT(i0 >= 0);
	ASSERT(i0 < this->size0);
	ASSERT(i1 >= 0);
	ASSERT(i1 < this->size1);
	ASSERT(i2 >= 0);
	ASSERT(i2 < this->size2);
	ASSERT(i3 >= 0);
	ASSERT(i3 < this->size3);
	int32_t i = (this->stride0 * i0) + (this->stride1 * i1) + (this->stride2 * i2) + (this->stride3 * i3);
	ASSERT(i >= 0);
	ASSERT(i < tensor_size(this));
	n += 1;
	return i;
}

void tensor_set_4d(Tensor* this, int32_t i0, int32_t i1, int32_t i2, int32_t i3, float value) {
	this->data[tensor_index_4d(this, i0, i1, i2, i3)] = value;
}

float tensor_get_4d(Tensor* this, int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
	return this->data[tensor_index_4d(this, i0, i1, i2, i3)];
}

void tensor_fill(Tensor* this, float value) {
	int32_t size = tensor_size(this);
	for(int32_t i = 0; i < size; i += 1) {
		this->data[i] = value;
	}
}

void tensor_print(Tensor* this) {
	printf("%i x %i x %i x %i\n", this->size0, this->size1, this->size2, this->size3);
	int32_t size = tensor_size(this);
	for(int32_t i = 0; i < size; i += 1) {
		printf("%0.5f ", this->data[i]);
	}
	printf("\n");
}

void tensor_fill_funny(Tensor* this, float add) {
	int32_t size = tensor_size(this);
	for(int32_t i = 0; i < size; i++) {
		this->data[i] = ((float)i / size) + add;
	}
}

void linear_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
	const int32_t layers = input->size2;
	const int32_t inputs = input->size3;
	const int32_t outputs = output->size3;
	
	//ASSERT(weight->size0 == inputs);
	//ASSERT(weight->size1 == outputs);
	//ASSERT(weight->size2 == 1);
	//ASSERT(weight->size3 == 1);
	
	//printf("%i %i\n", input->size2, output->size2);
	ASSERT(input->size2 == output->size2);
	
	//printf("%i %i\n", inputs, outputs);

	ASSERT(tensor_size(weight) == (inputs * outputs));
	ASSERT(tensor_size(bias) == outputs);
	
	for(int32_t l = 0; l < layers; l += 1) {
		for(int32_t o = 0; o < outputs; o += 1) {
			float sum = tensor_get_4d(bias, 0, 0, 0, o);
			for(int32_t i = 0; i < inputs; i += 1) {
				sum += tensor_get_4d(input, 0, 0, l, i) * tensor_get_4d(weight, 0, 0, o, i);
			}
			tensor_set_4d(output, 0, 0, l, o, sum);
		}
	}
	
	/*
	int32_t w = 0;
	for(int32_t o = 0; o < outputs; o += 1) {
		output->data[o] = bias->data[o];
		for(int32_t i = 0; i < inputs; i += 1, w += 1) {
			output->data[o] += input->data[i] * weight->data[w];
		}
	}
	*/
}

void test_linear() {
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t outputs = 2;
	
	Storage s1;
	Tensor image;
	tensor_make_4d(&image, s1, 1, 1, w, h);
	tensor_fill_funny(&image, 0.1);
	
	Tensor input;
	tensor_make_4d(&input, s1, 1, 1, 1, w * h);
	
	Storage s2;
	Tensor weight;
	tensor_make_4d(&weight, s2, 1, 1, outputs, w * h);
	tensor_fill_funny(&weight, 0.2);
	
	Storage s3;
	Tensor bias;
	tensor_make_4d(&bias, s3, 1, 1, 1, outputs);
	tensor_fill_funny(&bias, 0.3);
	
	Storage s4;
	Tensor output;
	tensor_make_4d(&output, s4, 1, 1, 1, outputs);
	
	linear_forward(&input, &weight, &bias, &output);

	//tensor_print(&input);
	//tensor_print(&weight);
	//tensor_print(&bias);
	tensor_print(&output);
}

void convolution_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
	const int32_t w = input->size2;
	const int32_t h = input->size3;
	const int32_t output_planes = weight->size0;
	const int32_t input_planes = weight->size1;
	const int32_t kw = weight->size2;
	const int32_t kh = weight->size3;
	const int32_t w2 = w - kw + 1;
	const int32_t h2 = h - kh + 1;
	
	ASSERT(kw >= 1);
	ASSERT(kh >= 1);
	ASSERT(input->size0 == 1);
	ASSERT(input->size1 == input_planes);
	ASSERT(tensor_size(bias) == output_planes);
	ASSERT(output->size0 == 1);
	ASSERT(output->size1 == output_planes);
	ASSERT(output->size2 == w2);
	ASSERT(output->size3 == h2);

	for(int32_t k = 0; k < output_planes; k += 1) {
		for(int32_t j = 0; j < w2; j += 1) {
			for(int32_t i = 0; i < h2; i += 1) {
				float sum = tensor_get_4d(bias, 0, 0, 0, k);
				for(int32_t l = 0; l < input_planes; l += 1) {
					for(int32_t s = 0; s < kw; s += 1) {
						for(int32_t t = 0; t < kh; t += 1) {
							float ww = tensor_get_4d(weight, k, l, s, t);
							float ii = tensor_get_4d(input, 0, l, j + s, i + t);
							sum += ww * ii;
						}
					}
				}
				tensor_set_4d(output, 0, k, j, i, sum);
			}
		}
	}
}

void test_convolution() {
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t kw = 3;
	const int32_t kh = 3;
	const int32_t w2 = w - kw + 1;
	const int32_t h2 = h - kh + 1;
	const int32_t input_planes = 2;
	const int32_t output_planes = 3;
	
	Storage s1;
	Tensor input;
	tensor_make_4d(&input, s1, 1, input_planes, w, h);
	tensor_fill_funny(&input, 0.4);
	
	Storage s2;
	Tensor weight;
	tensor_make_4d(&weight, s2, output_planes, input_planes, kw, kh);
	tensor_fill_funny(&weight, 0.5);
	//tensor_fill(&weight, 0.0);
	
	Storage s3;
	Tensor bias;
	tensor_make_4d(&bias, s3, 1, 1, 1, output_planes);
	tensor_fill_funny(&bias, 0.6);
	
	Storage s4;
	Tensor output;
	tensor_make_4d(&output, s4, 1, output_planes, w2, h2);
	
	convolution_forward(&input, &weight, &bias, &output);
		
	//tensor_print(&input);
	//tensor_print(&weight);
	//tensor_print(&bias);
	tensor_print(&output);
}

void test_linear_convolution() {
	const int32_t w = 106;
	const int32_t h = 104;
	const int32_t kw = 3;
	const int32_t kh = 3;
	const int32_t w2 = w - kw + 1;
	const int32_t h2 = h - kh + 1;
	const int32_t input_planes = 2;
	const int32_t output_planes = 3;
	const int32_t outputs = 2;
	
	// convolution
	
	Storage s1;
	Tensor input;
	tensor_make_4d(&input, s1, 1, input_planes, w, h);
	tensor_fill_funny(&input, 0.7);
	
	Storage s2;
	Tensor convolution_weight;
	tensor_make_4d(&convolution_weight, s2, output_planes, input_planes, kw, kh);
	tensor_fill_funny(&convolution_weight, 0.8);
	
	Storage s3;
	Tensor convolution_bias;
	tensor_make_4d(&convolution_bias, s3, 1, 1, 1, output_planes);
	tensor_fill_funny(&convolution_bias, 0.9);
	
	Storage s4;
	Tensor convolution_output;
	tensor_make_4d(&convolution_output, s4, 1, output_planes, w2, h2);

	// linear

	Tensor linear_input;
	tensor_make_4d(&linear_input, s4, 1, 1, output_planes, w2 * h2);

	Storage s5;
	Tensor linear_weight;
	tensor_make_4d(&linear_weight, s5, 1, 1, outputs, w2 * h2);
	tensor_fill_funny(&linear_weight, 1.0);
	
	Storage s6;
	Tensor linear_bias;
	tensor_make_4d(&linear_bias, s6, 1, 1, 1, outputs);
	tensor_fill_funny(&linear_bias, 1.1);

	Storage s7;
	Tensor output;
	tensor_make_4d(&output, s7, 1, 1, output_planes, outputs);
	
	convolution_forward(&input, &convolution_weight, &convolution_bias, &convolution_output);
	linear_forward(&linear_input, &linear_weight, &linear_bias, &output);
		
	//tensor_print(&input);
	//tensor_print(&linear_input);
	tensor_print(&output);
}

int32_t main() {
	printf("Helloo NN\n");

	test_linear();
	test_convolution();
	test_linear_convolution();
	
	//printf("N:%i\n", n);
	
	printf("ByeBye NN\n");

	return 0;
}