
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define float double
#define TENSOR_PRINT_ENABLE

typedef struct {
	float* data;
	int32_t size;
} Storage;

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
//#define ASSERT(condition)

void storage_make(Storage* this, float* data, int32_t size) {
	this->data = data;
	this->size = size;
}

void tensor_make_4d(Tensor* this, Storage* storage, int32_t size0, int32_t size1, int32_t size2, int32_t size3) {
	int32_t size = size0 * size1 * size2 * size3; 
	if(storage == NULL) {
		this->data = malloc(size * sizeof(float));
	} else {
		this->data = storage->data;
		ASSERT(size <= storage->size);
	}
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
#ifdef TENSOR_PRINT_ENABLE
	printf("%i x %i x %i x %i\n", this->size0, this->size1, this->size2, this->size3);
	int32_t size = tensor_size(this);
	for(int32_t i = 0; i < size; i += 1) {
		printf("%0.5f ", this->data[i]);
	}
	printf("\n");
#endif
}

void tensor_fill_funny(Tensor* this, float add) {
	int32_t size = tensor_size(this);
	for(int32_t i = 0; i < size; i++) {
		this->data[i] = ((float)i / size) + add;
	}
}

// https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear

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
				float ww = tensor_get_4d(weight, 0, 0, o, i);
				float ii = tensor_get_4d(input, 0, 0, l, i);
				sum += ww * ii;
			}
			tensor_set_4d(output, 0, 0, l, o, sum);
		}
	}
}

void test_linear() {
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t outputs = 2;
	
	Tensor image;
	tensor_make_4d(&image, NULL, 1, 1, w, h);
	tensor_fill_funny(&image, 0.1);
	
	Tensor input;
	tensor_make_4d(&input, NULL, 1, 1, 1, w * h);
	
	Tensor weight;
	tensor_make_4d(&weight, NULL, 1, 1, outputs, w * h);
	tensor_fill_funny(&weight, 0.2);
	
	Tensor bias;
	tensor_make_4d(&bias, NULL, 1, 1, 1, outputs);
	tensor_fill_funny(&bias, 0.3);
	
	Tensor output;
	tensor_make_4d(&output, NULL, 1, 1, 1, outputs);
	
	linear_forward(&input, &weight, &bias, &output);

	//tensor_print(&input);
	//tensor_print(&weight);
	//tensor_print(&bias);
	tensor_print(&output);
}

// https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution

void spatial_convolution_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
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

void test_spatial_convolution() {
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t kw = 3;
	const int32_t kh = 3;
	const int32_t w2 = w - kw + 1;
	const int32_t h2 = h - kh + 1;
	const int32_t input_planes = 2;
	const int32_t output_planes = 3;
	
	Tensor input;
	tensor_make_4d(&input, NULL, 1, input_planes, w, h);
	tensor_fill_funny(&input, 0.4);
	
	Tensor weight;
	tensor_make_4d(&weight, NULL, output_planes, input_planes, kw, kh);
	tensor_fill_funny(&weight, 0.5);
	
	Tensor bias;
	tensor_make_4d(&bias, NULL, 1, 1, 1, output_planes);
	tensor_fill_funny(&bias, 0.6);
	
	Tensor output;
	tensor_make_4d(&output, NULL, 1, output_planes, w2, h2);
	
	spatial_convolution_forward(&input, &weight, &bias, &output);
		
	//tensor_print(&input);
	//tensor_print(&weight);
	//tensor_print(&bias);
	tensor_print(&output);
}

float max(float value1, float value2) {
	if(value1 > value2) {
		return value1;
	} else {
		return value2;
	}
}

void relu_forward(Tensor* input, Tensor* output) {
	ASSERT(input->size0 == input->size0);
	ASSERT(input->size1 == input->size1);
	ASSERT(input->size2 == input->size2);
	ASSERT(input->size3 == input->size3);

	int32_t count = tensor_size(input);
	for(int32_t i = 0; i < count; i += 1) {
		output->data[i] = max(0.0, input->data[i]);
	}
}

void test_relu() {
	Tensor input;
	tensor_make_4d(&input, NULL, 2, 1, 2, 3);
	tensor_fill_funny(&input, -0.5);
	
	Tensor output;
	tensor_make_4d(&output, NULL, 2, 1, 2, 3);

	relu_forward(&input, &output);
	
	tensor_print(&output);
}

// https://github.com/torch/nn/blob/master/doc/simple.md#nn.BatchNormalization

void batch_normalization_forward(Tensor* input, Tensor* mean, Tensor* var, Tensor* output) {
	const float epsilon = 1.0e-5;
	
	const int32_t n = input->size2;
	const int32_t d = input->size3;
	
	ASSERT(input->size0 == 1);
	ASSERT(input->size1 == 1);
	ASSERT(output->size0 == 1);
	ASSERT(output->size1 == 1);
	
	ASSERT(input->size2 == output->size2);
	ASSERT(input->size3 == output->size3);

	ASSERT(tensor_size(mean) == d);
	ASSERT(tensor_size(var) == d);
	
	for(int32_t i = 0; i < n; i += 1) {
		for(int32_t j = 0; j < d; j += 1) {
			float ii = tensor_get_4d(input, 0, 0, i, j);
			float mm = tensor_get_4d(mean, 0, 0, 0, j);
			float vv = tensor_get_4d(var, 0, 0, 0, j);
			float oo = (ii - mm) / sqrt(vv + epsilon);
			tensor_set_4d(output, 0, 0, i, j, oo);
		}
	}
}

void test_batch_normalization() {
	Tensor input;
	tensor_make_4d(&input, NULL, 1, 1, 3, 5);
	tensor_fill_funny(&input, 0.1);
	
	Tensor output;
	tensor_make_4d(&output, NULL, 1, 1, 3, 5);

	Tensor mean;
	tensor_make_4d(&mean, NULL, 1, 1, 1, 5);
	tensor_fill_funny(&mean, 0.2);

	Tensor var;
	tensor_make_4d(&var, NULL, 1, 1, 1, 5);
	tensor_fill_funny(&var, 0.3);

	batch_normalization_forward(&input, &mean, &var, &output);
	
	//tensor_print(&input);
	//tensor_print(&mean);
	//tensor_print(&var);
	tensor_print(&output);
}

void spatial_batch_normalization() {
}

void test_spatial_batch_normalization() {
}

void test_all() {
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t kw = 3;
	const int32_t kh = 3;
	const int32_t w2 = w - kw + 1;
	const int32_t h2 = h - kh + 1;
	const int32_t input_planes = 2;
	const int32_t output_planes = 3;
	const int32_t outputs = 2;
	
	// convolution
	
	Tensor input;
	tensor_make_4d(&input, NULL, 1, input_planes, w, h);
	tensor_fill_funny(&input, 0.7);
	
	Tensor convolution_weight;
	tensor_make_4d(&convolution_weight, NULL, output_planes, input_planes, kw, kh);
	tensor_fill_funny(&convolution_weight, 0.8);
	
	Tensor convolution_bias;
	tensor_make_4d(&convolution_bias, NULL, 1, 1, 1, output_planes);
	tensor_fill_funny(&convolution_bias, 0.9);
	
	Tensor convolution_output;
	tensor_make_4d(&convolution_output, NULL, 1, output_planes, w2, h2);
	
	// spatial batch normalization
	
	// TODO
	
	// relu
	
	// TODO

	// linear
	
	Storage s;
	storage_make(&s, convolution_output.data, tensor_size(&convolution_output));
	Tensor linear_input;
	tensor_make_4d(&linear_input, &s, 1, 1, output_planes, w2 * h2);

	Tensor linear_weight;
	tensor_make_4d(&linear_weight, NULL, 1, 1, outputs, w2 * h2);
	tensor_fill_funny(&linear_weight, 1.0);
	
	Tensor linear_bias;
	tensor_make_4d(&linear_bias, NULL, 1, 1, 1, outputs);
	tensor_fill_funny(&linear_bias, 1.1);

	Tensor output;
	tensor_make_4d(&output, NULL, 1, 1, output_planes, outputs);
	
	spatial_convolution_forward(&input, &convolution_weight, &convolution_bias, &convolution_output);
	linear_forward(&linear_input, &linear_weight, &linear_bias, &output);
		
	//tensor_print(&input);
	//tensor_print(&linear_input);
	tensor_print(&output);
}

int32_t main() {
	printf("Helloo NN\n");

	test_linear();
	test_spatial_convolution();
	test_relu();
	test_batch_normalization();
	test_spatial_batch_normalization();
	test_all();
	
	//printf("N:%i\n", n);
	
	printf("ByeBye NN\n");

	return 0;
}
