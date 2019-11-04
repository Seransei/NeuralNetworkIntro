#pragma once
#include "include/fann.h"

void getNumInOut(const char*);

int main()
{
	const char* filename = "langue.data";

	unsigned int num_input;
	unsigned int num_output;

	getNumInOut(filename);

	const unsigned int num_layers = 2;
	const unsigned int num_neurons_layers = 5;

	const float  desired_error = 0.001f;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;

	struct fann* ann = fann_create_standard(num_layers, num_input, num_neurons_layers, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "output.net");

	fann_destroy(ann);

	return 0;
}

void getNumInOut(const char* filename) 
{

}