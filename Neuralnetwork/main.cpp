#pragma once
#include "include/fann.h"
#include <string>
#include <iostream>
#include <fstream>

void getNumInOut();
void testLanguage();

const char* trainingFile = "langue.data";

unsigned int num_input;
unsigned int num_output;

const unsigned int num_layers = 3;
const unsigned int num_neurons_hidden = 2;

const float  desired_error = 0.001f;
const unsigned int max_epochs = 500000;
const unsigned int epochs_between_reports = 1000;

struct fann* ann;

int main()
{
	getNumInOut();

	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, trainingFile, max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "output.net");

	testLanguage();

	fann_destroy(ann);

	return 0;
}

void getNumInOut() 
{
	std::ifstream in;
	in.open(trainingFile);

	if (!in)
		exit(1);

	std::string buffer;
	in >> buffer;

	in >> num_input;
	in >> num_output;
}

void testLanguage()
{
	ann = fann_create_from_file("output.net");

	fann_type *calc_out;
	fann_type input[5];
	input[0] = 12.2;
	input[1] = 9.84;
	input[2] = 8.66;
	input[3] = 6.3;
	input[4] = 8.66;

	calc_out = fann_run(ann, input);

	printf("%f %f %f\n", calc_out[0], calc_out[1], calc_out[2]);
}