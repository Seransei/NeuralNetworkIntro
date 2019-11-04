#pragma once
#include "include/fann.h"
#include <string>
#include <iostream>
#include <fstream>

void getNumInOut(const char*);

const char* filename = "langue.data";

unsigned int num_input;
unsigned int num_output;


const unsigned int num_layers = 3;
const unsigned int num_neurons_hidden = 2;

const float  desired_error = 0.001f;
const unsigned int max_epochs = 500000;
const unsigned int epochs_between_reports = 1000;

int main()
{
	getNumInOut(filename);

	struct fann* ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, filename, max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "output.net");

	fann_destroy(ann);

	return 0;
}

void getNumInOut(const char* filename) 
{
	std::ifstream in;
	in.open(filename);

	if (!in)
		exit(1);

	std::string buffer;
	in >> buffer;

	in >> num_input;
	in >> num_output;
}