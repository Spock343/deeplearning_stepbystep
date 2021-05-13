#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>

#ifndef OVERFLOW
#define OVERFLOW -2
#endif

const int INITDEPTH = 20;
const int ADDLENGTH = 10;
const int MAX = 256;
const double PI = 3.1415926;
int maxdistance = 0;
int positioncode = 0;

enum Status { TRUE = 1, FALSE = 0, OK = 1, ERROR = 0, INFEASIBLE = -1 };

typedef struct {
	double** W;
	int h;
	int w;
}weight;

typedef struct {
	double* b;
	int length;
}bias;

typedef struct {
	double**** K;
	int h;
	int w;
	int d;
	int n;
}kernel;

typedef struct {
	double*** caches;
	int h;
	int w;
	int d;
}tensor3d, at, ct, hid_variable, yt;

typedef struct {
	char name;
	weight W;
	kernel K;
	bias b;
	weight dW;
	kernel dK;
	bias db;
	int ksize;
	int stride;
	char pool;
	char activation;
}layers;

typedef struct {
	layers* layer;
	double***** caches;
	double***** dcaches;
	int** shape;
	int depth;
	int modelsize;
	char optimizer;
	char loss;
	double keep_drop;
	double lambda;
	double clip;
}model;

typedef struct {
	double**** data;
	int* dim;
	int totaldim;
	char name;
}dataset;

typedef struct {
	layers* layer;
}parameter_v, parameter_s, parameter;

// 对于dat，最后只有dat[0]要使用，并不需要将后面的dat[1 到 T_x - 1]的真实导数算出来，最终与da一起进行反向传播即可
typedef struct {
	int T_x;
	int n_x;
	int n_a;
	int n_y;
	int m;
	double*** a;       // [T_x + 1, n_a, m]
	double*** da;      // [T_x, n_a, m]
	double*** dat;     // [T_x + 1, n_a, m]
	double** Waa;      // [n_a, n_a]
	double** dWaa;     // [n_a, n_a]
	double** Wax;      // [n_a, n_x]
	double** dWax;     // [n_a, n_x]
	double* ba;        // [n_a,]
	double* dba;       // [n_a,]
	double** Wya;      // [n_y, n_a]
	double** dWya;     // [n_y, n_a]
	double* by;        // [n_y,]
	double* dby;       // [n_y,]
	double*** dx;      // [T_x, n_x, m]
	double*** outputs; // [T_x, n_y, m]
}BasicRnnCell;

typedef struct {
	int T_x;
	int n_x;
	int n_a;
	int n_y;
	int m;
	double*** a;       // [T_x + 1, n_a, m]
	double*** da;      // [T_x, n_a, m]
	double*** dat;     // [T_x + 1, n_a, m]
	double*** c;       // [T_x + 1, n_a, m]
	double*** dct;     // [T_x + 1, n_a, m]
	double** Wf;       // [n_a, n_a + n_x]
	double** dWf;      // [n_a, n_a + n_x]
	double* bf;        // [n_a,]
	double* dbf;       // [n_a,]
	double** Wi;       // [n_a, n_a + n_x]
	double** dWi;      // [n_a, n_a + n_x]
	double* bi;        // [n_a,]
	double* dbi;       // [n_a,]
	double** Wc;       // [n_a, n_a + n_x]
	double** dWc;      // [n_a, n_a + n_x]
	double* bc;        // [n_a,]
	double* dbc;       // [n_a,]
	double** Wo;       // [n_a, n_a + n_x]
	double** dWo;      // [n_a, n_a + n_x]
	double* bo;        // [n_a,]
	double* dbo;       // [n_a,]
	double** Wy;       // [n_y, n_a]
	double** dWy;      // [n_y, n_a]
	double* by;        // [n_y,]
	double* dby;       // [n_y,]
	double*** ft;      // [T_x, n_a, m]
	double*** it;      // [T_x, n_a, m]
	double*** cct;     // [T_x, n_a, m]
	double*** ot;      // [T_x, n_a, m]
	double*** dx;      // [n_T, n_x, m]
	double*** outputs; // [n_T, n_y, m]
}LSTMCell;

typedef struct {
	double** dWc;
	double** dWi;
	double** dWo;
	double** dWf;
	double** dWy;
	double* dbc;
	double* dbi;
	double* dbo;
	double* dbf;
	double* dby;
}LSTMparameter;

typedef struct {
	LSTMCell cell;
	double** a0;
	double clip;
	int issample;
	int train_input_start, train_input_end;
	int train_output_start, train_output_end;
	int test_input_start, test_input_end;
}single_rnn_layer;

Status Initmodel(model& model);
Status Addlayer(model& model, char name, int* dim, int ksize = 0, int step = 0, char pool = '0', char activation = '0', char init = 'x');
Status Reducelayer(model& model);
void modelsummary(model model);
Status xavierInit(double** W, int h, int w, char activation);
Status xavierInit(double**** kernel, int h, int w, int d, int n, char activation);
Status kaimingInit(double** W, int h, int w, double a = 0);
Status kaimingInit(double**** kernel, int h, int w, int d, int n, double a = 0);
weight Initweight(int* dim);
Status setweight(int* dim, weight& W, char activation, char init);
double* zerovector(int length);
bias Initbias(int length);
kernel Initkernel(int* dim);
Status setkernel(kernel& K, char activation, char init);
tensor3d Inittensor3d(int h, int w, int d);
Status Initparameter(model& model, parameter& parameter);
double** creatematrix(int h, int w);
Status zeromatrix(double** matrix, int h, int w);
double** copymatrix(double** matrix, int h, int w);
Status Freematrix(double** matrix, int h);
double**** createtensor(int h, int w, int d, int c);
double**** copytensor(double**** tensor, int h, int w, int d, int n);
Status Freetensor(double**** tensor, int h, int w, int d);
double*** createtensor(int h, int w, int d);
Status zerotensor(double*** tensor, int h, int w, int d);
double*** copytensor(double*** tensor, int h, int w, int d);
Status Freetensor(double*** tensor, int h, int w);
double*** datadimconver(double*** X, int dim1, int dim2, int dim3, int h, int w, int d);
double** Flatten(double**** tensor, int* dim);
double**** unfold(double** tensor, int* dim);
double** transpose(double** X, int h, int w);
double** matmul(double** X, double** W, int* dim);
double** matmultranspose(double** X, double** W, int* dim);
double** transposematmul(double** X, double** W, int* dim);
double** add(double** X, double* b, int* dim);
double** add_v2(double** X, double* b, int h, int w);
double** add_v3(double** A, double** B, int h, int w);
Status add(double** A, double** B, int h, int w);
Status sum(double* A, double** B, int h, int w, int axis);
double** multiply(double** A, double** B, int h, int w);
double** concat(double** A, int n_a, double** B, int n_b, int m);
double** slice(double** A, int start, int end, int h, int w, int axis);
double** relu(double** X, int* dim);
double**** relu(double**** X, int* dim);
double** sigmoid(double** X, int* dim);
double** tanh(double** X, int* dim);
double** softmax(double** X, int* dim);
double** softmax(double** X, int h, int m);
double** scoretoinference(double** score, int n_y, int m);
double**** sigmoid(double**** X, int* dim);
double**** tanh(double**** X, int* dim);
double**** zeropad(double**** X, int* dim, int pad);
double conv2dsinglestep(double**** X, kernel K, bias b, int m, int n_C, int horiz_start, int ver_start);
double averagepool(double**** X, int m, int n_C, int horiz_start, int ver_start, int ksize);
double maxpool(double**** X, int m, int n_C, int horiz_start, int ver_start, int ksize);
double** relubackward(double** dA, double** cache, int* dim);
double**** relubackward(double**** dA, double**** cache, int* dim);
double** sigmoidbackward(double** dA, double** cache, int* dim);
double** tanhbackward(double** dA, double** cache, int* dim);
double** softmaxbackward(double** dA, double** cache, int* dim);
double** softmaxbackward(double** dA, double** cache, int h, int w);
double**** sigmoidbackward(double**** dA, double**** cache, int* dim);
double** loaddata(int batch, int feature);
double**** loaddata(int batch, int h, int w, int d);
double** createYS(double** X, int batch, int feature, int classes);
double** createY(double**** X, int batch, int h, int w, int d, int classes);
double** createY(double** X, double** matrix, int* ndim);
double** createY(double** X, int batch, int feature, int classes);
double** loaddataY(int batch, int classes);
Status linearactivationforward(double** X, model& model, int depth, int* dim);
Status conv2dforward(double**** X, model& model, int depth, int* dim);
Status poolforward(double**** X, model& model, int depth, int* dim);
Status modelforward(double**** inputs, model& model, int batch, int h, int w, int d);
Status Clearcaches(model& model);
Status linearactivationbackward(model& model, int depth, int batch);
Status conv2dbackwardsinglestep_dA(double**** dA_pad, kernel K, double dZ, int horiz_start, int ver_start, int m, int n_C);
Status conv2dbackwardsinglestep_dK(kernel dK, double**** A_pad, double dZ, int horiz_start, int ver_start, int m, int n_C);
Status conv2dbackward(model& model, int depth);
Status maxpoolingbackward(double**** dA, double dZ, int m, int n_C, int horiz_start, int ver_start, int ksize, double**** A, double max);
Status averagepoolingbackward(double**** dA, double dZ, int m, int n_C, int horiz_start, int ver_start, int ksize);
Status poolbackward(model& model, int depth);
Status modelbackward(model& model, int batch, double**** X, double**** Y, int h, int w, int d);
Status Cleardcaches(model& model);
Status InitRnnCell(BasicRnnCell& cell, int m, int n_x, int n_a, int T_x, int n_y);
Status rnncellforward(BasicRnnCell& cell, double** xt, int t);
Status rnnforward(BasicRnnCell& cell, double*** X, double** a0, int n);
Status rnnforward_v2(BasicRnnCell& cell, double*** X, double** a0, int n);
Status ClearRnnCell_v1(BasicRnnCell& cell);
Status rnncellbackward(BasicRnnCell& cell, double** xt, int t);
Status rnnbackward(BasicRnnCell& cell, double*** X, int n);
Status rnnbackward_v2(BasicRnnCell& cell, double*** X, int n);
Status ClearRnnCell_v2(BasicRnnCell& cell);
Status InitLSTMCell(LSTMCell& cell, int m, int n_x, int n_a, int T_x, int n_y);
Status LSTMcellforward(LSTMCell& cell, double** xt, int t);
Status LSTMforward(LSTMCell& cell, double*** X, double** a0, int n);
Status LSTMforward_v2(LSTMCell& cell, double*** X, double** a0, int n);
Status ClearLSTMCell_v1(LSTMCell& cell);
Status LSTMcellbackward(LSTMCell& cell, double** xt, int t);
Status LSTMbackward(LSTMCell& cell, double*** X, int n);
Status LSTMbackward_v2(LSTMCell& cell, double*** X, int n);
Status ClearLSTMCell_v2(LSTMCell& cell);
Status InitLSTMparameter(LSTMparameter& p, LSTMCell cell);
Status Initrnnlayer(single_rnn_layer& layer, int m, int n_x, int n_a, int T_x, int n_y, int tris, int trie, int tros, int troe, int teis, int teie, int issample, double clip);
Status rnnlayerforward(single_rnn_layer& layer, double*** X, int istrain, int issample);
Status rnnlayerbackward(single_rnn_layer& layer, double*** X, double*** Y, int issample);
Status clip(single_rnn_layer& layer, double clip = 10000.0);
Status updaternnlayer(single_rnn_layer& layer, double*** X, double*** Y, int t, int issample, LSTMparameter v, LSTMparameter s, double lr);
Status rnnlayerfit(single_rnn_layer& layer, dataset X_train, dataset Y_train, dataset X_val, dataset Y_val,
	double lr, int epochs, int testbatch, int minibatch, int acc, int feq = 1, int earlystop = 10000, double decay = 0.0);
double crossentropy(double**** predict, double**** Y, int* size, int batch);
double crossentropy(double*** predict, double*** logits, int h, int w, int d, int batch);
double crossentropy_v2(double**** predict, double**** Y, int* size, int batch);
double squaredif(double**** predict, double**** Y, int* size, int batch);
double**** minibatchfortensor(double**** tensor, int* curdim, int batch, int start);
double**** minibatchformatrix(double**** matrix, int h, int w, int batch, int start);
double*** minibatch_v2(double*** X, int minibatch, int T_x, int n, int m, int start);
Status updatekernel(kernel K, kernel dK, double lr);
Status updateweight(weight W, weight dW, double lr);
Status updatebias(bias b, bias db, double lr);
Status updatekernel_v2(kernel K, kernel v, kernel s, double lr);
Status updateweight_v2(weight W, weight v, weight s, double lr);
Status updatebias_v2(bias b, bias v, bias s, double lr);
Status updateweight(double** W, double** v, double** s, double lr, int t, int h, int w);
Status updatebias(double* b, double* v, double* s, double lr, int t, int length);
Status clip(model& model);
Status updatewithgradient(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d);
Status updatewithmomentum(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d, parameter_v& v);
Status updatewithAdam(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d, parameter_v& v, parameter_s& s);
Status modelcompile(model& model, char optimizer, char loss, double keep_drop = 1.0, double lambda = 0.0, double clip = 10000.0);
Status modelfit(dataset X, dataset Y, model& model, double lr, int epochs, int minibatch, int acc, int feq, dataset X_t,
	dataset Y_t, int earlystop = 10000, double attenuation = 1.0);
double accuracy(double** Y, double** predict, int h, int w);
Status Initdata(dataset& dataset, int* dim, int totaldim, char name);
Status Addnoise(double**** data, int* dim);
Status setzeropointone(double**** data, int start, int end, int h, int w, int n_C);
Status square(double**** data, int start, int end, int h, int w, int n_C);
Status square_v2(double**** data, int start, int end, int h, int w, int n_C);
Status circular(double**** data, int start, int end, int h, int w, int n_C);
Status triangle(double**** data, int start, int end, int h, int w, int n_C);
Status triangle_v2(double**** data, int start, int end, int h, int w, int n_C);
Status parallelogram(double**** data, int start, int end, int h, int w, int n_C);
Status rectangle(double**** data, int start, int end, int h, int w, int n_C);
Status Initdata_v1(dataset& X, dataset& Y, int batch);
Status Initdata_v2(dataset& X, dataset& Y, int batch);
Status Initdata_v3(dataset& X, dataset& Y, int batch);
Status Initdata_v4(dataset& X, dataset& Y, int batch);
Status Addnoise(double*** X, int T_x, int n_x, int m);
Status setzeropointone(double*** X, int T_x, int n_x, int m);
Status sin_v1(double*** data, int T_x, int n_x, int start, int end);
Status sin_v2(double*** data, int T_x, int n_x, int start, int end);
Status cos_v1(double*** data, int T_x, int n_x, int start, int end);
Status cos_v2(double*** data, int T_x, int n_x, int start, int end);
Status triangular_wave(double*** data, int T_x, int n_x, int start, int end);
Status square_wave(double*** data, int T_x, int n_x, int start, int end);
Status sawtooch_wave(double*** data, int T_x, int n_x, int start, int end);
Status Initdata_v5(dataset& X, dataset& Y, int batch);
Status numseq(double*** data, int T_x, int n_x, int start, int end, int firstnum);
Status Initdata_v6(dataset& X, dataset& Y);
Status word2vec(char word, int& num);
Status vec2word(int num, char& word);
Status Initdata_v7(dataset& X, dataset& Y);
double* Ex(double** data, int* dim);
double* Ex(double**** data, int* dim);
double* Var(double** data, int* dim, double* Ex);
Status datanormalized(dataset& dataset);

double** creatematrix(int h, int w)
{
	double** result = (double**)malloc(h * sizeof(double*));
	if (result == NULL)exit(OVERFLOW);
	for (int i = 0; i < h; i++) {
		result[i] = (double*)malloc(w * sizeof(double));
		if (result[i] == NULL)exit(OVERFLOW);
	}
	return result;
}

Status zeromatrix(double** matrix, int h, int w)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			matrix[i][j] = 0;
		}
	}
	return OK;
}

double** copymatrix(double** matrix, int h, int w)
{
	double** copy = creatematrix(h, w);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			copy[i][j] = matrix[i][j];
		}
	}
	return copy;
}

Status Freematrix(double** matrix, int h)
{
	for (int i = 0; i < h; i++) {
		free(matrix[i]);
	}
	free(matrix);
	return OK;
}

double**** createtensor(int h, int w, int d, int c)
{
	double**** result;
	result = (double****)malloc(h * sizeof(double***));
	if (result == NULL)exit(OVERFLOW);
	for (int i = 0; i < h; i++) {
		result[i] = (double***)malloc(w * sizeof(double**));
		if (result[i] == NULL)exit(OVERFLOW);
		for (int j = 0; j < w; j++) {
			result[i][j] = (double**)malloc(d * sizeof(double*));
			if (result[i][j] == NULL)exit(OVERFLOW);
			for (int k = 0; k < d; k++) {
				result[i][j][k] = (double*)malloc(c * sizeof(double));
				if (result[i][j][k] == NULL)exit(OVERFLOW);
			}
		}
	}
	return result;
}

double**** copytensor(double**** tensor, int h, int w, int d, int n)
{
	double**** copy = createtensor(h, w, d, n);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				for (int c = 0; c < n; c++) {
					copy[i][j][k][c] = tensor[i][j][k][c];
				}
			}
		}
	}
	return copy;
}

Status Freetensor(double**** tensor, int h, int w, int d)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				free(tensor[i][j][k]);
			}
			free(tensor[i][j]);
		}
		free(tensor[i]);
	}
	free(tensor);
	return OK;
}

double*** createtensor(int h, int w, int d)
{
	double*** result = (double***)malloc(h * sizeof(double**));
	if (result == NULL)exit(OVERFLOW);
	for (int i = 0; i < h; i++) {
		result[i] = (double**)malloc(w * sizeof(double*));
		if (result[i] == NULL)exit(OVERFLOW);
		for (int j = 0; j < w; j++) {
			result[i][j] = (double*)malloc(d * sizeof(double));
			if (result[i][j] == NULL)exit(OVERFLOW);
		}
	}
	return result;
}

Status zerotensor(double*** tensor, int h, int w, int d)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				tensor[i][j][k] = 0;
			}
		}
	}
	return OK;
}

double*** copytensor(double*** tensor, int h, int w, int d)
{
	double*** copy = createtensor(h, w, d);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				copy[i][j][k] = tensor[i][j][k];
			}
		}
	}
	return copy;
}

Status Freetensor(double*** tensor, int h, int w)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			free(tensor[i][j]);
		}
		free(tensor[i]);
	}
	free(tensor);
	return OK;
}

// 维度位置转换，比如dim1 = 2，dim2 = 3，dim3 = 1，则维度从[h, w, d]变为[w, d, h]
double*** datadimconver(double*** X, int dim1, int dim2, int dim3, int h, int w, int d)
{
	int nh, nw, nd;
	switch (dim1) {
	case 1:nh = h; break;
	case 2:nh = w; break;
	case 3:nh = d; break;
	default:printf("the dim1 is worry"); exit(INFEASIBLE);
	}
	switch (dim2) {
	case 1:nw = h; break;
	case 2:nw = w; break;
	case 3:nw = d; break;
	default:printf("the dim2 is worry"); exit(INFEASIBLE);
	}
	switch (dim3) {
	case 1:nd = h; break;
	case 2:nd = w; break;
	case 3:nd = d; break;
	default:printf("the dim3 is worry"); exit(INFEASIBLE);
	}
	double*** result = createtensor(nh, nw, nd);
	if (dim1 == 2 && dim2 == 1 && dim3 == 3) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int k = 0; k < d; k++) {
					result[j][i][k] = X[i][j][k];
				}
			}
		}
	}
	else if (dim1 == 1 && dim2 == 3 && dim3 == 2) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int k = 0; k < d; k++) {
					result[i][k][j] = X[i][j][k];
				}
			}
		}
	}
	else if (dim1 == 2 && dim2 == 3 && dim3 == 1) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int k = 0; k < d; k++) {
					result[j][k][i] = X[i][j][k];
				}
			}
		}
	}
	else if (dim1 == 3 && dim2 == 1 && dim3 == 2) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int k = 0; k < d; k++) {
					result[k][i][j] = X[i][j][k];
				}
			}
		}
	}
	else if (dim1 == 3 && dim2 == 2 && dim3 == 1) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				for (int k = 0; k < d; k++) {
					result[k][j][i] = X[i][j][k];
				}
			}
		}
	}
	else {
		printf("please use copytensor");
		exit(INFEASIBLE);
	}
	return result;
}

Status Initmodel(model& model)
{
	model.layer = (layers*)malloc(INITDEPTH * sizeof(layers));
	if (model.layer == NULL)exit(OVERFLOW);
	model.caches = (double*****)malloc(INITDEPTH * sizeof(double****));
	if (model.caches == NULL)exit(OVERFLOW);
	model.dcaches = (double*****)malloc(INITDEPTH * sizeof(double****));
	if (model.dcaches == NULL)exit(OVERFLOW);
	model.shape = (int**)malloc(INITDEPTH * sizeof(int*));
	if (model.shape == NULL)exit(OVERFLOW);
	model.depth = 0;
	model.modelsize = INITDEPTH;
	return OK;
}

void modelsummary(model model)
{
	int depth = model.depth;
	char name;
	printf("The depth of model is %d\n", depth);
	for (int i = 0; i < depth; i++) {
		name = model.layer[i].name;
		printf("Layer %d: %c\n", i + 1, name);
		switch (name) {
		case 'f':printf("The shape of weight is [%d, %d], the activation is %c\n",
			model.layer[i].W.h, model.layer[i].W.w, model.layer[i].activation); break;
		case 'c':printf("The shape of kernel is [%d, %d, %d, %d], the stride is %d, the activation is %c\n",
			model.layer[i].K.h, model.layer[i].K.w, model.layer[i].K.d, model.layer[i].K.n, model.layer[i].stride, model.layer[i].activation); break;
		case 'p':printf("The ksize is %d, stride is %d, way of pooling is %c, activation is %c\n",
			model.layer[i].ksize, model.layer[i].stride, model.layer[i].pool, model.layer[i].activation); break;
		case 'l':printf("This is Flatten layer\n"); break;
		default:printf("your model is worry"); exit(INFEASIBLE);
		}
	}
	printf("\n");
}

Status xavierInit(double** W, int h, int w, char activation)
{
	double gain = 1.0;
	switch (activation) {
	case 'n':
	case 's':
	case 'x':gain = 1.0; break;
	case 't':gain = 5.0 / 3.0; break;
	case 'r':gain = sqrt(2.0); break;
	default:printf("your activation is worry"); exit(INFEASIBLE);
	}
	double initmax = (double)((6.0 / (double)(h + w)));
	initmax = gain * sqrt(initmax);
	double increase = 2 * initmax / ((double)(h * w));
	initmax = -initmax;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			W[i][j] = initmax;
			initmax += increase;
		}
	}
	return OK;
}

Status xavierInit(double**** kernel, int h, int w, int d, int n, char activation)
{
	double gain = 1.0;
	switch (activation) {
	case 'n':
	case 's':
	case 'x':gain = 1.0; break;
	case 't':gain = 5.0 / 3.0; break;
	case 'r':gain = sqrt(2.0); break;
	default:printf("your activation is worry"); exit(INFEASIBLE);
	}
	int fan_in = h * w * d;
	int fan_out = h * w * n;
	double initmax = (double)((6.0 / (double)(fan_in + fan_out)));
	initmax = gain * sqrt(initmax);
	double increase = 2 * initmax / ((double)(h * w * d * n));
	initmax = -initmax;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				for (int m = 0; m < n; m++) {
					kernel[i][j][k][m] = initmax;
					initmax += increase;
				}
			}
		}
	}
	return OK;
}

Status kaimingInit(double** W, int h, int w, double a)
{
	double initmax = 6.0 / ((1 + a * a) * h);
	initmax = sqrt(initmax);
	double increase = 2 * initmax / (h * w);
	initmax = -initmax;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			W[i][j] = initmax;
			initmax += increase;
		}
	}
	return OK;
}

Status kaimingInit(double**** kernel, int h, int w, int d, int n, double a)
{
	int fan_in = h * w * d;
	double initmax = 6.0 / ((1 + a * a) * fan_in);
	initmax = sqrt(initmax);
	double increase = 2 * initmax / (h * w * d * n);
	initmax = -initmax;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				for (int m = 0; m < n; m++) {
					kernel[i][j][k][m] = initmax;
					initmax += increase;
				}
			}
		}
	}
	return OK;
}

weight Initweight(int* dim)
{
	double** W = creatematrix(dim[0], dim[1]);
	weight w = { W, dim[0], dim[1] };
	return w;
}

Status setweight(int* dim, weight& W, char activation, char init)
{
	if (init == 'x') {
		xavierInit(W.W, dim[0], dim[1], activation);
	}
	else if (init == 'k') {
		kaimingInit(W.W, dim[0], dim[1]);
	}
	else {
		printf("the init is worry");
		exit(INFEASIBLE);
	}
	return OK;
}

double* zerovector(int length)
{
	double* b = (double*)malloc(length * sizeof(double));
	if (b == NULL)exit(OVERFLOW);
	for (int i = 0; i < length; i++) {
		b[i] = 0;
	}
	return b;
}

bias Initbias(int length)
{
	bias B;
	B.length = length;
	B.b = zerovector(length);
	return B;
}

kernel Initkernel(int* dim)
{
	kernel K;
	K.K = createtensor(dim[0], dim[1], dim[2], dim[3]);
	K.h = dim[0]; K.w = dim[1]; K.d = dim[2]; K.n = dim[3];
	return K;
}

Status setkernel(kernel& K, char activation, char init)
{
	if (init == 'x') {
		xavierInit(K.K, K.h, K.w, K.d, K.n, activation);
	}
	else if (init == 'k') {
		kaimingInit(K.K, K.h, K.w, K.d, K.n);
	}
	else {
		printf("the init is worry");
		exit(INFEASIBLE);
	}
	return OK;
}

tensor3d Inittensor3d(int h, int w, int d)
{
	tensor3d result;
	result.h = h; result.w = w; result.d = d;
	result.caches = createtensor(h, w, d);
	return result;
}

Status Initparameter(model& model, parameter& parameter)
{
	int length = model.depth;
	parameter.layer = (layers*)malloc(length * sizeof(layers));
	if (parameter.layer == NULL)exit(OVERFLOW);
	int kdim[4] = { 0, 0, 0, 0 };
	int wdim[2] = { 0, 0 };
	int j, k, n, m;
	for (int i = 0; i < length; i++) {
		switch (model.layer[i].name) {
		case 'f':
			wdim[0] = model.layer[i].dW.h;
			wdim[1] = model.layer[i].dW.w;
			parameter.layer[i].dW = Initweight(wdim);
			for (j = 0; j < wdim[0]; j++) {
				for (k = 0; k < wdim[1]; k++) {
					parameter.layer[i].dW.W[j][k] = 0;
				}
			}
			parameter.layer[i].db = Initbias(wdim[1]);
			break;
		case 'c':
			kdim[0] = model.layer[i].dK.h;
			kdim[1] = model.layer[i].dK.w;
			kdim[2] = model.layer[i].dK.d;
			kdim[3] = model.layer[i].dK.n;
			parameter.layer[i].dK = Initkernel(kdim);
			for (j = 0; j < kdim[0]; j++) {
				for (k = 0; k < kdim[1]; k++) {
					for (n = 0; n < kdim[2]; n++) {
						for (m = 0; m < kdim[3]; m++) {
							parameter.layer[i].dK.K[j][k][n][m] = 0;
						}
					}
				}
			}
			parameter.layer[i].db = Initbias(kdim[3]);
			break;
		case 'l':
			break;
		case 'p':
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

// 这里的dim是权重的dim或者conv2d的kernel的dim，权重的dim为[h, w], conv2d的dim为[h, w, d, n_C]
// 维度的大小自行计算，较复杂的公式如下
// n_H = (h - model.layer[depth].ksize + 2 * pad) / model.layer[depth].stride + 1
// n_W = (w - model.layer[depth].ksize + 2 * pad) / model.layer[depth].stride + 1
// n_H = (h - ksize) / stride + 1;
// n_W = (w - ksize) / stride + 1;
// ksize是conv2dlayer或者poollayer的kernel大小，step是步数
// 用f代表全连接层，用c代表卷积层，用p代表池化层，用l代表扁平化层
// 这里的pool用a代表平均池化，用m代表最大池化
// 用s代表sigmoid激活函数，用r代表relu激活函数，用t代表tanh激活函数，用n代表没有激活函数，(这里全连接层必须要有激活函数)
// 用x代表softmax计分函数
// 用A代表Adam优化器，用g代表gradient优化器，用m代表monmentum优化器
// 用c代表交叉熵代价函数，用s代表平方差代价函数
Status Addlayer(model& model, char name, int* dim, int ksize, int step, char pool, char activation, char init)
{
	if (model.modelsize <= model.depth) {
		model.layer = (layers*)realloc(model.layer, (model.modelsize + ADDLENGTH) * sizeof(layers));
		if (model.layer == NULL)exit(OVERFLOW);
		model.caches = (double*****)realloc(model.caches, (model.modelsize + ADDLENGTH) * sizeof(double****));
		if (model.caches == NULL)exit(OVERFLOW);
		model.dcaches = (double*****)realloc(model.dcaches, (model.modelsize + ADDLENGTH) * sizeof(double****));
		if (model.dcaches == NULL)exit(OVERFLOW);
		model.shape = (int**)realloc(model.shape, (model.modelsize + ADDLENGTH) * sizeof(int*));
		if (model.shape = NULL)exit(OVERFLOW);
	}
	model.shape[model.depth] = (int*)malloc(4 * sizeof(int));
	if (model.shape[model.depth] == NULL)exit(OVERFLOW);
	switch (name) {
	case 'f':
		model.caches[model.depth] = (double****)malloc(1 * sizeof(double***));
		if (model.caches[model.depth] == NULL)exit(OVERFLOW);
		model.caches[model.depth][0] = (double***)malloc(1 * sizeof(double**));
		if (model.caches[model.depth][0] == NULL)exit(OVERFLOW);
		model.dcaches[model.depth] = (double****)malloc(1 * sizeof(double***));
		if (model.dcaches[model.depth] == NULL)exit(OVERFLOW);
		model.dcaches[model.depth][0] = (double***)malloc(1 * sizeof(double**));
		if (model.dcaches[model.depth][0] == NULL)exit(OVERFLOW);
		model.layer[model.depth].name = 'f';
		model.layer[model.depth].W = Initweight(dim);
		setweight(dim, model.layer[model.depth].W, activation, init);
		model.layer[model.depth].dW = Initweight(dim);
		model.layer[model.depth].b = Initbias(dim[1]);
		model.layer[model.depth].db = Initbias(dim[1]);
		model.layer[model.depth].activation = activation;
		break;
	case 'c':
		model.layer[model.depth].name = 'c';
		model.layer[model.depth].K = Initkernel(dim);
		setkernel(model.layer[model.depth].K, activation, init);
		model.layer[model.depth].dK = Initkernel(dim);
		model.layer[model.depth].b = Initbias(dim[3]);
		model.layer[model.depth].db = Initbias(dim[3]);
		model.layer[model.depth].ksize = dim[0];
		model.layer[model.depth].stride = step;
		model.layer[model.depth].activation = activation;
		break;

	case 'p':
		model.layer[model.depth].name = 'p';
		model.layer[model.depth].ksize = ksize;
		model.layer[model.depth].stride = step;
		model.layer[model.depth].pool = pool;
		model.layer[model.depth].activation = activation;
		break;
	case 'l':
		model.caches[model.depth] = (double****)malloc(1 * sizeof(double***));
		if (model.caches[model.depth] == NULL)exit(OVERFLOW);
		model.caches[model.depth][0] = (double***)malloc(1 * sizeof(double**));
		if (model.caches[model.depth][0] == NULL)exit(OVERFLOW);
		model.dcaches[model.depth] = (double****)malloc(1 * sizeof(double***));
		if (model.dcaches[model.depth] == NULL)exit(OVERFLOW);
		model.dcaches[model.depth][0] = (double***)malloc(1 * sizeof(double**));
		if (model.dcaches[model.depth][0] == NULL)exit(OVERFLOW);
		model.layer[model.depth].name = 'l';
		break;
	default:printf("your layer is worry"); exit(INFEASIBLE);
	}
	model.depth++;
	return OK;
}

Status Reducelayer(model& model)
{
	if (model.depth <= 0) {
		printf("your model is empty");
		return ERROR;
	}
	model.depth--;
	switch (model.layer[model.depth].name) {
	case 'f':
		Freetensor(model.caches[model.depth], 1, 1, model.shape[model.depth][0]);
		Freematrix(model.layer[model.depth].W.W, model.layer[model.depth].W.h);
		Freematrix(model.layer[model.depth].dW.W, model.layer[model.depth].dW.h);
		free(model.layer[model.depth].b.b);
		free(model.layer[model.depth].db.b);
		break;
	case 'l':
		Freetensor(model.caches[model.depth], 1, 1, model.shape[model.depth][0]);
		break;
	case 'c':
		Freetensor(model.caches[model.depth], model.shape[model.depth][0], model.shape[model.depth][1], model.shape[model.depth][2]);
		Freetensor(model.layer[model.depth].K.K, model.layer[model.depth].K.h, model.layer[model.depth].K.w, model.layer[model.depth].K.d);
		Freetensor(model.layer[model.depth].dK.K, model.layer[model.depth].dK.h, model.layer[model.depth].dK.w, model.layer[model.depth].dK.d);
		free(model.layer[model.depth].b.b);
		free(model.layer[model.depth].db.b);
		break;
	case 'p':
		Freetensor(model.caches[model.depth], model.shape[model.depth][0], model.shape[model.depth][1], model.shape[model.depth][2]);
		break;
	}
	free(model.shape[model.depth]);
	return OK;
}

double** Flatten(double**** tensor, int* dim)
{
	double** output = (double**)malloc(dim[0] * sizeof(double*));
	if (output == NULL)exit(OVERFLOW);
	int length = dim[1] * dim[2] * dim[3];
	int temp = 0;
	for (int i = 0; i < dim[0]; i++) {
		output[i] = (double*)malloc(length * sizeof(double));
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int d = 0; d < dim[3]; d++) {
					temp = j * dim[2] * dim[3] + k * dim[3] + d;
					output[i][temp] = tensor[i][j][k][d];
				}
			}
		}
	}
	return output;
}

double**** unfold(double** output, int* dim)
{
	kernel tempk = Initkernel(dim);
	int temp = 0;
	double**** tensor = tempk.K;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					temp = j * dim[2] * dim[3] + k * dim[3] + n;
					tensor[i][j][k][n] = output[i][temp];
				}
			}
		}
	}
	return tensor;
}

double** transpose(double** X, int h, int w)
{
	double** result = creatematrix(w, h);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			result[j][i] = X[i][j];
		}
	}
	return result;
}

// [dim[0], dim[1]] * [dim[1], dim[2]]
double** matmul(double** X, double** W, int* dim)
{
	double** result = creatematrix(dim[0], dim[2]);
	double sum = 0;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[2]; j++) {
			sum = 0;
			for (int k = 0; k < dim[1]; k++) {
				sum += X[i][k] * W[k][j];
			}
			result[i][j] = sum;
		}
	}
	return result;
}

// [dim[0], dim[1]] * [dim[2], dim[1]].T
double** matmultranspose(double** X, double** W, int* dim)
{
	double** result = creatematrix(dim[0], dim[2]);
	double sum = 0;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[2]; j++) {
			sum = 0;
			for (int k = 0; k < dim[1]; k++) {
				sum += X[i][k] * W[j][k];
			}
			result[i][j] = sum;
		}
	}
	return result;
}

// [dim[1], dim[0]].T * [dim[1], dim[2]]
double** transposematmul(double** X, double** W, int* dim)
{
	double** result = creatematrix(dim[0], dim[2]);
	double sum = 0;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[2]; j++) {
			sum = 0;
			for (int k = 0; k < dim[1]; k++) {
				sum += X[k][i] * W[k][j];
			}
			result[i][j] = sum;
		}
	}
	return result;
}

// 第一维度是批次大小
double** add(double** X, double* b, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			result[i][j] = X[i][j] + b[j];
		}
	}
	return result;
}

// 第二维度是批次大小
double** add_v2(double** X, double* b, int h, int w)
{
	double** result = creatematrix(h, w);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			result[j][i] = X[j][i] + b[j];
		}
	}
	return result;
}

// 矩阵相加
double** add_v3(double** A, double** B, int h, int w)
{
	double** result = creatematrix(h, w);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			result[i][j] = A[i][j] + B[i][j];
		}
	}
	return result;
}

// 矩阵相加，储存到前面的矩阵上面
Status add(double** A, double** B, int h, int w)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			A[i][j] += B[i][j];
		}
	}
	return OK;
}

// 对矩阵的某一维度加和到向量中
// axis = 0时对第一维度加和，axis = 1时对第二维度加和
// 即axis = 0时，A的长度为w，axis = 1时，A的长度为h
Status sum(double* A, double** B, int h, int w, int axis)
{
	if (axis == 0) {
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				A[i] += B[j][i];
			}
		}
	}
	else if (axis == 1) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				A[i] += B[i][j];
			}
		}
	}
	else {
		printf("the axis is worry");
		exit(INFEASIBLE);
	}
	return OK;
}

double** multiply(double** A, double** B, int h, int w)
{
	double** result = creatematrix(h, w);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			result[i][j] = A[i][j] * B[i][j];
		}
	}
	return result;
}

double** concat(double** A, int n_a, double** B, int n_b, int m)
{
	int i;
	double** result = creatematrix(n_a + n_b, m);
	for (i = 0; i < n_a; i++) {
		for (int j = 0; j < m; j++) {
			result[i][j] = A[i][j];
		}
	}
	for (i = n_a; i < n_a + n_b; i++) {
		for (int j = 0; j < m; j++) {
			result[i][j] = B[i - n_a][j];
		}
	}
	return result;
}

// axis = 0时对矩阵的第一维度切片，axis = 1时对矩阵的第二维度切片
// 切片从start 到 end，包括start，不包括end，即[start, end)
double** slice(double** A, int start, int end, int h, int w, int axis)
{
	double** result;
	if (axis == 0) {
		result = creatematrix(end - start, w);
		for (int i = start; i < end; i++) {
			for (int j = 0; j < w; j++) {
				result[i - start][j] = A[i][j];
			}
		}
	}
	else if (axis == 1) {
		result = creatematrix(h, end - start);
		for (int i = 0; i < h; i++) {
			for (int j = start; j < end; j++) {
				result[i][j - start] = A[i][j];
			}
		}
	}
	else {
		printf("your axis is worry");
		exit(INFEASIBLE);
	}
	return result;
}

double** relu(double** X, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			result[i][j] = X[i][j] > 0.0 ? X[i][j] : 0;
		}
	}
	return result;
}

double**** relu(double**** X, int* dim)
{
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					result[i][j][k][n] = X[i][j][k][n] > 0 ? X[i][j][k][n] : 0;
				}
			}
		}
	}
	return result;
}

double** sigmoid(double** X, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			result[i][j] = 1 / (1 + exp(-X[i][j]));
		}
	}
	return result;
}

double** tanh(double** X, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	double s;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			s = exp(X[i][j]);
			result[i][j] = (s - 1 / s) / (s + 1 / s);
		}
	}
	return result;
}

// 第一维度是批次大小
double** softmax(double** X, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	double sum = 0;
	int i, j;
	for (i = 0; i < dim[0]; i++) {
		sum = 0;
		for (j = 0; j < dim[1]; j++) {
			sum += exp(X[i][j]);
		}
		for (j = 0; j < dim[1]; j++) {
			result[i][j] = exp(X[i][j]) / sum;
		}
	}
	return result;
}

//第二维度是批次大小
double** softmax(double** X, int h, int m)
{
	double** result = creatematrix(h, m);
	double sum = 0;
	int i, j;
	for (i = 0; i < m; i++) {
		sum = 0;
		for (j = 0; j < h; j++) {
			sum += exp(X[j][i]);
		}
		for (j = 0; j < h; j++) {
			result[j][i] = exp(X[j][i]) / sum;
		}
	}
	return result;
}

double** scoretoinference(double** score, int n_y, int m)
{
	double** inference = creatematrix(n_y, m);
	zeromatrix(inference, n_y, m);
	double max = 0;
	int index = 0;
	for (int i = 0; i < m; i++) {
		max = score[0][i];
		index = 0;
		for (int j = 1; j < n_y; j++) {
			if (score[j][i] > max) {
				index = j;
				max = score[j][i];
			}
		}
		inference[index][i] = 1;
	}
	return inference;
}

double**** sigmoid(double**** X, int* dim)
{
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					result[i][j][k][n] = 1 / (1 + exp(-X[i][j][k][n]));
				}
			}
		}
	}
	return result;
}

double**** tanh(double**** X, int* dim)
{
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	double s;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					s = exp(X[i][j][k][n]);
					result[i][j][k][n] = (s - 1 / s) / (s + 1 / s);
				}
			}
		}
	}
	return result;
}

double**** zeropad(double**** X, int* dim, int pad)
{
	double**** pads = createtensor(dim[0], dim[1] + 2 * pad, dim[2] + 2 * pad, dim[3]);
	int i, n, j, k;
	for (i = 0; i < dim[0]; i++) {
		for (n = 0; n < dim[3]; n++) {
			for (j = 0; j < dim[1]; j++) {
				for (k = 0; k < dim[1]; k++) {
					pads[i][j + pad][k + pad][n] = X[i][j][k][n];
				}
			}
			for (j = 0; j < pad; j++) {
				for (k = 0; k < dim[2] + 2 * pad; k++) {
					pads[i][j][k][n] = 0;
				}
			}
			for (j = dim[1] + pad; j < dim[1] + 2 * pad; j++) {
				for (k = 0; k < dim[2] + 2 * pad; k++) {
					pads[i][j][k][n] = 0;
				}
			}
			for (k = 0; k < pad; k++) {
				for (j = pad; j < dim[1] + pad; j++) {
					pads[i][j][k][n] = 0;
				}
			}
			for (k = dim[2] + pad; k < dim[2] + 2 * pad; k++) {
				for (j = pad; j < dim[1] + pad; j++) {
					pads[i][j][k][n] = 0;
				}
			}
		}
	}
	return pads;
}

//m指的是inputs的第几个样例，n_C指的是第n_C个通道，horiz_start和ver_start分别指的是从inputs的开始卷积的垂直位置和水平位置 
double conv2dsinglestep(double**** X, kernel K, bias b, int m, int n_C, int horiz_start, int ver_start)
{
	double result = 0;
	for (int i = 0, h_start = horiz_start; i < K.h; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < K.w; j++, w_start++) {
			for (int k = 0; k < K.d; k++) {
				result += K.K[i][j][k][n_C] * X[m][h_start][w_start][k];
			}
		}
	}
	result += b.b[n_C];
	return result;
}

double averagepool(double**** X, int m, int d, int horiz_start, int ver_start, int ksize)
{
	double avg = 0;
	for (int i = 0, h_start = horiz_start; i < ksize; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < ksize; j++, w_start++) {
			avg += X[m][h_start][w_start][d];
		}
	}
	avg = avg / (ksize * ksize);
	return avg;
}

double maxpool(double**** X, int m, int d, int horiz_start, int ver_start, int ksize)
{
	double max = X[m][horiz_start][ver_start][d];
	for (int i = 0, h_start = horiz_start; i < ksize; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < ksize; j++, w_start++) {
			max = (X[m][h_start][w_start][d] > max) ? X[m][h_start][w_start][d] : max;
		}
	}
	return max;
}

double** relubackward(double** dA, double** cache, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			result[i][j] = cache[i][j] <= 0 ? 0 : dA[i][j];
		}
	}
	return result;
}

double**** relubackward(double**** dA, double**** cache, int* dim)
{
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					result[i][j][k][n] = cache[i][j][k][n] <= 0 ? 0 : dA[i][j][k][n];
				}
			}
		}
	}
	return result;
}

double** sigmoidbackward(double** dA, double** cache, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	double s = 0;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			s = cache[i][j];
			result[i][j] = dA[i][j] * s * (1 - s);
		}
	}
	return result;
}

double** tanhbackward(double** dA, double** cache, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	double s = 0;
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			s = cache[i][j];
			result[i][j] = dA[i][j] * (1 - s * s);
		}
	}
	return result;
}

double** softmaxbackward(double** dA, double** cache, int* dim)
{
	double** result = creatematrix(dim[0], dim[1]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			result[i][j] = 0;
			for (int k = 0; k < dim[1]; k++) {
				if (j == k) {
					result[i][j] += dA[i][k] * cache[i][j] * (1 - cache[i][j]);
				}
				else {
					result[i][j] -= dA[i][k] * cache[i][j] * cache[i][k];
				}
			}
		}
	}
	return result;
}

double** softmaxbackward(double** dA, double** cache, int h, int w)
{
	double** result = creatematrix(h, w);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			result[j][i] = 0;
			for (int k = 0; k < h; k++) {
				if (j == k) {
					result[j][i] += dA[k][i] * cache[j][i] * (1 - cache[j][i]);
				}
				else {
					result[j][i] -= dA[k][i] * cache[j][i] * cache[k][i];
				}
			}
		}
	}
	return result;
}

double**** sigmoidbackward(double**** dA, double**** cache, int* dim)
{
	double s = 0;
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					s = cache[i][j][k][n];
					result[i][j][k][n] = dA[i][j][k][n] * s * (1 - s);
				}
			}
		}
	}
	return result;
}

double**** tanhbackward(double**** dA, double**** cache, int* dim)
{
	double s = 0;
	double**** result = createtensor(dim[0], dim[1], dim[2], dim[3]);
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					s = cache[i][j][k][n];
					result[i][j][k][n] = dA[i][j][k][n] * (1 - s * s);
				}
			}
		}
	}
	return result;
}

double** loaddata(int batch, int feature)
{
	srand((unsigned)time(NULL));
	double** result = creatematrix(batch, feature);
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < feature; j++) {
			result[i][j] = (double)(rand() % MAX) / MAX * 2 - 1;
		}
	}
	return result;
}

double**** loaddata(int batch, int h, int w, int d)
{
	srand((unsigned)time(NULL));
	double**** result = createtensor(batch, h, w, d);
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < w; k++) {
				for (int n = 0; n < d; n++) {
					result[i][j][k][n] = (double)(rand() % MAX) / MAX * 2 - 1;
				}
			}
		}
	}
	return result;
}

double** loaddataY(int batch, int classes)
{
	double** Y = creatematrix(batch, classes);
	srand((unsigned)time(NULL));
	int index = 0;
	for (int i = 0; i < batch; i++) {
		index = (int)(rand() % classes);
		for (int j = 0; j < classes; j++) {
			Y[i][j] = 0;
		}
		Y[i][index] = 1;
	}
	return Y;
}

//该数据集分类标签较差，导致贝叶斯风险很高
double** createYS(double** X, int batch, int feature, int classes)
{
	double** result = creatematrix(batch, classes);
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < classes; j++) {
			result[i][j] = 0;
		}
	}
	int count = 0;
	int temp = 0;
	for (i = 0; i < batch; i++) {
		count = 0;
		for (j = 0; j < feature; j++) {
			if (X[i][j] > 0)count++;
		}
		temp = (count * classes) / feature;
		if (temp == classes)temp--;
		result[i][temp] = 1;
	}
	return result;
}

double** createY(double**** X, int batch, int h, int w, int d, int classes)
{
	int i, c, j, k, n;
	double** result = creatematrix(batch, classes);
	for (i = 0; i < batch; i++) {
		for (j = 0; j < classes; j++) {
			result[i][j] = 0;
		}
	}
	double* count = (double*)malloc(classes * sizeof(classes));
	if (count == NULL)exit(OVERFLOW);
	double max;
	int index;
	int temp;
	for (i = 0; i < batch; i++) {
		for (c = 0; c < classes; c++) {
			count[c] = 0;
		}
		for (j = 0; j < h; j++) {
			for (k = 0; k < w; k++) {
				for (n = 0; n < d; n++) {
					temp = k / 10 + j / 4;
					count[temp] += X[i][j][k][n];
				}
			}
		}
		max = count[0];
		index = 0;
		for (c = 1; c < classes; c++) {
			if (count[c] > max) {
				max = count[c];
				index = c;
			}
		}
		result[i][index] = 1;
	}
	free(count);
	return result;
}

double** createY(double** X, double** matrix, int* ndim)
{
	double** Y = matmul(X, matrix, ndim);
	double max;
	int index;
	for (int i = 0; i < ndim[0]; i++) {
		max = Y[i][0];
		index = 0;
		Y[i][0] = 0;
		for (int j = 1; j < ndim[2]; j++) {
			if (Y[i][j] > max) {
				max = Y[i][j];
				index = j;
			}
			Y[i][j] = 0;
		}
		Y[i][index] = 1;
	}
	return Y;
}

double** createY(double** X, int batch, int feature, int classes)
{
	double** result = creatematrix(batch, classes);
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < classes; j++) {
			result[i][j] = 0;
		}
	}
	double sum = 0;
	int temp = 0;
	for (i = 0; i < batch; i++) {
		sum = 0;
		for (j = 0; j < feature; j++) {
			sum += X[i][j];
		}
		temp = (int)sum;
		temp %= 10;
		result[i][temp] = 1;
	}
	return result;
}

//传入的depth从0到depth-1
Status linearactivationforward(double** X, model& model, int depth, int* dim)
{
	char activation = model.layer[depth].activation;
	int ndim[2] = { dim[0], dim[2] };
	if (fabs(model.keep_drop - 1.0) >= 0.001) {
		double num;
		double** X_dropout = copymatrix(X, dim[0], dim[1]);
		srand((unsigned)time(NULL));
		for (int i = 0; i < dim[1]; i++) {
			num = (double)((double)(rand() % MAX) / MAX);
			if (num > model.keep_drop) {
				for (int j = 0; j < dim[0]; j++) {
					X_dropout[j][i] = 0;
				}
			}
			else {
				for (int j = 0; j < dim[0]; j++) {
					X_dropout[j][i] /= model.keep_drop;
				}
			}
		}
		model.caches[depth][0][0] = matmul(X_dropout, model.layer[depth].W.W, dim);
		Freematrix(X_dropout, dim[0]);
	}
	else {
		model.caches[depth][0][0] = matmul(X, model.layer[depth].W.W, dim);
	}
	double** ptrs = model.caches[depth][0][0];
	model.caches[depth][0][0] = add(model.caches[depth][0][0], model.layer[depth].b.b, ndim);
	Freematrix(ptrs, ndim[0]);
	double** ptrs2 = model.caches[depth][0][0];
	switch (activation) {
	case 'r':model.caches[depth][0][0] = relu(model.caches[depth][0][0], ndim); break;
	case 't':model.caches[depth][0][0] = tanh(model.caches[depth][0][0], ndim); break;
	case 'x':model.caches[depth][0][0] = softmax(model.caches[depth][0][0], ndim); break;
	case 's':model.caches[depth][0][0] = sigmoid(model.caches[depth][0][0], ndim); break;
	default:printf("your activation is worry!"); exit(INFEASIBLE);
	}
	Freematrix(ptrs2, ndim[0]);
	return OK;
}

//dim为X的维度 
Status conv2dforward(double**** X, model& model, int depth, int* dim)
{
	int m = dim[0];
	int h = dim[1];
	int w = dim[2];
	int d = dim[3];
	int pad = model.layer[depth].ksize / 2;
	int ksize = model.layer[depth].ksize;
	int stride = model.layer[depth].stride;
	int horiz_start;
	int ver_start;
	int n_H = (h - ksize + 2 * pad) / stride + 1;
	int n_W = (w - ksize + 2 * pad) / stride + 1;
	int n_C = model.layer[depth].K.n;
	model.caches[depth] = createtensor(m, n_H, n_W, n_C);
	model.shape[depth][0] = m;
	model.shape[depth][1] = n_H;
	model.shape[depth][2] = n_W;
	model.shape[depth][3] = n_C;
	double**** X_pad = zeropad(X, dim, pad);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n_H; j++) {
			for (int k = 0; k < n_W; k++) {
				for (int n = 0; n < n_C; n++) {
					horiz_start = j * stride;
					ver_start = k * stride;
					model.caches[depth][i][j][k][n] = conv2dsinglestep(X_pad, model.layer[depth].K, model.layer[depth].b, i, n, horiz_start, ver_start);
				}
			}
		}
	}
	Freetensor(X_pad, dim[0], dim[1] + 2 * pad, dim[2] + 2 * pad);
	double**** ptrs = model.caches[depth];
	switch (model.layer[depth].activation) {
	case 'r':
		model.caches[depth] = relu(model.caches[depth], model.shape[depth]);
		break;
	case 's':
		model.caches[depth] = sigmoid(model.caches[depth], model.shape[depth]);
		break;
	case 't':
		model.caches[depth] = tanh(model.caches[depth], model.shape[depth]);
		break;
	case 'n':
		break;
	default:
		printf("your activation is worry");
		exit(INFEASIBLE);
	}
	if (model.layer[depth].activation != 'n') {
		Freetensor(ptrs, model.shape[depth][0], model.shape[depth][1], model.shape[depth][2]);
	}
	return OK;
}

Status poolforward(double**** X, model& model, int depth, int* dim)
{
	int m = dim[0];
	int h = dim[1];
	int w = dim[2];
	int d = dim[3];
	int ksize = model.layer[depth].ksize;
	int stride = model.layer[depth].stride;
	int n_H = (h - ksize) / stride + 1;
	int n_W = (w - ksize) / stride + 1;
	double(*pooling)(double**** X, int m, int d, int horiz_start, int ver_start, int ksize);
	switch (model.layer[depth].pool) {
	case 'a':
		pooling = averagepool;
		break;
	case 'm':
		pooling = maxpool;
		break;
	default:
		printf("your way of pooling is worry");
		exit(INFEASIBLE);
	}
	model.caches[depth] = createtensor(m, n_H, n_W, d);
	model.shape[depth][0] = m;
	model.shape[depth][1] = n_H;
	model.shape[depth][2] = n_W;
	model.shape[depth][3] = d;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n_H; j++) {
			for (int k = 0; k < n_W; k++) {
				for (int n = 0; n < d; n++) {
					int horiz_start = j * stride;
					int ver_start = k * stride;
					model.caches[depth][i][j][k][n] = pooling(X, i, n, horiz_start, ver_start, ksize);
				}
			}
		}
	}
	double**** ptrs = model.caches[depth];
	switch (model.layer[depth].activation) {
	case 'r':
		model.caches[depth] = relu(model.caches[depth], model.shape[depth]);
		break;
	case 's':
		model.caches[depth] = sigmoid(model.caches[depth], model.shape[depth]);
		break;
	case 't':
		model.caches[depth] = tanh(model.caches[depth], model.shape[depth]);
		break;
	case 'n':
		break;
	default:
		printf("your activation is worry");
		exit(INFEASIBLE);
	}
	if (model.layer[depth].activation != 'n') {
		Freetensor(ptrs, model.shape[depth][0], model.shape[depth][1], model.shape[depth][2]);
	}
	return OK;
}

//如果输入层是四维，则维度为[batch, h, w, d]，如果是二维，则维度为[batch, model.layer[0].W.h]；并将数据储存于inputs[0][0]中
Status modelforward(double**** inputs, model& model, int batch, int h, int w, int d)
{
	if (model.depth <= 0)return ERROR;
	char name = model.layer[0].name;
	int dim[3] = { 0,0,0 };
	int ldim[4] = { 0,0,0,0 };
	switch (name) {
	case 'f':
		dim[0] = batch; dim[1] = model.layer[0].W.h; dim[2] = model.layer[0].W.w;
		linearactivationforward(inputs[0][0], model, 0, dim);
		model.shape[0][0] = batch; model.shape[0][1] = dim[2];
		break;
	case 'c':
		ldim[0] = batch; ldim[1] = h; ldim[2] = w; ldim[3] = d;
		conv2dforward(inputs, model, 0, ldim);
		break;
	case 'p':
		ldim[0] = batch; ldim[1] = h; ldim[2] = w; ldim[3] = d;
		poolforward(inputs, model, 0, ldim);
		break;
	case 'l':
		ldim[0] = batch; ldim[1] = h; ldim[2] = w; ldim[3] = d;
		model.caches[0][0][0] = Flatten(inputs, ldim);
		model.shape[0][0] = batch; model.shape[0][1] = h * w * d;
		break;
	default:
		printf("your model is worry!");
		exit(INFEASIBLE);
	}
	for (int i = 1; i < model.depth; i++) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			dim[0] = batch; dim[1] = model.layer[i].W.h; dim[2] = model.layer[i].W.w;
			linearactivationforward(model.caches[i - 1][0][0], model, i, dim);
			model.shape[i][0] = batch; model.shape[i][1] = dim[2];
			break;
		case 'c':
			conv2dforward(model.caches[i - 1], model, i, model.shape[i - 1]);
			break;
		case 'p':
			poolforward(model.caches[i - 1], model, i, model.shape[i - 1]);
			break;
		case 'l':
			model.caches[i][0][0] = Flatten(model.caches[i - 1], model.shape[i - 1]);
			model.shape[i][0] = batch; model.shape[i][1] = model.shape[i - 1][1] * model.shape[i - 1][2] * model.shape[i - 1][3];
			break;
		default:
			printf("your model is worry!");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status Clearcaches(model& model)
{
	for (int i = 0; i < model.depth; i++) {
		switch (model.layer[i].name) {
		case 'l':
		case 'f':
			Freematrix(model.caches[i][0][0], model.shape[i][0]);
			break;
		case 'c':
		case 'p':
			Freetensor(model.caches[i], model.shape[i][0], model.shape[i][1], model.shape[i][2]);
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

//传入的depth从1到model.depth-1(不可为0)
Status linearactivationbackward(model& model, int depth, int batch)
{
	if (depth <= 0) {
		printf("your depth is worry");
		return ERROR;
	}
	int i, j;
	int adim[2] = { batch, model.shape[depth][1] };
	char activation = model.layer[depth].activation;
	double** ptrs = model.dcaches[depth][0][0];
	switch (activation) {
	case 'r':
		model.dcaches[depth][0][0] = relubackward(model.dcaches[depth][0][0], model.caches[depth][0][0], adim);
		break;
	case 't':
		model.dcaches[depth][0][0] = tanhbackward(model.dcaches[depth][0][0], model.caches[depth][0][0], adim);
		break;
	case 'x':
		model.dcaches[depth][0][0] = softmaxbackward(model.dcaches[depth][0][0], model.caches[depth][0][0], adim);
		break;
	case 's':
		model.dcaches[depth][0][0] = sigmoidbackward(model.dcaches[depth][0][0], model.caches[depth][0][0], adim);
		break;
	default:
		printf("your activation is worry");
		exit(INFEASIBLE);
	}
	Freematrix(ptrs, batch);
	int dim[3] = { model.layer[depth].W.h, batch, model.layer[depth].W.w };
	double** ptrs2 = model.layer[depth].dW.W;
	model.layer[depth].dW.W = transposematmul(model.caches[depth - 1][0][0], model.dcaches[depth][0][0], dim);
	Freematrix(ptrs2, model.layer[depth].W.h);
	for (i = 0; i < dim[0]; i++) {
		for (j = 0; j < dim[2]; j++) {
			model.layer[depth].dW.W[i][j] /= batch;
		}
	}
	if (fabs(model.lambda) >= 0.001) {
		for (i = 0; i < dim[0]; i++) {
			for (j = 0; j < dim[2]; j++) {
				model.layer[depth].dW.W[i][j] += model.lambda * model.layer[depth].W.W[i][j];
			}
		}
	}
	double sum = 0;
	for (i = 0; i < dim[2]; i++) {
		sum = 0;
		for (j = 0; j < batch; j++) {
			sum += model.dcaches[depth][0][0][j][i];
		}
		sum /= batch;
		model.layer[depth].db.b[i] = sum;
	}
	dim[0] = batch; dim[1] = model.shape[depth][1]; dim[2] = model.shape[depth - 1][1];
	model.dcaches[depth - 1][0][0] = matmultranspose(model.dcaches[depth][0][0], model.layer[depth].W.W, dim);
	return OK;
}

Status conv2dbackwardsinglestep_dA(double**** dA_pad, kernel K, double dZ, int horiz_start, int ver_start, int m, int n_C)
{
	for (int i = 0, h_start = horiz_start; i < K.h; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < K.w; j++, w_start++) {
			for (int k = 0; k < K.d; k++) {
				dA_pad[m][h_start][w_start][k] += K.K[i][j][k][n_C] * dZ;
			}
		}
	}
	return OK;
}

Status conv2dbackwardsinglestep_dK(kernel dK, double**** A_pad, double dZ, int horiz_start, int ver_start, int m, int n_C)
{
	for (int i = 0, h_start = horiz_start; i < dK.h; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < dK.w; j++, w_start++) {
			for (int k = 0; k < dK.d; k++) {
				dK.K[i][j][k][n_C] += A_pad[m][h_start][w_start][k] * dZ;
			}
		}
	}
	return OK;
}

//传入的depth从1到depth-1
Status conv2dbackward(model& model, int depth)
{
	int i, j, k, n;
	if (depth <= 0) {
		printf("your depth is worry");
		return ERROR;
	}
	char activation = model.layer[depth].activation;
	double**** ptrs = model.dcaches[depth];
	switch (activation) {
	case 'r':
		model.dcaches[depth] = relubackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 's':
		model.dcaches[depth] = sigmoidbackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 't':
		model.dcaches[depth] = tanhbackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 'n':
		break;
	default:
		printf("your activation is worry");
		exit(INFEASIBLE);
	}
	if (activation != 'n') {
		Freetensor(ptrs, model.shape[depth][0], model.shape[depth][1], model.shape[depth][2]);
	}
	int horiz_start;
	int ver_start;
	int m = model.shape[depth][0];
	int n_H = model.shape[depth][1];
	int n_W = model.shape[depth][2];
	int n_C = model.shape[depth][3];
	int p_H = model.shape[depth - 1][1];
	int p_W = model.shape[depth - 1][2];
	int p_C = model.shape[depth - 1][3];
	model.dcaches[depth - 1] = createtensor(m, p_H, p_W, p_C);
	int kdim[4] = { model.layer[depth].K.h, model.layer[depth].K.w, model.layer[depth].K.d, model.layer[depth].K.n };
	model.layer[depth].dK = Initkernel(kdim);
	for (i = 0; i < kdim[0]; i++) {
		for (j = 0; j < kdim[1]; j++) {
			for (k = 0; k < kdim[2]; k++) {
				for (n = 0; n < kdim[3]; n++) {
					model.layer[depth].dK.K[i][j][k][n] = 0;
				}
			}
		}
	}
	model.layer[depth].db = Initbias(n_C);
	for (i = 0; i < kdim[3]; i++) {
		model.layer[depth].db.b[i] = 0;
	}
	int stride = model.layer[depth].stride;
	int ksize = model.layer[depth].K.h;
	int pad = ksize / 2;
	double**** A_pad = zeropad(model.caches[depth - 1], model.shape[depth - 1], pad);
	double**** dA_pad = createtensor(m, p_H + 2 * pad, p_W + 2 * pad, p_C);
	for (i = 0; i < m; i++) {
		for (j = 0; j < p_H + 2 * pad; j++) {
			for (k = 0; k < p_W + 2 * pad; k++) {
				for (n = 0; n < p_C; n++) {
					dA_pad[i][j][k][n] = 0;
				}
			}
		}
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n_H; j++) {
			for (k = 0; k < n_W; k++) {
				for (n = 0; n < n_C; n++) {
					horiz_start = j * stride;
					ver_start = k * stride;
					conv2dbackwardsinglestep_dA(dA_pad, model.layer[depth].K, model.dcaches[depth][i][j][k][n], horiz_start, ver_start, i, n);
					conv2dbackwardsinglestep_dK(model.layer[depth].dK, A_pad, model.dcaches[depth][i][j][k][n], horiz_start, ver_start, i, n);
					model.layer[depth].db.b[n] += model.dcaches[depth][i][j][k][n];
				}
			}
		}
	}
	Freetensor(A_pad, model.shape[depth - 1][0], model.shape[depth - 1][1] + 2 * pad, model.shape[depth - 1][2] + 2 * pad);
	for (i = 0; i < m; i++) {
		for (j = 0; j < p_H; j++) {
			for (k = 0; k < p_W; k++) {
				for (n = 0; n < p_C; n++) {
					model.dcaches[depth - 1][i][j][k][n] = dA_pad[i][j + pad][k + pad][n];
				}
			}
		}
	}
	Freetensor(dA_pad, m, p_H + 2 * pad, p_W + 2 * pad);
	return OK;
}

Status maxpoolingbackward(double**** dA, double dZ, int m, int n_C, int horiz_start, int ver_start, int ksize, double**** A, double max)
{
	for (int i = 0, h_start = horiz_start; i < ksize; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < ksize; j++, w_start++) {
			if (max == A[m][h_start][w_start][n_C]) {
				dA[m][h_start][w_start][n_C] += dZ;
			}
		}
	}
	return OK;
}

Status averagepoolingbackward(double**** dA, double dZ, int m, int n_C, int horiz_start, int ver_start, int ksize)
{
	double average = dZ / (ksize * ksize);
	for (int i = 0, h_start = horiz_start; i < ksize; i++, h_start++) {
		for (int j = 0, w_start = ver_start; j < ksize; j++, w_start++) {
			dA[m][h_start][w_start][n_C] += average;
		}
	}
	return OK;
}

//传入的depth从1到depth-1
Status poolbackward(model& model, int depth)
{
	int i, j, k, n;
	int m = model.shape[depth][0];
	int n_H = model.shape[depth][1];
	int n_W = model.shape[depth][2];
	int n_C = model.shape[depth][3];
	int p_H = model.shape[depth - 1][1];
	int p_W = model.shape[depth - 1][2];
	int stride = model.layer[depth].stride;
	int ksize = model.layer[depth].ksize;
	int ver_start, horiz_start;
	double Z, dZ;
	double**** ptrs = model.dcaches[depth];
	switch (model.layer[depth].activation) {
	case 'r':
		model.dcaches[depth] = relubackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 's':
		model.dcaches[depth] = sigmoidbackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 't':
		model.dcaches[depth] = tanhbackward(model.dcaches[depth], model.caches[depth], model.shape[depth]);
		break;
	case 'n':
		break;
	default:
		printf("your activation is worry");
		exit(INFEASIBLE);
	}
	if (model.layer[depth].activation != 'n') {
		Freetensor(ptrs, model.shape[depth][0], model.shape[depth][1], model.shape[depth][2]);
	}
	model.dcaches[depth - 1] = createtensor(m, p_H, p_W, n_C);
	for (i = 0; i < m; i++) {
		for (j = 0; j < p_H; j++) {
			for (k = 0; k < p_W; k++) {
				for (n = 0; n < n_C; n++) {
					model.dcaches[depth - 1][i][j][k][n] = 0;
				}
			}
		}
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n_H; j++) {
			for (k = 0; k < n_W; k++) {
				for (n = 0; n < n_C; n++) {
					horiz_start = j * stride;
					ver_start = k * stride;
					Z = model.caches[depth][i][j][k][n];
					dZ = model.dcaches[depth][i][j][k][n];
					switch (model.layer[depth].pool) {
					case 'm':
						maxpoolingbackward(model.dcaches[depth - 1], dZ, i, n, horiz_start, ver_start, ksize, model.caches[depth - 1], Z);
						break;
					case 'a':
						averagepoolingbackward(model.dcaches[depth - 1], dZ, i, n, horiz_start, ver_start, ksize);
						break;
					default:
						printf("your way of pooling is worry");
						exit(INFEASIBLE);
					}
				}
			}
		}
	}
	return OK;
}

Status modelbackward(model& model, int batch, double**** X, double**** Y, int h, int w, int d)
{
	if (model.depth <= 0) {
		printf("your model is worry");
		return ERROR;
	}
	char name = model.layer[model.depth - 1].name;
	int yh, yw, yd;
	switch (name) {
	case 'f':
		model.dcaches[model.depth - 1][0][0] = creatematrix(batch, model.shape[model.depth - 1][1]);
		if (model.loss == 'c') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < model.shape[model.depth - 1][1]; j++) {
					model.dcaches[model.depth - 1][0][0][i][j] = -Y[0][0][i][j] / model.caches[model.depth - 1][0][0][i][j];
					if (model.layer[model.depth - 1].activation == 's' || model.layer[model.depth - 1].W.w == 1) {
						model.dcaches[model.depth - 1][0][0][i][j] += (1 - Y[0][0][i][j]) / (1 - model.caches[model.depth - 1][0][0][i][j]);
					}
				}
			}
		}
		else if (model.loss == 's') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < model.shape[model.depth - 1][1]; j++) {
					model.dcaches[model.depth - 1][0][0][i][j] = model.caches[model.depth - 1][0][0][i][j] - Y[0][0][i][j];
				}
			}
		}
		else {
			printf("your model should compile first");
			return ERROR;
		}
		break;
	case 'c':
	case 'p':
		yh = model.shape[model.depth - 1][1];
		yw = model.shape[model.depth - 1][2];
		yd = model.shape[model.depth - 1][3];
		model.dcaches[model.depth - 1] = createtensor(batch, h, w, d);
		if (model.loss == 'c') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < h; j++) {
					for (int k = 0; k < w; k++) {
						for (int n = 0; n < d; n++) {
							model.dcaches[model.depth - 1][i][j][k][n] = -Y[i][j][k][n] / model.caches[model.depth - 1][i][j][k][n];
						}
					}
				}
			}
		}
		else if (model.loss == 's') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < h; j++) {
					for (int k = 0; k < w; k++) {
						for (int n = 0; n < d; n++) {
							model.dcaches[model.depth - 1][i][j][k][n] = model.caches[model.depth - 1][i][j][k][n] - Y[i][j][k][n];
						}
					}
				}
			}
		}
		else {
			printf("your model should compile first");
			return ERROR;
		}
		break;
	case 'l':
		if (model.depth <= 1) {
			return OK;
		}
		model.dcaches[model.depth - 1][0][0] = creatematrix(batch, model.shape[model.depth - 1][1]);
		if (model.loss == 'c') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < model.shape[model.depth - 1][1]; j++) {
					model.dcaches[model.depth - 1][0][0][i][j] = -Y[0][0][i][j] / model.caches[model.depth - 1][0][0][i][j];
				}
			}
		}
		else if (model.loss == 's') {
			for (int i = 0; i < batch; i++) {
				for (int j = 0; j < model.shape[model.depth - 1][1]; j++) {
					model.dcaches[model.depth - 1][0][0][i][j] = model.caches[model.depth - 1][0][0][i][j] - Y[0][0][i][j];
				}
			}
		}
		else {
			printf("your model should compile first");
			return ERROR;
		}
		model.dcaches[model.depth - 2] = unfold(model.dcaches[model.depth - 1][0][0], model.shape[model.depth - 2]);
		break;
	default:
		printf("your layer is worry");
		exit(INFEASIBLE);
	}
	char activation = model.layer[model.depth - 1].activation;
	int pad;
	int Xdim[4];
	int i, j, k, n;
	int dim[4] = { 0,0,0,0 };
	double**** A_pad;
	for (i = model.depth - 1; i >= 1; i--) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			linearactivationbackward(model, i, batch);
			break;
		case 'c':
			conv2dbackward(model, i);
			break;
		case 'p':
			poolbackward(model, i);
			break;
		case 'l':
			dim[0] = batch; dim[1] = model.shape[i - 1][1]; dim[2] = model.shape[i - 1][2]; dim[3] = model.shape[i - 1][3];
			model.dcaches[i - 1] = unfold(model.dcaches[i][0][0], dim);
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	name = model.layer[0].name;
	activation = model.layer[0].activation;
	double sum = 0;
	double**** ptrs;
	double** ptrs2;
	switch (name) {
	case 'f':
		ptrs2 = model.dcaches[0][0][0];
		dim[0] = batch; dim[1] = model.shape[0][1];
		switch (activation) {
		case 'r':
			model.dcaches[0][0][0] = relubackward(model.dcaches[0][0][0], X[0][0], dim);
			break;
		case 't':
			model.dcaches[0][0][0] = tanhbackward(model.dcaches[0][0][0], X[0][0], dim);
			break;
		case 'x':
			model.dcaches[0][0][0] = softmaxbackward(model.dcaches[0][0][0], X[0][0], dim);
			break;
		case 's':
			model.dcaches[0][0][0] = sigmoidbackward(model.dcaches[0][0][0], X[0][0], dim);
			break;
		default:
			printf("your activation is worry");
			exit(INFEASIBLE);
		}
		Freematrix(ptrs2, dim[0]);
		dim[0] = model.layer[0].W.h; dim[1] = batch; dim[2] = model.shape[0][1];
		ptrs2 = model.layer[0].dW.W;
		model.layer[0].dW.W = transposematmul(X[0][0], model.dcaches[0][0][0], dim);
		Freematrix(ptrs2, model.layer[0].dW.h);
		for (i = 0; i < dim[0]; i++) {
			for (j = 0; j < dim[2]; j++) {
				model.layer[0].dW.W[i][j] /= batch;
			}
		}
		if (fabs(model.lambda) >= 0.001) {
			for (i = 0; i < dim[0]; i++) {
				for (j = 0; j < dim[2]; j++) {
					model.layer[0].dW.W[i][j] += model.lambda * model.layer[0].W.W[i][j];
				}
			}
		}
		for (i = 0; i < dim[2]; i++) {
			sum = 0;
			for (j = 0; j < batch; j++) {
				sum += model.dcaches[0][0][0][j][i];
			}
			sum /= batch;
			model.layer[0].db.b[i] = sum;
		}
		break;
	case 'c':
		ptrs = model.dcaches[0];
		switch (activation) {
		case 'r':
			model.dcaches[0] = relubackward(model.dcaches[0], model.caches[0], model.shape[0]);
			break;
		case 't':
			model.dcaches[0] = tanhbackward(model.dcaches[0], model.caches[0], model.shape[0]);
			break;
		case 's':
			model.dcaches[0] = sigmoidbackward(model.dcaches[0], model.caches[0], model.shape[0]);
			break;
		case 'n':
			break;
		default:
			printf("your activation is worry");
			exit(INFEASIBLE);
		}
		if (activation != 'n') {
			Freetensor(ptrs, model.shape[0][0], model.shape[0][1], model.shape[0][2]);
		}
		Xdim[0] = batch; Xdim[1] = h; Xdim[2] = w; Xdim[3] = d;
		pad = model.layer[0].ksize / 2;
		A_pad = zeropad(X, Xdim, pad);
		for (i = 0; i < model.layer[0].dK.h; i++) {
			for (j = 0; j < model.layer[0].dK.w; j++) {
				for (k = 0; k < model.layer[0].dK.d; k++) {
					for (n = 0; n < model.layer[0].dK.n; n++) {
						model.layer[0].dK.K[i][j][k][n] = 0;
					}
				}
			}
		}
		for (i = 0; i < model.layer[0].db.length; i++) {
			model.layer[0].db.b[i] = 0;
		}
		for (i = 0; i < batch; i++) {
			for (j = 0; j < model.shape[0][1]; j++) {
				for (k = 0; k < model.shape[0][2]; k++) {
					for (n = 0; n < model.shape[0][3]; n++) {
						int horiz_start = j * model.layer[0].stride;
						int ver_start = k * model.layer[0].stride;
						conv2dbackwardsinglestep_dK(model.layer[0].dK, A_pad, model.dcaches[0][i][j][k][n], horiz_start, ver_start, i, n);
						model.layer[0].db.b[n] += model.dcaches[0][i][j][k][n];
					}
				}
			}
		}
		Freetensor(A_pad, Xdim[0], Xdim[1], Xdim[2]);
		break;
	case 'p':
		break;
	case 'l':
		break;
	default:
		printf("your layer is worry");
		exit(INFEASIBLE);
	}
	return OK;
}

Status Cleardcaches(model& model)
{
	for (int i = 0; i < model.depth; i++) {
		switch (model.layer[i].name) {
		case 'l':
		case 'f':
			Freematrix(model.dcaches[i][0][0], model.shape[i][0]);
			break;
		case 'c':
		case 'p':
			Freetensor(model.dcaches[i], model.shape[i][0], model.shape[i][1], model.shape[i][2]);
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status InitRnnCell(BasicRnnCell& cell, int m, int n_x, int n_a, int T_x, int n_y)
{
	cell.m = m;
	cell.n_x = n_x;
	cell.n_a = n_a;
	cell.T_x = T_x;
	cell.n_y = n_y;
	cell.a = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.a == NULL)exit(OVERFLOW);
	cell.da = (double***)malloc(T_x * sizeof(double**));
	if (cell.da == NULL)exit(OVERFLOW);
	cell.dat = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.dat == NULL)exit(OVERFLOW);
	cell.dx = (double***)malloc(T_x * sizeof(double**));
	if (cell.dx == NULL)exit(OVERFLOW);
	cell.outputs = (double***)malloc(T_x * sizeof(double**));
	if (cell.outputs == NULL)exit(OVERFLOW);
	cell.Waa = creatematrix(n_a, n_a);
	xavierInit(cell.Waa, n_a, n_a, 't');
	cell.Wax = creatematrix(n_a, n_x);
	xavierInit(cell.Wax, n_a, n_x, 't');
	cell.Wya = creatematrix(n_y, n_a);
	xavierInit(cell.Wya, n_y, n_a, 'n');
	cell.ba = zerovector(n_a);
	cell.by = zerovector(n_y);
	return OK;
}

// t从0到T_x-1取值，a从t迭代到t+1
Status rnncellforward(BasicRnnCell& cell, double** xt, int t)
{
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	int dim[3] = { n_a, n_x, m };
	int ndim[3] = { n_a, n_a, m };
	int adim[2] = { n_a, m };
	double** ptrs1 = matmul(cell.Wax, xt, dim);
	double** ptrs2 = matmul(cell.Waa, cell.a[t], ndim);
	double** ptrs3 = add_v3(ptrs1, ptrs2, n_a, m);
	double** ptrs4 = add_v2(ptrs3, cell.ba, n_a, m);
	cell.a[t + 1] = tanh(ptrs4, adim);
	ndim[0] = n_y;
	double** ptrs5 = matmul(cell.Wya, cell.a[t + 1], ndim);
	double** ptrs6 = add_v2(ptrs5, cell.by, n_y, m);
	cell.outputs[t] = softmax(ptrs6, n_y, m);
	Freematrix(ptrs1, n_a);
	Freematrix(ptrs2, n_a);
	Freematrix(ptrs3, n_a);
	Freematrix(ptrs4, n_a);
	Freematrix(ptrs5, n_y);
	Freematrix(ptrs6, n_y);
	return OK;
}

Status rnnforward(BasicRnnCell& cell, double*** X, double** a0, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_y = cell.n_y;
	int m = cell.m;
	cell.a[0] = copymatrix(a0, cell.n_a, cell.m);
	for (i = 0; i < n; i++) {
		rnncellforward(cell, X[i], i);
	}
	double*** temp = (double***)malloc(T_x * sizeof(double**));
	for (i = n; i < T_x; i++) {
		temp[i] = scoretoinference(cell.outputs[i - 1], n_y, m);
		rnncellforward(cell, temp[i], i);
	}
	for (i = n; i < T_x; i++) {
		Freematrix(temp[i], n_y);
	}
	free(temp);
	return OK;
}

Status rnnforward_v2(BasicRnnCell& cell, double*** X, double** a0, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	cell.a[0] = copymatrix(a0, cell.n_a, cell.m);
	for (i = 0; i < n; i++) {
		rnncellforward(cell, X[i], i);
	}
	double** temp = creatematrix(n_x, m);
	zeromatrix(temp, n_x, m);
	for (i = n; i < T_x; i++) {
		rnncellforward(cell, temp, i);
	}
	Freematrix(temp, n_x);
	return OK;
}

Status ClearRnnCell_v1(BasicRnnCell& cell)
{
	int i;
	for (i = 0; i <= cell.T_x; i++) {
		Freematrix(cell.a[i], cell.n_a);
	}
	for (i = 0; i < cell.T_x; i++) {
		Freematrix(cell.outputs[i], cell.n_y);
	}
	return OK;
}

// t从0到T_x-1取值，dat从t+1迭代到t，其中dat[t]从da[t]+dat[t+1]中迭代
// da值不变，因为并不关心它的取值
Status rnncellbackward(BasicRnnCell& cell, double** xt, int t)
{
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int m = cell.m;
	int dim[2] = { n_a, m };
	int tdim[3] = { n_x, n_a, m };
	int wdim[3] = { n_a, m, n_x };
	double** da_correct = add_v3(cell.dat[t + 1], cell.da[t], n_a, m);
	double** dtanh = tanhbackward(da_correct, cell.a[t + 1], dim);
	cell.dx[t] = transposematmul(cell.Wax, dtanh, tdim);
	double** dWax = matmultranspose(dtanh, xt, wdim);
	add(cell.dWax, dWax, n_a, n_x);
	tdim[0] = n_a;
	cell.dat[t] = transposematmul(cell.Waa, dtanh, tdim);
	wdim[2] = n_a;
	double** dWaa = matmultranspose(dtanh, cell.a[t], wdim);
	add(cell.dWaa, dWaa, n_a, n_a);
	sum(cell.dba, dtanh, n_a, m, 1);
	Freematrix(da_correct, n_a);
	Freematrix(dtanh, n_a);
	Freematrix(dWax, n_a);
	Freematrix(dWaa, n_a);
	return OK;
}

// 默认从y方向传过来的da已经算出来了
// dWya与dby为全连接层计算的梯度，在这里不计算这两个值
Status rnnbackward(BasicRnnCell& cell, double*** X, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	double*** temp = (double***)malloc(T_x * sizeof(double**));
	cell.dWax = creatematrix(n_a, n_x);
	zeromatrix(cell.dWax, n_a, n_x);
	cell.dWaa = creatematrix(n_a, n_a);
	zeromatrix(cell.dWaa, n_a, n_a);
	cell.dba = zerovector(n_a);
	cell.dat[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dat[T_x], n_a, m);
	for (i = T_x - 1; i >= n; i--) {
		temp[i] = scoretoinference(cell.outputs[i - 1], n_y, m);
		rnncellbackward(cell, temp[i], i);
	}
	for (i = T_x - 1; i >= n; i--) {
		Freematrix(temp[i], n_y);
	}
	for (i = n - 1; i >= 0; i--) {
		rnncellbackward(cell, X[i], i);
	}
	free(temp);
	return OK;
}

Status rnnbackward_v2(BasicRnnCell& cell, double*** X, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	double** temp = creatematrix(n_x, m);
	zeromatrix(temp, n_x, m);
	cell.dWax = creatematrix(n_a, n_x);
	zeromatrix(cell.dWax, n_a, n_x);
	cell.dWaa = creatematrix(n_a, n_a);
	zeromatrix(cell.dWaa, n_a, n_a);
	cell.dba = zerovector(n_a);
	cell.dat[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dat[T_x], n_a, m);
	for (i = T_x - 1; i >= n; i--) {
		rnncellbackward(cell, temp, i);
	}
	for (i = n - 1; i >= 0; i--) {
		rnncellbackward(cell, X[i], i);
	}
	Freematrix(temp, n_x);
	return OK;
}

Status ClearRnnCell_v2(BasicRnnCell& cell)
{
	int i;
	for (i = 0; i <= cell.T_x; i++) {
		Freematrix(cell.dat[i], cell.n_a);
	}
	for (i = 0; i < cell.T_x; i++) {
		Freematrix(cell.da[i], cell.n_a);
	}
	for (i = 0; i < cell.T_x; i++) {
		Freematrix(cell.dx[i], cell.n_x);
	}
	return OK;
}

Status InitLSTMCell(LSTMCell& cell, int m, int n_x, int n_a, int T_x, int n_y)
{
	cell.m = m;
	cell.n_x = n_x;
	cell.n_a = n_a;
	cell.T_x = T_x;
	cell.n_y = n_y;
	cell.a = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.a == NULL)exit(OVERFLOW);
	cell.da = (double***)malloc(T_x * sizeof(double**));
	if (cell.da == NULL)exit(OVERFLOW);
	cell.dat = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.dat == NULL)exit(OVERFLOW);
	cell.c = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.c == NULL)exit(OVERFLOW);
	cell.dct = (double***)malloc((1 + T_x) * sizeof(double**));
	if (cell.dct == NULL)exit(OVERFLOW);
	cell.ft = (double***)malloc(T_x * sizeof(double**));
	if (cell.ft == NULL)exit(OVERFLOW);
	cell.it = (double***)malloc(T_x * sizeof(double**));
	if (cell.it == NULL)exit(OVERFLOW);
	cell.cct = (double***)malloc(T_x * sizeof(double**));
	if (cell.cct == NULL)exit(OVERFLOW);
	cell.ot = (double***)malloc(T_x * sizeof(double**));
	if (cell.ot == NULL)exit(OVERFLOW);
	cell.dx = (double***)malloc(T_x * sizeof(double**));
	if (cell.dx == NULL)exit(OVERFLOW);
	cell.outputs = (double***)malloc(T_x * sizeof(double**));
	if (cell.outputs == NULL)exit(OVERFLOW);
	cell.Wf = creatematrix(n_a, n_a + n_x);
	xavierInit(cell.Wf, n_a, n_a + n_x, 's');
	cell.Wi = creatematrix(n_a, n_a + n_x);
	xavierInit(cell.Wi, n_a, n_a + n_x, 's');
	cell.Wc = creatematrix(n_a, n_a + n_x);
	xavierInit(cell.Wc, n_a, n_a + n_x, 't');
	cell.Wo = creatematrix(n_a, n_a + n_x);
	xavierInit(cell.Wo, n_a, n_a + n_x, 's');
	cell.Wy = creatematrix(n_y, n_a);
	xavierInit(cell.Wy, n_y, n_a, 'x');
	cell.bf = zerovector(n_a);
	cell.bi = zerovector(n_a);
	cell.bc = zerovector(n_a);
	cell.bo = zerovector(n_a);
	cell.by = zerovector(n_y);
	return OK;
}

// 传入的t从0到T_x-1
Status LSTMcellforward(LSTMCell& cell, double** xt, int t)
{
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	double** concats = concat(cell.a[t], n_a, xt, n_x, m);
	int dim[3] = { n_a, n_a + n_x, m };
	int ndim[3] = { n_y, n_a, m };
	int adim[2] = { n_a, m };
	double** ptrs1 = matmul(cell.Wf, concats, dim);
	double** ptrs2 = add_v2(ptrs1, cell.bf, n_a, m);
	cell.ft[t] = sigmoid(ptrs2, adim);
	double** ptrs3 = matmul(cell.Wi, concats, dim);
	double** ptrs4 = add_v2(ptrs3, cell.bi, n_a, m);
	cell.it[t] = sigmoid(ptrs4, adim);
	double** ptrs5 = matmul(cell.Wc, concats, dim);
	double** ptrs6 = add_v2(ptrs5, cell.bc, n_a, m);
	cell.cct[t] = tanh(ptrs6, adim);
	double** ptrs7 = multiply(cell.ft[t], cell.c[t], n_a, m);
	double** ptrs8 = multiply(cell.it[t], cell.cct[t], n_a, m);
	cell.c[t + 1] = add_v3(ptrs7, ptrs8, n_a, m);
	double** ptrs9 = matmul(cell.Wo, concats, dim);
	double** ptrs10 = add_v2(ptrs9, cell.bo, n_a, m);
	cell.ot[t] = sigmoid(ptrs10, adim);
	double** ptrs11 = tanh(cell.c[t + 1], adim);
	cell.a[t + 1] = multiply(cell.ot[t], ptrs11, n_a, m);
	double** ptrs12 = matmul(cell.Wy, cell.a[t + 1], ndim);
	double** ptrs13 = add_v2(ptrs12, cell.by, n_y, m);
	cell.outputs[t] = softmax(ptrs13, n_y, m);
	Freematrix(ptrs1, n_a);
	Freematrix(ptrs2, n_a);
	Freematrix(ptrs3, n_a);
	Freematrix(ptrs4, n_a);
	Freematrix(ptrs5, n_a);
	Freematrix(ptrs6, n_a);
	Freematrix(ptrs7, n_a);
	Freematrix(ptrs8, n_a);
	Freematrix(ptrs9, n_a);
	Freematrix(ptrs10, n_a);
	Freematrix(ptrs11, n_a);
	Freematrix(ptrs12, n_y);
	Freematrix(ptrs13, n_y);
	return OK;
}

// Sampling
Status LSTMforward(LSTMCell& cell, double*** X, double** a0, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_y = cell.n_y;
	int m = cell.m;
	double*** temp = (double***)malloc(T_x * sizeof(double**));
	cell.a[0] = copymatrix(a0, n_a, m);
	cell.c[0] = creatematrix(n_a, m);
	zeromatrix(cell.c[0], n_a, m);
	for (i = 0; i < n; i++) {
		LSTMcellforward(cell, X[i], i);
	}
	for (i = n; i < T_x; i++) {
		temp[i] = scoretoinference(cell.outputs[i - 1], n_y, m);
		LSTMcellforward(cell, temp[i], i);
	}
	for (i = n; i < T_x; i++) {
		Freematrix(temp[i], n_y);
	}
	free(temp);
	return OK;
}

Status LSTMforward_v2(LSTMCell& cell, double*** X, double** a0, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_x = cell.n_x;
	int n_a = cell.n_a;
	int n_y = cell.n_y;
	int m = cell.m;
	double** temp = creatematrix(n_x, m);
	zeromatrix(temp, n_x, m);
	cell.a[0] = copymatrix(a0, n_a, m);
	cell.c[0] = creatematrix(n_a, m);
	zeromatrix(cell.c[0], n_a, m);
	for (i = 0; i < n; i++) {
		LSTMcellforward(cell, X[i], i);
	}
	for (i = n; i < T_x; i++) {
		LSTMcellforward(cell, temp, i);
	}
	Freematrix(temp, n_x);
	return OK;
}

Status ClearLSTMCell_v1(LSTMCell& cell)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_y = cell.n_y;
	for (i = 0; i < T_x; i++) {
		Freematrix(cell.ft[i], n_a);
		Freematrix(cell.it[i], n_a);
		Freematrix(cell.cct[i], n_a);
		Freematrix(cell.ot[i], n_a);
		Freematrix(cell.outputs[i], n_y);
	}
	for (i = 0; i <= T_x; i++) {
		Freematrix(cell.a[i], n_a);
		Freematrix(cell.c[i], n_a);
	}
	return OK;
}

// t从0到T_x-1取值，dat从t+1迭代到t，其中dat[t]从da[t]+dat[t+1]中迭代，dct从t+1迭代到t
// da值不变，因为并不关心它的取值
Status LSTMcellbackward(LSTMCell& cell, double** xt, int t)
{
	int i, j;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int m = cell.m;
	int adim[2] = { n_a, m };
	int dim[3] = { n_a, m, n_a + n_x };
	int ndim[3] = { n_a, n_a, m };
	int xdim[3] = { n_x, n_a, m };
	double** da_correct = add_v3(cell.da[t], cell.dat[t + 1], n_a, m);
	double** ptrs1 = sigmoidbackward(da_correct, cell.ot[t], adim);
	double** ptrs2 = tanh(cell.c[t + 1], adim);
	double** dot = multiply(ptrs1, ptrs2, n_a, m);
	double** ptrs3 = tanh(cell.c[t + 1], adim);
	double** ptrs4 = tanhbackward(da_correct, ptrs3, adim);
	double** ptrs5 = multiply(ptrs4, cell.ot[t], n_a, m);
	double** ptrs6 = add_v3(ptrs5, cell.dct[t + 1], n_a, m);
	double** ptrs7 = tanhbackward(cell.it[t], cell.cct[t], adim);
	double** dcct = multiply(ptrs6, ptrs7, n_a, m);
	double** ptrs8 = sigmoidbackward(cell.cct[t], cell.it[t], adim);
	double** dit = multiply(ptrs6, ptrs8, n_a, m);
	double** ptrs9 = sigmoidbackward(cell.c[t], cell.ft[t], adim);
	double** dft = multiply(ptrs6, ptrs9, n_a, m);
	double** concats = concat(cell.a[t], n_a, xt, n_x, m);
	double** dWft = matmultranspose(dft, concats, dim);
	double** dWit = matmultranspose(dit, concats, dim);
	double** dWct = matmultranspose(dcct, concats, dim);
	double** dWot = matmultranspose(dot, concats, dim);
	add(cell.dWf, dWft, n_a, n_a + n_x);
	add(cell.dWi, dWit, n_a, n_a + n_x);
	add(cell.dWc, dWct, n_a, n_a + n_x);
	add(cell.dWo, dWot, n_a, n_a + n_x);
	sum(cell.dbf, dft, n_a, m, 1);
	sum(cell.dbi, dit, n_a, m, 1);
	sum(cell.dbc, dcct, n_a, m, 1);
	sum(cell.dbo, dot, n_a, m, 1);
	double** Wfa = slice(cell.Wf, 0, n_a, n_a, n_a + n_x, 1);
	double** ptrs10 = transposematmul(Wfa, dft, ndim);
	double** Wia = slice(cell.Wi, 0, n_a, n_a, n_a + n_x, 1);
	double** ptrs11 = transposematmul(Wia, dit, ndim);
	double** Wca = slice(cell.Wc, 0, n_a, n_a, n_a + n_x, 1);
	double** ptrs12 = transposematmul(Wca, dcct, ndim);
	double** Woa = slice(cell.Wo, 0, n_a, n_a, n_a + n_x, 1);
	double** ptrs13 = transposematmul(Woa, dot, ndim);
	cell.dat[t] = creatematrix(n_a, m);
	for (i = 0; i < n_a; i++) {
		for (j = 0; j < m; j++) {
			cell.dat[t][i][j] = ptrs10[i][j] + ptrs11[i][j] + ptrs12[i][j] + ptrs13[i][j];
		}
	}
	cell.dct[t] = multiply(ptrs6, cell.ft[t], n_a, m);
	double** Wfx = slice(cell.Wf, n_a, n_a + n_x, n_a, n_a + n_x, 1);
	double** ptrs14 = transposematmul(Wfx, dft, xdim);
	double** Wix = slice(cell.Wi, n_a, n_a + n_x, n_a, n_a + n_x, 1);
	double** ptrs15 = transposematmul(Wix, dit, xdim);
	double** Wcx = slice(cell.Wc, n_a, n_a + n_x, n_a, n_a + n_x, 1);
	double** ptrs16 = transposematmul(Wcx, dcct, xdim);
	double** Wox = slice(cell.Wo, n_a, n_a + n_x, n_a, n_a + n_x, 1);
	double** ptrs17 = transposematmul(Wox, dot, xdim);
	cell.dx[t] = creatematrix(n_x, m);
	for (i = 0; i < n_x; i++) {
		for (j = 0; j < m; j++) {
			cell.dx[t][i][j] = ptrs14[i][j] + ptrs15[i][j] + ptrs16[i][j] + ptrs17[i][j];
		}
	}
	Freematrix(concats, n_a + n_x);
	Freematrix(da_correct, n_a);
	Freematrix(dot, n_a);
	Freematrix(dit, n_a);
	Freematrix(dcct, n_a);
	Freematrix(dft, n_a);
	Freematrix(dWft, n_a);
	Freematrix(dWit, n_a);
	Freematrix(dWct, n_a);
	Freematrix(dWot, n_a);
	Freematrix(Wfa, n_a);
	Freematrix(Wia, n_a);
	Freematrix(Wca, n_a);
	Freematrix(Woa, n_a);
	Freematrix(Wfx, n_a);
	Freematrix(Wix, n_a);
	Freematrix(Wcx, n_a);
	Freematrix(Wox, n_a);
	Freematrix(ptrs1, n_a);
	Freematrix(ptrs2, n_a);
	Freematrix(ptrs3, n_a);
	Freematrix(ptrs4, n_a);
	Freematrix(ptrs5, n_a);
	Freematrix(ptrs6, n_a);
	Freematrix(ptrs7, n_a);
	Freematrix(ptrs8, n_a);
	Freematrix(ptrs9, n_a);
	Freematrix(ptrs10, n_a);
	Freematrix(ptrs11, n_a);
	Freematrix(ptrs12, n_a);
	Freematrix(ptrs13, n_a);
	Freematrix(ptrs14, n_x);
	Freematrix(ptrs15, n_x);
	Freematrix(ptrs16, n_x);
	Freematrix(ptrs17, n_x);
	return OK;
}

// 默认从y方向传过来的da已经算出来了
// dWya与dby为全连接层计算的梯度，在这里不计算这两个值
Status LSTMbackward(LSTMCell& cell, double*** X, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	double*** temp = (double***)malloc(T_x * sizeof(double**));
	cell.dat[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dat[T_x], n_a, m);
	cell.dct[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dct[T_x], n_a, m);
	cell.dWf = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWf, n_a, n_a + n_x);
	cell.dWi = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWi, n_a, n_a + n_x);
	cell.dWc = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWc, n_a, n_a + n_x);
	cell.dWo = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWo, n_a, n_a + n_x);
	cell.dbf = zerovector(n_a);
	cell.dbi = zerovector(n_a);
	cell.dbc = zerovector(n_a);
	cell.dbo = zerovector(n_a);
	for (i = T_x - 1; i >= n; i--) {
		temp[i] = scoretoinference(cell.outputs[i - 1], n_y, m);
		LSTMcellbackward(cell, temp[i], i);
	}
	for (i = T_x - 1; i >= n; i--) {
		Freematrix(temp[i], n_y);
	}
	for (i = n - 1; i >= 0; i--) {
		LSTMcellbackward(cell, X[i], i);
	}
	free(temp);
	return OK;
}

Status LSTMbackward_v2(LSTMCell& cell, double*** X, int n)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	int m = cell.m;
	double** temp = creatematrix(n_x, m);
	zeromatrix(temp, n_x, m);
	cell.dat[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dat[T_x], n_a, m);
	cell.dct[T_x] = creatematrix(n_a, m);
	zeromatrix(cell.dct[T_x], n_a, m);
	cell.dWf = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWf, n_a, n_a + n_x);
	cell.dWi = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWi, n_a, n_a + n_x);
	cell.dWc = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWc, n_a, n_a + n_x);
	cell.dWo = creatematrix(n_a, n_a + n_x);
	zeromatrix(cell.dWo, n_a, n_a + n_x);
	cell.dbf = zerovector(n_a);
	cell.dbi = zerovector(n_a);
	cell.dbc = zerovector(n_a);
	cell.dbo = zerovector(n_a);
	for (i = T_x - 1; i >= n; i--) {
		LSTMcellbackward(cell, temp, i);
	}
	for (i = n - 1; i >= 0; i--) {
		LSTMcellbackward(cell, X[i], i);
	}
	Freematrix(temp, n_x);
	return OK;
}

Status ClearLSTMCell_v2(LSTMCell& cell)
{
	int i;
	int T_x = cell.T_x;
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	for (i = 0; i < T_x; i++) {
		Freematrix(cell.da[i], n_a);
		Freematrix(cell.dx[i], n_x);
	}
	for (i = 0; i <= T_x; i++) {
		Freematrix(cell.dat[i], n_a);
		Freematrix(cell.dct[i], n_a);
	}
	return OK;
}

// 从start位置开始到end，包括start不包括end
Status Initrnnlayer(single_rnn_layer& layer, int m, int n_x, int n_a, int T_x, int n_y, int tris, int trie, int tros, int troe, int teis, int teie, int issample, double clip)
{
	layer.issample = issample;
	layer.clip = clip;
	InitLSTMCell(layer.cell, m, n_x, n_a, T_x, n_y);
	layer.train_input_start = tris;
	layer.train_input_end = trie;
	layer.train_output_start = tros;
	layer.train_output_end = troe;
	layer.test_input_start = teis;
	layer.test_input_end = teie;
	layer.a0 = creatematrix(n_a, m);
	zeromatrix(layer.a0, n_a, m);
	return OK;
}

// istrain为1时是训练，istrain为0时是测试
Status rnnlayerforward(single_rnn_layer& layer, double*** X, int istrain, int issample)
{
	int inputsize;
	if (istrain) {
		inputsize = layer.train_input_end - layer.train_input_start;
	}
	else {
		inputsize = layer.test_input_end - layer.test_input_start;
	}
	if (issample) {
		LSTMforward(layer.cell, X, layer.a0, inputsize);
	}
	else {
		LSTMforward_v2(layer.cell, X, layer.a0, inputsize);
	}
	return OK;
}

Status rnnlayerbackward(single_rnn_layer& layer, double*** X, double*** Y, int issample)
{
	int i, j, k, count;
	double sum = 0;
	int T_x = layer.cell.T_x;
	int n_a = layer.cell.n_a;
	int n_y = layer.cell.n_y;
	int m = layer.cell.m;
	int dim[3] = { n_y, m, n_a };
	int ndim[3] = { n_a, n_y, m };
	int length = layer.train_output_end - layer.train_output_start;
	double*** d_crossentropy = createtensor(T_x, n_y, m);
	double*** dy = (double***)malloc(T_x * sizeof(double**));
	double*** ptrs = (double***)malloc(T_x * sizeof(double**));
	zerotensor(d_crossentropy, T_x, n_y, m);
	layer.cell.dWy = creatematrix(n_y, m);
	zeromatrix(layer.cell.dWy, n_y, m);
	layer.cell.dby = zerovector(n_y);
	for (i = T_x - 1, count = 0; count < length; count++, i--) {
		for (j = 0; j < m; j++) {
			for (k = 0; k < n_y; k++) {
				d_crossentropy[i][k][j] = -Y[i][k][j] / layer.cell.outputs[i][k][j];
			}
		}
	}
	for (i = T_x - 1, count = 0; count < length; count++, i--) {
		dy[i] = softmaxbackward(d_crossentropy[i], layer.cell.outputs[i], n_y, m);
		layer.cell.da[i] = transposematmul(layer.cell.Wy, dy[i], ndim);
		for (j = 0; j < n_y; j++) {
			sum = 0;
			for (k = 0; k < m; k++) {
				sum += dy[i][j][k];
			}
			layer.cell.dby[j] += sum / m;
		}
		ptrs[i] = matmultranspose(dy[i], layer.cell.a[i + 1], dim);
		for (j = 0; j < n_y; j++) {
			for (k = 0; k < n_a; k++) {
				layer.cell.dWy[j][k] += ptrs[i][j][k] / m;
			}
		}
	}
	Freetensor(d_crossentropy, T_x, n_y);  // 用v6数据集时会报错（原因未知）
	for (i = T_x - 1, count = 0; count < length; count++, i--) {
		Freematrix(dy[i], n_y);  // 用v6数据集时会报错（原因未知）
		Freematrix(ptrs[i], n_y);
	}
	free(dy);
	free(ptrs);
	for (i = T_x - 1 - length; i >= 0; i--) {
		layer.cell.da[i] = creatematrix(n_a, m);
		zeromatrix(layer.cell.da[i], n_a, m);
	}
	LSTMbackward_v2(layer.cell, X, T_x - 1);
	return OK;
}

Status clip(single_rnn_layer& layer, double clip)
{
	int i, j;
	int n_a = layer.cell.n_a;
	int n_x = layer.cell.n_x;
	int n_y = layer.cell.n_y;
	for (i = 0; i < n_y; i++) {
		for (j = 0; j < n_a; j++) {
			layer.cell.dWy[i][j] = layer.cell.dWy[i][j] > clip ? clip : layer.cell.dWy[i][j];
			layer.cell.dWy[i][j] = layer.cell.dWy[i][j] < -clip ? -clip : layer.cell.dWy[i][j];
		}
		layer.cell.dby[i] = layer.cell.dby[i] > clip ? clip : layer.cell.dby[i];
		layer.cell.dby[i] = layer.cell.dby[i] < -clip ? -clip : layer.cell.dby[i];
	}
	for (i = 0; i < n_a; i++) {
		for (j = 0; j < n_a + n_x; j++) {
			layer.cell.dWc[i][j] = layer.cell.dWc[i][j] > clip ? clip : layer.cell.dWc[i][j];
			layer.cell.dWc[i][j] = layer.cell.dWc[i][j] < -clip ? -clip : layer.cell.dWc[i][j];
			layer.cell.dWi[i][j] = layer.cell.dWi[i][j] > clip ? clip : layer.cell.dWi[i][j];
			layer.cell.dWi[i][j] = layer.cell.dWi[i][j] < -clip ? -clip : layer.cell.dWi[i][j];
			layer.cell.dWo[i][j] = layer.cell.dWo[i][j] > clip ? clip : layer.cell.dWo[i][j];
			layer.cell.dWo[i][j] = layer.cell.dWo[i][j] < -clip ? -clip : layer.cell.dWo[i][j];
			layer.cell.dWf[i][j] = layer.cell.dWf[i][j] > clip ? clip : layer.cell.dWf[i][j];
			layer.cell.dWf[i][j] = layer.cell.dWf[i][j] < -clip ? -clip : layer.cell.dWf[i][j];
		}
		layer.cell.dbc[i] = layer.cell.dbc[i] > clip ? clip : layer.cell.dbc[i];
		layer.cell.dbc[i] = layer.cell.dbc[i] < -clip ? -clip : layer.cell.dbc[i];
		layer.cell.dbi[i] = layer.cell.dbi[i] > clip ? clip : layer.cell.dbi[i];
		layer.cell.dbi[i] = layer.cell.dbi[i] < -clip ? -clip : layer.cell.dbi[i];
		layer.cell.dbo[i] = layer.cell.dbo[i] > clip ? clip : layer.cell.dbo[i];
		layer.cell.dbo[i] = layer.cell.dbo[i] < -clip ? -clip : layer.cell.dbo[i];
		layer.cell.dbf[i] = layer.cell.dbf[i] > clip ? clip : layer.cell.dbf[i];
		layer.cell.dbf[i] = layer.cell.dbf[i] < -clip ? -clip : layer.cell.dbf[i];
	}
	return OK;
}

Status updaternnlayer(single_rnn_layer& layer, double*** X, double*** Y, int t, int issample, LSTMparameter v, LSTMparameter s, double lr)
{
	double beta1 = 0.9;
	double beta2 = 0.999;
	int j, k;
	rnnlayerforward(layer, X, 1, issample);
	rnnlayerbackward(layer, X, Y, issample);
	clip(layer, layer.clip);
	int n_a = layer.cell.n_a;
	int n_x = layer.cell.n_x;
	int n_y = layer.cell.n_y;
	for (j = 0; j < n_a; j++) {
		for (k = 0; k < n_a + n_x; k++) {
			v.dWc[j][k] = beta1 * v.dWc[j][k] + (1 - beta1) * layer.cell.dWc[j][k];
			s.dWc[j][k] = beta2 * s.dWc[j][k] + (1 - beta2) * pow(layer.cell.dWc[j][k], 2.0);
			v.dWi[j][k] = beta1 * v.dWi[j][k] + (1 - beta1) * layer.cell.dWi[j][k];
			s.dWi[j][k] = beta2 * s.dWi[j][k] + (1 - beta2) * pow(layer.cell.dWi[j][k], 2.0);
			v.dWo[j][k] = beta1 * v.dWo[j][k] + (1 - beta1) * layer.cell.dWo[j][k];
			s.dWo[j][k] = beta2 * s.dWo[j][k] + (1 - beta2) * pow(layer.cell.dWo[j][k], 2.0);
			v.dWf[j][k] = beta1 * v.dWf[j][k] + (1 - beta1) * layer.cell.dWf[j][k];
			s.dWf[j][k] = beta2 * s.dWf[j][k] + (1 - beta2) * pow(layer.cell.dWf[j][k], 2.0);
		}
	}
	for (j = 0; j < n_a; j++) {
		v.dbc[j] = beta1 * v.dbc[j] + (1 - beta1) * layer.cell.dbc[j];
		s.dbc[j] = beta2 * s.dbc[j] + (1 - beta2) * pow(layer.cell.dbc[j], 2.0);
		v.dbi[j] = beta1 * v.dbi[j] + (1 - beta1) * layer.cell.dbi[j];
		s.dbi[j] = beta2 * s.dbi[j] + (1 - beta2) * pow(layer.cell.dbi[j], 2.0);
		v.dbo[j] = beta1 * v.dbo[j] + (1 - beta1) * layer.cell.dbo[j];
		s.dbo[j] = beta2 * s.dbo[j] + (1 - beta2) * pow(layer.cell.dbo[j], 2.0);
		v.dbf[j] = beta1 * v.dbf[j] + (1 - beta1) * layer.cell.dbf[j];
		s.dbf[j] = beta2 * s.dbf[j] + (1 - beta2) * pow(layer.cell.dbf[j], 2.0);
	}
	for (j = 0; j < n_y; j++) {
		for (k = 0; k < n_a; k++) {
			v.dWy[j][k] = beta1 * v.dWy[j][k] + (1 - beta1) * layer.cell.dWy[j][k];
			s.dWy[j][k] = beta2 * s.dWy[j][k] + (1 - beta2) * pow(layer.cell.dWy[j][k], 2.0);
		}
	}
	for (j = 0; j < n_y; j++) {
		v.dby[j] = beta1 * v.dby[j] + (1 - beta1) * layer.cell.dby[j];
		s.dby[j] = beta2 * s.dby[j] + (1 - beta2) * pow(layer.cell.dby[j], 2.0);
	}
	updateweight(layer.cell.Wc, v.dWc, s.dWc, lr, t, n_a, n_a + n_x);
	updateweight(layer.cell.Wi, v.dWi, s.dWi, lr, t, n_a, n_a + n_x);
	updateweight(layer.cell.Wo, v.dWo, s.dWo, lr, t, n_a, n_a + n_x);
	updateweight(layer.cell.Wf, v.dWf, s.dWf, lr, t, n_a, n_a + n_x);
	updateweight(layer.cell.Wy, v.dWy, s.dWy, lr, t, n_y, n_a);
	updatebias(layer.cell.bc, v.dbc, s.dbc, lr, t, n_a);
	updatebias(layer.cell.bi, v.dbi, s.dbi, lr, t, n_a);
	updatebias(layer.cell.bo, v.dbo, s.dbo, lr, t, n_a);
	updatebias(layer.cell.bf, v.dbf, s.dbf, lr, t, n_a);
	updatebias(layer.cell.by, v.dby, s.dby, lr, t, n_y);
	return OK;
}

Status InitLSTMparameter(LSTMparameter& p, LSTMCell cell)
{
	int n_a = cell.n_a;
	int n_x = cell.n_x;
	int n_y = cell.n_y;
	p.dWc = creatematrix(n_a, n_a + n_x);
	zeromatrix(p.dWc, n_a, n_a + n_x);
	p.dWi = creatematrix(n_a, n_a + n_x);
	zeromatrix(p.dWi, n_a, n_a + n_x);
	p.dWo = creatematrix(n_a, n_a + n_x);
	zeromatrix(p.dWo, n_a, n_a + n_x);
	p.dWf = creatematrix(n_a, n_a + n_x);
	zeromatrix(p.dWf, n_a, n_a + n_x);
	p.dWy = creatematrix(n_y, n_a);
	zeromatrix(p.dWy, n_y, n_a);
	p.dbc = zerovector(n_a);
	p.dbi = zerovector(n_a);
	p.dbo = zerovector(n_a);
	p.dbf = zerovector(n_a);
	p.dby = zerovector(n_y);
	return OK;
}

Status rnnlayerfit(single_rnn_layer& layer, dataset X_train, dataset Y_train, dataset X_val, dataset Y_val,
	double lr, int epochs, int testbatch, int minibatch, int acc, int feq, int earlystop, double decay)
{
	int mark = 0;
	int start = 0;
	double lastacc = 0;
	double testacc = 0;
	double testcost = 0;
	int T_x = layer.cell.T_x;
	int n_a = layer.cell.n_a;
	int n_x = layer.cell.n_x;
	int n_y = layer.cell.n_y;
	int m = layer.cell.m;
	double** p;
	double** q;
	LSTMparameter v, s;
	InitLSTMparameter(v, layer.cell);
	InitLSTMparameter(s, layer.cell);
	double**** X_temp = (double****)malloc(1 * sizeof(double***));
	if (!X_temp)exit(OVERFLOW);
	double**** Y_temp = (double****)malloc(1 * sizeof(double***));
	if (!Y_temp)exit(OVERFLOW);
	for (int i = 0; i < epochs; i++) {
		if (i % feq == 0) {
			srand((unsigned)time(NULL));
			start = rand() % (m - minibatch);
			X_temp[0] = minibatch_v2(X_train.data[0], minibatch, T_x, n_x, m, start);
			Y_temp[0] = minibatch_v2(Y_train.data[0], minibatch, T_x, n_y, m, start);
			layer.cell.m = minibatch;
			rnnlayerforward(layer, X_temp[0], 0, layer.issample);
			printf("\nepoch %d: train cost = %f", i, crossentropy(layer.cell.outputs, Y_temp[0], T_x, n_y, minibatch, minibatch));
			if (acc) {
				p = transpose(Y_temp[0][T_x - 1], n_y, minibatch);
				q = transpose(layer.cell.outputs[T_x - 1], n_y, minibatch);
				printf(" train acc = %f", accuracy(p, q, minibatch, n_y));
				Freematrix(p, minibatch);
				Freematrix(q, minibatch);
			}
			Freetensor(X_temp[0], T_x, n_x);
			Freetensor(Y_temp[0], T_x, n_y);
			ClearLSTMCell_v1(layer.cell);
			layer.cell.m = testbatch;
			rnnlayerforward(layer, X_val.data[0], 0, layer.issample);
			testcost = crossentropy(layer.cell.outputs, Y_val.data[0], T_x, n_y, testbatch, testbatch);
			printf(" test cost = %f", testcost);
			if (acc) {
				lastacc = testacc;
				p = transpose(Y_val.data[0][T_x - 1], n_y, testbatch);
				q = transpose(layer.cell.outputs[T_x - 1], n_y, testbatch);
				testacc = accuracy(p, q, testbatch, n_y);
				printf(" test acc = %f", testacc);
				Freematrix(p, testbatch);
				Freematrix(q, testbatch);
			}
			ClearLSTMCell_v1(layer.cell);
			if (testacc > lastacc) {
				mark++;
			}
			else {
				mark = 0;
			}
			if (mark >= earlystop || fabs(testacc - 1.0) < 0.001) {
				printf("\nearlystop");
				return OK;
			}
			printf("\n");
		}
		printf("->");
		srand((unsigned)time(NULL));
		start = rand() % (m - minibatch);
		X_temp[0] = minibatch_v2(X_train.data[0], minibatch, T_x, n_x, m, start);
		Y_temp[0] = minibatch_v2(Y_train.data[0], minibatch, T_x, n_y, m, start);
		layer.cell.m = minibatch;
		updaternnlayer(layer, X_temp[0], Y_temp[0], i + 1, 0, v, s, lr);
		printf("\b->");
		Freetensor(X_temp[0], T_x, n_x);
		Freetensor(Y_temp[0], T_x, n_y);
		ClearLSTMCell_v1(layer.cell);  // 用v6数据集时有概率会报错（原因未知）
		ClearLSTMCell_v2(layer.cell);  // 用v6数据集时有概率会报错（原因未知）
		lr *= (1 - decay);
	}
	layer.cell.m = testbatch;
	rnnlayerforward(layer, X_val.data[0], 0, layer.issample);
	printf("\ncost = %f", crossentropy(layer.cell.outputs, Y_val.data[0], T_x, n_y, testbatch, testbatch));
	if (acc) {
		p = transpose(Y_val.data[0][T_x - 1], n_y, testbatch);
		q = transpose(layer.cell.outputs[T_x - 1], n_y, testbatch);
		testacc = accuracy(p, q, testbatch, n_y);
		printf(" test acc = %f", testacc);
		Freematrix(p, testbatch);
		Freematrix(q, testbatch);
	}
	return OK;
}

double crossentropy(double**** predict, double**** Y, int* size, int batch)
{
	double cost = 0;
	for (int i = 0; i < size[0]; i++) {
		for (int j = 0; j < size[1]; j++) {
			for (int k = 0; k < size[2]; k++) {
				for (int n = 0; n < size[3]; n++) {
					cost -= Y[i][j][k][n] * log(predict[i][j][k][n]);
				}
			}
		}
	}
	cost = cost / batch;
	return cost;
}

double crossentropy(double*** predict, double*** logits, int h, int w, int d, int batch)
{
	double cost = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < d; k++) {
				cost -= logits[i][j][k] * log(predict[i][j][k]);
			}
		}
	}
	cost = cost / batch;
	return cost;
}

double crossentropy_v2(double**** predict, double**** Y, int* size, int batch)
{
	double cost = 0;
	for (int i = 0; i < size[0]; i++) {
		for (int j = 0; j < size[1]; j++) {
			for (int k = 0; k < size[2]; k++) {
				for (int n = 0; n < size[3]; n++) {
					cost += (Y[i][j][k][n] - 1) * log(1 - predict[i][j][k][n]);
					cost -= Y[i][j][k][n] * log(predict[i][j][k][n]);
				}
			}
		}
	}
	cost = cost / batch;
	return cost;
}

double squaredif(double**** predict, double**** Y, int* size, int batch)
{
	double cost = 0;
	for (int i = 0; i < size[0]; i++) {
		for (int j = 0; j < size[1]; j++) {
			for (int k = 0; k < size[2]; k++) {
				for (int n = 0; n < size[3]; n++) {
					cost += pow((Y[i][j][k][n] - predict[i][j][k][n]), 2.0);
				}
			}
		}
	}
	cost = cost / (batch * 2);
	return cost;
}

double**** minibatchfortensor(double**** tensor, int* curdim, int batch, int start)
{
	double**** newX = createtensor(batch, curdim[1], curdim[2], curdim[3]);
	for (int i = 0; i < batch; i++, start++) {
		for (int j = 0; j < curdim[1]; j++) {
			for (int k = 0; k < curdim[2]; k++) {
				for (int n = 0; n < curdim[3]; n++) {
					newX[i][j][k][n] = tensor[start][j][k][n];
				}
			}
		}
	}
	return newX;
}

double**** minibatchformatrix(double**** matrix, int h, int w, int batch, int start)
{
	double**** newY = createtensor(1, 1, batch, w);
	for (int i = 0; i < batch; i++, start++) {
		for (int j = 0; j < w; j++) {
			newY[0][0][i][j] = matrix[0][0][start][j];
		}
	}
	return newY;
}

double*** minibatch_v2(double*** X, int minibatch, int T_x, int n, int m, int start)
{
	double*** result = createtensor(T_x, n, minibatch);
	for (int k = 0; k < minibatch; start++, k++) {
		for (int i = 0; i < T_x; i++) {
			for (int j = 0; j < n; j++) {
				result[i][j][k] = X[i][j][start];
			}
		}
	}
	return result;
}

Status updatekernel(kernel K, kernel dK, double lr)
{
	for (int i = 0; i < K.h; i++) {
		for (int j = 0; j < K.w; j++) {
			for (int k = 0; k < K.d; k++) {
				for (int n = 0; n < K.n; n++) {
					K.K[i][j][k][n] -= lr * dK.K[i][j][k][n];
				}
			}
		}
	}
	return OK;
}

Status updateweight(weight W, weight dW, double lr)
{
	for (int i = 0; i < W.h; i++) {
		for (int j = 0; j < W.w; j++) {
			W.W[i][j] -= lr * dW.W[i][j];
		}
	}
	return OK;
}

Status updatebias(bias b, bias db, double lr)
{
	for (int i = 0; i < b.length; i++) {
		b.b[i] -= lr * db.b[i];
	}
	return OK;
}

Status updatekernel_v2(kernel K, kernel v, kernel s, double lr, int t)
{
	double epsilon = 0.0000001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	double v_correct;
	double s_correct;
	for (int i = 0; i < K.h; i++) {
		for (int j = 0; j < K.w; j++) {
			for (int k = 0; k < K.d; k++) {
				for (int n = 0; n < K.n; n++) {
					v_correct = v.K[i][j][k][n] / (1 - pow(beta1, 1.0 * t));
					s_correct = s.K[i][j][k][n] / (1 - pow(beta2, 1.0 * t));
					K.K[i][j][k][n] -= lr * v_correct / (sqrt(s_correct) + epsilon);
				}
			}
		}
	}
	return OK;
}

Status updateweight_v2(weight W, weight v, weight s, double lr, int t)
{
	double epsilon = 0.0000001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int i, j;
	double v_correct;
	double s_correct;
	for (i = 0; i < W.h; i++) {
		for (j = 0; j < W.w; j++) {
			v_correct = v.W[i][j] / (1 - pow(beta1, 1.0 * t));
			s_correct = s.W[i][j] / (1 - pow(beta2, 1.0 * t));
			W.W[i][j] -= lr * v_correct / (sqrt(s_correct) + epsilon);
		}
	}
	return OK;
}

Status updatebias_v2(bias b, bias v, bias s, double lr, int t)
{
	double epsilon = 0.0000001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int i;
	double v_correct;
	double s_correct;
	for (i = 0; i < b.length; i++) {
		v_correct = v.b[i] / (1 - pow(beta1, 1.0 * t));
		s_correct = s.b[i] / (1 - pow(beta2, 1.0 * t));
		b.b[i] -= lr * v_correct / (sqrt(s_correct) + epsilon);
	}
	return OK;
}

Status updateweight(double** W, double** v, double** s, double lr, int t, int h, int w)
{
	double epsilon = 0.00000001;
	double beta1 = 0.9;
	int i, j;
	double v_correct;
	double s_correct;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			v_correct = v[i][j] / (1 - pow(beta1, 1.0 * t));
			s_correct = s[i][j] / (1 - pow(beta1, 1.0 * t));
			W[i][j] -= lr * v_correct / (sqrt(s_correct) + epsilon);
		}
	}
	return OK;
}

Status updatebias(double* b, double* v, double* s, double lr, int t, int length)
{
	double epsilon = 0.00000001;
	double beta1 = 0.9;
	int i;
	double v_correct;
	double s_correct;
	for (i = 0; i < length; i++) {
		v_correct = v[i] / (1 - pow(beta1, 1.0 * t));
		s_correct = s[i] / (1 - pow(beta1, 1.0 * t));
		b[i] -= lr * v_correct / (sqrt(s_correct) + epsilon);
	}
	return OK;
}

Status clip(model& model)
{
	int depth = model.depth;
	int i, j, k, n, m;
	double clip = model.clip;
	char name;
	for (i = 0; i < depth; i++) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			for (j = 0; j < model.layer[i].dW.h; j++) {
				for (k = 0; k < model.layer[i].dW.w; k++) {
					model.layer[i].dW.W[j][k] = model.layer[i].dW.W[j][k] > clip ? clip : model.layer[i].dW.W[j][k];
					model.layer[i].dW.W[j][k] = model.layer[i].dW.W[j][k] < (-clip) ? (-clip) : model.layer[i].dW.W[j][k];
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				model.layer[i].db.b[j] = model.layer[i].db.b[j] > clip ? clip : model.layer[i].db.b[j];
				model.layer[i].db.b[j] = model.layer[i].db.b[j] < (-clip) ? (-clip) : model.layer[i].db.b[j];
			}
			break;
		case 'c':
			for (m = 0; m < model.layer[i].dK.h; m++) {
				for (j = 0; j < model.layer[i].dK.w; j++) {
					for (k = 0; k < model.layer[i].dK.d; k++) {
						for (n = 0; n < model.layer[i].dK.n; n++) {
							model.layer[i].dK.K[m][j][k][n] = model.layer[i].dK.K[m][j][k][n] > clip ? clip : model.layer[i].dK.K[m][j][k][n];
							model.layer[i].dK.K[m][j][k][n] = model.layer[i].dK.K[m][j][k][n] < (-clip) ? (-clip) : model.layer[i].dK.K[m][j][k][n];
						}
					}
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				model.layer[i].db.b[j] = model.layer[i].db.b[j] > clip ? clip : model.layer[i].db.b[j];
				model.layer[i].db.b[j] = model.layer[i].db.b[j] < (-clip) ? (-clip) : model.layer[i].db.b[j];
			}
			break;
		case 'l':
			break;
		case 'p':
			break;
		default:
			printf("your model is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status updatewithgradient(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d)
{
	char name;
	int i;
	modelforward(X, model, batch, h, w, d);
	modelbackward(model, batch, X, Y, h, w, d);
	clip(model);
	for (i = 0; i < model.depth; i++) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			updateweight(model.layer[i].W, model.layer[i].dW, lr);
			updatebias(model.layer[i].b, model.layer[i].db, lr);
			break;
		case 'c':
			updatekernel(model.layer[i].K, model.layer[i].dK, lr);
			updatebias(model.layer[i].b, model.layer[i].db, lr);
			break;
		case 'p':
			break;
		case 'l':
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status updatewithmomentum(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d, parameter_v& v)
{
	char name;
	double beta = 0.9;
	int i, j, k, n, m;
	modelforward(X, model, batch, h, w, d);
	modelbackward(model, batch, X, Y, h, w, d);
	clip(model);
	for (i = 0; i < model.depth; i++) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			for (j = 0; j < model.layer[i].dW.h; j++) {
				for (k = 0; k < model.layer[i].dW.w; k++) {
					v.layer[i].dW.W[j][k] = beta * v.layer[i].dW.W[j][k] + (1 - beta) * model.layer[i].dW.W[j][k];
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				v.layer[i].db.b[j] = beta * v.layer[i].db.b[j] + (1 - beta) * model.layer[i].db.b[j];
			}
			updateweight(model.layer[i].W, v.layer[i].dW, lr);
			updatebias(model.layer[i].b, v.layer[i].db, lr);
			break;
		case 'c':
			for (m = 0; m < model.layer[i].dK.h; m++) {
				for (j = 0; j < model.layer[i].dK.w; j++) {
					for (k = 0; k < model.layer[i].dK.d; k++) {
						for (n = 0; n < model.layer[i].dK.n; n++) {
							v.layer[i].dK.K[m][j][k][n] = beta * v.layer[i].dK.K[m][j][k][n] + (1 - beta) * model.layer[i].dK.K[m][j][k][n];
						}
					}
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				v.layer[i].db.b[j] = beta * v.layer[i].db.b[j] + (1 - beta) * model.layer[i].db.b[j];
			}
			updatekernel(model.layer[i].K, model.layer[i].dK, lr);
			updatebias(model.layer[i].b, model.layer[i].db, lr);
			break;
		case 'p':
			break;
		case 'l':
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status updatewithAdam(double**** X, double**** Y, model& model, double lr, int batch, int h, int w, int d, int t, parameter_v& v, parameter_s& s)
{
	char name;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int i, j, k, n, m;
	modelforward(X, model, batch, h, w, d);
	modelbackward(model, batch, X, Y, h, w, d);
	clip(model);
	for (i = 0; i < model.depth; i++) {
		name = model.layer[i].name;
		switch (name) {
		case 'f':
			for (j = 0; j < model.layer[i].dW.h; j++) {
				for (k = 0; k < model.layer[i].dW.w; k++) {
					v.layer[i].dW.W[j][k] = beta1 * v.layer[i].dW.W[j][k] + (1 - beta1) * model.layer[i].dW.W[j][k];
					s.layer[i].dW.W[j][k] = beta2 * s.layer[i].dW.W[j][k] + (1 - beta2) * pow(model.layer[i].dW.W[j][k], 2);
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				v.layer[i].db.b[j] = beta1 * v.layer[i].db.b[j] + (1 - beta1) * model.layer[i].db.b[j];
				s.layer[i].db.b[j] = beta2 * s.layer[i].db.b[j] + (1 - beta2) * pow(model.layer[i].db.b[j], 2);
			}
			updateweight_v2(model.layer[i].W, v.layer[i].dW, s.layer[i].dW, lr, t);
			updatebias_v2(model.layer[i].b, v.layer[i].db, s.layer[i].db, lr, t);
			break;
		case 'c':
			for (m = 0; m < model.layer[i].dK.h; m++) {
				for (j = 0; j < model.layer[i].dK.w; j++) {
					for (k = 0; k < model.layer[i].dK.d; k++) {
						for (n = 0; n < model.layer[i].dK.n; n++) {
							v.layer[i].dK.K[m][j][k][n] = beta1 * v.layer[i].dK.K[m][j][k][n] + (1 - beta1) * model.layer[i].dK.K[m][j][k][n];
							s.layer[i].dK.K[m][j][k][n] = beta2 * s.layer[i].dK.K[m][j][k][n] + (1 - beta2) * pow(model.layer[i].dK.K[m][j][k][n], 2);
						}
					}
				}
			}
			for (j = 0; j < model.layer[i].db.length; j++) {
				v.layer[i].db.b[j] = beta1 * v.layer[i].db.b[j] + (1 - beta1) * model.layer[i].db.b[j];
				s.layer[i].db.b[j] = beta2 * s.layer[i].db.b[j] + (1 - beta2) * pow(model.layer[i].db.b[j], 2);
			}
			updatekernel_v2(model.layer[i].K, v.layer[i].dK, s.layer[i].dK, lr, t);
			updatebias_v2(model.layer[i].b, v.layer[i].db, s.layer[i].db, lr, t);
			break;
		case 'p':
			break;
		case 'l':
			break;
		default:
			printf("your layer is worry");
			exit(INFEASIBLE);
		}
	}
	return OK;
}

Status modelcompile(model& model, char optimizer, char loss, double keep_drop, double lambda, double clip)
{
	model.optimizer = optimizer;
	model.loss = loss;
	model.keep_drop = keep_drop;
	model.lambda = lambda;
	model.clip = clip;
	return OK;
}

double accuracy(double** Y, double** predict, int h, int w)
{
	double** logits = creatematrix(h, w);
	int index = 0;
	double max = 0;
	double acc = 0;
	int i, j;
	for (i = 0; i < h; i++) {
		index = 0;
		max = predict[i][0];
		for (j = 0; j < w; j++) {
			if (predict[i][j] > max) {
				index = j;
				max = predict[i][j];
			}
			logits[i][j] = 0;
		}
		logits[i][index] = 1;
	}
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			if (logits[i][j] && Y[i][j]) {
				acc += 1;
			}
		}
	}
	acc /= h;
	return acc;
}

// 这里用X代表输入，用Y代表输出
//如果data的totaldim = 2，则dim为[batch, feature]或者[batch, classes]，如果totaldim = 4，则dim为[batch, h, w, d]
Status Initdata(dataset& dataset, int* dim, int totaldim, char name)
{
	if (name != 'Y' && name != 'X') {
		printf("the name of dataset is worry");
		return ERROR;
	}
	dataset.name = name;
	if (totaldim != 2 && totaldim != 4) {
		printf("your totaldim is worry");
		return ERROR;
	}
	dataset.totaldim = totaldim;
	dataset.dim = (int*)malloc(4 * sizeof(int));
	if (dataset.dim == NULL)exit(OVERFLOW);
	dataset.dim[0] = dim[0]; dataset.dim[1] = dim[1]; dataset.dim[2] = dim[2]; dataset.dim[3] = dim[3];
	switch (name) {
	case 'X':
		switch (totaldim) {
		case 2:
			dataset.data = (double****)malloc(1 * sizeof(double***));
			if (dataset.data == NULL)exit(OVERFLOW);
			dataset.data[0] = (double***)malloc(1 * sizeof(double**));
			if (dataset.data[0] == NULL)exit(OVERFLOW);
			dataset.data[0][0] = loaddata(dim[0], dim[1]);
			break;
		case 4:
			dataset.data = loaddata(dim[0], dim[1], dim[2], dim[3]);
			break;
		}
		break;
	case 'Y':
		switch (totaldim) {
		case 2:
			dataset.data = (double****)malloc(1 * sizeof(double***));
			if (dataset.data == NULL)exit(OVERFLOW);
			dataset.data[0] = (double***)malloc(1 * sizeof(double**));
			if (dataset.data[0] == NULL)exit(OVERFLOW);
			dataset.data[0][0] = loaddataY(dim[0], dim[1]);
			break;
		case 4:
			dataset.data = loaddata(dim[0], dim[1], dim[2], dim[3]);
			break;
		}
		break;
	}
	return OK;
}

Status Addnoise(double**** data, int* dim)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int n = 0; n < dim[3]; n++) {
					data[i][j][k][n] += (double)((double)(rand() % MAX) / MAX * 0.2 - 0.1);
				}
			}
		}
	}
	return OK;
}

Status setzeropointone(double**** data, int start, int end, int h, int w, int n_C)
{
	for (int i = start; i < end; i++) {
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < w; k++) {
				for (int n = 0; n < n_C; n++) {
					data[i][j][k][n] = 0.1;
				}
			}
		}
	}
	return OK;
}

Status square(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_maxdistance = h / 4;
	int w_maxdistance = w / 4;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (int j = h_maxdistance + hposition; j < h - h_maxdistance + hposition; j++) {
		for (int k = w_maxdistance + wposition; k < w - w_maxdistance + wposition; k++) {
			for (int i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status square_v2(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_maxdistance = h / 6;
	int w_maxdistance = w / 6;
	int half = h / 2;
	int j, k, i;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_maxdistance + hposition; j < half + hposition; j++) {
		for (k = half - (j - hposition) + w_maxdistance + wposition; k < half + (j - hposition) - w_maxdistance + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}

		}
	}
	for (j = half + hposition; j < h - h_maxdistance + hposition; j++) {
		for (k = (j - hposition) - half + w_maxdistance + wposition; k < w - (j - hposition) + 2 * w_maxdistance + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status circular(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_maxdistance = h / 6;
	int w_maxdistance = w / 6;
	int half = h / 2;
	int j, k, i;
	int temp = (int)((h * h) / 9);
	int w_start, w_end;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_maxdistance + hposition; j < h - h_maxdistance + hposition; j++) {
		w_start = (int)(half - sqrt(1.0 * (temp - ((j - hposition) - half) * ((j - hposition) - half))));
		w_end = (int)(half + sqrt(1.0 * (temp - ((j - hposition) - half) * ((j - hposition) - half))));
		for (k = w_start + wposition; k < w_end + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status triangle(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_start = h / 4;
	int h_end = h - h_start;
	int w_start, w_end;
	int j, k, i;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_start + hposition; j < h_end + hposition; j++) {
		w_start = 5 * h / 8 - (j - hposition) / 2;
		w_end = 3 * h / 8 + (j - hposition) / 2;
		for (k = w_start + wposition; k < w_end + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status triangle_v2(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_start = h / 4;
	int h_end = h - h_start;
	int w_start, w_end;
	int j, k, i;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_start + hposition; j < h_end + hposition; j++) {
		w_start = h / 8 + (j - hposition) / 2;
		w_end = 7 * h / 8 - (j - hposition) / 2;
		for (k = w_start + wposition; k < w_end + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status parallelogram(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_start = h / 4;
	int h_end = h - h_start;
	int w_start, w_end;
	int j, k, i;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_start + hposition; j < h_end + hposition; j++) {
		w_start = 5 * h / 8 - (j - hposition) / 2;
		w_end = w_start + h / 4;
		for (k = w_start + wposition; k < w_end + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status rectangle(double**** data, int start, int end, int h, int w, int n_C)
{
	int h_start = h / 3;
	int h_end = h - h_start;
	int w_start = w / 6;
	int w_end = 5 * w / 6;
	int j, k, i;
	int hposition = positioncode % (maxdistance + 1);
	hposition = hposition - maxdistance / 2;
	int wposition = positioncode / (maxdistance + 1);
	wposition = wposition - maxdistance / 2;
	positioncode = (positioncode + 1) % ((maxdistance + 1) * (maxdistance + 1));
	for (j = h_start + hposition; j < h_end + hposition; j++) {
		for (k = w_start + wposition; k < w_end + wposition; k++) {
			for (i = start; i < end; i++) {
				data[i][j][k][n_C] = 0.8;
			}
		}
	}
	return OK;
}

Status Initdata_v1(dataset& X, dataset& Y, int batch)
{
	int temp = 0;
	int i, j;
	X.data = createtensor(batch, 28, 28, 1);
	X.name = 'X';
	X.totaldim = 4;
	X.dim = (int*)malloc(4 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.dim[0] = batch; X.dim[1] = 28; X.dim[2] = 28; X.dim[3] = 1;
	Y.data = createtensor(1, 1, batch, 7);
	Y.name = 'Y';
	Y.totaldim = 2;
	Y.dim = (int*)malloc(2 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.dim[0] = batch; Y.dim[1] = 7;
	setzeropointone(X.data, 0, batch, 28, 28, 1);
	for (i = 0; i < batch; i++) {
		for (j = 0; j < 7; j++) {
			Y.data[0][0][i][j] = 0;
		}
	}
	for (i = 0; i < batch; i++) {
		temp = i % 7;
		switch (temp) {
		case 0:
			square(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][0] = 1;
			break;
		case 1:
			square_v2(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][1] = 1;
			break;
		case 2:
			circular(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][2] = 1;
			break;
		case 3:
			triangle(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][3] = 1;
			break;
		case 4:
			triangle_v2(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][4] = 1;
			break;
		case 5:
			parallelogram(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][5] = 1;
			break;
		case 6:
			rectangle(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][6] = 1;
			break;
		}
	}
	Addnoise(X.data, X.dim);
	return OK;
}

Status Initdata_v2(dataset& X, dataset& Y, int batch)
{
	int temp = 0;
	int i, j;
	X.data = createtensor(batch, 28, 28, 3);
	X.dim = (int*)malloc(4 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.dim[0] = batch; X.dim[1] = 28; X.dim[2] = 28; X.dim[3] = 3;
	X.totaldim = 4;
	X.name = 'X';
	Y.data = createtensor(1, 1, batch, 9);
	Y.dim = (int*)malloc(2 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.dim[0] = batch; Y.dim[1] = 9;
	Y.totaldim = 2;
	Y.name = 'Y';
	setzeropointone(X.data, 0, batch, 28, 28, 3);
	for (i = 0; i < batch; i++) {
		for (j = 0; j < 9; j++) {
			Y.data[0][0][i][j] = 0;
		}
	}
	for (i = 0; i < batch; i++) {
		temp = i % 9;
		switch (temp) {
		case 0:
			square(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][0] = 1;
			break;
		case 1:
			triangle(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][1] = 1;
			break;
		case 2:
			circular(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][2] = 1;
			break;
		case 3:
			square(X.data, i, i + 1, 28, 28, 1);
			Y.data[0][0][i][3] = 1;
			break;
		case 4:
			triangle(X.data, i, i + 1, 28, 28, 1);
			Y.data[0][0][i][4] = 1;
			break;
		case 5:
			circular(X.data, i, i + 1, 28, 28, 1);
			Y.data[0][0][i][5] = 1;
			break;
		case 6:
			square(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][6] = 1;
			break;
		case 7:
			triangle(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][7] = 1;
			break;
		case 8:
			circular(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][8] = 1;
			break;
		}
	}
	Addnoise(X.data, X.dim);
	return OK;
}

Status Initdata_v3(dataset& X, dataset& Y, int batch)
{
	int temp = 0;
	int i, j;
	X.data = createtensor(batch, 28, 28, 3);
	X.dim = (int*)malloc(4 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.dim[0] = batch; X.dim[1] = 28; X.dim[2] = 28; X.dim[3] = 3;
	X.totaldim = 4;
	X.name = 'X';
	Y.data = createtensor(1, 1, batch, 7);
	Y.dim = (int*)malloc(2 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.dim[0] = batch; Y.dim[1] = 7;
	Y.totaldim = 2;
	Y.name = 'Y';
	setzeropointone(X.data, 0, batch, 28, 28, 3);
	for (i = 0; i < batch; i++) {
		for (j = 0; j < 7; j++) {
			Y.data[0][0][i][j] = 0;
		}
	}
	for (i = 0; i < batch; i++) {
		temp = i % 7;
		switch (temp) {
		case 0:
			square(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][0] = 1;
			break;
		case 1:
			square(X.data, i, i + 1, 28, 28, 1);
			Y.data[0][0][i][1] = 1;
			break;
		case 2:
			square(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][2] = 1;
			break;
		case 3:
			square(X.data, i, i + 1, 28, 28, 0);
			square(X.data, i, i + 1, 28, 28, 1);
			Y.data[0][0][i][3] = 1;
			break;
		case 4:
			square(X.data, i, i + 1, 28, 28, 0);
			square(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][4] = 1;
			break;
		case 5:
			square(X.data, i, i + 1, 28, 28, 1);
			square(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][5] = 1;
			break;
		case 6:
			square(X.data, i, i + 1, 28, 28, 0);
			square(X.data, i, i + 1, 28, 28, 1);
			square(X.data, i, i + 1, 28, 28, 2);
			Y.data[0][0][i][6] = 1;
		}
	}
	Addnoise(X.data, X.dim);
	return OK;
}

Status Initdata_v4(dataset& X, dataset& Y, int batch)
{
	int temp = 0;
	int i, j;
	X.data = createtensor(batch, 28, 28, 1);
	X.name = 'X';
	X.totaldim = 4;
	X.dim = (int*)malloc(4 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.dim[0] = batch; X.dim[1] = 28; X.dim[2] = 28; X.dim[3] = 1;
	Y.data = createtensor(1, 1, batch, 5);
	Y.name = 'Y';
	Y.totaldim = 2;
	Y.dim = (int*)malloc(2 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.dim[0] = batch; Y.dim[1] = 5;
	setzeropointone(X.data, 0, batch, 28, 28, 1);
	for (i = 0; i < batch; i++) {
		for (j = 0; j < 5; j++) {
			Y.data[0][0][i][j] = 0;
		}
	}
	for (i = 0; i < batch; i++) {
		temp = i % 10;
		switch (temp) {
		case 0:
			square(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][0] = 1;
			break;
		case 1:
			square_v2(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][0] = 1;
			break;
		case 2:
		case 3:
			circular(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][1] = 1;
			break;
		case 4:
			triangle(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][2] = 1;
			break;
		case 5:
			triangle_v2(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][2] = 1;
			break;
		case 6:
		case 7:
			parallelogram(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][3] = 1;
			break;
		case 8:
		case 9:
			rectangle(X.data, i, i + 1, 28, 28, 0);
			Y.data[0][0][i][4] = 1;
			break;
		}
	}
	Addnoise(X.data, X.dim);
	return OK;
}

Status Addnoise(double*** X, int T_x, int n_x, int m)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < T_x; i++) {
		for (int j = 0; j < n_x; j++) {
			for (int k = 0; k < m; k++) {
				X[i][j][k] += (double)((double)(rand() % MAX) / MAX * 0.2 - 0.1);
			}
		}
	}
	return OK;
}

Status setzeropointone(double*** X, int T_x, int n_x, int m)
{
	for (int i = 0; i < T_x; i++) {
		for (int j = 0; j < n_x; j++) {
			for (int k = 0; k < m; k++) {
				X[i][j][k] = 0.1;
			}
		}
	}
	return OK;
}

Status sin_v1(double*** data, int T_x, int n_x, int start, int end)
{
	double temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = (sin(PI * j / 10) + 1) * (n_x - 1) / 2;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
	}
	return OK;
}

Status sin_v2(double*** data, int T_x, int n_x, int start, int end)
{
	double temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = (sin(PI * j / 15) + 1) * (n_x - 1) / 2;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
	}
	return OK;
}

Status cos_v1(double*** data, int T_x, int n_x, int start, int end)
{
	double temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = (cos(PI * j / 10) + 1) * (n_x - 1) / 2;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
	}
	return OK;
}

Status cos_v2(double*** data, int T_x, int n_x, int start, int end)
{
	double temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = (cos(PI * j / 15) + 1) * (n_x - 1) / 2;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
	}
	return OK;
}

Status triangular_wave(double*** data, int T_x, int n_x, int start, int end)
{
	int i, j;
	double temp = 0;
	for (i = start; i < end; i++) {
		for (j = 0; j < T_x / 3; j++) {
			temp = n_x - 1 - 3.0 * j * (n_x - 1) / T_x;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
		for (j = T_x / 3; j < 2 * T_x / 3; j++) {
			temp = 1 - n_x + 3.0 * j * (n_x - 1) / T_x;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
		for (j = 2 * T_x / 3; j < T_x; j++) {
			temp = 3 * n_x - 1 - 3.0 * j * (n_x - 1) / T_x;
			temp = 0.0001 > temp ? 0.0001 : temp;
			temp = n_x - 1 < temp ? n_x - 1 : temp;
			data[j][(int)temp][i] = 0.9;
		}
	}
	return OK;
}

Status square_wave(double*** data, int T_x, int n_x, int start, int end)
{
	int temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = j / (T_x / 6);
			if (temp % 2) {
				data[j][n_x / 6][i] = 0.9;
			}
			else {
				data[j][5 * n_x / 6][i] = 0.9;
			}
		}
	}
	return OK;
}

Status sawtooch_wave(double*** data, int T_x, int n_x, int start, int end)
{
	int temp = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < T_x; j++) {
			temp = (3 * n_x - 1 - j * (3 * n_x - 1) / (T_x - 1)) % n_x;
			data[j][temp][i] = 0.9;
		}
	}
	return OK;
}

Status Initdata_v5(dataset& X, dataset& Y, int batch)
{
	X.dim = (int*)malloc(3 * sizeof(int));
	if (!X.dim)exit(OVERFLOW);
	X.dim[0] = 30;
	X.dim[1] = 30;
	X.dim[2] = batch;
	X.totaldim = 3;
	X.name = 'X';
	Y.dim = (int*)malloc(3 * sizeof(int));
	if (!Y.dim)exit(OVERFLOW);
	Y.totaldim = 3;
	Y.name = 'Y';
	Y.dim[0] = 1;
	Y.dim[1] = 7;
	Y.dim[2] = batch;
	X.data = createtensor(1, 30, 30, batch);
	setzeropointone(X.data[0], 30, 30, batch);
	Y.data = createtensor(1, 30, 7, batch);
	zerotensor(Y.data[0], 30, 7, batch);
	int temp = 0;
	for (int i = 0; i < batch; i++) {
		temp = i % 7;
		Y.data[0][29][temp][i] = 1;
		switch (temp) {
		case 0:
			sin_v1(X.data[0], 30, 30, i, i + 1);
			break;
		case 1:
			sin_v2(X.data[0], 30, 30, i, i + 1);
			break;
		case 2:
			cos_v1(X.data[0], 30, 30, i, i + 1);
			break;
		case 3:
			cos_v2(X.data[0], 30, 30, i, i + 1);
			break;
		case 4:
			triangular_wave(X.data[0], 30, 30, i, i + 1);
			break;
		case 5:
			square_wave(X.data[0], 30, 30, i, i + 1);
			break;
		case 6:
			sawtooch_wave(X.data[0], 30, 30, i, i + 1);
			break;
		}
	}
	Addnoise(X.data[0], 30, 30, batch);
	return OK;
}

Status numseq(double*** data, int T_x, int n_x, int start, int end, int firstnum)
{
	firstnum %= n_x;
	int temp = firstnum;
	for (int i = start; i < end; i++) {
		temp = firstnum % n_x;
		for (int j = 0; j < T_x; j++) {
			data[j][temp][i] = 1;
			temp = (temp + 1) % n_x;
		}
	}
	return OK;
}

Status Initdata_v6(dataset& X, dataset& Y)
{
	X.totaldim = 3;
	X.dim = (int*)malloc(3 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.name = 'X';
	X.data = createtensor(1, 50, 30, 50);
	Y.totaldim = 3;
	Y.dim = (int*)malloc(3 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.name = 'Y';
	Y.data = createtensor(1, 50, 30, 50);
	zerotensor(X.data[0], 50, 30, 50);
	zerotensor(Y.data[0], 50, 30, 50);
	for (int i = 0; i < 50; i++) {
		numseq(X.data[0], 50, 30, i, i + 1, i);
		numseq(Y.data[0], 50, 30, i, i + 1, i + 1);
	}
	return OK;
}

Status word2vec(char word, int& num)
{
	if (word == '#') {
		num = 0;
	}
	else if (word == ' ') {
		num = 27;
	}
	else if (word == '\'') {
		num = 28;
	}
	else if (word == '-') {
		num = 29;
	}
	else if (word == ',') {
		num = 30;
	}
	else if (word == ':') {
		num = 31;
	}
	else if (word == ';') {
		num = 32;
	}
	else if (word == '\n') {
		num = 33;
	}
	else if (word == '.') {
		num = 34;
	}
	else if (word == '?') {
		num = 35;
	}
	else if (word >= 'A' && word <= 'Z') {
		num = word - 'A' + 1;
	}
	else if (word >= 'a' && word <= 'z') {
		num = word - 'a' + 1;
	}
	else {
		num = 36;
	}
	return OK;
}

Status vec2word(int num, char& word)
{
	if (num == 0) {
		word = '#';
	}
	else if (num == 27) {
		word = ' ';
	}
	else if (num == 28) {
		word = '\'';
	}
	else if (num == 29) {
		word = '-';
	}
	else if (num == 30) {
		word = ',';
	}
	else if (num == 31) {
		word = ':';
	}
	else if (num == 32) {
		word = ';';
	}
	else if (num == 33) {
		word = '\n';
	}
	else if (num == 34) {
		word = '.';
	}
	else if (num == 35) {
		word = '?';
	}
	else if (num >= 1 && num <= 26) {
		word = num + 'a' - 1;
	}
	else {
		word = '@';
	}
	return OK;
}

Status Initdata_v7(dataset& X, dataset& Y)
{
	int i, j, k;
	int id;
	char ch;
	int char_num = 37;
	int datalength = 12367;
	FILE* fp;
	char* a = (char*)malloc((datalength + 1) * sizeof(char));
	if (!a)exit(OVERFLOW);
	memset(a, '\0', (datalength + 1) * sizeof(char));

	// vs写法
//	if (fopen_s(&fp, "D:/data1.txt", "r")) {
//		printf("can't open data1.txt\n");
//		free(a);
//		exit(ERROR);
//	}
//	fread_s(a, sizeof(char) * datalength, sizeof(char), datalength, fp);

	// 其他写法
	if (!(fp = fopen("D:/data1.txt", "r"))) {
		printf("can't open data1.txt\n");
		free(a);
		exit(ERROR);
	}
	fread(a, sizeof(char), datalength, fp);

	fclose(fp);

	int length = strlen(a);
	int maxlength = 0, totallength = 0, curlength = 0;
	int count = 0, temp, t = 0;
	for (i = 0; i < length; i++) {
		if (a[i] >= 'A' && a[i] <= 'Z') {
			a[i] += 'a' - 'A';
		}
		if (a[i] == '#') {
			totallength++;
		}
	}
	for (i = 0; i < length; i++) {
		if (a[i] == '#') {
			if (curlength > maxlength) {
				maxlength = curlength;
			}
			curlength = 0;
		}
		else {
			curlength++;
		}
	}
	maxlength++;
	X.totaldim = 3;
	X.name = 'X';
	X.dim = (int*)malloc(3 * sizeof(int));
	if (X.dim == NULL)exit(OVERFLOW);
	X.dim[0] = maxlength + 3; X.dim[1] = char_num; X.dim[2] = totallength;
	X.data = createtensor(1, maxlength + 3, char_num, totallength);
	Y.totaldim = 3;
	Y.name = 'Y';
	Y.dim = (int*)malloc(3 * sizeof(int));
	if (Y.dim == NULL)exit(OVERFLOW);
	Y.dim[0] = maxlength + 3; Y.dim[1] = char_num; Y.dim[2] = totallength;
	Y.data = createtensor(1, maxlength + 3, char_num, totallength);
	zerotensor(X.data[0], maxlength + 3, char_num, totallength);
	zerotensor(Y.data[0], maxlength + 3, char_num, totallength);
	for (i = 0; i < length; i++) {
		word2vec(a[i], temp);
		X.data[0][t][temp][count] = 1;
		Y.data[0][t + 3][temp][count] = 1;
		if (a[i] != '#') {
			t++;
		}
		else {
			t = 0;
			count++;
		}
	}
	for (i = 0; i < X.dim[2]; i++) {
		for (j = 0; j < X.dim[0]; j++) {
			id = 0;
			for (k = 0; k < X.dim[1]; k++) {
				if (X.data[0][j][k][i] > 0.5) {
					id = k;
				}
			}
			vec2word(id, ch);
			if (ch == '#') {
				printf("\n");
				break;
			}
			printf("%c", ch);
		}
		printf("\n");
	}
	return OK;
}

double* Ex(double** data, int* dim)
{
	double* result = (double*)malloc(dim[1] * sizeof(double));
	for (int i = 0; i < dim[1]; i++) {
		result[i] = 0;
		for (int j = 0; j < dim[0]; j++) {
			result[i] += data[j][i];
		}
		result[i] /= dim[0];
	}
	return result;
}

double* Ex(double**** data, int* dim)
{
	double** Ldata = Flatten(data, dim);
	int Ldim[2] = { dim[0], dim[1] * dim[2] * dim[3] };
	double* result = Ex(Ldata, Ldim);
	return result;
}

double* Var(double** data, int* dim, double* Ex)
{
	double* result = (double*)malloc(dim[1] * sizeof(double));
	for (int i = 0; i < dim[1]; i++) {
		result[i] = 0;
		for (int j = 0; j < dim[0]; j++) {
			result[i] += pow((data[j][i] - Ex[i]), 2.0);
		}
		result[i] /= dim[0];
	}
	return result;
}

Status datanormalized(dataset& dataset)
{
	if (dataset.totaldim == 2) {
		double* ex = Ex(dataset.data[0][0], dataset.dim);
		double* var = Var(dataset.data[0][0], dataset.dim, ex);
		for (int i = 0; i < dataset.dim[0]; i++) {
			for (int j = 0; j < dataset.dim[1]; j++) {
				dataset.data[0][0][i][j] -= ex[j];
				dataset.data[0][0][i][j] /= sqrt(var[j]);
			}
		}
		return OK;
	}
	else if (dataset.totaldim == 4) {
		double* ex = Ex(dataset.data, dataset.dim);
		double** Ldata = Flatten(dataset.data, dataset.dim);
		int Ldim[2] = { dataset.dim[0], dataset.dim[1] * dataset.dim[2] * dataset.dim[3] };
		double* var = Var(Ldata, Ldim, ex);
		int temp;
		for (int i = 0; i < dataset.dim[0]; i++) {
			for (int j = 0; j < dataset.dim[1]; j++) {
				for (int k = 0; k < dataset.dim[2]; k++) {
					for (int n = 0; n < dataset.dim[3]; n++) {
						temp = n + k * dataset.dim[3] + j * dataset.dim[3] * dataset.dim[2];
						dataset.data[i][j][k][n] -= ex[temp];
						dataset.data[i][j][k][n] /= sqrt(var[temp]);
					}
				}
			}
		}
		return OK;
	}
	else {
		printf("the dimention of dataset is worry");
		exit(INFEASIBLE);
	}
}

Status modelfit(dataset X, dataset Y, model& model, double lr, int epochs, int minibatch, int acc, int feq,
	dataset X_t, dataset Y_t, int earlystop, double attenuation)
{
	int tempdim[4] = { 1,1, minibatch, Y.dim[1] };
	int dim[4] = { 1, 1, X_t.dim[0], Y_t.dim[1] };
	double test_acc = 0.0;
	double last_acc = 0.0;
	double test_cost = 0.0;
	int mark = 0;
	double keep_drop = model.keep_drop;
	if (Y.totaldim == 4) {
		tempdim[0] = minibatch; tempdim[1] = Y.dim[1]; tempdim[2] = Y.dim[2]; tempdim[3] = Y.dim[3];
		dim[0] = Y_t.dim[0]; dim[1] = Y_t.dim[1]; dim[2] = Y_t.dim[2]; dim[3] = Y_t.dim[3];
		acc = 0;
	}
	parameter_v v;
	parameter_s s;
	switch (model.optimizer) {
	case 'A':
		Initparameter(model, v);
		Initparameter(model, s);
		break;
	case 'm':
		Initparameter(model, v);
		break;
	}
	char op = model.optimizer;
	char loss = model.loss;
	double(*crossfunction)(double**** predict, double**** Y, int* size, int batch);
	switch (loss) {
	case 'c':
		if (model.layer[model.depth - 1].activation == 'x') {
			crossfunction = crossentropy;
		}
		else {
			crossfunction = crossentropy_v2;
		}
		break;
	case 's':crossfunction = squaredif; break;
	default:printf("your way of compile is worry"); exit(INFEASIBLE);
	}
	double**** miniX;
	double**** miniY;
	for (int i = 0; i < epochs; i++) {
		srand((unsigned)time(NULL));
		int start = (int)(rand() % (X.dim[0] - minibatch + 1));
		if (X.totaldim == 2) {
			miniX = minibatchformatrix(X.data, X.dim[0], X.dim[1], minibatch, start);
		}
		else if (X.totaldim == 4) {
			miniX = minibatchfortensor(X.data, X.dim, minibatch, start);
		}
		else {
			printf("the totaldim is worry");
			exit(INFEASIBLE);
		}
		if (Y.totaldim == 2) {
			miniY = minibatchformatrix(Y.data, Y.dim[0], Y.dim[1], minibatch, start);
		}
		else if (Y.totaldim == 4) {
			miniY = minibatchfortensor(Y.data, Y.dim, minibatch, start);
		}
		else {
			printf("the totaldim is worry");
			exit(INFEASIBLE);
		}
		if (i % feq == 0) {
			if (acc) {
				last_acc = test_acc;
			}
			model.keep_drop = 1.0;
			modelforward(miniX, model, minibatch, X.dim[1], X.dim[2], X.dim[3]);
			printf("\nepoch %d: train_cost = %f ", i, crossfunction(model.caches[model.depth - 1], miniY, tempdim, minibatch));
			if (acc) {
				printf("train_acc = %f ", accuracy(miniY[0][0], model.caches[model.depth - 1][0][0], minibatch, Y.dim[1]));
			}
			Clearcaches(model);
			modelforward(X_t.data, model, X_t.dim[0], X_t.dim[1], X_t.dim[2], X_t.dim[3]);
			test_cost = crossfunction(model.caches[model.depth - 1], Y_t.data, dim, Y_t.dim[0]);
			printf("test_cost = %f ", test_cost);
			if (acc) {
				test_acc = accuracy(Y_t.data[0][0], model.caches[model.depth - 1][0][0], Y_t.dim[0], Y_t.dim[1]);
				printf("test_acc = %f", test_acc);

				//								for(int k = 0; k < Y.dim[1]; k++){  //用于查看变化 
				//									printf("\n");
				//									for(int j = 0; j < Y.dim[1]; j++){
				//										printf("%.3f ", model.caches[model.depth - 1][0][0][k][j]);
				//									}
				//									printf("\n");
				//									for(int j = 0; j < Y.dim[1]; j++){
				//										printf("%.3f ", Y_t.data[0][0][k][j]);
				//									}
				//									printf("\n");
				//								}
				//								printf("\n");

				if (test_acc < last_acc) {
					mark++;
				}
				else {
					mark = 0;
				}
				if (mark >= earlystop || fabs(test_acc - 1.0) < 0.001) {
					printf("\nEarly stop");
					return OK;
				}
			}
			printf("\n");
			Clearcaches(model);
		}
		model.keep_drop = keep_drop;
		printf("->");
		switch (op) {
		case 'A':
			updatewithAdam(miniX, miniY, model, lr, minibatch, X.dim[1], X.dim[2], X.dim[3], i + 1, v, s);
			break;
		case 'm':
			updatewithmomentum(miniX, miniY, model, lr, minibatch, X.dim[1], X.dim[2], X.dim[3], v);
			break;
		case 'g':
			updatewithgradient(miniX, miniY, model, lr, minibatch, X.dim[1], X.dim[2], X.dim[3]);
			break;
		default:
			printf("the optimizer is worry");
			exit(INFEASIBLE);
		}
		// lr = lr * attenuation;
		Clearcaches(model);
		Cleardcaches(model);
		printf("\b->");
		if (X.totaldim == 2) {
			Freetensor(miniX, 1, 1, minibatch);
		}
		else if (X.totaldim == 4) {
			Freetensor(miniX, minibatch, X.dim[1], X.dim[2]);
		}
		else {
			printf("the totaldim is worry");
			exit(INFEASIBLE);
		}
		if (Y.totaldim == 2) {
			Freetensor(miniY, 1, 1, minibatch);
		}
		else if (Y.totaldim == 4) {
			Freetensor(miniY, minibatch, Y.dim[1], Y.dim[2]);
		}
		else {
			printf("the totaldim is worry");
			exit(INFEASIBLE);
		}
	}
	model.keep_drop = 1.0;
	modelforward(X_t.data, model, X_t.dim[0], X_t.dim[1], X_t.dim[2], X_t.dim[3]);
	printf("\ncost = %f\n", crossfunction(model.caches[model.depth - 1], Y_t.data, dim, Y_t.dim[0]));
	if (acc) {
		printf("accuracy = %f\n", accuracy(Y_t.data[0][0], model.caches[model.depth - 1][0][0], Y_t.dim[0], Y_t.dim[1]));
	}
	return OK;
}

int main(void)
{
	int i;
	int batch = 5000;
	int testbatch = 1000;
	int feature = 400;
	int classes = 10;
	int h = 14;
	int w = 14;
	int d = 3;
	int epochs = 3;
	double lr = 0.1;
	model mymodel;
	Initmodel(mymodel);
	dataset inputs;
	dataset outputs;
	dataset X_test, Y_test;


	//单层感知器
	/*
	maxdistance = 2;
	batch = 7000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 5;
	epochs = 50;
	lr = 0.0002;
	Initdata_v4(inputs, outputs, batch);
	Initdata_v4(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int dim[2] = { h * w * d, classes };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 5);
	modelsummary(mymodel);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 5, X_test, Y_test);
	*/

	//两层神经网络
	/*
	maxdistance = 2;
	batch = 7000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 5;
	epochs = 100;
	lr = 0.0005;
	Initdata_v4(inputs, outputs, batch);
	Initdata_v4(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int dim[2] = { h * w * d, 16 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r', 'k');
	dim[0] = 16; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelcompile(mymodel, 'A', 'c', 0.7, 0, 5);
	modelsummary(mymodel);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 10, X_test, Y_test);
	*/

	//简单的CNN
	/*
	maxdistance = 2;
	batch = 7000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 5;
	epochs = 40;
	lr = 0.0002;
	Initdata_v4(inputs, outputs, batch);
	Initdata_v4(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int kdim[4] = { 3,3,d,2 };
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	Addlayer(mymodel, 'p', kdim, 2, 2, 'm', 'r');
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim);
	dim[0] = h * w * 2 / 4; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 4, X_test, Y_test, 100, 0.95);
	*/

	//自编码器(好像有点难训练)
	/*
	maxdistance = 0;
	batch = 5000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 7;
	epochs = 300;
	lr = 0.0002;
	Initdata_v1(inputs, outputs, batch);
	outputs.dim[0] = batch;
	outputs.dim[1] = h * w * d;
	Freematrix(outputs.data[0][0], batch);
	outputs.data[0][0] = Flatten(inputs.data, inputs.dim);
	Initdata_v1(X_test, Y_test, testbatch);
	Y_test.dim[0] = testbatch;
	Y_test.dim[1] = h * w * d;
	Freematrix(Y_test.data[0][0], testbatch);
	Y_test.data[0][0] = Flatten(X_test.data, X_test.dim);
	int Xdim[4] = { batch, h, w, d };
	outputs.data[0][0] = Flatten(inputs.data, Xdim);
	int dim[2] = { h * w * d, 128 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = 128, dim[1] = 64;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = 64, dim[1] = 32;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = 32, dim[1] = 64;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = 64; dim[1] = 128;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = 128, dim[1] = h * w * d;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 's');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 's', 1.0, 0.0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 7, 0, 25, X_test, Y_test);
	for (int m = 0; m < 14; m++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (Y_test.data[0][0][m][i * w + j] >= 0.8) {
					printf("77");
				}
				else if (Y_test.data[0][0][m][i * w + j] >= 0.7) {
					printf("11");
				}
				else if (Y_test.data[0][0][m][i * w + j] >= 0.6) {
					printf("7 ");
				}
				else if (Y_test.data[0][0][m][i * w + j] >= 0.5) {
					printf("1 ");
				}
				else {
					printf("0 ");
				}
			}
			printf("\n");
		}
		printf("\n");
		for (int i = 0; i < 32; i++) {
			printf("%.1f ", mymodel.caches[3][0][0][m][i]);
		}
		printf("\n\n");
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.8) {
					printf("77");
				}
				else if (mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.7) {
					printf("11");
				}
				else if (mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.6) {
					printf("7 ");
				}
				else if (mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.5) {
					printf("1 ");
				}
				else {
					printf("0 ");
				}
			}
			printf("\n");
		}
		printf("\n");
	}
	*/

	// 双层的比较容易收敛 
	/*
	maxdistance = 0;
	batch = 5000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 7;
	epochs = 500;
	lr = 0.0002;
	Initdata_v1(inputs, outputs, batch);
	outputs.dim[0] = batch;
	outputs.dim[1] = h * w * d;
	Freematrix(outputs.data[0][0], batch);
	outputs.data[0][0] = Flatten(inputs.data, inputs.dim);
	Initdata_v1(X_test, Y_test, testbatch);
	Y_test.dim[0] = testbatch;
	Y_test.dim[1] = h * w * d;
	Freematrix(Y_test.data[0][0], testbatch);
	Y_test.data[0][0] = Flatten(X_test.data, X_test.dim);
	int Xdim[4] = { batch, h, w, d };
	outputs.data[0][0] = Flatten(inputs.data, Xdim);
	int dim[2] = { h * w * d, 32 };
	Addlayer(mymodel, 'l', dim);
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'r');
	dim[0] = dim[1]; dim[1] = h * w * d;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 's');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 's', 1.0, 0.0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 7, 0, 25, X_test, Y_test);
	for (int m = 0; m < 14; m++) {
		for (i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (Y_test.data[0][0][m][i * w + j] >= 0.8) {
					printf("77");
				}
				else if(Y_test.data[0][0][m][i * w + j] >= 0.7){
					printf("11");
				}
				else if(Y_test.data[0][0][m][i * w + j] >= 0.6){
					printf("7 ");
				}
				else if(Y_test.data[0][0][m][i * w + j] >= 0.5){
					printf("1 ");
				}
				else{
					printf("0 ");
				}
			}
			printf("\n");
		}
		printf("\n");
		for (i = 0; i < 32; i++) {
			printf("%.1f ", mymodel.caches[mymodel.depth - 2][0][0][m][i]);
		}
		printf("\n\n");
		for (i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.8) {
					printf("77");
				}
				else if(mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.7){
					printf("11");
				}
				else if(mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.6){
					printf("7 ");
				}
				else if(mymodel.caches[mymodel.depth - 1][0][0][m][i * w + j] >= 0.5){
					printf("1 ");
				}
				else{
					printf("0 ");
				}
			}
			printf("\n");
		}
		printf("\n");
	}
	*/

	//一种图形七种颜色分类 
	/*
	maxdistance = 8;
	batch = 7000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 3;
	classes = 7;
	epochs = 100;
	lr = 0.0002;
	Initdata_v3(inputs, outputs, batch);
	Initdata_v3(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	dim[0] = h * w * d; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 10, X_test, Y_test);
	*/

	//三种图形的分别三种颜色分类（共九类）
	/*
	maxdistance = 8;
	batch = 9000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 3;
	classes = 9;
	epochs = 100;
	lr = 0.0001;
	Initdata_v2(inputs, outputs, batch);
	Initdata_v2(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	dim[0] = h * w * d; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 10, X_test, Y_test, 100, 0.96);
	*/

	//七种图形分类
	/*
	maxdistance = 8;
	batch = 7000;
	testbatch = 100;
	h = 28;
	w = 28;
	d = 1;
	classes = 7;
	epochs = 2000;
	lr = 0.0002;
	Initdata_v1(inputs, outputs, batch);
	Initdata_v1(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim);
	dim[0] = h * w * d; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 5);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 100, X_test, Y_test, 100, 0.99);
	*/

	// 采用卷积神经网络
	
	maxdistance = 8;
	batch = 7000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 1;
	classes = 7;
	epochs = 200;
	lr = 0.0001;
	Initdata_v1(inputs, outputs, batch);
	Initdata_v1(X_test, Y_test, testbatch);
	datanormalized(inputs);
	datanormalized(X_test);
	int kdim[4] = { 3,3,d,4 };
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	kdim[2] = kdim[3]; kdim[3] = 8;
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	Addlayer(mymodel, 'p', kdim, 2, 2, 'm', 'r');
	kdim[2] = kdim[3]; kdim[3] = 16;
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	Addlayer(mymodel, 'p', kdim, 2, 2, 'm', 'r');
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim);
	dim[0] = h * w * 16 / 4 / 4; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.5, 0, 10);
	modelfit(inputs, outputs, mymodel, lr, epochs, 64, 1, 5, X_test, Y_test, 100, 0.99);
	

	//迁移学习的一种简单示例 

	//(预训练)
	/*
	maxdistance = 8;
	batch = 9000;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 3;
	classes = 9;
	epochs = 15;
	lr = 0.0003;
	Initdata_v2(inputs, outputs, batch);
	Initdata_v2(X_test, Y_test, testbatch);
	int kdim[4] = { 3,3,d,6 };
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	Addlayer(mymodel, 'p', kdim, 2, 2, 'm', 'r');
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	dim[0] = h * w * 6 / 4; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.7, 0, 10);
	modelfit(inputs, outputs, mymodel, lr, epochs, 128, 1, 3, X_test, Y_test);
	Reducelayer(mymodel);
	Reducelayer(mymodel);
	Freetensor(inputs.data, inputs.dim[0], inputs.dim[1], inputs.dim[2]);
	free(inputs.dim);
	Freetensor(X_test.data, X_test.dim[0], X_test.dim[1], X_test.dim[2]);
	free(X_test.dim);
	Freetensor(outputs.data, 1, 1, outputs.dim[0]);
	free(outputs.dim);
	Freetensor(Y_test.data, 1, 1, Y_test.dim[0]);
	free(Y_test.dim);
	batch = 200;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 3;
	classes = 7;
	epochs = 200;
	lr = 0.0003;
	Initdata_v3(inputs, outputs, batch);
	Initdata_v3(X_test, Y_test, testbatch);
	modelforward(inputs.data, mymodel, batch, h, w, d);
	Freetensor(inputs.data, inputs.dim[0], inputs.dim[1], inputs.dim[2]);
	inputs.data = copytensor(mymodel.caches[mymodel.depth - 1], batch, 14, 14, 6);
	Clearcaches(mymodel);
	inputs.dim[1] = inputs.dim[2] = 14;
	inputs.dim[3] = 6;
	modelforward(X_test.data, mymodel, testbatch, h, w, d);
	Freetensor(X_test.data, X_test.dim[0], X_test.dim[1], X_test.dim[2]);
	X_test.data = copytensor(mymodel.caches[mymodel.depth - 1], testbatch, 14, 14, 6);
	Clearcaches(mymodel);
	X_test.dim[1] = X_test.dim[2] = 14;
	X_test.dim[3] = 6;
	model newmodel;
	Initmodel(newmodel);
	dim[0] = h * w * 6 / 4; dim[1] = classes;
	Addlayer(newmodel, 'l', dim);
	Addlayer(newmodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(newmodel);
	modelcompile(newmodel, 'A', 'c', 0.7, 0, 10);
	modelfit(inputs, outputs, newmodel, lr, epochs, 128, 1, 10, X_test, Y_test);
	*/

	// 效果对比
	/*
	maxdistance = 8;
	batch = 200;
	testbatch = 1000;
	h = 28;
	w = 28;
	d = 3;
	classes = 7;
	epochs = 30;
	lr = 0.0003;
	Initdata_v3(inputs, outputs, batch);
	Initdata_v3(X_test, Y_test, testbatch);
	int kdim[4] = { 3,3,d,6 };
	Addlayer(mymodel, 'c', kdim, 3, 1, '0', 'n');
	Addlayer(mymodel, 'p', kdim, 2, 2, 'm', 'r');
	int dim[2] = { 0,0 };
	Addlayer(mymodel, 'l', dim, 0, 0, '0', '0');
	dim[0] = h * w * 6 / 4; dim[1] = classes;
	Addlayer(mymodel, 'f', dim, 0, 0, '0', 'x');
	modelsummary(mymodel);
	modelcompile(mymodel, 'A', 'c', 0.7, 0, 10);
	modelfit(inputs, outputs, mymodel, lr, epochs, 128, 1, 5, X_test, Y_test);
	*/

	/*
	int T_x = 30;
	int n_x = 30;
	int n_a = 50;
	int n_y = 7;
	int m = 2000;
	int testbatch = 700;
	int minibatch = 64;
	int epochs = 200;
	int feq = 10;
	double lr = 0.01;
	dataset X;
	dataset Y;
	dataset X_val;
	dataset Y_val;
	single_rnn_layer mylayer;
	*/

	/*
	Initdata_v5(X, Y, m);
	Initdata_v5(X_val, Y_val, testbatch);
	Initrnnlayer(mylayer, m, n_x, n_a, T_x, n_y, 0, T_x, T_x - 1, T_x, 0, T_x, 0, 100);
	rnnlayerfit(mylayer, X, Y, X_val, Y_val, lr, epochs, testbatch, minibatch, 1, feq, 100, 0.02);
	*/

	/*
	T_x = 50;
	n_x = 30;
	n_a = 50;
	n_y = n_x;
	m = 50;
	testbatch = 50;
	minibatch = 30;
	epochs = 200;
	feq = 5;
	lr = 0.005;
	Initdata_v6(X, Y);
	Initdata_v6(X_val, Y_val);
	Initrnnlayer(mylayer, m, n_x, n_a, T_x, n_y, 0, T_x, T_x - 1, T_x, 0, T_x, 0, 100);
	rnnlayerfit(mylayer, X, Y, X_val, Y_val, lr, epochs, testbatch, minibatch, 1, feq, 100, 0.02);
	*/

	return OK;
}
