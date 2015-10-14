#python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
ctypedef np.int_t DTYPE_t


def frac_neighbours(np.ndarray[np.int8_t, ndim=1] gameState, np.ndarray[np.uint8_t, ndim=1] n_mask):
	assert gameState.shape[0] == n_mask.shape[0]

	cdef int imax = gameState.shape[0]
	cdef int n_coop = 0
	cdef int n = 0
	cdef int idx
	for idx in range(imax):
		if n_mask[idx] == 1:
			n_coop += (gameState[idx]==1)
			n += 1

	return float(n_coop)/float(n)

def mean_neighbours(np.ndarray[np.int16_t, ndim=1] state, np.ndarray[np.uint8_t, ndim=1] n_mask):
	assert state.shape[0] == n_mask.shape[0]

	cdef int imax = state.shape[0]
	cdef int count = 0
	cdef int n = 0
	cdef int idx
	for idx in range(imax):
		if n_mask[idx] == 1:
			count += state[idx]
			n += 1

	return float(count)/n

def sum_mask_nn(np.ndarray[np.int8_t, ndim=1] gameState, np.ndarray[np.int8_t, ndim=1] n_mask):
	assert gameState.shape[0] == n_mask.shape[0]

	cdef int imax = gameState.shape[0]
	cdef float n_help = 0
	cdef int idx
	cdef int count = 0
	for idx in range(imax):
		if n_mask[idx] != 0 and gameState[idx] > 0:
			n_help += n_mask[idx]
			count += 1

	return n_help,count

def count_neighbours(np.ndarray[np.int8_t, ndim=1] gameState, np.ndarray[np.uint8_t, ndim=1] n_mask):
	assert gameState.shape[0] == n_mask.shape[0]

	cdef int imax = gameState.shape[0]
	cdef int n_coop = 0
	cdef int n = 0
	cdef int idx
	for idx in range(imax):
		if n_mask[idx] == 1:
			n_coop += (gameState[idx]==1)
			n += 1

	return float(n_coop)