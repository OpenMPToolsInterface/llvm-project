#include <Python.h>
#include <omp-tools.h>
// #include <ompd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

extern void* ompd_library;

struct _ompd_aspace_cont {int id;};
struct _ompd_thread_cont {int id;};
ompd_address_space_context_t context = {42};
ompd_address_space_context_t invalidcontext = {99};

//call back functions for ompd_initialize
ompd_rc_t _alloc(ompd_size_t bytes, void** ptr);
ompd_rc_t _free(void* ptr);
ompd_rc_t _sizes(ompd_address_space_context_t * _acontext, ompd_device_type_sizes_t * sizes);
ompd_rc_t _sym_addr(ompd_address_space_context_t *context, ompd_thread_context_t *tcontext, const char* symbol_name, ompd_address_t *symbol_addr, const char* file_name);
ompd_rc_t _read (ompd_address_space_context_t *context, ompd_thread_context_t *tcontext,const ompd_address_t *addr, ompd_size_t nbytes, void* buffer);
ompd_rc_t _read_string (ompd_address_space_context_t *context, ompd_thread_context_t *tcontext, const ompd_address_t *addr, ompd_size_t nbytes, void* buffer);
ompd_rc_t _endianess(ompd_address_space_context_t *address_space_context, const void *input, ompd_size_t unit_size, ompd_size_t count, void *output);
ompd_rc_t _thread_context(ompd_address_space_context_t *context, ompd_thread_id_t kind, ompd_size_t sizeof_thread_id, const void *thread_id,ompd_thread_context_t **thread_context);
ompd_rc_t _print(const char* str, int category);


/*
	Test API: ompd_get_thread_handle

		ompdtestapi threadandparallel

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
		ompd init
		b 7
		c
		ompdtestapi ompd_get_thread_handle

		for ompd_rc_unavailable:
		ompd init
		ompdtestapi ompd_get_thread_handle
*/

PyObject* test_ompd_get_thread_handle(PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_thread_handle\"...\n");

	PyObject* addrSpaceTup = PyTuple_GetItem(args, 0);
	ompd_address_space_handle_t* addr_handle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceTup, "AddressSpace");

	PyObject* threadIdTup = PyTuple_GetItem(args, 1);
	uint64_t threadID = (uint64_t) PyLong_AsLong(threadIdTup);

	ompd_size_t sizeof_thread_id = sizeof (threadID);
	ompd_thread_handle_t* thread_handle;

	// should be successful
	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_thread_handle (addr_handle, 1/*lwp*/, sizeof_thread_id, &threadID, &thread_handle);

	if (rc == ompd_rc_unavailable) {
		// ompd_rc_unavailable if the thread is not an OpenMP thread.
		printf ("Success. ompd_rc_unavailable, OpenMP is disabled.\n");
		printf ("This is not a Parallel Region, No more testing is possible.\n");
		return Py_None;
	} else if (rc != ompd_rc_ok)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");


	// as in ompd-types.h, only 0-3 are valid for thread kind
	// ompd_rc_unsupported if thread kind is not supported.
	printf ("Test: Unsupported thread kind.\n");
	rc = ompd_get_thread_handle (addr_handle, 4 , sizeof_thread_id, &threadID, &thread_handle);
	if (rc != ompd_rc_unsupported)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");


	// ompd_rc_bad_input: if a different value in sizeof_thread_id is expected for a thread kind.
	// sizeof_thread_id is validated at thread_context which is call back function "_thread_context" where we expect size to be sizeof(long int)
	printf ("Test: Wrong value for sizeof threadID.\n");
	rc = ompd_get_thread_handle (addr_handle, 1/*lwp*/,  sizeof_thread_id - 1, &threadID, &thread_handle);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	// Random checks with null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:	is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:		is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#ifdef ABORTHANDLED
	rc = ompd_get_thread_handle (0xdeadbeef, 1/*lwp*/, sizeof_thread_id, &threadID, &thread_handle);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting ompd_rc_bad_input for NULL thread_handle.\n");
	rc = ompd_get_thread_handle (addr_handle, 1/*lwp*/, sizeof_thread_id, &threadID, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or stale_handle for NULL addr_handle.\n");
	rc = ompd_get_thread_handle (NULL, 1/*lwp*/, sizeof_thread_id, &threadID, &thread_handle);
	if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
}

/*
	Test API: ompd_get_curr_parallel_handle.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.			}
		9.			   return 0;
		10.		}

	GDB Commands:
		ompd init
		b 7
		omptestapi ompd_get_curr_parallel_handle

		for ompd_rc_unavailable
		ompd init
		omptestapi ompd_get_curr_parallel_handle (or break at line 4 before this)
*/

PyObject* test_ompd_get_curr_parallel_handle (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_curr_parallel_handle\"...\n");

	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* thread_handle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));

	ompd_parallel_handle_t* parallel_handle;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_curr_parallel_handle(thread_handle, &parallel_handle);
	if (rc == ompd_rc_unavailable) {
		//ompd_rc_unavailable if the thread is not currently part of a team

		// ToCheck: Even in non parallel region, error code is stale_handle
		// Need to find a test case for ompd_rc_unavailable ?????
		printf ("Success. ompd_rc_unavailable, Not in parallel region\n");
		printf ("No more testing is possible.\n");
		return Py_None;
	}else if (rc == ompd_rc_stale_handle) {
		printf ("Failed. stale_handle, may be in non-parallel region.\n");
		printf ("No more testing is possible.\n");
		return Py_None;
	}else if (rc != ompd_rc_ok)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");


	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_get_curr_parallel_handle(0xdeadbeef, &parallel_handle);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting ompd_rc_bad_input for NULL parallel_handle.\n");
	rc = ompd_get_curr_parallel_handle(thread_handle, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or stale_handle for NULL thread_handle.\n");
	rc = ompd_get_curr_parallel_handle(NULL, &parallel_handle);
	if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;

}

/*
		Test API: ompd_get_thread_in_parallel.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(3);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
				ompd init
				b 7
				omptestapi ompd_get_thread_in_parallel
*/
PyObject* test_ompd_get_thread_in_parallel (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_thread_in_parallel\"...\n");

	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallel_handle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	ompd_thread_handle_t* thread_handle;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_thread_in_parallel (parallel_handle, 1 /* lesser than team-size-var*/, &thread_handle);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}else
		printf ("Success.\n");

	//ompd_rc_bad_input: if the thread_num argument is greater than or equal to the team-size-var ICV or negative
	printf ("Test: Invalid thread num (199).\n");
	rc = ompd_get_thread_in_parallel (parallel_handle, 199, &thread_handle);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Invalid thread num (-5).\n");
	rc = ompd_get_thread_in_parallel (parallel_handle, -5, &thread_handle);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_get_thread_in_parallel (0xdeadbeef, 1, &thread_handle);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting ompd_rc_bad_input for NULL thread_handle.\n");
	rc = ompd_get_thread_in_parallel (parallel_handle, 1, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or stale_handle for NULL parallel_handle.\n");
	rc = ompd_get_thread_in_parallel (NULL, 1, &thread_handle);
	if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;

}

/*
		Test API: ompd_thread_handle_compare.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(4);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
				ompd init
				b 7
				omptestapi ompd_thread_handle_compare
*/

PyObject* test_ompd_thread_handle_compare (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_thread_handle_compare\"...\n");

	PyObject* threadHandlePy1 = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* thread_handle1 = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy1, "ThreadHandle"));
	PyObject* threadHandlePy2 = PyTuple_GetItem(args, 1);
	ompd_thread_handle_t* thread_handle2 = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy2, "ThreadHandle"));

	int cmp_value;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_thread_handle_compare (thread_handle1, thread_handle2, &cmp_value);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}
	else
		printf ("Success.\n");

	if (cmp_value == 0) {
		printf ("Threads are Equal.\n");
	} else {
		// a value less than, equal to, or greater than 0 indicates that the thread corresponding to thread_handle_1 is,
		// respectively, less than, equal to, or greater than that corresponding to thread_handle_2.
		if (cmp_value <= 0) {
			printf ("Thread 1 is lesser than thread 2, cmp_val = %d\n", cmp_value);
			printf ("Test: Changing the order.\n");
			rc = ompd_thread_handle_compare (thread_handle2, thread_handle1, &cmp_value);
			if (rc != ompd_rc_ok) {
				printf ("Failed, with return code = %d\n", rc);
				return Py_None;
			}
			if (cmp_value >= 0)
				printf ("Success now cmp_value is greater, %d.\n", cmp_value);
			else
				printf ("Failed.\n");
		} else {
			printf ("Thread 1 is greater than thread 2.\n");
			printf ("Test: Changing the order.\n");
			rc = ompd_thread_handle_compare (thread_handle2, thread_handle1, &cmp_value);
			if (rc != ompd_rc_ok) {
				printf ("Failed, with return code = %d\n", rc);
				return Py_None;
			}
			if (cmp_value <= 0)
				printf ("Success now cmp_value is lesser, %d.\n", cmp_value);
			else
				printf ("Failed.\n");
		}

		// Random checks with  null and invalid args.
		/*
			ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
			ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
			ompd_rc_error:			is returned when a fatal error occurred;
		*/
		printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
		#if ABORTHANDLED
		rc = ompd_thread_handle_compare (thread_handle2, 0xdeadbeef, &cmp_value);
		if (rc != ompd_rc_stale_handle)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
		#else
		printf ("Skipped. Aborted, not handled.\n");
		#endif

		printf ("Test: Expecting ompd_rc_bad_input for NULL cmp_value.\n");
		#if ABORTHANDLED
		rc = ompd_thread_handle_compare (thread_handle2, thread_handle1, NULL);
		if (rc != ompd_rc_bad_input)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
		#else
		printf ("Skipped. Aborted, not handled.\n");
		#endif

		printf ("Test: Expecting ompd_rc_error or stale_handle for NULL thread_handle.\n");
		rc = ompd_thread_handle_compare (NULL, thread_handle1, &cmp_value);
		if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
	}

	return Py_None;
}

/*
		Test API: ompd_get_thread_id.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
				ompd init
				b 7
				omptestapi ompd_get_thread_id
*/

PyObject* test_ompd_get_thread_id (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_thread_id\"...\n");

	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* thread_handle = (ompd_thread_handle_t*) (PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));

	uint64_t threadID;
	ompd_size_t sizeof_thread_id = sizeof (threadID);

	printf ("Test: With Correct Arguments.\n ");
	ompd_rc_t rc = ompd_get_thread_id(thread_handle, 1/*lwp*/, sizeof_thread_id, &threadID);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}else
		printf ("Success. Thread id = %d\n", threadID);

	//ompd_rc_bad_input: if a different value in sizeof_thread_id is expected for a thread kind of kind
	printf ("Test: Wrong sizeof_thread_id.\n");
	rc = ompd_get_thread_id(thread_handle, 1/*lwp*/, sizeof_thread_id-1, &threadID);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	//ompd_rc_unsupported: if the kind of thread is not supported
	printf ("Test: Unsupported thread kind.\n");
	// thread kind currently support from 0-3, refer in ompd-types.h
	rc = ompd_get_thread_id(thread_handle, 4, sizeof_thread_id-1, &threadID);
	if (rc != ompd_rc_unsupported)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	rc = ompd_get_thread_id(0xdeadbeef, 1/*lwp*/, sizeof_thread_id, &threadID);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_bad_input for NULL threadID.\n");
	rc = ompd_get_thread_id(thread_handle, 1/*lwp*/, sizeof_thread_id, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error for NULL thread_handle.\n");
	rc = ompd_get_thread_id(NULL, 1/*lwp*/, sizeof_thread_id, &threadID);
	if (rc != ompd_rc_error)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
}

/*
		Test API: ompd_rel_thread_handle

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
				ompd init
				b 7
				omptestapi ompd_rel_thread_handle
*/

//	TODO: This might not be the right way to do,as this handle comes from python not generated by ompd API

PyObject* test_ompd_rel_thread_handle (PyObject* self, PyObject* args)
{
	printf ("Testing Not enabled for \"ompd_rel_thread_handle\"...\n");
	printf ("Skipping.\n");
	return Py_None;

	#if 0
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* thread_handle = (ompd_thread_handle_t*) (PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));

	printf ("Test: with correct Args.\n");
	ompd_rc_t rc;
	#if ABORTHANDLED
	rc = ompd_rel_thread_handle (thread_handle);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled. Testing function needs to be modified\n");
	return Py_None;
	#endif


	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_rel_thread_handle (0xdeadbeef);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting stale handle or bad_input for NULL thread_handle.\n");
	rc = ompd_rel_thread_handle (NULL);
	if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
	#endif

}

/*
	Test API: ompd_get_enclosing_parallel_handle.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.						omp_set_num_threads(3);
		9.						#pragma omp parallel
		10.						{
		11.								printf ("Parallel level 2, thread num = %d", omp_get_thread_num());
		12.						}
		13.				}
		14.				return 0;
		15.		}

	GDB Commands:
		ompd init
		b 11
		ompdtestapi ompd_get_enclosing_parallel_handle

		for "ompd_rc_unavailable":
				ompd init
				omptestapi ompd_get_enclosing_parallel_handle (or break at line 4 before this)


*/

PyObject* test_ompd_get_enclosing_parallel_handle (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_enclosing_parallel_handle\"...\n");

	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallel_handle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	ompd_parallel_handle_t* enclosing_parallel_handle;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_enclosing_parallel_handle(parallel_handle, &enclosing_parallel_handle);
	if (rc == ompd_rc_unavailable) {
		//ompd_rc_unavailable: if no enclosing parallel region exists.
		printf ("Success. return code is ompd_rc_unavailable, Not in parallel region\n");
		printf ("No more testing is possible.\n");
		return Py_None;
	}else if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}else
		printf ("Success.\n");

	// Random checks with  null and invalid args.
	/*
	   ompd_rc_stale_handle:   is returned when the specified handle is no longer valid;
	   ompd_rc_bad_input:	   is returned when the input parameters (other than handle) are invalid;
	   ompd_rc_error:		   is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_get_enclosing_parallel_handle(0xdeadbeef, &enclosing_parallel_handle);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting ompd_rc_bad_input for NULL enclosing_parallel_handle.\n");
	rc = ompd_get_enclosing_parallel_handle(parallel_handle, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or stale_handle for NULL parallel_handle.\n");
	rc = ompd_get_enclosing_parallel_handle(NULL, &enclosing_parallel_handle);
	if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
}

/*
		Test API: ompd_parallel_handle_compare.

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.						omp_set_num_threads(3);
		9.						#pragma omp parallel
		10.						{
		11.								printf ("Parallel level 2, thread num = %d", omp_get_thread_num());
		12.						}
		13.				}
		14.				return 0;
		15.		}

		GDB Commands:
				ompd init
				b 11
				ompdtestapi ompd_parallel_handle_compare

*/

PyObject* test_ompd_parallel_handle_compare (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_parallel_handle_compare\"...\n");

	PyObject* parallelHandlePy1 = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallel_handle1 = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy1, "ParallelHandle"));
	PyObject* parallelHandlePy2 = PyTuple_GetItem(args, 1);
	ompd_parallel_handle_t* parallel_handle2 = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy2, "ParallelHandle"));

	int cmp_value;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_parallel_handle_compare (parallel_handle1, parallel_handle2, &cmp_value);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}
	else
		printf ("Success.\n");

	if (cmp_value == 0) {
		printf ("Parallel regions are Same.\n");
	} else {
		// A value less than, equal to, or greater than 0 indicates that the region corresponding to parallel_handle_1 is,
		// respectively, less than, equal to, or greater than that corresponding to parallel_handle_2
		if (cmp_value <= 0) {
			printf ("Parallel handle 1 is lesser than handle 2, cmp_val = %d\n", cmp_value);
			printf ("Test: Changing the order.\n");
			rc = ompd_parallel_handle_compare (parallel_handle2, parallel_handle1, &cmp_value);
			if (rc != ompd_rc_ok) {
				printf ("Failed, with return code = %d\n", rc);
				return Py_None;
			}
			if (cmp_value >= 0)
				printf ("Success now cmp_value is greater, %d.\n", cmp_value);
			else
				printf ("Failed.\n");
		} else {
			printf ("Parallel 1 is greater than handle 2.\n");
			printf ("Test: Changing the order.\n");
			rc = ompd_parallel_handle_compare (parallel_handle2, parallel_handle1, &cmp_value);
			if (rc != ompd_rc_ok) {
				printf ("Failed, with return code = %d\n", rc);
				return Py_None;
			}
			if (cmp_value <= 0)
				printf ("Success now cmp_value is lesser, %d.\n", cmp_value);
			else
				printf ("Failed.\n");
			}

		// Random checks with  null and invalid args.
		/*
			ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
			ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
			ompd_rc_error:			is returned when a fatal error occurred;
		*/
		printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
		#if ABORTHANDLED
		rc = ompd_parallel_handle_compare (parallel_handle2, 0xdeadbeef, &cmp_value);
		if (rc != ompd_rc_stale_handle)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
		#else
		printf ("Skipped. Aborted, not handled.\n");
		#endif

		printf ("Test: Expecting ompd_rc_bad_input for NULL cmp_value.\n");
		#if ABORTHANDLED
		rc = ompd_parallel_handle_compare (parallel_handle2, parallel_handle1, NULL);
		if (rc != ompd_rc_bad_input)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
		#else
		printf ("Skipped. Aborted, not handled.\n");
		#endif

		printf ("Test: Expecting ompd_rc_error or stale_handle for NULL thread_handle.\n");
		rc = ompd_parallel_handle_compare (NULL, parallel_handle1, &cmp_value);
		if (rc != ompd_rc_error && rc != ompd_rc_stale_handle)
			printf ("Failed, with return code = %d\n", rc);
		else
			printf ("Success.\n");
	}

	return Py_None;
}


/*
		Test API: ompd_rel_parallel_handle

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
				ompd init
				b 7
				omptestapi ompd_rel_parallel_handle
*/

// TODO: Same as thread_rel_handle, might not be a right way to test
// What released should be provided by ompd API, this address is actually from python
PyObject* test_ompd_rel_parallel_handle (PyObject* self, PyObject* args)
{
	printf ("Testing NOT enabled for \"ompd_rel_parallel_handle\"...\n");
	printf ("Skipping.\n");
	return Py_None;

	#if 0
	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallel_handle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));

	printf ("Should be Skipped, might have to modify this function.\n");
	printf ("Test: with correct Args.\n");
	ompd_rc_t rc = ompd_rel_parallel_handle (parallel_handle);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	}else
		 printf ("Success.\n");


	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_rel_parallel_handle (0xdeadbeef);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting stale handle or bad_input for NULL thread_handle.\n");
	// Freeing NULL may not be a wise check !!!
	#if ABORTHANDLED
	rc = ompd_rel_parallel_handle (NULL);
	if ((rc != ompd_rc_bad_input) || (rc != ompd_rc_stale_handle))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	return Py_None;
	#endif

}

/*
		Test API: ompd_initialize

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
			b 4
			ompdtestapi ompd_initialize\
*/
PyObject* test_ompd_initialize (PyObject* self, PyObject* noargs)
{
	printf ("Testing \"test_ompd_initialize\"...\n");

	ompd_word_t version;
	ompd_rc_t rc = ompd_get_api_version (&version);
	if (rc != ompd_rc_ok) {
		printf ("Failed in \"ompd_get_api_version\".\n");
		return Py_None;
	}

	static ompd_callbacks_t table =
		{
			_alloc,
			_free,
			_print,
			_sizes,
			_sym_addr,
			_read,
			NULL,
			_read_string,
			_endianess,
			_endianess,
			_thread_context
		};

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t (*my_ompd_init)(ompd_word_t version, ompd_callbacks_t*) = dlsym(ompd_library, "ompd_initialize");
	rc = my_ompd_init(version, &table);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success.\n");

	static ompd_callbacks_t invalid_table =  {
			NULL, /*			_alloc, */
			NULL, /*			_free, */
			NULL, /*			_print,*/
			NULL, /*			_sizes, */
			NULL, /*			_sym_addr, */
			NULL, /*			_read,*/
			NULL,
			NULL, /*			_read_string, */
			NULL, /*			_endianess, */
			NULL, /*			_endianess, */
			NULL, /*			_thread_context */
		};

	// ompd_rc_bad_input: if invalid callbacks are provided
	printf ("Test: Invalid callbacks.\n");
	rc = my_ompd_init(version, &invalid_table);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	//ompd_rc_unsupported: if the requested API version cannot be provided
	printf ("Test: Wrong API version.\n");
	rc = my_ompd_init(150847, &table);
	if (rc != ompd_rc_unsupported)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/

	printf ("Test: Expecting ompd_rc_bad_input for NULL table.\n");
	rc = my_ompd_init(version, NULL);
	if (rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL\n");
	rc = my_ompd_init(NULL, &table);
	if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
}

/*
		Test API: ompd_get_api_version

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
			ompd init
			b 7
			ompdtestapi ompd_get_version

*/

PyObject* test_ompd_get_api_version(PyObject* self, PyObject* noargs)
{
	printf ("Testing \"ompd_get_api_version\"...\n");

	ompd_word_t version;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_api_version (&version);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success. API version is %d\n", version);

	printf ("Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL version\n");
	#if ABORTHANDLED
	rc = ompd_get_api_version (NULL);
	if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");;
	#endif


	return Py_None;
}

/*
		Test API: ompd_get_version_string

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
			ompd init
			b 7
			omptestapi ompd_get_version_string

*/

PyObject* test_ompd_get_version_string (PyObject* self, PyObject* noargs)
{
	printf ("Testing \"ompd_get_version_string\"...\n");

	const char *string;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_version_string (&string);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success. API version is %s\n", string);

	#if ABORTHANDLED
	printf ("Test: Expecting ompd_rc_error or ompd_rc_bad_input for NULL version\n");
	rc = ompd_get_version_string (NULL);
	if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
		printf ("Skipped. Aborted, not handled.\n");
	#endif

	return Py_None;
}


/*
		Test API: ompd_finalize

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
		ompd init
		b 7
		ompdtestapi ompd_finalize


		b 4
		r
		ompdtestapi ompd_finalize

*/

PyObject* test_ompd_finalize (PyObject* self, PyObject* noargs)
{
	printf ("Testing \"ompd_finalize\"...\n");

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_finalize();
	if (rc == ompd_rc_ok)
		printf ("Ret code: ompd_rc_ok, Success if ompd is initialized.\n");
	// ompd_rc_unsupported: if the OMPD library is not initialized.
	else if (rc == ompd_rc_unsupported)
		printf ("Ret code: ompd_rc_unsupported, Success if ompd is NOT initialized.\n");
	else
		 printf ("Failed: Return code is %d.\n", rc);

	return Py_None;
}

/*
		Test API: ompd_process_initialize

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:

*/

PyObject* test_ompd_process_initialize (PyObject* self, PyObject* noargs)
{

	printf ("Testing \"ompd_process_initializ\"....\n");

	ompd_address_space_handle_t* addr_handle;

	//	ompd_address_space_context_t context = {42};

	printf ("Test: with correct Args.\n");
	ompd_rc_t rc = ompd_process_initialize (&context, &addr_handle);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success.\n");

	printf ("Test: With Unsupported library.\n");
	printf ("Skipping, Have to test manually with 32 and 64 bit combination.\n");


	//	ompd_address_space_context_t invalidcontext = {99};
	printf ("Test: with wrong context value.\n");
	rc = ompd_process_initialize (&invalidcontext, &addr_handle);
	if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_incompatible))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");


	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/

	printf ("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
	rc = ompd_process_initialize (&context, NULL);
	if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	return Py_None;
}

/*
		Test API: ompd_device_initialize

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:

*/

PyObject* test_ompd_device_initialize (PyObject* self, PyObject* noargs)
{
	printf ("Testing Not enabled for \"ompd_device_initialize\".\n");
	printf ("Skipping.\n");

	return Py_None;
}


/*
		Test API: ompd_rel_address_space_handle

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:

*/
PyObject* test_ompd_rel_address_space_handle (PyObject* self, PyObject* noargs)
{
	printf ("Testing Not enabled for \"ompd_rel_address_space_handle\".\n");
	printf ("Skipping.\n");

	return Py_None;
}


/*
		Test API: ompd_get_omp_version

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
			ompd init
			b 10
			c
			ompdtestapi ompd_get_omp_version

*/
PyObject* test_ompd_get_omp_version (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_omp_version\" ...\n");

	PyObject* addrSpaceTup = PyTuple_GetItem(args, 0);
	ompd_address_space_handle_t* addr_handle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceTup, "AddressSpace");

	ompd_word_t omp_version;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_omp_version(addr_handle, &omp_version);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success. API version is %ld\n", omp_version);

	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_get_omp_version(0xdeadbeef, &omp_version);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
	rc = ompd_get_omp_version(NULL, &omp_version);
	if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	printf ("Test: Expecting ompd_rc_error or bad_input for NULL omp_version.\n");
	#if ABORTHANDLED
	rc = ompd_get_omp_version(addr_handle, NULL);
	if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	return Py_None;
}


/*
		Test API: ompd_get_omp_version_string

		program:
		1.		#include <stdio.h>
		2.		#include <omp.h>
		3.		int main () {
		4.				omp_set_num_threads(2);
		5.				#pragma omp parallel
		6.				{
		7.						printf ("Parallel level 1, thread num = %d", omp_get_thread_num());
		8.				}
		9.			   return 0;
		10.		}

		GDB Commands:
			ompd init
			b 7
			ompdtestapi ompd_get_omp_version_string

*/
PyObject* test_ompd_get_omp_version_string (PyObject* self, PyObject* args)
{
	printf ("Testing \"ompd_get_omp_version\" ...\n");

	PyObject* addrSpaceTup = PyTuple_GetItem(args, 0);
	ompd_address_space_handle_t* addr_handle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceTup, "AddressSpace");

	const char *string;

	printf ("Test: With Correct Arguments.\n");
	ompd_rc_t rc = ompd_get_omp_version_string(addr_handle, &string);
	if (rc != ompd_rc_ok) {
		printf ("Failed, with return code = %d\n", rc);
		return Py_None;
	} else
		printf ("Success. API version is %s\n", string);

	// Random checks with  null and invalid args.
	/*
		ompd_rc_stale_handle:	is returned when the specified handle is no longer valid;
		ompd_rc_bad_input:		is returned when the input parameters (other than handle) are invalid;
		ompd_rc_error:			is returned when a fatal error occurred;
	*/
	printf ("Test: Expecting stale handle for 0xdeadbeef.\n");
	#if ABORTHANDLED
	rc = ompd_get_omp_version_string(0xdeadbeef, &string);
	if (rc != ompd_rc_stale_handle)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	printf ("Test: Expecting stale handle or bad_input for NULL addr_handle.\n");
	rc = ompd_get_omp_version_string(NULL, &string);
	if ((rc != ompd_rc_bad_input) && (rc != ompd_rc_stale_handle))
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");

	#if ABORTHANDLED
	printf ("Test: Expecting ompd_rc_error or bad_input for NULL omp_version.\n");
	rc = ompd_get_omp_version_string(addr_handle, NULL);
	if (rc != ompd_rc_error && rc != ompd_rc_bad_input)
		printf ("Failed, with return code = %d\n", rc);
	else
		printf ("Success.\n");
	#else
	printf ("Skipped. Aborted, not handled.\n");
	#endif

	return Py_None;
}

