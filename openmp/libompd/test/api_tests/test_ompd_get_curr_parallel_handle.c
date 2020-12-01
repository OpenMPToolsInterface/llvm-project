// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %S/test_ompd_get_curr_parallel_handle.cmd %t 2>&1 | tee %t.out | FileCheck %s
// RUN: %gdb-test -x %S/test_ompd_get_curr_parallel_handle.cmd2 %t 2>&1 | tee %t.out2 | FileCheck %s
#include <stdio.h>
#include <omp.h>

int main () {
	omp_set_num_threads(2);
	#pragma omp parallel
	{
		printf ("Parallel level 1, thread num = %d.\n", omp_get_thread_num());
	}
	return 0;
}
// CHECK-NOT: Failed
// CHECK-NOT: Skip
