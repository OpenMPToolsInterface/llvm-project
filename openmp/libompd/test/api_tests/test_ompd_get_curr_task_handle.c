// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %S/test_ompd_get_curr_task_handle.cmd %t 2>&1 | tee %t.out | FileCheck %s


#include <stdio.h>
#include <omp.h>
int get_fib_num (int num)
{
    int t1, t2;
    if (num < 2)
        return num;
    else {
        #pragma omp task shared(t1)
        t1 = get_fib_num(num-1);
        #pragma omp task shared(t2)
        t2 = get_fib_num(num-2);
        #pragma omp taskwait
        return t1+t2;
    }
}

int main () {
    int ret = 0;
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        ret = get_fib_num(10);
    }
    printf ("Fib of 10 is %d", ret);
   return 0;
}

// CHECK-NOT: Failed
// CHECK-NOT: Skip

