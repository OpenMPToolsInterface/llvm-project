// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %S/ompd_icvs.cmd %t 2>&1 | tee %t.out | FileCheck %s

#include <stdio.h>
#include <omp.h>
int main (void)
{
  omp_set_max_active_levels(3);
  omp_set_dynamic(0);
  omp_set_num_threads(9);
  #pragma omp parallel
  {
    omp_set_num_threads(5);
    #pragma omp parallel
    {
      #pragma omp single
      {
        /*
        * If OMP_NUM_THREADS=2,3 was set, the following should print:
        * Inner: num_thds=3
        * Inner: num_thds=3
        *
        * If nesting is not supported, the following should print:
        * Inner: num_thds=1
        * Inner: num_thds=1
        */
        printf ("Inner: num_thds=%d\n", omp_get_num_threads());
      }
    }
    #pragma omp barrier
    omp_set_max_active_levels(0);
    #pragma omp parallel
    {
      #pragma omp single
      {
        /*
        * Even if OMP_NUM_THREADS=2,3 was set, the following should
        * print, because nesting is disabled:
        * Inner: num_thds=1
        * Inner: num_thds=1
        */
        printf ("Inner: num_thds=%d\n", omp_get_num_threads());
      }
    }
    #pragma omp barrier
    #pragma omp single
    {
      /*
      * If OMP_NUM_THREADS=2,3 was set, the following should print:
      * Outer: num_thds=2
      */
      printf ("Outer: num_thds=%d\n", omp_get_num_threads());
    }
  }
  return 0;
}
// CHECK: Loaded OMPD lib successfully!

// CHECK: levels-var                      parallel                   2
// CHECK: active-levels-var               parallel                   2
// CHECK: ompd-team-size-var              parallel                   5

// CHECK: levels-var                      parallel                   2
// CHECK: active-levels-var               parallel                   1
// CHECK: ompd-team-size-var              parallel                   1

// CHECK: levels-var                      parallel                   1
// CHECK: active-levels-var               parallel                   1
// CHECK: ompd-team-size-var              parallel                   9

// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
// CHECK-NOT: No such file or directory
