! Ensure argument -I works as expected with module files.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

!-----------------------------------------
! EXPECTED OUTPUT FOR MISSING MODULE FILE
!-----------------------------------------
! SINGLEINCLUDE:Error reading module file for module 'basictestmoduletwo'
! SINGLEINCLUDE-NOT:Error reading module file for module 'basictestmoduletwo'
! SINGLEINCLUDE-NOT:error: Derived type 't1' not found
! SINGLEINCLUDE:error: Derived type 't2' not found

!---------------------------------------
! EXPECTED OUTPUT FOR ALL MODULES FOUND
!---------------------------------------
! INCLUDED-NOT:Error reading module file
! INCLUDED-NOT:error: Derived type 't1' not found
! INCLUDED:error: Derived type 't2' not found

program test_dash_I_with_mod_files
    USE basictestmoduleone
    USE basictestmoduletwo
    type(t1) :: x1 ! t1 defined in Inputs/basictestmoduleone.mod
    type(t2) :: x2 ! t2 defined in Inputs/module-dir/basictestmoduleone.mod
end
