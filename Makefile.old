
MACHINETYPE = cineca|enea|ubuntu
MPIFLAVOUR = intel|ch|open
LIBTYPE  = mkl|src|sys

PROJECT_DIR = $(CURDIR)
BIN_DIR = $(PROJECT_DIR)/bin
SRC_DIR = $(PROJECT_DIR)/src
TST_DIR = $(SRC_DIR)/testers
LAPACK_LIB_DIR    = $(TST_DIR)/LAPACK/lapack-3.9.0
SCALAPACK_LIB_DIR = $(TST_DIR)/ScaLAPACK/scalapack-2.1.0.mod
FTLA_LIB_DIR      = $(TST_DIR)/FTLA/ftla-rSC13.mod

OPTIMIZATION = -O3
DEBUG = -g
NO_WARN_UNUSED = -Wno-unused-but-set-variable -Wno-unused-variable
CFLAGS = $(OPTIMIZATION) $(DEBUG) -DINJECT -Wall -w3 -wd1418 -wd2259
FFLAGS = $(OPTIMIZATION) $(DEBUG)


## check machine/platform, library type and mpi flavour
ifeq ($(machine),)
  $(error Unspecified machine, please specify 'machine = $(MACHINETYPE)')
else
  ifneq ($(machine),$(filter $(machine),$(subst |, ,$(MACHINETYPE))))
    $(error Unknown machine '$(machine)', please specify 'machine = $(MACHINETYPE)')
  endif
endif

ifeq ($(library),)
  $(error Unspecified library type, please specify 'library = $(LIBTYPE)')
else
  ifneq ($(library),$(filter $(library),$(subst |, ,$(LIBTYPE))))
    $(error Unknown machine '$(library)', please specify 'library = $(LIBTYPE)')
  endif
endif

ifeq ($(mpi),)
  $(error Unspecified mpi flavour, please specify 'mpi = $(MPIFLAVOUR)')
else
  ifneq ($(mpi),$(filter $(mpi),$(subst |, ,$(MPIFLAVOUR))))
    $(error Unknown mpi flavour '$(mpi)', please specify 'mpi = $(MPIFLAVOUR)')
  endif
endif


## set compilers
ifeq ($(mpi),intel)
  MPICC = mpiicc
  MPIFC = mpiifort
endif
ifeq ($(mpi),open)
  MPICC = mpicc
  MPIFC = mpif77
endif
ifeq ($(mpi),ch)
  MPICC = mpicc
  MPIFC = mpif77
endif

## machine/platform flags
##
## how to make a smart switch: https://stackoverflow.com/questions/200205/good-way-to-do-a-switch-in-a-makefile

# marconi|galileo -> cineca
SEQ_MFLAGS_cineca_intel_mkl = -I$(MKL_INC) -L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm
PAR_MFLAGS_cineca_intel_mkl = -I$(MKL_INC) -L$(MKL_LIB) -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_gf_lp64 -lmkl_core -lmkl_sequential -lm

SEQ_MFLAGS_cineca_intel_src = $(LAPACK_LIB_DIR)/liblapack.a $(LAPACK_LIB_DIR)/librefblas.a -lm
PAR_MFLAGS_cineca_intel_src = $(SCALAPACK_LIB_DIR)/libscalapack.a $(LAPACK_LIB_DIR)/liblapack.a $(LAPACK_LIB_DIR)/librefblas.a

PAR_MFLAGS_cineca_ch_mkl    = -I$(MKL_INC) -L$(MKL_LIB) -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_core -lmkl_sequential -lm
PAR_MFLAGS_cineca_open_sys  = -L$(SCALAPACK_LIB) -lscalapack -L$(LAPACK_LIB) -llapack -L$(BLAS_LIB) -lblas -lifcore -lm
FTLAMAKEFILE_cineca_intel = Makefile.cineca.mk

# cresco6/cresco4 -> enea
SEQ_MFLAGS_enea_intel_mkl = -I$(MKLROOT)/include -L$(MKLROOT)/lib -mkl -ldl -lm
PAR_MFLAGS_enea_intel_mkl = -I$(MKLROOT)/include -L$(MKLROOT)/lib -mkl -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -ldl -lm

SEQ_MFLAGS_enea_intel_src = $(LAPACK_LIB_DIR)/liblapack.a $(LAPACK_LIB_DIR)/librefblas.a -lm
PAR_MFLAGS_enea_intel_src = $(SCALAPACK_LIB_DIR)/libscalapack.a $(LAPACK_LIB_DIR)/liblapack.a $(LAPACK_LIB_DIR)/librefblas.a
FTLAMAKEFILE_enea_intel = Makefile.enea.mk

# ubuntu
  # TODO

PAR_MACHINEFLAGS = $(PAR_MFLAGS_$(machine)_$(mpi)_$(library))
SEQ_MACHINEFLAGS = $(SEQ_MFLAGS_$(machine)_$(mpi)_$(library))
FTLAMAKEFILE     = $(FTLAMAKEFILE_$(machine)_$(mpi))

## set targets
SEQ_EXE = # compare_solve
PAR_EXE = run_IMe-SV run_IMe-SV_omp run_SPK-SV_mkl run_SPK-SV_src run_SPK-TRF_cp# run_SPK-SV_mkl_stat # compare_all_p compare_checkpointing compare_solve_p
EXE = $(addprefix $(BIN_DIR)/, $(SEQ_EXE) $(PAR_EXE) )

PAR_STD_DEP = $(SRC_DIR)/pDGEIT_WX.h $(TST_DIR)/test_*.h $(TST_DIR)/tester_head_p.c $(TST_DIR)/tester_shoulder_p.c $(TST_DIR)/tester_tail_p.c $(SRC_DIR)/helpers/*.h
SEQ_STD_DEP = $(TST_DIR)/tester_head.c $(TST_DIR)/tester_shoulder.c $(TST_DIR)/tester_tail.c $(SRC_DIR)/helpers/*.h

all: $(LAPACK_LIB_DIR)/librefblas.a \
		$(LAPACK_LIB_DIR)/liblapack.a \
		$(SCALAPACK_LIB_DIR)/libscalapack.a \
		$(FTLA_LIB_DIR)/libftla.a \
		$(EXE) \
		| $(BIN_DIR)
.PHONY: all

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LAPACK_LIB_DIR)/librefblas.a:
	cd $(LAPACK_LIB_DIR) && $(MAKE) blaslib

$(LAPACK_LIB_DIR)/liblapack.a: $(LAPACK_LIB_DIR)/librefblas.a
	cd $(LAPACK_LIB_DIR) && $(MAKE) lapacklib

# do not use "-j" flag: compilation inconsistency!
# ScaLAPACK's makefile has been modified to accept variable for pointing to the local LAPACK lib in this repository
$(SCALAPACK_LIB_DIR)/libscalapack.a: $(LAPACK_LIB_DIR)/librefblas.a $(LAPACK_LIB_DIR)/liblapack.a
	cd $(SCALAPACK_LIB_DIR) && $(MAKE) LAPACK_DIR=$(LAPACK_LIB_DIR) lib
	
$(FTLA_LIB_DIR)/libftla.a:
	cd $(FTLA_LIB_DIR) && $(MAKE) -f $(FTLAMAKEFILE)

$(TST_DIR)/ScaLAPACK/%.o: $(TST_DIR)/ScaLAPACK/%.f
	$(MPIFC) $(FFLAGS) -c $< -o $@ $(PAR_MACHINEFLAGS)

$(BIN_DIR)/run_IMe-SV: $(TST_DIR)/run_IMe-SV.c \
				$(TST_DIR)/test_IMe_pviDGESV.h \
				$(SRC_DIR)/pviDGESV_WO.h \
				$(SRC_DIR)/pDGEIT_WX.h \
				$(SRC_DIR)/pviDGESV_WO_1D.h \
				$(SRC_DIR)/pDGEIT_WX_1D.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -wd3180 -lifcore -o $(BIN_DIR)/run_IMe-SV $(TST_DIR)/run_IMe-SV.c $(PAR_MACHINEFLAGS)

$(BIN_DIR)/run_IMe-SV_omp: $(TST_DIR)/run_IMe-SV.c \
				$(TST_DIR)/test_IMe_pviDGESV.h \
				$(SRC_DIR)/pviDGESV_WO.h \
				$(SRC_DIR)/pDGEIT_WX.h \
				$(SRC_DIR)/pviDGESV_WO_1D.h \
				$(SRC_DIR)/pDGEIT_WX_1D.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -qopenmp -lifcore -o $(BIN_DIR)/run_IMe-SV_omp $(TST_DIR)/run_IMe-SV.c $(PAR_MACHINEFLAGS)
	
$(BIN_DIR)/run_SPK-SV_mkl: $(TST_DIR)/run_SPK-SV.c \
				$(TST_DIR)/test_ScaLAPACK_pDGESV.h \
				$(TST_DIR)/ScaLAPACK/ScaLAPACK_pDGESV.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_SPK-SV_mkl $(TST_DIR)/run_SPK-SV.c $(PAR_MFLAGS_$(machine)_$(mpi)_mkl)


$(BIN_DIR)/run_SPK-TRF_cp: $(TST_DIR)/run_SPK-TRF_cp.c \
				$(TST_DIR)/test_ScaLAPACK_pDGETRF_cp_ft1.h \
				$(TST_DIR)/ScaLAPACK/ScaLAPACK_pDGETRF_cp_ft1_sim.h \
				$(TST_DIR)/ScaLAPACK/pdgetrf_cp.o \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_SPK-TRF_cp $(TST_DIR)/ScaLAPACK/pdgetrf_cp.o $(TST_DIR)/run_SPK-TRF_cp.c $(PAR_MFLAGS_$(machine)_$(mpi)_mkl)
	
# static linking experiment
#
#$(BIN_DIR)/run_SPK-SV_mkl_stat: $(TST_DIR)/run_SPK-SV.c \
				$(TST_DIR)/test_ScaLAPACK_pDGESV.h \
				$(TST_DIR)/ScaLAPACK/ScaLAPACK_pDGESV.h \
				| $(BIN_DIR)
#	$(MPICC) $(CFLAGS)  -DMKL_ILP64 -I${MKLROOT}/include -lifcore -o $(BIN_DIR)/run_SPK-SV_mkl_stat $(TST_DIR)/run_SPK-SV.c $(PAR_MFLAGS_$(machine)_$(mpi)_mkl) ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm -ldl

#$(BIN_DIR)/run_SPK-SV_mkl_stat: $(TST_DIR)/run_SPK-SV.c \
				$(TST_DIR)/test_ScaLAPACK_pDGESV.h \
				$(TST_DIR)/ScaLAPACK/ScaLAPACK_pDGESV.h \
				| $(BIN_DIR)
#	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_SPK-SV_mkl_stat $(TST_DIR)/run_SPK-SV.c -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -static_mpi -Wl,--end-group -lpthread -lm
#
# => static linking: ueseless on cresco6 and galileo

$(BIN_DIR)/run_SPK-SV_src: $(TST_DIR)/run_SPK-SV.c \
				$(TST_DIR)/test_ScaLAPACK_pDGESV.h \
				$(TST_DIR)/ScaLAPACK/ScaLAPACK_pDGESV.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_SPK-SV_src $(TST_DIR)/run_SPK-SV.c $(PAR_MFLAGS_$(machine)_$(mpi)_src)

$(BIN_DIR)/compare_solve: $(TST_DIR)/compare_solve.c \
				$(TST_DIR)/test_IMe_DGESV.h \
				$(TST_DIR)/test_GaussianElimination_GE.h \
				$(TST_DIR)/test_LAPACK_DGESV.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/compare_solve $(TST_DIR)/compare_solve.c $(SEQ_MACHINEFLAGS)

$(BIN_DIR)/compare_solve_p: $(TST_DIR)/compare_solve_p.c \
				$(PAR_STD_DEP) \
				$(SRC_DIR)/pviDGESV*.h \
				$(SRC_DIR)/ft-simulated/*.h \
				$(TST_DIR)/ScaLAPACK/*.h \
				$(TST_DIR)/ScaLAPACK/pdgetrf_cp.o \
				$(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore $< -o $(BIN_DIR)/compare_solve_p $(TST_DIR)/ScaLAPACK/pdgetrf_cp.o $(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o -L$(TST_DIR)/ScaLAPACK $(PAR_MACHINEFLAGS)

$(BIN_DIR)/compare_all_p: $(TST_DIR)/compare_all_p.c \
				$(PAR_STD_DEP) \
				$(SRC_DIR)/pviDGESV*.h \
				$(SRC_DIR)/pviDGEF*.h \
				$(SRC_DIR)/ft-simulated/*.h \
				$(TST_DIR)/ScaLAPACK/*.h \
				$(TST_DIR)/ScaLAPACK/pdgetrf_cp.o \
				$(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o \
				$(TST_DIR)/FTLA/*.h \
				| $(FTLA_LIB_DIR)/libftla.a \
				$(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore $< -o $(BIN_DIR)/compare_all_p $(TST_DIR)/ScaLAPACK/pdgetrf_cp.o $(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o $(FTLA_LIB_DIR)/libftla.a $(FTLA_LIB_DIR)/helpersftla.a -L$(TST_DIR)/ScaLAPACK $(PAR_MACHINEFLAGS)

$(BIN_DIR)/compare_checkpointing: $(TST_DIR)/compare_checkpointing.c \
				$(PAR_STD_DEP) \
				$(SRC_DIR)/pviDGEF_WO.h \
				$(TST_DIR)/ScaLAPACK/*.h \
				$(TST_DIR)/ScaLAPACK/pdgetrf_cp.o \
				$(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore $< -o $(BIN_DIR)/compare_checkpointing $(TST_DIR)/ScaLAPACK/pdgetrf_cp.o $(TST_DIR)/ScaLAPACK/pdgeqrf_cp.o -L$(TST_DIR)/ScaLAPACK $(PAR_MACHINEFLAGS)

# old versions
#
#$(BIN_DIR)/compare_svxk: $(TST_DIR)/compare_svxk.c \
				$(PAR_STD_DEP) \
				$(SRC_DIR)/pviDGEF_WO.h \
				$(TST_DIR)/../*.h \
				$(TST_DIR)/*.h \
				$(TST_DIR)/ScaLAPACK/*.h \
				| $(BIN_DIR)
#	$(MPICC) $(CFLAGS) -lifcore $< -o $(BIN_DIR)/compare_svxk -L$(TST_DIR)/ScaLAPACK $(PAR_MACHINEFLAGS)

#$(BIN_DIR)/compare_pviDGEF: $(TST_DIR)/compare_pviDGEF.c \
				$(PAR_STD_DEP) \
				$(SRC_DIR)/pviDGEF*.h \
				$(TST_DIR)/../*.h \
				$(TST_DIR)/*.h \
				$(TST_DIR)/ScaLAPACK/*.h \
				| $(BIN_DIR)
#	$(MPICC) $(CFLAGS) -lifcore $< -o $(BIN_DIR)/compare_pviDGEF -L$(TST_DIR)/ScaLAPACK $(PAR_MACHINEFLAGS)

#$(BIN_DIR)/%: $(SRC_DIR)/%.c $(SRC_DIR)/*.h $(SRC_DIR)/../ft-simulated/*.h $(SRC_DIR)/ScaLAPACK/*.o $(SRC_DIR)/ScaLAPACK/*.f $(SRC_DIR)/ScaLAPACK/*.h $(SRC_DIR)/FTLA/libftla.a $(SRC_DIR)/FTLA/helpersftla.a
#	$(MPICC) $(CFLAGS)  -lifcore $< -o $@ $(SRC_DIR)/FTLA/helpersftla.a $(SRC_DIR)/FTLA/libftla.a -L$(SRC_DIR)/ScaLAPACK  $(PAR_MACHINEFLAGS)

# cleanup
#
clean:
	rm -f $(EXE)
	cd $(TST_DIR)/ScaLAPACK && rm -f *.o

clean_ftla:
	cd $(FTLA_LIB_DIR) && $(MAKE) clean

clean_lapack:
	cd $(LAPACK_LIB_DIR) && $(MAKE) clean

clean_scalapack:
	cd $(SCALAPACK_LIB_DIR) && $(MAKE) clean

clean_all: clean clean_ftla clean_lapack clean_scalapack
