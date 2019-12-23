# machine	= marconi|zavai|cresco6
# mpi		= intel|ch|open

PROJECT_DIR = $(CURDIR)
BIN_DIR = $(PROJECT_DIR)/bin
SRC_DIR = $(PROJECT_DIR)/src/testers
NO_WARN_UNUSED = -Wno-unused-but-set-variable -Wno-unused-variable

CC = icc

DEBUG = -g -O3
CFLAGS = -Wall $(NO_WARN_UNUSED) $(DEBUG)
FFLAGS = $(DEBUG)

SEQ_EXE = $(BIN_DIR)/compare_DGESV
PAR_EXE = $(BIN_DIR)/compare_pDGESV $(BIN_DIR)/compare_pDGESV_versions
EXE = $(SEQ_EXE) $(PAR_EXE)

ifeq ($(mpi),)
  $(error Unspecified mpi flavour, please specify 'mpi = intel|ch|open')
endif
ifeq ($(machine),)
  $(error Unspecified machine, please specify 'machine = marconi|zavai|cresco6')
endif

ifeq ($(machine),marconi) # if marconi
  ifeq ($(mpi),intel)
    # marconi intelmpi
    MPICC = mpiicc
    MPIFC = mpiifort
    MACHINEFLAGS= -I$(MKL_INC) -L$(MKL_LIB) -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_core -lmkl_sequential -lmkl_gf_lp64 -lm
  else
    ifeq ($(mpi),ch)
      # marconi mpich
      MPICC = mpicc
      MACHINEFLAGS= -I$(MKL_INC) -L$(MKL_LIB) -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_core -lmkl_sequential -lm
    else
      ifeq ($(mpi),open)
        # marconi openmpi
        MPICC = mpicc
        MACHINEFLAGS= -L${SCALAPACK_LIB} -lscalapack -L${LAPACK_LIB} -llapack -L${BLAS_LIB} -lblas -lifcore -lm
      else
        $(error Unknown mpi flavour '$(mpi)', please specify 'mpi = intel|ch|open')
      endif
    endif
  endif # end marconi
else
  ifeq ($(machine),zavai) # if zavai
    MACHINEFLAGS=
  else
    ifeq ($(machine),cresco6) # if cresco6
      MACHINEFLAGS=
    else
      $(error Unknown machine '$(machine)', please specify 'machine = marconi|zavai|cresco6')
    endif
  endif
endif

all: $(BIN_DIR) $(EXE)
.PHONY: all

$(SRC_DIR)/ScaLAPACK/%.o: $(SRC_DIR)/ScaLAPACK/%.f
	$(MPIFC) $(FFLAGS) -c $< -o $@ $(MACHINEFLAGS)
	
$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	
$(BIN_DIR)/compare_DGESV : $(SRC_DIR)/compare_DGESV.c
	$(CC) $(CFLAGS) $(SRC_DIR)/compare_DGESV.c -o $(BIN_DIR)/compare_DGESV -I$(MKL_INC) -L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm

$(BIN_DIR)/%: $(SRC_DIR)/%.c $(SRC_DIR)/*.h $(SRC_DIR)/../ft-simulated/*.h $(SRC_DIR)/ScaLAPACK/*.o $(SRC_DIR)/ScaLAPACK/*.f $(SRC_DIR)/ScaLAPACK/*.h
	$(MPICC) $(CFLAGS) $< -o $@ -L$(SRC_DIR)/ScaLAPACK $(MACHINEFLAGS)
	
clean:
	rm -f $(EXE)