
$(BIN_DIR)/run_IMe-SV-early: $(TST_DIR)/run_IMe-SV-early.c \
				$(TST_DIR)/test_IMe_pviDGESV.early.h \
				$(SRC_DIR)/pviDGESV_WO.early.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_IMe-SV-early $(TST_DIR)/run_IMe-SV-early.c $(PAR_MACHINEFLAGS)

$(BIN_DIR)/run_IMe-SV-early2: $(TST_DIR)/run_IMe-SV-early2.c \
				$(TST_DIR)/test_IMe_pviDGESV.early2.h \
				$(SRC_DIR)/pviDGESV_WO.early2.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_IMe-SV-early2 $(TST_DIR)/run_IMe-SV-early2.c $(PAR_MACHINEFLAGS)

	$(BIN_DIR)/run_IMe-SV-scalarization: $(TST_DIR)/run_IMe-SV-scalarization.c \
				$(TST_DIR)/test_IMe_pviDGESV.scalarization.h \
				$(SRC_DIR)/pviDGESV_WO.scalarization.h \
				| $(BIN_DIR)
	$(MPICC) $(CFLAGS) -lifcore -o $(BIN_DIR)/run_IMe-SV-scalarization $(TST_DIR)/run_IMe-SV-scalarization.c $(PAR_MACHINEFLAGS)