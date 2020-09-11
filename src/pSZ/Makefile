all: 
	@$(MAKE) psz -j
	@echo "---------------------------------------------------------------------------------------"
	@echo "Regardless, you can try \"./psz2d r2r 1 -4 yesblk dq cesm ../../data/sample-cesm-CLDHGH\""
	@echo "---------------------------------------------------------------------------------------"

OBJ_TO_LINK := ../constants.o ../SDRB.o ../types.o ../format.o ../verify.o
ADDED_PATH  := -I..
SRC_CODE    := psz_14.hh psz_14blocked.hh psz_dualquant.hh psz_workflow.hh psz_exe.cc

psz: psz1d psz2d psz3d

psz1d: $(SRC_CODE)
	clang++ psz_exe.cc $(OBJ_TO_LINK) $(ADDED_PATH) -D_1D -o psz1d
psz2d: $(SRC_CODE)
	clang++ psz_exe.cc $(OBJ_TO_LINK) $(ADDED_PATH) -D_2D -o psz2d
psz3d: $(SRC_CODE)
	clang++ psz_exe.cc $(OBJ_TO_LINK) $(ADDED_PATH) -D_3D -o psz3d


clean:
	$(RM) psz1d psz2d psz3d
