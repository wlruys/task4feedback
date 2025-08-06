#pragma once
#include <iostream>

#ifdef ENABLE_PARMETIS
#include <mpi.h>
#include <parmetis.h>
class ParMETIS_wrapper {
private:
  MPI_Comm comm; // Duplicate of the global communicator
public:
  ParMETIS_wrapper() {
    // MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    int npes = 0;
    int mype = 0;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &mype);
  }

  ParMETIS_wrapper(int comm_f) {
    // Duplicate the global communicator for sanity checking
    MPI_Comm comm_original = MPI_Comm_f2c(comm_f);
    MPI_Comm_dup(comm_original, &comm);
    int npes = 0;
    int mype = 0;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &mype);
  }

  void print_info() {
    int npes = 0;
    int mype = 0;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &mype);
    std::cout << "ParMETIS_wrapper - npes: " << npes << ", mype: " << mype << std::endl;
  }

  void callParMETIS(int32_t *vtxdist, int32_t *xadj, int32_t *adjncy, int32_t *vwgt, int32_t *vsize,
                    int32_t *adjwgt, int32_t wgtflag, int32_t numflag, int32_t ncon, float *tpwgts,
                    float *ubvec, float itr, int32_t *part) {
    int npes = 0;
    int mype = 0;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &mype);
    int32_t nparts = npes; // partition into as many parts as processors

    int32_t options[4] = {1, 0, 42, PARMETIS_PSR_COUPLED}; // verbose=0, random seed=42
    int32_t edgecut;

    int status = ParMETIS_V3_AdaptiveRepart(vtxdist, xadj, adjncy, vwgt, vsize, adjwgt, &wgtflag,
                                            &numflag, &ncon, &nparts, tpwgts, ubvec, &itr, options,
                                            &edgecut, part, &comm);

    if (status != METIS_OK) {
      std::cout << "ParMETIS error on rank " << mype << std::endl;
    }
  }
#else
class ParMETIS_wrapper {
public:
  ParMETIS_wrapper() {
    std::cerr << "[ParMETIS] error: support was disabled at compile time\n";
  }
  ParMETIS_wrapper(int /*unused*/) {
    std::cerr << "[ParMETIS] error: support was disabled at compile time\n";
  }
  static void print_info() {
    std::cerr << "[ParMETIS] error: support was disabled at compile time\n";
  }
  static void callParMETIS(...) {
    std::cerr << "[ParMETIS] error: support was disabled at compile time\n";
  }
#endif
};
