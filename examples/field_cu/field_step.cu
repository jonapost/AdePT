// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostoalakis  12 Nov 2020

#include <AdePT/BlockData.h>

#include "ConstBzFieldStepper.h"

struct SimpleTrack {
  int index{0};
  int pdg{0};
  double kineticEnergy{0};
  double position[3]{0};
  double direction[3]{0};
  bool flag1;
  bool flag2;
};

template<unsigned int N>
struct FieldPropagationBuffer
{
  int    charge[N];
  float  position[3][N];
  float  momentum[3][N];
  int    index[N];
  bool   active[N];
};

constexpr float ElectronMass = 0.511;  // MeV / c^2

// VECCORE_ATT_HOST_DEVICE
__host__  __device__ 
void EvaluateField( const float position[3], float fieldValue[3] )
{

}

// this GPU kernel function is used to initialize 
//     .. the particles' state ?
__global__ void init() // curandState_t *states)
{
  /* we have to initialize the state */
  // curand_init(0, 0, 0, states);
}


constexpr float BzValue = 0.1 ; //  * Tesla ; 

// V1 -- one per warp
__global__ void moveInField(adept::BlockData<SimpleTrack> *block,
                            adept::BlockData<float>   *stepSize,
                            int maxIndex)
{
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // check if you are not outside the used block
  if (pclIdx > maxIndex) return;

  SimpleTrack &track= (*block)[particle_index];
  float  step= (*stepSize)[particle_index];
  int    charge = (track.pdg == -11) - (track.pdg == 11);
  // float pclPosition[3]   
  // Vector3D<float> pclPosition( track.position[0], track.position[1], track.position[2] );

  // Evaluate initial field value
  EvaluateField( pclPosition, fieldVector );

  // float restMass = ElectronMass;  // For now ... 
  float kinE = track.kineticEnergy;
  float momentumMag = sqrt( kinE * ( kinE + 2.0 * ElectronMass) );
  
  // Collect position, momentum
  // float momentum[3] = { momentumMag * track.direction[0], 
  //                       momentumMag * track.direction[1], 
  //                       momentumMag * track.direction[2] } ;
  // Vector3D<float> dir( track.direction[0], 
  //                      track.direction[1], 
  //                      track.direction[2] ); 

  float xOut, yOut, zOut, dirX, dirY, dirZ;

  ConstBzFieldStepper  helixBz(BzValue);

  // For now all particles ( e-, e+, gamma ) can be propagated using this
  //   for gammas  charge = 0 works, and ensures that it goes straight.
  helixBz.DoStep( track.position[0], track.position[1], track.position[2],
                  track.direction[0], track.direction[1], track.direction[2],
                  charge, momentumMag, step,
                  xOut, yOut, zOut, dirX, dirY, dirZ );
  // helixBz.DoStep( ); 

  // Update position, direction
  track.position[0] = xOut;
  track.position[1] = yOut;
  track.position[2] = zOut;
  track.direction[0] = dirX;
  track.direction[1] = dirY;
  track.direction[2] = dirZ;
}

// V2 -- a work does work of more > 1
//
// kernel function that does transportation
__global__ void transport(int n, adept::BlockData<MyTrack> *block, curandState_t *states, Queue_t *queues)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    // transport particles
    for (int xyz = 0; xyz < 3; xyz++) {
      (*block)[i].pos[xyz] = (*block)[i].pos[xyz] + (*block)[i].energy * (*block)[i].dir[xyz];
    }
  }
}
