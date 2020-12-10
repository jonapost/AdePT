// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostoalakis  12 Nov 2020

#include <AdePT/BlockData.h>

#include <CopCore/SystemOfUnits.h>
#include <CopCore/PhysicalConstants.h>

#include "ConstBzFieldStepper.h"

using floatX_t = double;  //  float type for X = position
using floatE_t = double;  //  float type for E = energy  & momentum

struct SimpleTrack {
  int      index{0};
  int      pdg{0};
  floatE_t kineticEnergy{0};
  floatX_t position[3]{0};
  floatX_t direction[3]{0};
  bool     flag1;
  bool     flag2;
};

template<unsigned int N>
struct FieldPropagationBuffer
{
  int      charge[N];
  floatX_t position[3][N];
  floatE_t momentum[3][N];
  int      index[N];
  bool     active[N];
};

constexpr float ElectronMass = copcore::units::kElectronMassC2;

constexpr floatX_t  minX = -2.0 * meter, maxX = 2.0 * meter;
constexpr floatX_t  minY = -3.0 * meter, maxY = 3.0 * meter;
constexpr floatX_t  minZ = -5.0 * meter, maxZ = 5.0 * meter;

constexpr floatE_t  maxP = 1.0 * GeV;

__device__ void initOneTrack(int index,   SimpleTrack &track )
{
  float  r = curand(); 
  // track.charge = ( r < 0.45 ? -1 : ( r< 0.9 ? 0 : +1 ) );
  constexpr  int  pgdElec = 11 , pdgGamma = 22;
  track.pdg = ( r < 0.45 ? pdgElec : ( r< 0.9 ? pdgGamma : -pdgElec ) );

  track.position[0] = minX + curand() * ( maxX - minX );
  track.position[1] = minY + curand() * ( maxY - minY );
  track.position[2] = minZ + curand() * ( maxZ - minZ );

  floatE_t  px, py, pz;
  px = maxP * 2.0 * ( curand() - 0.5 );   // -maxP to +maxP
  py = maxP * 2.0 * ( curand() - 0.5 );
  pz = maxP * 2.0 * ( curand() - 0.5 );

  floatE_t  pmag2 =  px*px + py*py + pz*pz;
  floatE_t  inv_pmag = 1.0 * sqrt(pmag2);
  track.direction[0] = px * inv_pmag; 
  track.direction[1] = pY * inv_pmag; 
  track.direction[2] = pZ * inv_pmag;

  floatE_t  mass = ( pdg == pdgGamma ) ?  0.0 : kElectronMassC2 ; // rest mass
  track.kineticEnergy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
}

// this GPU kernel function is used to initialize 
//     .. the particles' state ?

__global__ void initTracks(adept::BlockData<SimpleTrack> *trackBlock,
                           adept::BlockData<floatX_t>    *stepSize
                          )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pclIdx >= maxIndex) return;

  SimpleTrack* pTrack =   trackBlock->NextElement();

  initOneTrack( pclIdx, *pTrack);
}

__global__ void initCurand(curandState_t *states)
{
  /* initialize the state */
  curand_init(0, 0, 0, states);
}


constexpr float BzValue = 0.1 * copcore::units::Tesla ; 

// VECCORE_ATT_HOST_DEVICE
__host__  __device__ 
void EvaluateField( const floatX_t position[3], float fieldValue[3] )
{
    fieldValue[0]= 0.0;
    fieldValue[1]= 0.0;
    fieldValue[2]= BzValue;        
}

// V1 -- one per warp
__global__ void moveInField(adept::BlockData<SimpleTrack> *trackBlock,
                            adept::BlockData<floatX_t>   *stepSize,
                            int maxIndex)
{
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  SimpleTrack &track= (*trackBlock)[pclIdx];

  // check if you are not outside the used block
  if (pclIdx >= maxIndex || !track.active ) return;
 
  float  step= (*stepSize)[pclIdx];

  // Charge for e+ / e-  only    ( gamma / other neutrals also ok.) 
  int    charge = (track.pdg == -11) - (track.pdg == 11);
  // double pclPosition[3]   
  // Vector3D<double> pclPosition( track.position[0], track.position[1], track.position[2] );

  // Evaluate initial field value
  // EvaluateField( pclPosition, fieldVector );

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
  // Alternative: load into local variables ?
  // float xIn= track.position[0], yIn= track.position[1], zIn = track.position[2];
  // float dirXin= track.direction[0], dirYin = track.direction[1], dirZin = track.direction[2];

  // helixBz.DoStep( ); 

  // Update position, direction
  track.position[0] = xOut;
  track.position[1] = yOut;
  track.position[2] = zOut;
  track.direction[0] = dirX;
  track.direction[1] = dirY;
  track.direction[2] = dirZ;
}

int main()
{
  // Initialize Curand
  curandState_t *state;
  cudaMalloc((void **)&state, sizeof(curandState_t));
  init<<<1, 1>>>(state);
  cudaDeviceSynchronize();

  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block
  using Block_t    = adept::BlockData<MyTrack>;
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaMallocManaged(&buffer2, blocksize);
  auto block = Block_t::MakeInstanceAt(capacity, buffer2);



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
